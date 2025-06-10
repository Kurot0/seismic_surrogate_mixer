import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x):
        return self.proj(x)


class MaskedPatchEmbed(PatchEmbed):
    def __init__(self, patch_size, in_chans, embed_dim, hide_prob, mask_random, mask_by_row):
        super().__init__(patch_size, in_chans, embed_dim)
        self.hide_prob = hide_prob
        self.mask_random = mask_random
        self.mask_by_row = mask_by_row

    def forward(self, x, epoch=None):
        features = super().forward(x)
        
        # 検証時 (epoch=None) はマスクを適用しない
        if epoch is not None:
            if self.mask_by_row:
                # 行ごと（パッチ次元ごと）のマスク
                B, P, C = features.shape
                row_mask = torch.rand(B, P, device=features.device) < self.hide_prob
                mask = row_mask.unsqueeze(-1).expand(-1, -1, C)
            else:
                # ピクセルごとのマスク (要素ごとのマスク)
                mask = torch.rand_like(features) < self.hide_prob
            
            # マスク部分の置き換え
            if self.mask_random:
                # Mask elements (or rows) with random values between 0 and 1
                features = torch.where(mask, torch.rand_like(features), features)
            else:
                # Mask elements (or rows) with 0
                features = torch.where(mask, torch.zeros_like(features), features)
        
        return features


class PatchExpansion(nn.Module):
    def __init__(self, embed_dim, patch_size, out_chans, img_size_in):
        super().__init__()
        self.proj_transpose = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange("b (h w) c -> b c h w", h=img_size_in[0] // patch_size[0]),
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_size)
        )
    
    def forward(self, x):
        return self.proj_transpose(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.model(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=0.):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b p c -> b c p"),
            Mlp(num_patches, tokens_mlp_dim, drop=drop),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, channels_mlp_dim, drop=drop)
        )
    
    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class FilmLayer(nn.Module):
    def __init__(self, dim):
        super(FilmLayer, self).__init__()
        self.gamma_fc = nn.Linear(dim, dim)
        self.beta_fc = nn.Linear(dim, dim)
        
    def forward(self, features, conditioning_information):
        gamma = self.gamma_fc(conditioning_information)
        beta = self.beta_fc(conditioning_information)

        adjusted_features = gamma * features + beta
        return adjusted_features
    

class Branch(nn.Module):
    """
    1 チャンネル (=1 周期) 分のサブネット
    FiLM → latter-Mixer(s) → PatchExpansion(1ch)
    """
    def __init__(self, embed_dim, num_patches,
                 tokens_mlp_dim, channels_mlp_dim,
                 latter_blocks, patch_size, img_size_in,
                 drop_rate):
        super().__init__()

        # self.film = FilmLayer(embed_dim)
        self.film_xy = FilmLayer(embed_dim)
        self.film_yx = FilmLayer(embed_dim)

        self.latter = nn.Sequential(
            *[MixerBlock(embed_dim, num_patches,
                         tokens_mlp_dim, channels_mlp_dim,
                         drop=drop_rate)
              for _ in range(latter_blocks)]
        )

        self.decoder = PatchExpansion(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_chans=1,               # 各ブランチは 1ch 出力
            img_size_in=img_size_in,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # FiLM を対称に 2 回適用し平均
        # x = (self.film(x, cond) + self.film(cond, x)) / 2
        # x = (self.film_xy(x, cond) + self.film_yx(cond, x)) / 2
        x = self.film_xy(x, cond)
        x = self.latter(x)
        x = self.decoder(x)            # (B,1,H',W')
        return x


class Network(nn.Module):
    """
    ランダムマスク付き & 周期別ブランチ構造
    """
    def __init__(self,
                 img_size_in, img_size_out, patch_size,
                 in_chans, out_chans,
                 embed_dim, tokens_mlp_dim, channels_mlp_dim,
                 num_blocks, drop_rate,
                 underground_channels,
                 hide_prob, mask_random, mask_by_row, **kwargs):
        super().__init__()

        # パッチ数
        num_patches = (img_size_in[0] // patch_size[0]) * (img_size_in[1] // patch_size[1])

        # ────────────── 共有エンコーダ ──────────────
        self.patch_embed = MaskedPatchEmbed(
            patch_size, in_chans, embed_dim,
            hide_prob, mask_random, mask_by_row
        )

        self.initial_mixer = nn.Sequential(
            *[MixerBlock(embed_dim, num_patches,
                         tokens_mlp_dim, channels_mlp_dim,
                         drop=drop_rate)
              for _ in range(num_blocks // 2)]
        )

        # ────────────── 地下構造パス ──────────────
        self.upsample_ug = nn.Upsample(size=img_size_in, mode='bilinear', align_corners=True)
        self.ug_patch_embed = PatchEmbed(patch_size, len(underground_channels), embed_dim)
        self.ug_mixer = MixerBlock(embed_dim, num_patches,
                                   tokens_mlp_dim, channels_mlp_dim,
                                   drop=drop_rate)

        # ────────────── 周期別ブランチ ──────────────
        latter_blocks = num_blocks - num_blocks // 2
        self.branches = nn.ModuleList([
            Branch(embed_dim, num_patches,
                   tokens_mlp_dim, channels_mlp_dim,
                   latter_blocks, patch_size, img_size_in,
                   drop_rate)
            for _ in range(out_chans)
        ])

        # 出力サイズ調整
        self.downsample = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    # ----------------------------------------------------------
    def forward(self, x, underground_data, epoch: int | None = None):
        """
        Parameters
        ----------
        x : (B, in_chans, H_in, W_in)
        underground_data : (B, |underground_channels|, H_ug, W_ug)
        epoch : int | None
            学習中は int を渡し，検証/推論時は None
        """
        # --- 地下構造特徴 ---
        ug = self.upsample_ug(underground_data.to(x.device))
        ug = self.ug_patch_embed(ug)          # (B,P,D)
        ug = self.ug_mixer(ug)

        # --- 入力特徴 (マスクあり/なし) ---
        if epoch is None:
            x = self.patch_embed(x)           # 検証・推論 (マスク無効)
        else:
            x = self.patch_embed(x, epoch)    # 学習 (マスク適用)

        x = self.initial_mixer(x)             # (B,P,D)

        # --- 周期ごとに並列処理 ---
        outs = [br(x, ug.expand_as(x)) for br in self.branches]  # list[(B,1,H',W')]
        x = torch.cat(outs, dim=1)            # (B,out_chans,H',W')

        x = self.downsample(x)                # 指定解像度へ
        return x
