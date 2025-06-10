import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # GUI 不要
import matplotlib.pyplot as plt


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x):
        return self.proj(x)


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
    """FiLM : γ・β を学習し features を γ*x + β で変調"""
    def __init__(self, dim):
        super().__init__()
        self.gamma_fc = nn.Linear(dim, dim)
        self.beta_fc  = nn.Linear(dim, dim)

    def forward(self, features, conditioning_information):
        gamma = self.gamma_fc(conditioning_information)
        beta  = self.beta_fc(conditioning_information)
        return gamma * features + beta         # γ・βは返さない
    

class Branch(nn.Module):
    """
    1 周期 (=1 チャネル) 分のサブネット。
    FiLM を方向ごとに独立させ，パラメータを .pt と .png で保存。
    """
    def __init__(self, embed_dim, num_patches,
                 tokens_mlp_dim, channels_mlp_dim, latter_blocks,
                 patch_size, img_size_in, drop_rate,
                 branch_idx):
        super().__init__()
        self.branch_idx = branch_idx          # 出力チャネル番号
        self._saved = False                   # 初回だけ保存

        # —— FiLM を方向別に 2 個 —— #
        self.film_x2ug = FilmLayer(embed_dim)   # x  → ug
        self.film_ug2x = FilmLayer(embed_dim)   # ug → x

        # 後段 Mixer 群
        self.latter = nn.Sequential(
            *[MixerBlock(embed_dim, num_patches,
                         tokens_mlp_dim, channels_mlp_dim,
                         drop=drop_rate)
              for _ in range(latter_blocks)]
        )

        # パッチ列 → 画像(1ch)
        self.decoder = PatchExpansion(embed_dim, patch_size, 1, img_size_in)

    # ---------- FiLM パラメータを保存するユーティリティ ----------
    @staticmethod
    def _to_png(arr: np.ndarray, path: str):
        vmax = np.abs(arr).max() + 1e-8        # 正負対称スケール
        plt.figure(figsize=(4, 3), dpi=150)
        im = plt.imshow(arr, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _save_layer(self, layer: FilmLayer, direction: str):
        out_dir = "film_params"
        os.makedirs(out_dir, exist_ok=True)

        for name, lin in [("gamma", layer.gamma_fc), ("beta", layer.beta_fc)]:
            weight = lin.weight.data.cpu()
            bias   = lin.bias.data.cpu()

            # ① .pt 保存
            torch.save({"weight": weight, "bias": bias},
                       f"{out_dir}/branch{self.branch_idx}_{direction}_{name}.pt")

            # ② .png 保存（weight 行列と bias を別々に可視化）
            self._to_png(weight.numpy(),
                         f"{out_dir}/branch{self.branch_idx}_{direction}_{name}_weight.png")
            self._to_png(bias.unsqueeze(0).numpy(),
                         f"{out_dir}/branch{self.branch_idx}_{direction}_{name}_bias.png")
    # ----------------------------------------------------------------

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 双方向で独立 FiLM を適用
        feat1 = self.film_x2ug(x,   cond)   # x → ug
        feat2 = self.film_ug2x(cond, x)     # ug → x
        x = (feat1 + feat2) / 2

        x = self.latter(x)
        x = self.decoder(x)

        # 推論モードで初回だけパラメータ保存
        if (not self.training) and (not self._saved):
            self._save_layer(self.film_x2ug, "x2ug")
            self._save_layer(self.film_ug2x, "ug2x")
            self._saved = True

        return x
    

class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans, out_chans, embed_dim,
                 tokens_mlp_dim, channels_mlp_dim, num_blocks, drop_rate, underground_channels, **kwargs):
        super().__init__()

        # パッチ数 (P) は共有
        num_patches = (img_size_in[0] // patch_size[0]) * (img_size_in[1] // patch_size[1])

        # ───────────────── encoder (共有) ───────────────────
        self.patch_embed   = PatchEmbed(patch_size, in_chans, embed_dim)
        self.initial_mixer = nn.Sequential(
            *[
                MixerBlock(embed_dim, num_patches,
                           tokens_mlp_dim, channels_mlp_dim,
                           drop=drop_rate)
                for _ in range(num_blocks // 2)      # 前半を共有
            ]
        )

        # ───────────────── underground branch (共有) ─────────
        self.upsample_ug = nn.Upsample(size=img_size_in, mode='bilinear', align_corners=True)
        self.ug_patch_embed = PatchEmbed(patch_size, len(underground_channels), embed_dim)
        self.ug_mixer       = MixerBlock(embed_dim, num_patches,
                                         tokens_mlp_dim, channels_mlp_dim)

        # ───────────────── per‑period branches ───────────────
        latter_blocks = num_blocks - num_blocks // 2          # 後半 depth
        self.branches = nn.ModuleList([
            Branch(
                embed_dim=embed_dim,
                num_patches=num_patches,
                tokens_mlp_dim=tokens_mlp_dim,
                channels_mlp_dim=channels_mlp_dim,
                latter_blocks=latter_blocks,
                patch_size=patch_size,
                img_size_in=img_size_in,
                drop_rate=drop_rate,
                branch_idx=i,
            )
            for i in range(out_chans)                         # ← 周期＝チャネル数
        ])

        # ───────────────── 出力サイズ調整 ──────────────────
        self.downsample = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    # ----------------------------------------------------------
    def forward(self, x, underground_data):
        # 1. 地下構造をパッチ特徴へ
        ug = self.upsample_ug(underground_data.to(x.device))
        ug = self.ug_patch_embed(ug)
        ug = self.ug_mixer(ug)                               # (B,P,D)

        # 2. 入力画像を共有エンコーダ
        x = self.patch_embed(x)                              # (B,P,D)
        x = self.initial_mixer(x)                            # (B,P,D)

        # 3. 各周期ブランチを並列実行
        branch_outputs = [
            branch(x, ug.expand_as(x))                       # (B,1,H',W')
            for branch in self.branches
        ]
        x = torch.cat(branch_outputs, dim=1)                 # (B,out_chans,H',W')

        # 4. 必要ならリサイズして返す
        x = self.downsample(x)
        return x
