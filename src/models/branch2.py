import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# 既存 import に追加
import os
import numpy as np
import matplotlib.pyplot as plt        # ★ カラーマップ保存用
from PIL import Image                   # PIL はもう不要になった場合は削除しても可



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
    def __init__(self, dim):
        super().__init__()
        self.gamma_fc = nn.Linear(dim, dim)
        self.beta_fc  = nn.Linear(dim, dim)

    # γ・β も返す
    def forward(self, features, cond):
        gamma = self.gamma_fc(cond)
        beta  = self.beta_fc(cond)
        out   = gamma * features + beta
        return out, gamma.detach(), beta.detach()
    

class Branch(nn.Module):
    def __init__(self, embed_dim, num_patches, tokens_mlp_dim,
                 channels_mlp_dim, latter_blocks, patch_size, img_size_in,
                 drop_rate, img_size_out, branch_idx):
        super().__init__()
        self.branch_idx = branch_idx
        self.h_p = img_size_in[0] // patch_size[0]
        self.w_p = img_size_in[1] // patch_size[1]
        self.img_size_out = img_size_out

        # ★ 方向別に FiLM を 2 個持つ
        self.film_x2ug = FilmLayer(embed_dim)   # 震源→地下
        self.film_ug2x = FilmLayer(embed_dim)   # 地下→震源

        self.latter = nn.Sequential(*[
            MixerBlock(embed_dim, num_patches,
                       tokens_mlp_dim, channels_mlp_dim,
                       drop=drop_rate)
            for _ in range(latter_blocks)
        ])
        self.decoder = PatchExpansion(embed_dim, patch_size, 1, img_size_in)

        # 保存ディレクトリ
        self.save_dir = "film_weights"
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- 保存用ヘルパ ----------
    def _save_pt(self, g, b, tag, samp):
        torch.save({'gamma': g, 'beta': b},
                   f"{self.save_dir}/br{self.branch_idx}_{tag}_s{samp}.pt")

    def _save_heatmap(self, arr2d: torch.Tensor, fname: str,
                    target_size: tuple | None = None):
        """
        arr2d : 2-D numpy / tensor  (H , W)
        target_size : (H_tgt , W_tgt)  → そのピクセル数で保存
                    None            → arr.shape のまま保存
        """
        arr2d = arr2d.cpu().numpy()
        h_src, w_src = arr2d.shape
        if target_size is None:
            h_tgt, w_tgt = h_src, w_src
        else:
            h_tgt, w_tgt = target_size                    # 例: (400,400)

        dpi = 100                                         # 固定
        plt.figure(figsize=(w_tgt / dpi, h_tgt / dpi), dpi=dpi)
        plt.axis("off")
        plt.imshow(arr2d, cmap="bwr", aspect="auto", interpolation="nearest")
        plt.tight_layout(pad=0)
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()
    # ----------------------------------

    def forward(self, x, cond):
        # --- 双方向 FiLM ---
        x2ug, g1, b1 = self.film_x2ug(x,   cond)
        ug2x, g2, b2 = self.film_ug2x(cond, x)
        x = (x2ug + ug2x) / 2
        x = self.latter(x)
        x = self.decoder(x)           # (B,1,H',W')

        # -------- 予測モードで保存 --------
        if not self.training:
            for n,(g,b,tag) in enumerate([(g1,b1,"x2ug"), (g2,b2,"ug2x")]):
                for s in range(g.size(0)):           # バッチの全サンプル
                    gs, bs = g[s].cpu(), b[s].cpu()  # (P,D)
                    # 1) .pt
                    self._save_pt(gs, bs, tag, s)
                    # 2‑a) パッチ平均 → (H_p,W_p)
                    mean_g = gs.mean(-1).reshape(self.h_p, self.w_p)
                    mean_b = bs.mean(-1).reshape(self.h_p, self.w_p)
                    # ── mean (パッチ平均) は 出力画像サイズで保存 ───────────
                    self._save_heatmap(mean_g,
                        f"{self.save_dir}/br{self.branch_idx}_{tag}_gamma_mean_s{s}.png",
                        target_size=self.img_size_out)      # ← 400×400 等
                    self._save_heatmap(mean_b,
                        f"{self.save_dir}/br{self.branch_idx}_{tag}_beta_mean_s{s}.png",
                        target_size=self.img_size_out)

                    # ── mat (元の P×D) は そのままの行列サイズで保存 ────────
                    self._save_heatmap(gs,
                        f"{self.save_dir}/br{self.branch_idx}_{tag}_gamma_mat_s{s}.png")   # target_size=None → 2520×64
                    self._save_heatmap(bs,
                        f"{self.save_dir}/br{self.branch_idx}_{tag}_beta_mat_s{s}.png")
        # ----------------------------------
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
            Branch(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim,
                latter_blocks, patch_size, img_size_in, drop_rate, img_size_out,
                branch_idx=i)                       # ★ 番号付与
            for i in range(out_chans)
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
