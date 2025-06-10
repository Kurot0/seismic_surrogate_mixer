import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")                # GUI 不要
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange


# ───────────────────── サブユーティリティ ──────────────────────
# 既存 save_png, save_param_png を整理して置き換え
def save_matrix_png(arr2d: np.ndarray, path: str,
                    show_axis: bool = True):
    """
    行列を赤青カラーマップで保存
    * show_axis=True  : 目盛り＋カラーバーあり
    * show_axis=False : 軸なし（film_weights/mat 用）
    """
    vmax = np.abs(arr2d).max() + 1e-8
    plt.figure(figsize=(4, 3), dpi=150)
    im = plt.imshow(arr2d, cmap="bwr",
                    vmin=-vmax, vmax=vmax, aspect="auto",
                    interpolation="nearest")
    if show_axis:
        plt.colorbar(im)
    else:
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
# ───────────────────────────────────────────────────────────────


# ───────────────────── 基本ブロック ────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )
    def forward(self, x): return self.proj(x)


class PatchExpansion(nn.Module):
    def __init__(self, embed_dim, patch_size, out_chans, img_size_in):
        super().__init__()
        self.proj_transpose = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange("b (h w) c -> b c h w",
                      h=img_size_in[0] // patch_size[0]),
            nn.ConvTranspose2d(embed_dim, out_chans,
                               kernel_size=patch_size, stride=patch_size)
        )
    def forward(self, x): return self.proj_transpose(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features   = out_features   or in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(), nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )
    def forward(self, x): return self.model(x)


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
# ───────────────────────────────────────────────────────────────


class FilmLayer(nn.Module):
    """FiLM : γ・β を学習し features を γ*x + β で変調。
       forward は (変調後, γ, β) を返す。"""
    def __init__(self, dim):
        super().__init__()
        self.gamma_fc = nn.Linear(dim, dim)
        self.beta_fc  = nn.Linear(dim, dim)

    def forward(self, features, cond):
        gamma = self.gamma_fc(cond)
        beta  = self.beta_fc(cond)
        return gamma * features + beta, gamma.detach(), beta.detach()


class Branch(nn.Module):
    """
    1 周期 (=1 チャネル) 用のサブネット。
    ・FiLM 2 方向 (x→ug, ug→x) で **γ/β 特徴マップ** を film_weights に保存  
    ・FiLM の **線形層パラメータ** (重み・バイアス) を film_params に保存
    """
    def __init__(self, embed_dim, num_patches,
                 tokens_mlp_dim, channels_mlp_dim, latter_blocks,
                 patch_size, img_size_in, img_size_out, drop_rate,
                 branch_idx):
        super().__init__()
        self.branch_idx = branch_idx
        self.h_p = img_size_in[0] // patch_size[0]
        self.w_p = img_size_in[1] // patch_size[1]
        self.img_size_out = img_size_out                 # (H_out, W_out)

        # FiLM (方向別)
        self.film_x2ug = FilmLayer(embed_dim)   # x  → ug
        self.film_ug2x = FilmLayer(embed_dim)   # ug → x

        # 後段 Mixer 群
        self.latter = nn.Sequential(*[
            MixerBlock(embed_dim, num_patches,
                       tokens_mlp_dim, channels_mlp_dim,
                       drop=drop_rate)
            for _ in range(latter_blocks)
        ])

        # デコーダ
        self.decoder = PatchExpansion(embed_dim, patch_size, 1, img_size_in)

        # 保存ディレクトリ
        self.dir_weights = "film_weights"
        self.dir_params  = "film_params"
        ensure_dir(self.dir_weights)
        ensure_dir(self.dir_params)

        # パラメータは 1 回だけ保存
        self._params_saved = False


    # -------- γ・β 特徴マップを保存 --------
    def _save_features(self, gamma, beta, direction, samp_idx):
        """
        gamma, beta : (P , D) tensor / cpu
        保存形式
         · .pt  そのまま
         · γ/β それぞれ mean (P→H_p×W_p) & matrix (P×D) を png
        """
        g = gamma.cpu()
        b = beta.cpu()

        # ① .pt
        torch.save({'gamma': g, 'beta': b},
                   f"{self.dir_weights}/br{self.branch_idx}_{direction}_s{samp_idx}.pt")

        # ② mean → 画像 (出力解像度で)
        mean_g = g.mean(-1).reshape(self.h_p, self.w_p).numpy()
        mean_b = b.mean(-1).reshape(self.h_p, self.w_p).numpy()
        save_matrix_png(mean_g,
            f"{self.dir_weights}/br{self.branch_idx}_{direction}_gamma_mean_s{samp_idx}.png",
            show_axis=True)
        save_matrix_png(mean_b,
            f"{self.dir_weights}/br{self.branch_idx}_{direction}_beta_mean_s{samp_idx}.png",
            show_axis=True)

        # ---- mat 画像 : 軸なし・行列サイズそのまま --------------
        save_matrix_png(g.numpy(),
            f"{self.dir_weights}/br{self.branch_idx}_{direction}_gamma_mat_s{samp_idx}.png",
            show_axis=False)
        save_matrix_png(b.numpy(),
            f"{self.dir_weights}/br{self.branch_idx}_{direction}_beta_mat_s{samp_idx}.png",
            show_axis=False)


    # -------- γ・β 線形層パラメータを保存 --------
    def _save_layer_params(self, layer: FilmLayer, direction: str):
        for name, lin in [("gamma", layer.gamma_fc), ("beta", layer.beta_fc)]:
            w = lin.weight.data.cpu()
            b = lin.bias.data.cpu()

            # ---- .pt 保存はそのまま ----
            torch.save({"weight": w, "bias": b},
                    f"{self.dir_params}/br{self.branch_idx}_{direction}_{name}.pt")

            # ---- PNG 可視化：weight, bias -------------------
            save_matrix_png(w.numpy(),
                f"{self.dir_params}/br{self.branch_idx}_{direction}_{name}_weight.png",
                show_axis=True)
            save_matrix_png(b.unsqueeze(0).numpy(),
                f"{self.dir_params}/br{self.branch_idx}_{direction}_{name}_bias.png",
                show_axis=True)


    # ------------- forward -------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        # FiLM 双方向
        out1, g1, b1 = self.film_x2ug(x,   cond)   # x → ug
        out2, g2, b2 = self.film_ug2x(cond, x)     # ug → x
        x = (out1 + out2) / 2

        x = self.latter(x)
        x = self.decoder(x)                        # (B,1,H',W')

        # ───── 推論モード：保存処理 ─────
        if not self.training:

            # 1) 特徴マップ (サンプルごと)
            for s in range(g1.size(0)):
                self._save_features(g1[s], b1[s], "x2ug", s)
                self._save_features(g2[s], b2[s], "ug2x", s)

            # 2) パラメータ (初回のみ)
            if not self._params_saved:
                self._save_layer_params(self.film_x2ug, "x2ug")
                self._save_layer_params(self.film_ug2x, "ug2x")
                self._params_saved = True
        # ──────────────────────────────
        return x
# ───────────────────────────────────────────────────────────────


class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans,
                 out_chans, embed_dim, tokens_mlp_dim, channels_mlp_dim,
                 num_blocks, drop_rate, underground_channels, **kwargs):
        super().__init__()

        # ---- パッチ数 ----
        num_patches = (img_size_in[0] // patch_size[0]) * \
                      (img_size_in[1] // patch_size[1])

        # ---- 共有エンコーダ ----
        self.patch_embed   = PatchEmbed(patch_size, in_chans, embed_dim)
        self.initial_mixer = nn.Sequential(*[
            MixerBlock(embed_dim, num_patches,
                       tokens_mlp_dim, channels_mlp_dim,
                       drop=drop_rate)
            for _ in range(num_blocks // 2)
        ])

        # ---- 地下構造パス ----
        self.upsample_ug   = nn.Upsample(size=img_size_in,
                                         mode='bilinear', align_corners=True)
        self.ug_patch_embed = PatchEmbed(patch_size,
                                         len(underground_channels), embed_dim)
        self.ug_mixer       = MixerBlock(embed_dim, num_patches,
                                         tokens_mlp_dim, channels_mlp_dim)

        # ---- 周期ブランチ ----
        latter_blocks = num_blocks - num_blocks // 2
        self.branches = nn.ModuleList([
            Branch(embed_dim, num_patches,
                   tokens_mlp_dim, channels_mlp_dim, latter_blocks,
                   patch_size, img_size_in, img_size_out, drop_rate,
                   branch_idx=i)
            for i in range(out_chans)
        ])

        # ---- 出力サイズ調整 ----
        self.downsample = nn.Upsample(size=img_size_out,
                                      mode='bilinear', align_corners=True)

    # ---------------- forward ----------------
    def forward(self, x, underground):
        # 地下→パッチ列
        ug = self.upsample_ug(underground.to(x.device))
        ug = self.ug_patch_embed(ug)
        ug = self.ug_mixer(ug)                      # (B,P,D)

        # 入力→共有エンコーダ
        x = self.patch_embed(x)                     # (B,P,D)
        x = self.initial_mixer(x)                   # (B,P,D)

        # 各周期ブランチ並列実行
        outs = [br(x, ug.expand_as(x)) for br in self.branches]
        x = torch.cat(outs, dim=1)                  # (B,out_chans,H',W')

        # 出力解像度へ
        return self.downsample(x)
