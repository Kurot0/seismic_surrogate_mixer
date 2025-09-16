import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


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
        super(FilmLayer, self).__init__()
        self.gamma_fc = nn.Linear(dim, dim)
        self.beta_fc = nn.Linear(dim, dim)
        
    def forward(self, features, conditioning_information):
        gamma = self.gamma_fc(conditioning_information)
        beta = self.beta_fc(conditioning_information)

        adjusted_features = gamma * features + beta
        return adjusted_features
    

class Branch(nn.Module):
    def __init__(self, embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, latter_blocks, patch_size, img_size_in, drop_rate):
        super().__init__()
        self.film_xy = FilmLayer(embed_dim)
        self.film_yx = FilmLayer(embed_dim)

        self.latter = nn.Sequential(
            *[
                MixerBlock(embed_dim, num_patches,
                           tokens_mlp_dim, channels_mlp_dim,
                           drop=drop_rate)
                for _ in range(latter_blocks)
            ]
        )

        self.decoder = PatchExpansion(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_chans=1,
            img_size_in=img_size_in,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = (self.film_xy(x, cond) + self.film_yx(cond, x)) / 2
        x = self.latter(x)
        x = self.decoder(x)
        return x
    

class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans, out_chans, embed_dim,
                 tokens_mlp_dim, channels_mlp_dim, num_blocks, drop_rate, underground_channels, **kwargs):
        super().__init__()
        num_patches = (img_size_in[0] // patch_size[0]) * (img_size_in[1] // patch_size[1])
        self.img_size_in = img_size_in
        self.out_chans = out_chans
        self.use_streams_inference = True
        self.use_streams_training  = True
        self._streams = None
        self._streams_device = None

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.initial_mixer = nn.Sequential(
            *[
                MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
                for _ in range(num_blocks // 2)
            ]
        )

        self.upsample_ug    = nn.Upsample(size=img_size_in, mode='bilinear', align_corners=True)
        self.ug_patch_embed = PatchEmbed(patch_size, len(underground_channels), embed_dim)
        self.ug_mixer       = MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim)

        latter_blocks = num_blocks - num_blocks // 2
        self.branches = nn.ModuleList([
            Branch(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim,
                   latter_blocks, patch_size, img_size_in, drop_rate)
            for _ in range(out_chans)
        ])

        self.downsample = nn.Upsample(size=img_size_out, mode='bicubic', align_corners=True)

    def _ensure_streams(self, device):
        dev = device if isinstance(device, torch.device) else torch.device(device)
        need_new = False
        if self._streams is None:
            need_new = True
        elif len(self._streams) != self.out_chans:
            need_new = True
        elif self._streams_device != dev:
            need_new = True
        if need_new:
            self._streams = [torch.cuda.Stream(device=dev) for _ in range(self.out_chans)]
            self._streams_device = dev
        return self._streams

    def forward(self, x, underground_data, **kwargs):
        x = x.contiguous(memory_format=torch.channels_last)

        ug = self.upsample_ug(underground_data.to(x.device).contiguous(memory_format=torch.channels_last))
        ug = self.ug_patch_embed(ug)
        ug = self.ug_mixer(ug)

        x = self.patch_embed(x)
        x = self.initial_mixer(x)

        ug_exp = ug.expand_as(x)

        B = x.size(0)
        Hin, Win = self.img_size_in
        device, dtype = x.device, x.dtype
        cuda_ok = torch.cuda.is_available() and x.is_cuda

        if (not self.training) and self.use_streams_inference and cuda_ok:
            x_buf = torch.empty(
                (B, self.out_chans, Hin, Win),
                device=device, dtype=dtype
            ).contiguous(memory_format=torch.channels_last)

            streams = self._ensure_streams(device)
            cur = torch.cuda.current_stream(device=device)
            for i, (br, st) in enumerate(zip(self.branches, streams)):
                st.wait_stream(cur)
                with torch.cuda.stream(st):
                    xi = br(x, ug_exp)
                    xi = xi.to(memory_format=torch.channels_last)
                    x_buf.narrow(1, i, 1).copy_(xi, non_blocking=True)
            for st in streams:
                cur.wait_stream(st)

            x = self.downsample(x_buf)
            return x

        if self.training and self.use_streams_training and cuda_ok:
            streams = self._ensure_streams(device)
            cur = torch.cuda.current_stream(device=device)

            outs = [None] * self.out_chans
            for i, (br, st) in enumerate(zip(self.branches, streams)):
                st.wait_stream(cur)
                with torch.cuda.stream(st):
                    outs[i] = br(x, ug_exp)
            for st in streams:
                cur.wait_stream(st)

            x = torch.cat(outs, dim=1)
            x = x.contiguous(memory_format=torch.channels_last)
            x = self.downsample(x)
            return x

        outs = []
        for br in self.branches:
            outs.append(br(x, ug_exp))
        x = torch.cat(outs, dim=1)
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.downsample(x)
        return x
