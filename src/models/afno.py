import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
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


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h, w, fno_blocks, fno_softshrink):
        super().__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w

        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        # self.bias = None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C).float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.softshrink)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, h, w, fno_blocks, fno_softshrink, channels_mlp_dim, drop=0.):
        super().__init__()
        self.spatial_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            AdaptiveFourierNeuralOperator(dim, h, w, fno_blocks, fno_softshrink)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, channels_mlp_dim, drop=drop)
        )
    
    def forward(self, x):
        x = x + self.spatial_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class Network(nn.Module):
    def __init__(
        self, img_size_in, img_size_out, patch_size,
        in_chans,
        out_chans,
        embed_dim,
        fno_blocks,
        fno_softshrink,
        channels_mlp_dim,
        num_blocks,
        underground_channels=None,
        drop_rate=0.,
        **kwargs
    ):
        super().__init__()
        self.img_size_in = tuple(img_size_in)

        if underground_channels is None:
            underground_channels = []
        self.underground_channels = list(underground_channels)
        total_in_chans = in_chans + len(self.underground_channels)

        h = img_size_in[0] // patch_size[0]
        w = img_size_in[1] // patch_size[1]
        num_patches = h * w

        self.patch_embed = PatchEmbed(patch_size, total_in_chans, embed_dim)

        self.mixer_layers = nn.Sequential(
            *[MixerBlock(embed_dim, num_patches, h, w, fno_blocks, fno_softshrink, channels_mlp_dim, drop=drop_rate)
              for _ in range(num_blocks)]
        )

        self.patch_expand = PatchExpansion(embed_dim, patch_size, out_chans, img_size_in)
        self.resize = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    def forward(self, x, underground_data, **kwargs):
        u = underground_data[:, self.underground_channels, :, :]
        u = F.interpolate(u, size=self.img_size_in, mode='bicubic', align_corners=True)
        u = u.repeat(x.shape[0], 1, 1, 1)

        x = torch.cat([x, u], dim=1)

        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.patch_expand(x)
        x = self.resize(x)
        return x

