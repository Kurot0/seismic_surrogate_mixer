import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim, factor=2, dropout=0.0):
        super().__init__()
        hid = dim * factor
        self.seq = nn.Sequential(
            nn.Linear(dim, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.seq(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, mlp_factor=2):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.modes_x, self.modes_y = modes_x, modes_y

        self.weight_y = nn.Parameter(torch.randn(in_dim, out_dim, modes_y, 2) * 0.02)
        self.weight_x = nn.Parameter(torch.randn(in_dim, out_dim, modes_x, 2) * 0.02)

        self.backcast = MLP(out_dim, mlp_factor)

    @staticmethod
    def complex_weight(w):
        return torch.view_as_complex(w)

    def forward_fourier(self, x):
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., : self.modes_y] = torch.einsum(
            "b i h k, i o k -> b o h k", x_ft[..., : self.modes_y], self.complex_weight(self.weight_y)
        )
        y_out = torch.fft.irfft(out_ft, n=W, dim=-1, norm="ortho")

        x_ft = torch.fft.rfft(x, dim=-2, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, : self.modes_x, :] = torch.einsum(
            "b i h k, i o h -> b o h k", x_ft[:, :, : self.modes_x, :], self.complex_weight(self.weight_x)
        )
        x_out = torch.fft.irfft(out_ft, n=H, dim=-2, norm="ortho")
        return x_out + y_out

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        x = self.forward_fourier(x)
        x = rearrange(x, "b c h w -> b h w c")
        return self.backcast(x)


class ElementwiseFusion(nn.Module):
    def forward(self, a, g):
        return torch.cat((a + g, a - g, a * g), dim=-1)


class ChannelWiseProjection(nn.Module):
    def __init__(self, in_dim, out_chans):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(out_chans)
        ])

    def forward(self, x):
        outs = [proj(x) for proj in self.projs]
        out = torch.cat(outs, dim=-1)
        return rearrange(out, "b h w c -> b c h w")


class Network(nn.Module):
    def __init__(self, modes_x, modes_y, width, img_size_out, input_dim, underground_channels, n_layers, mlp_factor, out_chans, **kwargs):
        super().__init__()
        assert n_layers % 2 == 0
        self.img_size_out = tuple(img_size_out)
        self.padding = 8
        self.half_layers = n_layers // 2
        self.width = width
        self.underground_dim = len(underground_channels)

        self.in_proj_src = nn.Linear(input_dim + 2, width)
        self.in_proj_geo = nn.Linear(self.underground_dim + 2, width)

        self.front_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_x, modes_y, mlp_factor) for _ in range(self.half_layers)
        ])

        self.fusion = ElementwiseFusion()
        self.post_fusion_fc = nn.Linear(width * 3, width)

        self.back_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_x, modes_y, mlp_factor) for _ in range(self.half_layers)
        ])

        self.projection = ChannelWiseProjection(width, out_chans)

    def _get_grid(self, shape, device):
        B, X, Y = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, X, device=device)[None, :, None, None]
        gy = torch.linspace(0, 1, Y, device=device)[None, None, :, None]
        gx = gx.repeat(B, 1, Y, 1)
        gy = gy.repeat(B, X, 1, 1)
        return torch.cat((gx, gy), dim=-1)

    def _encode(self, x, proj, resize: bool):
        if resize:
            x = F.interpolate(x, size=self.img_size_out, mode="bicubic", align_corners=True)
        x = x.permute(0, 2, 3, 1)
        grid = self._get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = proj(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x, underground, **kwargs):
        g = underground.expand(x.shape[0], -1, -1, -1)
        a = self._encode(x, self.in_proj_src, resize=True)
        g = self._encode(g, self.in_proj_geo, resize=False)

        for layer in self.front_layers:
            a = a + layer(a)
            g = g + layer(g)

        z = self.fusion(a, g)
        z = self.post_fusion_fc(z)

        for layer in self.back_layers:
            z = z + layer(z)
        z = z[..., :-self.padding, :-self.padding, :]

        return self.projection(z)
