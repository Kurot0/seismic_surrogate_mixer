import copy, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange


class WNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, wnorm=False):
        super().__init__(in_features, out_features, bias=bias)
        if wnorm:
            weight_norm(self)
        self._patch_deepcopy()
    def _patch_deepcopy(self):
        orig = getattr(self, "__deepcopy__", None)
        def _dcp(self, memo):
            tmp = {n: getattr(self, n) for h in self._forward_pre_hooks.values()
                   if h.__class__.__name__=="_WeightNorm" and hasattr(self, h.name)
                   for n in [h.name]}
            for n in tmp: delattr(self, n)
            if orig: self.__deepcopy__ = orig
            else: del self.__deepcopy__
            res = copy.deepcopy(self)
            for k,v in tmp.items(): setattr(self, k, v)
            self.__deepcopy__ = _dcp.__get__(self, self.__class__)
            return res
        self.__deepcopy__ = _dcp.__get__(self, self.__class__)

class MLP(nn.Module):
    def __init__(self, dim, factor=2, dropout=0.0):
        super().__init__()
        hid = dim*factor
        self.seq = nn.Sequential(
            nn.Linear(dim,hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid,dim), nn.Dropout(dropout))
    def forward(self,x): return self.seq(x)

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, factor):
        super().__init__()
        self.in_dim,self.out_dim = in_dim,out_dim
        self.modes_x,self.modes_y = modes_x,modes_y
        weight_y = torch.empty(in_dim, out_dim, modes_y, 2)
        nn.init.xavier_normal_(weight_y)
        self.weight_y = nn.Parameter(weight_y)
        weight_x = torch.empty(in_dim, out_dim, modes_x, 2)
        nn.init.xavier_normal_(weight_x)
        self.weight_x = nn.Parameter(weight_x)
        self.backcast = MLP(out_dim, factor)
    def complex_weight(self,w): 
        return torch.view_as_complex(w)
    def forward_fourier(self,x):
        B,C,H,W = x.shape
        x_ft = torch.fft.rfft(x,dim=-1,norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        out_ft[...,:self.modes_y] = torch.einsum(
            "b i h k, i o k -> b o h k",
            x_ft[...,:self.modes_y], self.complex_weight(self.weight_y))
        y_out = torch.fft.irfft(out_ft,n=W,dim=-1,norm="ortho")
        x_ft = torch.fft.rfft(x,dim=-2,norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        out_ft[:,:,:self.modes_x,:] = torch.einsum(
            "b i h k, i o h -> b o h k",
            x_ft[:,:,:self.modes_x,:], self.complex_weight(self.weight_x))
        x_out = torch.fft.irfft(out_ft,n=H,dim=-2,norm="ortho")
        return x_out + y_out
    def forward(self,x):
        x = rearrange(x,"b h w c -> b c h w")
        x = self.forward_fourier(x)
        x = rearrange(x,"b c h w -> b h w c")
        return self.backcast(x)

class ElementwiseFusion(nn.Module):
    def forward(self,a,g): return torch.cat((a+g, a-g, a*g), dim=-1)

class ChannelWiseProjection(nn.Module):
    def __init__(self, in_dim, out_chans, wnorm=False):
        super().__init__()
        self.projs = nn.ModuleList([WNLinear(in_dim,1,wnorm=wnorm) for _ in range(out_chans)])
    def forward(self,x):
        out = torch.cat([p(x) for p in self.projs], dim=-1)
        return rearrange(out,"b h w c -> b c h w")

class Network(nn.Module):
    def __init__(
        self, modes_x, modes_y, embed_dim,
        img_size_out,
        img_size_train=None,
        in_chans=1, out_chans=1,
        underground_channels=None,
        n_layers=8, factor=1,
        ff_weight_norm=False, **kwargs):
        super().__init__()
        assert n_layers%2==0, "n_layers must be even"
        self.img_size_out  = tuple(img_size_out)
        self.img_size_train= tuple(img_size_train) if img_size_train else self.img_size_out
        self.padding = 8
        self.half_layers = n_layers//2
        self.embed_dim = embed_dim
        self.underground_dim = len(underground_channels or [])

        self.in_proj_src = WNLinear(in_chans+2, embed_dim, wnorm=ff_weight_norm)
        self.in_proj_geo = WNLinear(self.underground_dim+2, embed_dim, wnorm=ff_weight_norm)

        self.front_layers_a = nn.ModuleList(
            [SpectralConv2d(embed_dim, embed_dim, modes_x, modes_y, factor)
             for _ in range(self.half_layers)])
        self.front_layers_g = nn.ModuleList(
            [SpectralConv2d(embed_dim, embed_dim, modes_x, modes_y, factor)
             for _ in range(self.half_layers)])

        self.fusion = ElementwiseFusion()
        self.fusion_linear = WNLinear(embed_dim * 3, embed_dim, wnorm=ff_weight_norm)

        self.back_layers = nn.ModuleList(
            [SpectralConv2d(embed_dim, embed_dim, modes_x, modes_y, factor)
             for _ in range(self.half_layers)])

        self.projection = ChannelWiseProjection(embed_dim, out_chans, wnorm=ff_weight_norm)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True, antialias=True)

    def _get_grid(self, shape, device):
        B,X,Y = shape[0], shape[1], shape[2]
        gx = torch.linspace(0,1,X,device=device)[None,:,None,None].repeat(B,1,Y,1)
        gy = torch.linspace(0,1,Y,device=device)[None,None,:,None].repeat(B,X,1,1)
        return torch.cat((gx,gy), dim=-1)

    def _encode(self, x, proj, target_size):
        if x.shape[-2:] != target_size:
            x = self._resize(x, target_size)
        x = x.permute(0,2,3,1)
        grid = self._get_grid(x.shape, x.device)
        x = proj(torch.cat((x,grid), dim=-1))
        x = x.permute(0,3,1,2)
        x = F.pad(x, [0,self.padding,0,self.padding])
        return x.permute(0,2,3,1)

    def forward(self, x, underground, *, train_flag: bool=False):
        work_size = self.img_size_train if train_flag else self.img_size_out

        g = underground.expand(x.size(0), -1, -1, -1)
        a = self._encode(x, self.in_proj_src,  work_size)
        g = self._encode(g, self.in_proj_geo, work_size)

        for layer_a, layer_g in zip(self.front_layers_a, self.front_layers_g):
            a = a + layer_a(a)
            g = g + layer_g(g)

        z = self.fusion(a, g)
        z = self.fusion_linear(z)

        for layer in self.back_layers:
            z = z + layer(z)

        z = z[..., :-self.padding, :-self.padding, :]

        out = self.projection(z)

        if train_flag and self.img_size_train != self.img_size_out:
            out = self._resize(out, self.img_size_out)

        return out
