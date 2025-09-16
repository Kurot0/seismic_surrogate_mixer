import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1 = None, modes2 = None):
        super(SpectralConv2d_Uno, self).__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))

    def compl_mul2d(self, input, weights):
        a, b = input.real, input.imag
        c, d = weights.real, weights.imag
        ac = torch.einsum("bixy,ioxy->boxy", a, c)
        bd = torch.einsum("bixy,ioxy->boxy", b, d)
        ad = torch.einsum("bixy,ioxy->boxy", a, d)
        bc = torch.einsum("bixy,ioxy->boxy", b, c)
        real = ac - bd
        imag = ad + bc
        return torch.complex(real, imag)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2),norm = 'forward')
        return x


class pointwise_op_2D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op_2D,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True, antialias=True)
        return x_out


class OperatorBlock_2D(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1, modes2, Normalize = False, Non_Lin = True):
        super(OperatorBlock_2D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)

    def forward(self, x, dim1 = None, dim2 = None):
        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class Network(nn.Module):
    def __init__(self, in_chans, out_chans, embed_dim, img_size_out, img_size_train, underground_channels, pad = 0, **kwargs):
        super().__init__()
        if underground_channels is None:
            underground_channels = []
        self.underground_channels = underground_channels

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.img_size_out = tuple(img_size_out)
        self.padding = pad

        fc_in_ch = self.in_chans + len(self.underground_channels)+ 4 
        self.fc = nn.Linear(fc_in_ch, self.embed_dim)

        self.img_size_out = tuple(img_size_out)
        self.img_size_train = tuple(img_size_train or img_size_out)
        
        H, W = self.img_size_out

        def _modes(d): return max(d//8, 1)

        self.L0 = OperatorBlock_2D(self.embed_dim, self.embed_dim//2, H//2, W//2, _modes(H//2), _modes(W//2))
        self.L1 = OperatorBlock_2D(self.embed_dim//2, self.embed_dim//4, H//4, W//4, _modes(H//4), _modes(W//4))
        self.L2 = OperatorBlock_2D(self.embed_dim//4, self.embed_dim//8, H//8, W//8, _modes(H//8), _modes(W//8))
        self.L3 = OperatorBlock_2D(self.embed_dim//8, self.embed_dim//16, H//16, W//16, _modes(H//16), _modes(W//16))

        self.L4 = OperatorBlock_2D(self.embed_dim//16, self.embed_dim//8, H//8, W//8, _modes(H//8), _modes(W//8))
        self.L5 = OperatorBlock_2D(self.embed_dim//4, self.embed_dim//4, H//4, W//4, _modes(H//4), _modes(W//4))
        self.L6 = OperatorBlock_2D(self.embed_dim//2, self.embed_dim//2, H//2, W//2, _modes(H//2), _modes(W//2))
        self.L7 = OperatorBlock_2D(self.embed_dim, self.embed_dim, H, W, _modes(H), _modes(W))

        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 1) for _ in range(out_chans)])

    @staticmethod
    def _get_grid(shape, device):
        B, H, W = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 2 * np.pi, H, dtype=torch.float, device=device).view(1, H, 1, 1).repeat(B, 1, W, 1)
        gridy = torch.linspace(0, 2 * np.pi, W, dtype=torch.float, device=device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        return torch.cat((torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)), dim=-1)
    
    @staticmethod
    def _down_up(x, size, mode='bicubic'):
        return F.interpolate(x, size=size, mode=mode, align_corners=True, antialias=True)

    def _operator_path(self, x, H, W):
        h2, w2 = H//2,  W//2
        h4, w4 = H//4,  W//4
        h8, w8 = H//8,  W//8
        h16,w16 = H//16, W//16

        x0 = self.L0(x, h2, w2)
        x1 = self.L1(x0, h4, w4)
        x2 = self.L2(x1, h8, w8)
        x3 = self.L3(x2, h16, w16)

        x4 = self.L4(x3, h8, w8)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.L5(x4, h4, w4)
        x5 = torch.cat([x5, x1], dim=1)
        x6 = self.L6(x5, h2, w2)
        x6 = torch.cat([x6, x0], dim=1)
        x7 = self.L7(x6, H, W)
        return x7

    def forward(self, x, underground_data, train_flag: bool = False):
        x = self._down_up(x, self.img_size_out)

        x = torch.cat([x, underground_data.repeat(x.size(0), 1, 1, 1)], dim=1)

        if train_flag and self.img_size_train != self.img_size_out:
            x = self._down_up(x, self.img_size_train)

        x = x.permute(0, 2, 3, 1)
        x = torch.cat([x, self._get_grid(x.shape, x.device)], dim=-1)
        x = self.fc(x).permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [self.padding]*4)

        H, W = x.shape[-2:]
        x = self._operator_path(x, H, W)
        if self.padding:
            x = x[..., self.padding:-self.padding, self.padding:-self.padding]

        x = x.permute(0, 2, 3, 1)
        outs = [fc(x) for fc in self.fc1]
        x = torch.cat(outs, dim=-1)
        x = x.permute(0, 3, 1, 2) 

        if train_flag and self.img_size_train != self.img_size_out:
            x = self._down_up(x, self.img_size_out)

        return x
