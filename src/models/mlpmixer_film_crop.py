import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


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
    

class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans, out_chans, embed_dim,
                 tokens_mlp_dim, channels_mlp_dim, num_blocks, drop_rate, underground_channels, **kwargs):
        super().__init__()
        img_size_in = [img_size_in[0] // 4, img_size_in[1] // 4]
        num_patches = (img_size_in[1] // patch_size[1]) * (img_size_in[0] // patch_size[0])
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        initial_layers = [ 
            MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
            for _ in range(num_blocks//2)]
        self.initial_mixer_layers = nn.Sequential(*initial_layers)
        latter_layers = [ 
            MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
            for _ in range(num_blocks//2)]
        self.latter_mixer_layers = nn.Sequential(*latter_layers)
        self.patch_expand = PatchExpansion(embed_dim, patch_size, out_chans, img_size_in)
        self.downsample = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

        self.upsample = nn.Upsample(size=img_size_in, mode='bilinear', align_corners=True)
        self.underground_patch_embed = PatchEmbed(patch_size, len(underground_channels), embed_dim)
        self.conditioning_network = MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim)
        self.film_layer = FilmLayer(embed_dim)

        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

    def forward(self, x, underground_data):
        underground = underground_data.to(x.device)
        underground = self.upsample(underground)
        underground = self.underground_patch_embed(underground)
        underground = self.conditioning_network(underground)
        
        x = self.downsample2(x) # 画像サイズを1/4にする
        x = self.patch_embed(x)
        x = self.initial_mixer_layers(x)
        x = (self.film_layer(x, underground.expand_as(x)) + self.film_layer(underground.expand_as(x), x)) / 2
        x = self.latter_mixer_layers(x)
        x = self.patch_expand(x)
        x = self.downsample(x)
        return x
