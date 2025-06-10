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


# class FilmLayer(nn.Module):
#     def __init__(self, dim):
#         super(FilmLayer, self).__init__()
#         self.gamma_fc = nn.Linear(dim, dim)
#         self.beta_fc = nn.Linear(dim, dim)
        
#     def forward(self, features, conditioning_information):
#         gamma = self.gamma_fc(conditioning_information)
#         beta = self.beta_fc(conditioning_information)

#         adjusted_features = gamma * features + beta
#         return adjusted_features
    

class Branch(nn.Module):
    def __init__(self, embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, latter_blocks, patch_size, img_size_in, drop_rate):
        super().__init__()
        self.merge_fc = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
        )

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

    def forward(self, x, cond):
        merged = torch.cat(
            [
                x + cond,
                x - cond,
                x * cond,
            ],
            dim=-1,
        )
        x = self.merge_fc(merged)
        x = self.latter(x)
        x = self.decoder(x)
        return x
    

class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans, out_chans, embed_dim,
                 tokens_mlp_dim, channels_mlp_dim, num_blocks, drop_rate, underground_channels, **kwargs):
        super().__init__()
        num_patches = (img_size_in[0] // patch_size[0]) * (img_size_in[1] // patch_size[1])

        self.patch_embed   = PatchEmbed(patch_size, in_chans, embed_dim)
        self.initial_mixer = nn.Sequential(
            *[
                MixerBlock(embed_dim, num_patches,
                           tokens_mlp_dim, channels_mlp_dim,
                           drop=drop_rate)
                for _ in range(num_blocks // 2)
            ]
        )

        self.upsample_ug = nn.Upsample(size=img_size_in, mode='bilinear', align_corners=True)
        self.ug_patch_embed = PatchEmbed(patch_size, len(underground_channels), embed_dim)
        self.ug_mixer       = MixerBlock(embed_dim, num_patches,
                                         tokens_mlp_dim, channels_mlp_dim)

        latter_blocks = num_blocks - num_blocks // 2
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
            )
            for _ in range(out_chans)
        ])

        self.downsample = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    def forward(self, x, underground_data, **kwargs):
        ug = self.upsample_ug(underground_data.to(x.device))
        ug = self.ug_patch_embed(ug)
        ug = self.ug_mixer(ug)

        x = self.patch_embed(x)
        x = self.initial_mixer(x)

        branch_outputs = [
            branch(x, ug.expand_as(x))
            for branch in self.branches
        ]
        x = torch.cat(branch_outputs, dim=1)

        x = self.downsample(x)
        return x
