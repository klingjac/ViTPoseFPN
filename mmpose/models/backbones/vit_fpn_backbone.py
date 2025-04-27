import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from functools import partial
from timm.models.layers import trunc_normal_
from ..builder import BACKBONES


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@BACKBONES.register_module()
class ViTFPNBackbone(BaseModule):
    def __init__(self, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], depths=[3, 4, 6, 3], init_cfg=None, frozen_stages=-1):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.frozen_stages = frozen_stages

        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()

        in_channels = 3
        for i in range(len(embed_dims)):
            patch_embed = PatchEmbed(in_channels, embed_dims[i], stride=4 if i == 0 else 2)
            blocks = nn.Sequential(*[Block(embed_dims[i], num_heads[i], mlp_ratios[i]) for _ in range(depths[i])])
            self.patch_embeds.append(patch_embed)
            self.stages.append(blocks)
            in_channels = embed_dims[i]

        self._freeze_stages()
        self.init_weights()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for stage_idx in range(self.frozen_stages + 1):
                self.patch_embeds[stage_idx].eval()
                for param in self.patch_embeds[stage_idx].parameters():
                    param.requires_grad = False
                self.stages[stage_idx].eval()
                for param in self.stages[stage_idx].parameters():
                    param.requires_grad = False

    def forward(self, x):
        outs = []
        for patch_embed, stage in zip(self.patch_embeds, self.stages):
            x, H, W = patch_embed(x)
            x = stage(x)
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(x)
        return outs
