# custom_fpn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from functools import partial
from mmcv.runner import BaseModule
from ..builder import BACKBONES


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)
    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.backbone = backbone

        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, *img_size))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]

        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@BACKBONES.register_module()
class ViTMultiOutput(BaseModule):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, out_indices=[3, 7, 11],
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, init_cfg=None, use_checkpoint=False,
                 hybrid_backbone=None):
        super().__init__(init_cfg=init_cfg)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.depth = depth
        self.frozen_stages = frozen_stages
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn

        if hybrid_backbone is not None:
            self.patch_embed = PatchEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            self.hybrid = True
            self.patch_shape = None
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
            self.hybrid = False
            self.patch_shape = self.patch_embed.patch_shape

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for m in self.blocks:
                m.attn.eval()
                m.norm1.eval()
                for p in m.attn.parameters():
                    p.requires_grad = False
                for p in m.norm1.parameters():
                    p.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for m in self.blocks:
                m.mlp.eval()
                m.norm2.eval()
                for p in m.mlp.parameters():
                    p.requires_grad = False
                for p in m.norm2.parameters():
                    p.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        B, _, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.shape[1], :]

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                outs.append(out)

        return outs
