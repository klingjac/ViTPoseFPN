# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone FPN‑style fusion neck for ViT multi‑level outputs.

This removes the fusion logic from the head and makes it an independent neck
module that can be placed between *any* backbone (e.g. ``ViTMultiOutput``)
and *any* head that expects a **single** high‑resolution feature map (e.g.
``TopdownHeatmapSimpleHead``).

Example in a MM‑Pose config
---------------------------
```
model = dict(
    type='TopDown',
    backbone=dict(
        type='ViTMultiOutput',
        ...,
        out_indices=[9,10,11],   # three ViT blocks
    ),
    neck=dict(
        type='ViTFusionFPN',      # <‑‑ this file!
        in_channels=[768, 768, 768],
        fpn_strides=[4, 2, 1],    # coarse → … → fine (last stride = 1)
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,          # because neck outputs 768‑ch feature
        ...
    ),
)
```
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, constant_init, normal_init
from mmcv.runner import BaseModule
from mmpose.models.utils.ops import resize
from ..builder import NECKS


@NECKS.register_module()
class ViTFusionFPN(BaseModule):
    """FPN‑like feature fusion neck with optional user‑defined strides.

    Parameters
    ----------
    in_channels : Sequence[int]
        Channel dimension of each input feature level (must all be equal).
    fpn_strides : Sequence[int] | None, default=None
        Spatial down‑sampling strides applied *before* up‑resizing.  Provide one
        per level, ordered **coarse → fine**.  If ``None`` the default is
        ``[3**i for i in reversed(range(n_levels))]``.
    out_channels : int, default=None
        If set, a 1×1 conv will be appended to change the channel dimension of
        the fused map; otherwise we keep the same channels as the inputs.
    align_corners : bool, default=False
        Argument passed to ``resize``.
    """

    def __init__(
        self,
        in_channels,
        *,
        fpn_strides: tuple[int, ...] | None = None,
        out_channels: int | None = None,
        align_corners: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if not isinstance(in_channels, (list, tuple)):
            raise TypeError("in_channels must be a list/tuple when using multi‑level FPN")
        if len(set(in_channels)) > 1:
            raise ValueError("All input levels must have the same channel dimension")

        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.base_channels = in_channels[0]
        self.align_corners = align_corners

        # 1×1 projection for each level (keeps channel count the same)
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, self.base_channels, kernel_size=1)
            for c in in_channels
        ])

        # stride‑specific 3×3 convs (down‑sample)
        if fpn_strides is None:
            fpn_strides = [3 ** i for i in reversed(range(self.num_levels))]
        if len(fpn_strides) != self.num_levels:
            raise ValueError("fpn_strides length must match number of levels")

        self.fpn_stride_convs = nn.ModuleList([
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=s, padding=1)
            for s in fpn_strides
        ])

        # smoothing convs (depth‑wise 3×3 + BN + ReLU)
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1, groups=self.base_channels),
                build_norm_layer(dict(type="BN"), self.base_channels)[1],
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_levels)
        ])

        # optional final 1×1 to change channel dim
        if out_channels is not None and out_channels != self.base_channels:
            self.out_conv = nn.Conv2d(self.base_channels, out_channels, 1)
            self.out_channels = out_channels
        else:
            self.out_conv = nn.Identity()
            self.out_channels = self.base_channels

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, inputs):
        """Expect ``inputs`` as *list* (ordered coarse → fine)."""
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("ViTFusionFPN expects a list of feature maps from the backbone")
        if len(inputs) != self.num_levels:
            raise ValueError("Number of inputs does not match initialisation")

        processed = []
        for lvl, feat in enumerate(inputs):
            x = self.proj_layers[lvl](feat)
            x = self.fpn_stride_convs[lvl](x)
            processed.append(x)

        # up‑sample to the finest resolution among inputs
        target_size = max([p.shape[-2:] for p in processed], key=lambda s: s[0] * s[1])
        for lvl, p in enumerate(processed):
            p = resize(p, size=target_size, mode="bilinear", align_corners=self.align_corners)
            processed[lvl] = self.smooth_convs[lvl](p)

        fused = torch.stack(processed, 0).sum(0)
        fused = self.out_conv(fused)
        return fused  # single tensor (C, H, W)

    # ------------------------------------------------------------------
    # weight init helpers
    # ------------------------------------------------------------------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
