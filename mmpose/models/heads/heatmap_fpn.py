# Copyright (c) OpenMMLab. All rights reserved.
"""Top‑down pose head with built‑in FPN fusion **plus 3 × 3 smoothing**.

New since the previous revision
-------------------------------
* `fpn_strides` argument lets you specify the stride for each level from the
  config (e.g. `[4, 2, 1]`).  If omitted we fall back to the previous `3**i`
  scheme (`…,9,3,1`).
* Order is **lowest‑resolution → highest‑resolution** exactly matching the
  `in_index` list you pass.  The last level therefore keeps *stride = 1* and
  remains the finest spatial map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownHeatmapSimpleHeadFPN(TopdownHeatmapBaseHead):
    """Simple heat‑map head with internal FPN‑style fusion.

    Parameters
    ----------
    in_channels : int | Sequence[int]
        Channels of each backbone feature fed in via ``in_index``.
    out_channels : int
        Number of predicted heat‑map channels (joints).
    fpn_strides : Sequence[int] | None, default=None
        Spatial down‑sampling factors applied *before* up‑resizing.  Provide one
        integer per level **from coarse to fine**.  If ``None`` we keep the old
        behaviour ``3**i`` (…, 9, 3, 1).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        num_deconv_layers: int = 3,
        num_deconv_filters: tuple[int, ...] = (256, 256, 256),
        num_deconv_kernels: tuple[int, ...] = (4, 4, 4),
        extra: dict | None = None,
        in_index=0,
        input_transform=None,
        align_corners: bool = False,
        loss_keypoint=None,
        train_cfg=None,
        test_cfg=None,
        upsample: int = 0,
        fpn_strides: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        # -------------------------------------------------------------- losses
        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get("target_type", "GaussianHeatmap")

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = list(in_index) if isinstance(in_index, (list, tuple)) else [in_index]
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra should be dict or None.")

        # ------------------------------------------------------- FPN building
        if isinstance(self.in_channels, (list, tuple)):
            base_c = self.in_channels[0]
            self.proj_layers = nn.ModuleList([
                nn.Conv2d(c, base_c, kernel_size=1) for c in self.in_channels
            ])

            # determine stride list → one per level, same order as feat list
            if fpn_strides is None:
                fpn_strides = [3 ** i for i in reversed(range(len(self.in_channels)))]
            if len(fpn_strides) != len(self.in_channels):
                raise ValueError("Length of fpn_strides must match number of levels")

            self.fpn_stride_convs = nn.ModuleList([
                nn.Conv2d(base_c, base_c, kernel_size=3, stride=s, padding=1)
                for s in fpn_strides
            ])

            # 3×3 depth‑wise smoothing convs
            self.smooth_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(base_c, base_c, kernel_size=3, padding=1, groups=base_c),
                    build_norm_layer(dict(type="BN"), base_c)[1],
                    nn.ReLU(inplace=True),
                )
                for _ in self.in_channels
            ])
        else:
            self.proj_layers = self.fpn_stride_convs = self.smooth_convs = None

        # ------------------------------------------------ deconv up‑projection
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters, num_deconv_kernels
            )
            conv_channels = num_deconv_filters[-1]
        else:
            self.deconv_layers = nn.Identity()
            conv_channels = self.in_channels if isinstance(self.in_channels, int) else self.in_channels[0]

        # --------------------------------------------------------- final conv
        kernel_size, padding, identity_final = 1, 0, False
        if extra and "final_conv_kernel" in extra:
            k = extra["final_conv_kernel"]
            if k not in (0, 1, 3):
                raise ValueError("final_conv_kernel must be 0, 1, or 3")
            kernel_size, padding, identity_final = (k, 1, False) if k == 3 else (k, 0, k == 0)

        if identity_final:
            self.final_layer = nn.Identity()
        else:
            layers: list[nn.Module] = []
            if extra:
                for k_size in extra.get("num_conv_kernels", []):
                    layers += [
                        build_conv_layer(dict(type="Conv2d"), conv_channels, conv_channels, k_size, 1, (k_size - 1) // 2),
                        build_norm_layer(dict(type="BN"), conv_channels)[1],
                        nn.ReLU(inplace=True),
                    ]
            layers.append(build_conv_layer(dict(type="Conv2d"), conv_channels, out_channels, kernel_size, 1, padding))
            self.final_layer = nn.Sequential(*layers)

    # ============================================================= forward
    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    # --------------------------------------------------- inference helper
    def inference_model(self, x, flip_pairs=None):
        output = self.forward(x)
        if flip_pairs is not None:
            output_heatmap = flip_back(output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type)
            if self.test_cfg.get("shift_heatmap", False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    # -------------------------------------------------------- internals
    def _init_inputs(self, in_channels, in_index, input_transform):
        self.input_transform = input_transform
        self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if not isinstance(inputs, list):
            return inputs

        feats = [inputs[i] for i in self.in_index]
        processed = []
        for lvl, feat in enumerate(feats):
            x = self.proj_layers[lvl](feat)
            x = self.fpn_stride_convs[lvl](x)
            processed.append(x)

        target_size = max([p.shape[-2:] for p in processed], key=lambda s: s[0] * s[1])
        for lvl, p in enumerate(processed):
            p = resize(p, target_size, mode="bilinear", align_corners=self.align_corners)
            processed[lvl] = self.smooth_convs[lvl](p)

        return torch.stack(processed, 0).sum(0)

    # --------------------------------------------------------- deconv stack
    def _make_deconv_layer(self, n, filters, kernels):
        if n != len(filters) or n != len(kernels):
            raise ValueError("Deconv cfg length mismatch")
        layers, in_c = [], self.in_channels if isinstance(self.in_channels, int) else self.in_channels[0]
        for planes, k in zip(filters, kernels):
            ksize, pad, out_pad = self._get_deconv_cfg(k)
            layers += [
                build_upsample_layer(dict(type="deconv"), in_c, planes, ksize, 2, pad, out_pad, False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            in_c = planes
        return nn.Sequential(*layers)

    # ----------------------------------------------------------- weight init
    def init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        if self.smooth_convs is not None:
            for m in self.smooth_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
