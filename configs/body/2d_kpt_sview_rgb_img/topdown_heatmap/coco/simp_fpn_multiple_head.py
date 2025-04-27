load_from = '/scratch/rob599w25_class_root/rob599w25_class/klingjac/ViTPose/base_weights/multilatw.pth'

# -----------------------------------------------------------------------------
# Base presets -----------------------------------------------------------------
# -----------------------------------------------------------------------------
_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py',
]

# -----------------------------------------------------------------------------
# Optimisation -----------------------------------------------------------------
# -----------------------------------------------------------------------------

evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        },
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200],
)

total_epochs = 200

target_type = 'GaussianHeatmap'

# -----------------------------------------------------------------------------
# Channel mapping --------------------------------------------------------------
# -----------------------------------------------------------------------------
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[[i for i in range(17)]],
    inference_channel=[i for i in range(17)],
)

# -----------------------------------------------------------------------------
# Model ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
model = dict(
    type='TopDown',
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        ratio=1,
        frozen_stages=11,
        freeze_attn=False,
        freeze_ffn=False,
    ),
    neck=dict(
        type='FeaturePyramidNetwork',
        in_channels=768,
        num_scales=4,
        strides=[1, 2, 4, 8],
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=[768, 768, 768, 768],  # four FPN maps
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True,
    ),
)

# -----------------------------------------------------------------------------
# Data cfg ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
)

# -----------------------------------------------------------------------------
# Pipelines --------------------------------------------------------------------
# -----------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type=target_type),
    dict(type='Collect', keys=['img', 'target', 'target_weight'], meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect', keys=['img'], meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs']),
]

test_pipeline = val_pipeline

# -----------------------------------------------------------------------------
# Data loaders -----------------------------------------------------------------
# -----------------------------------------------------------------------------

data_root = 'data/coco_subtrain'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_subtrain2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
)
