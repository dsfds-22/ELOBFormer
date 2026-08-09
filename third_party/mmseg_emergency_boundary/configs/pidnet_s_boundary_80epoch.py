"""PIDNet-S external baseline under the frozen protocol."""


_base_ = [
    "_base_/dataset.py",
    "../upstream_configs/default_runtime.py",
]
custom_imports = dict(
    imports=["third_party.mmseg_emergency_boundary"],
    allow_failed_imports=False,
)

_project_root = __import__("os").environ.get(
    "LANE_DETECTION_PROJECT_ROOT", __import__("os").getcwd()
)
pretrained_checkpoint = __import__("os").path.join(
    _project_root, "weights", "pretrained", "pidnet_s_cityscapes.pth"
)
work_dir = __import__("os").path.join(
    _project_root,
    "runs",
    "emergency_boundary",
    "pidnet_s_band_edge_grouped_fixed12_sqrt_scene_v1_80epoch_es15",
)

norm_cfg = dict(type="BN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(456, 800),
    test_cfg=dict(size=(456, 800)),
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="PIDNet",
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU", inplace=True),
        init_cfg=None,
    ),
    decode_head=dict(
        type="PIDHead",
        in_channels=128,
        channels=128,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU", inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type="BinaryBceDiceLoss",
                loss_weight=0.4,
                loss_name="loss_sem_p",
            ),
            dict(
                type="BinaryBceDiceLoss",
                loss_weight=1.0,
                loss_name="loss_sem_i",
            ),
            dict(
                type="BinaryBceDiceLoss",
                loss_weight=0.3,
                loss_name="loss_edge",
            ),
            dict(
                type="BinaryBceDiceLoss",
                loss_weight=1.0,
                loss_name="loss_sem_boundary",
            ),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=80, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
)
param_scheduler = []

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=5,
        max_keep_ckpts=3,
        save_optimizer=False,
        save_param_scheduler=False,
        save_last=False,
        save_best="boundary/iou",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", interval=1),
)
custom_hooks = [
    dict(
        type="LoadMatchingCheckpointHook",
        checkpoint=pretrained_checkpoint,
    ),
    dict(
        type="EarlyStoppingHook",
        monitor="boundary/iou",
        rule="greater",
        min_delta=0.0,
        patience=15,
        strict=True,
    ),
]

log_processor = dict(by_epoch=True, window_size=20)
randomness = dict(seed=42, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="gloo"),
)
resume = False
