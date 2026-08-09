"""Frozen 800x450 emergency-boundary dataset protocol."""


_project_root = __import__("os").environ.get(
    "LANE_DETECTION_PROJECT_ROOT", __import__("os").getcwd()
)
data_root = __import__("os").path.join(
    _project_root,
    "datasets",
    "emergency_outer_boundary_grouped_fixed12_v2_clean601",
)
dataset_type = "EmergencyBoundaryCsvDataset"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadEmergencyBoundaryAnnotations", expected_size=(450, 800)),
    dict(type="EmergencyBoundaryBasicAugmentation"),
    dict(
        type="PackSegInputs",
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
            "sample_id",
            "prefix",
            "capture_session",
            "reviewed_segment_id",
            "scene_group",
            "has_boundary",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadEmergencyBoundaryAnnotations", expected_size=(450, 800)),
    dict(
        type="PackSegInputs",
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
            "sample_id",
            "prefix",
            "capture_session",
            "reviewed_segment_id",
            "scene_group",
            "has_boundary",
        ),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        type="SqrtSceneSampler",
        replacement=True,
        num_samples=0,
        seed=42,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="splits/train.csv",
        pipeline=train_pipeline,
        serialize_data=True,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="splits/val.csv",
        pipeline=test_pipeline,
        test_mode=True,
        serialize_data=True,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="splits/test.csv",
        pipeline=test_pipeline,
        test_mode=True,
        serialize_data=True,
    ),
)

val_evaluator = dict(type="EmergencyBoundaryMetric")
test_evaluator = dict(type="EmergencyBoundaryMetric")
