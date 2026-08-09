# Third-party notices

## Ultra-Fast-Lane-Detection

The files under `third_party/ufld/` are derived from
[`cfzd/Ultra-Fast-Lane-Detection`](https://github.com/cfzd/Ultra-Fast-Lane-Detection),
copyright (c) 2020 cfzd, and remain available under the MIT License. They are
included only because the frozen V4.3 pipeline uses the project-specific UFLD
lane checkpoint.

## MMSegmentation

The files under `third_party/mmseg_emergency_boundary/` are project-specific
dataset, loss, hook, metric, and configuration adapters for
[`open-mmlab/mmsegmentation`](https://github.com/open-mmlab/mmsegmentation).
MMSegmentation is licensed under Apache License 2.0 and is installed as an
external dependency; the full upstream repository is not vendored here.

## SegFormer

The MiT/SegFormer architecture and generic pretraining weights originate from
[`NVlabs/SegFormer`](https://github.com/NVlabs/SegFormer). Generic upstream
pretraining checkpoints are not redistributed in this release.
