# Unified Pure-Model FPS Benchmark

This directory is the authoritative pure-model runtime comparison for the
clean601 protocol.

## Protocol

- Source input: 800 x 450 RGB
- Batch size: 1
- Precision: FP32
- Warm-up: 20 forwards per repetition
- Timed forwards: 100 per repetition
- Repetitions: 3 independent model loads
- Timing: `time.perf_counter()` with CUDA synchronization before and after
- Included: normalized tensor model forward only
- Excluded: disk I/O, image decoding, CPU preprocessing, visualization, and
  mask writing
- cuDNN benchmark: enabled

PIDNet-S receives the same 800 x 450 source input but requires an internal
bottom pad to 800 x 456 because of its architecture stride. That required
computation is included in its reported runtime.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- CPU: 13th Gen Intel Core i9-13900HX
- Python: 3.10.20 (`mmseg_boundary` environment)
- PyTorch: 2.9.1+cu128
- torchvision: 0.24.1+cu128
- CUDA runtime: 12.8
- cuDNN: 9.10.2
- MMEngine: 0.10.7
- NVIDIA driver: 572.84
- GPU power limit: not reported by `nvidia-smi`

The raw files `<model>_repeat<1..3>.csv` are retained beside the aggregate
`unified_fps_summary.csv`. A1 and A2 are both reported as complete training
structures and as deployment structures with their auxiliary heads removed.
