
# GPU vs CPU Benchmark Report: CNN on CIFAR-10

---

## Overview

This experiment benchmarks a simple CNN trained on the CIFAR-10 dataset using:
- **CPU** (1 thread per process)
- **GPU (CUDA)**

Both models were trained for 5 epochs using identical hyperparameters.

---

## Configuration Summary

| Parameter            | Value              |
|----------------------|--------------------|
| Epochs               | 5                  |
| Batch Size           | 64                 |
| Learning Rate        | 1e-3               |
| Model                | SimpleCNN          |
| Dataset              | CIFAR-10           |
| Framework            | PyTorch + Accelerate |
| W&B Project          | gpu-vs-cpu-benchmark |
| CPU Threads Used     | 1 (for apple-to-apple comparison) |
| Accelerate CLI Used  | Yes (`--num_cpu_threads_per_process=1`) |

---

## Model Artifacts

| Device | Model Path            | Metadata Path               |
|--------|------------------------|-----------------------------|
| CPU    | `models/model_cpu.pt` | `models/model_cpu_meta.json` |
| GPU    | `models/model_cuda.pt` | `models/model_cuda_meta.json` |

---
