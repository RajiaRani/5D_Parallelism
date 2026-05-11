# 5D Parallelism for Large Language Model Training

*A systematic exploration of distributed training techniques for LLMs, inspired by the Ultra-Scale Playbook.*

---

## Overview

Training foundation models at frontier scale demands a deep understanding of how computation, memory, and communication intersect across hardware hierarchies. This repository documents my deep-dive into the **key parallelism techniques** that power modern GPU cluster training.

---

## Topics Studied

| # | Topic | Description |
|---|-------|-------------|
| 01 | Training on One GPU | Baseline memory profiling and analysis |
| 02 | Data Parallelism | Distributed training with gradient synchronization |
| 03 | Activation & Gradient Recomputation | Memory-compute tradeoffs via checkpointing |
| 04 | Ring-All-Reduce | Efficient gradient aggregation across GPUs |
| 05 | ZeRO (Stage 1, 2, 3) | Memory optimization via sharded optimizer states |
| 06 | Tensor Parallelism | Fine-grained weight matrix partitioning |
| 07 | Sequence Parallelism | Distributing sequence dimension across GPUs |
| 08 | Context Parallelism | Long-context attention via sequence sharding |
| 09 | Pipeline Parallelism | Layer pipelining with 1F1B scheduling |
| 10 | Expert Parallelism | MoE expert routing across devices |

---

## Repository Structure

```
5D_Parallelism/
├── 01. Training_on_One_GPU/
│   ├── Llama_Memory_Usage_Buckets.ipynb
│   └── Llama_Memory_Track.ipynb
│
├── 02. Data_Parallelism/
│
├── 03. Activation_Recomputation_Gradient_Recomputation/
│   ├── Activation_Recomputation.ipynb
│   ├── ShallowSpeed_GPU_Benchmark.ipynb
│   └── ShallowSpeed_Advanced_Benchmark.ipynb
│
├── 04. Ring-All-Reduce/
│
├── 05. ZeRO-1,2,3/
│
├── 06. Tensor_Parallelism/
│
├── 07. Sequence_Parallelism/
│
├── 08. Context_Parallelism/
│
├── 09. Pipeline_Parallelism/
│
├── 10. Expert_Parallelism/
│
└── README.md
```

---

## Technical Depth

### Memory Hierarchy
- **Parameter memory**: ~2 bytes/parameter (fp16) → ~8 bytes (Adam optimizer state)
- **Activation memory**: O(batch_size × seq_len × hidden_dim × num_layers)
- **KV-cache**: O(2 × batch_size × seq_len × num_heads × head_dim) per layer

### Communication Patterns
- **Data Parallelism**: All-reduce for synchronized gradients
- **Ring-All-Reduce**: Reduced bandwidth via ring topology
- **ZeRO**: Partitioned optimizer states with all-gather
- **Tensor Parallelism**: All-reduce within layer computation
- **Pipeline Parallelism**: Point-to-point activations between stages

### Roofline & Utilization
Peak FLOPS are rarely achievable due to memory bandwidth saturation, communication-computation overlap limitations, and kernel launch overhead in fine-grained parallelism.

---

## About The Ultra-Scale Playbook

**[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://www.amazon.com/dp/B0XXXXX)** by Vijay bridges theory and production systems:

- Concrete benchmark results on real GPU clusters
- Failure modes and debugging strategies
- Systems-level thinking for cluster-scale training
- Reproducible experimental setups

This repository serves as my implementation companion — working through concepts, reproducing benchmarks, and adding my own analysis.

---

## Why This Matters

Critical for:
- **ML Engineers** building or fine-tuning frontier models
- **Infrastructure Engineers** optimizing cluster utilization
- **Researchers** designing model architectures with deployment constraints
- **Students** bridging academic ML and systems knowledge

---

## Setup

```bash
conda create -n parallelism python=3.10
conda activate parallelism
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyterlab
```

---

## Resources

- [Ultra-Scale Playbook on Amazon](https://www.amazon.com/dp/B0XXXXX)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Megatron-LM Paper](https://arxiv.org/abs/2104.04473)
- [DeepSpeed ZeRO Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-training/)
