# 5D Parallelism for Large Language Model Training

*A systematic exploration of distributed training techniques for LLMs, inspired by the Ultra-Scale Playbook.*

---

## Overview

Training foundation models at frontier scale — hundreds of billions to trillions of parameters — demands a sophisticated understanding of how computation, memory, and communication intersect across hardware hierarchies. This repository documents my deep-dive into the **five fundamental parallelism dimensions** that power modern GPU cluster training.

Each module builds upon the previous, progressing from single-GPU baseline profiling to advanced techniques like Mixture-of-Experts routing and pipeline interleaving. The goal is not just to implement these techniques, but to develop rigorous intuition for *why* they work and *when* to apply them.

---

## The Five Dimensions of Parallelism

Modern large-scale training systems do not rely on a single parallelism strategy. Instead, they compose multiple techniques to maximize hardware utilization while staying within memory and bandwidth constraints.

### 1. Data Parallelism
Stateless model replicas with synchronized gradients via all-reduce. Foundational to distributed training; scales throughput linearly with GPU count until communication or I/O becomes bottleneck.

### 2. Tensor Parallelism (Model Parallelism, Fine-Grained)
Row/column partitioning of linear layers across devices (as in Megatron-LM). Enables fitting models that exceed single-GPU memory but introduces all-reduce within layer forward/backward passes.

### 3. Pipeline Parallelism (Model Parallelism, Coarse-Grained)
Vertical slicing of transformer layers across stages. Allows inter-stage microbatches to overlap computation and communication, amortizing pipeline bubbles.

### 4. Context Parallelism
Attention computation sharded along the sequence dimension (e.g., FlashAttention with ring attention). Critical for long-context models where the KV-cache exceeds device memory.

### 5. Expert Parallelism
Routing mechanism in MoE architectures where experts are partitioned across devices. Combined with data/tensor parallelism to maintain model capacity while scaling parameters.

---

## Repository Structure

```
5D_Parallelism/
├── 01. Training_on_One_GPU/
│   ├── Llama_Memory_Usage_Buckets.ipynb
│   └── Llama_Memory_Track.ipynb
│
├── 02. Data_Parallelism/
│   └── (Baseline + distributed sampling strategies)
│
├── 03. Activation_Recomputation_Gradient_Recomputation/
│   ├── Activation_Recomputation.ipynb
│   ├── ShallowSpeed_GPU_Benchmark.ipynb
│   └── ShallowSpeed_Advanced_Benchmark.ipynb
│
├── 04. Tensor_Parallelism/
│   └── (Coming soon)
│
├── 05. Context_Parallelism/
│   └── (Coming soon)
│
├── 06. Pipeline_Parallelism/
│   └── (Coming soon)
│
└── 07. Expert_Parallelism/
    └── (Coming soon)
```

---

## Technical Depth

### Memory Hierarchy Analysis
Understanding memory footprint is prerequisite to parallelism design. This repo traces:
- **Parameter memory**: ~2 bytes/parameter (fp16) → ~8 bytes (Adam optimizer state)
- **Activation memory**: O(batch_size × seq_len × hidden_dim × num_layers)
- **KV-cache**: O(2 × batch_size × seq_len × num_heads × head_dim) per layer

### Communication Complexity
Every parallelism strategy introduces communication overhead:
- Data parallelism: O(gradient_size) per step (all-reduce)
- Tensor parallelism: O(activation_size) per layer per step (all-reduce)
- Pipeline parallelism: O(microbatch_size × hidden_dim) for activations/gradients between stages

### Roofline & Hardware Utilization
Peak FLOPS are rarely achievable due to:
- Memory bandwidth saturation (especially for large models)
- Communication-computation overlap limitations
- Kernel launch overhead in fine-grained parallelism

---

## About The Ultra-Scale Playbook

**[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://www.amazon.com/dp/B0XXXXX)** by Vijay is a practitioner's guide to distributed training infrastructure. Unlike academic papers that present idealized algorithms, the Playbook bridges theory and production systems by:

- Providing concrete benchmark results on real GPU clusters
- Discussing failure modes and debugging strategies
- Covering the systems-level thinking required for cluster-scale training
- Offering reproducible experimental setups

This repository serves as my implementation companion — working through the concepts, reproducing benchmarks, and adding my own analysis where the Playbook leaves exercises.

### Key Topics Covered in the Book

| Chapter | Topic | This Repo |
|---------|-------|-----------|
| Memory Foundations | Activation/parameter profiling | Module 01 |
| Data Parallelism | Gradient synchronization, bucketing | Module 02 |
| Activation Recomputation | Checkpointing strategies | Module 03 |
| Tensor Parallelism | 3D parallelism composition | Module 04 |
| Pipeline Parallelism | 1F1B scheduling, bubble minimization | Module 06 |
| Context Parallelism | Ring attention, sequence sharding | Module 05 |
| Expert Parallelism | MoE routing, capacity factors | Module 07 |

---

## Why This Matters

Understanding parallelism at this level is critical for:

- **ML Engineers** building or fine-tuning frontier models
- **Infrastructure Engineers** optimizing cluster utilization
- **Researchers** designing new model architectures with deployment constraints
- **Students** bridging academic ML and systems knowledge

The gap between "I can train a ResNet on one GPU" and "I understand why pipeline bubbles exist and how to minimize them" is exactly the gap this material targets.

---

## Setup

```bash
conda create -n parallelism python=3.10
conda activate parallelism
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyterlab
```

Each notebook is self-contained with explicit dependencies.

---

## Contributing

This is a personal study repository, but open to corrections and discussions. Open an issue if you spot an error or want to discuss a tradeoff.

---

## Resources

- [Ultra-Scale Playbook on Amazon](https://www.amazon.com/dp/B0XXXXX)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Megatron-LM Paper](https://arxiv.org/abs/2104.04473)
- [DeepSpeed MoE Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-moe/)
