---
title: "parallel image filtering"
tags:
  - project
  - github
  - parallel
  - image-processing
sources:
  - "sources/raw-github-parallel-image-filtering.md"
updated: "2026-05-17"
---

Overview

Comparative implementation and benchmarking of parallel computing approaches for image filtering across CPU and GPU architectures. See raw README for full details: [[sources/raw-github-parallel-image-filtering.md]].

Features (summary)

- Multiple image filtering algorithms implemented with parallel backends.
- Performance benchmarking and comparison tools across architectures.
- Cross-architecture optimization notes and scalability testing.
- Timing analysis and performance metrics collection.

Parallelism models implemented

- OpenMP: shared-memory parallelization for multi-core CPUs.
- MPI: message-passing for distributed multi-process setups.
- CUDA: GPU acceleration for NVIDIA hardware.

Prerequisites (high-level)

- OpenMP: GCC with OpenMP support (compile with -fopenmp), multi-core CPU.
- MPI: OpenMPI or MPICH and multiple CPU cores or nodes.
- CUDA: NVIDIA GPU, CUDA Toolkit and compatible drivers (raw file lists "version X.X or higher" — verify exact version in repo).

Installation (brief)

Clone the repository and follow model-specific instructions in the README:

  git clone https://github.com/NEGU93/Parallel-Image-FIltering.git

For package installs the raw README includes example apt/yum commands for OpenMP and MPI, and points to NVIDIA Developer for CUDA installation.

Supported filter types

- Gaussian Blur
- Edge detection (Sobel, Laplacian)
- Median filter
- Convolution filters
- Custom kernel filters

Key insights (from raw README)

- OpenMP performs well on shared-memory, CPU-bound tasks.
- MPI is suitable for multi-node distributed workloads.
- CUDA provides large speedups for highly-parallel algorithms but may be limited by GPU memory bandwidth for memory-intensive filters.
- Results are hardware-dependent; benchmarking is included in the repo to reproduce comparisons.

Cross-references

- Raw README / source: [[sources/raw-github-parallel-image-filtering.md]]

Contradictions with existing wiki

- No contradictions detected. This project is distinct from other image-processing or neural-network projects currently in the wiki.

Recommended follow-ups

- Ingest benchmark outputs, plots, or example timing CSVs if present in the repository to enrich the project page.
- Capture exact CUDA toolkit minimum version from repository or CI files and update prerequisites.
