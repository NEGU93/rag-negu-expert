---
title: "CVNN-PolSAR"
tags:
  - project
  - github
  - cvnn
  - polsar
sources:
  - "sources/raw-github-cvnn-polsar.md"
updated: "2026-05-17"
---

CVNN-PolSAR: code repository for complex- and real-valued neural network segmentation of PolSAR images.

Summary

CVNN-PolSAR implements training and evaluation pipelines for real- and complex-valued neural networks applied to Polarimetric SAR (PolSAR) semantic segmentation and pixel-wise classification. The repository includes multiple model architectures, dataset handler templates, Monte Carlo experiment runner, result viewer (Qt), and plotting/export utilities.

Key features

- Supports both complex-valued and equivalent real-valued model modes (several real-mode encodings: real_imag, amplitude_phase, amplitude_only, real_only).
- Multiple model architectures available (e.g., cao, small-unet, zhang, haensch, cnn, mlp, tan, and user-provided models).
- Experiment output structure with tensorboard, checkpoints, evaluation CSVs, training histories, and predicted images for reproducibility.
- Monte Carlo runner to execute repeated experiments from JSON configuration files.
- Qt-based result viewer (qt_app.py) for quick visualization of saved runs (requires configuration of root_drive and dataset paths).

How to run (high-level)

- Single simulation: use principal_simulation.py with CLI args. Example options include dataset, model, epochs, learning_rate, balance strategy, real_mode, coherency boxcar size, etc.
- Monte Carlo: python runner.py -I <iterations> -CF <config.json>
- View results: run qt_app.py after setting required path variables.

Datasets

The repo expects users to supply dataset files and implement a dataset handler subclass of PolsarDatasetHandler (see src/dataset_reader.py). Supported dataset names in the code: SF-AIRSAR, OBER, FLEVOLAND, BRET, GARON. Note: BRET and GARON are marked as ONERA-proprietary in the raw source — they are not publicly distributable from the repo.

Publications

Core code from this repository was used in multiple publications by Barrachina et al. (2021–2022). For full citation details and links, see the raw source page: [[sources/raw-github-cvnn-polsar.md]]. The project is closely related to the [[projects/cvnn.md]] project page (general CVNN library and related work).

Related pages

- Project family / complex-valued NN tools: [[projects/cvnn.md]]
- Raw source summary: [[sources/raw-github-cvnn-polsar.md]]

Contradictions with existing wiki

- No direct contradictions found. There is content overlap with [[projects/cvnn.md]] (both concern complex-valued neural networks). Consider merging or keeping cross-references to avoid duplication.

