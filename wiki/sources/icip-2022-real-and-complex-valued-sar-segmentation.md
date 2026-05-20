---
title: "Real- and Complex-Valued Neural Networks for SAR Image Segmentation through Different Polarimetric Representations"
authors:
  - "J. A. Barrachina"
  - "C. Ren"
  - "G. Vieillard"
  - "C. Morisseau"
  - "J.-P. Ovarlez"
tags:
  - publications
  - conference
  - ICIP
  - CVNN
  - PolSAR
  - segmentation
sources:
  - "raw/publications/ICIP_2022.pdf"
  - "https://hal.science/hal-03841977"
updated: "2026-05-17"
---

Summary

This is the ICIP 2022 conference paper by J. A. Barrachina et al. that compares Complex-Valued Fully Convolutional Neural Network (CV-FCNN) and an equivalent-capacity Real-Valued FCNN (RV-FCNN) for pixel-wise segmentation of polarimetric SAR (PolSAR) images. Two input polarimetric representations are evaluated: the Pauli vector (complex 3D) and the Hermitian coherency matrix (averaged, leading to a 6D non-redundant vector after reshape). Experiments use the San Francisco AIRSAR PolSAR dataset (PolSF) with 5 semantic classes (Mountain, Water, Urban, Vegetation, Bare soil).

Key points / findings

- Motivation: coherency matrix uses local averaging (loss of resolution) and mixes adjacent pixels; Pauli vector keeps per-pixel complex information and is a natural input for CVNNs.
- Architectures: a U-Net inspired Complex-Valued FCNN (CV-FCNN) implemented with the authors' open-source cvnn toolbox and a real-valued parameter-equivalent RV-FCNN (following the r-scaling / equivalence methodology of the authors).
- Complex activations: Type-A (CReLU = ReLU on real and imaginary parts). Complex batch-norm and max-pooling (magnitude-based comparison) are used; categorical cross-entropy for complex nets is computed by averaging the loss on real and imaginary parts.
- Dataset / preprocessing: sliding window patches of 128×128, stride and window size following [7]; generated patches split 80% train, 10% validation, 10% test. Training: 400 epochs; Adam optimizer (lr 0.01). Experiments: four configurations (CV-FCNN/RV-FCNN × Pauli/coherency) repeated 50 times each with random dataset splits to obtain statistical intervals.

Quantitative results (test set, Table 1, reported values in %)

- Average Accuracy (AA) median (± reported interval):
  - CV-FCNN + Pauli: 98.00 ± 0.27 (mean 97.55 ± 0.15)
  - CV-FCNN + Coherency: 96.80 ± 0.25 (mean 96.54 ± 0.12)
  - RV-FCNN + Pauli: 96.75 ± 0.32 (mean 96.39 ± 0.18)
  - RV-FCNN + Coherency: 95.20 ± 0.44 (mean 94.98 ± 0.21)

- Overall Accuracy (OA) median (± reported interval):
  - CV-FCNN + Pauli: 99.64 ± 0.01 (mean 99.64 ± 0.01)
  - CV-FCNN + Coherency: 99.45 ± 0.02 (mean 99.44 ± 0.01)
  - RV-FCNN + Pauli: 99.40 ± 0.02 (mean 99.40 ± 0.01)
  - RV-FCNN + Coherency: 99.19 ± 0.03 (mean 99.18 ± 0.02)

- Main conclusions: CV-FCNNs outperform capacity-equivalent RV-FCNNs in segmentation accuracy on this dataset; using the Pauli vector as input consistently improves performance (higher AA and lower variance) relative to the coherency matrix for both complex- and real-valued models. The authors encourage using Pauli vector when available because it avoids averaging-induced resolution loss and requires less memory (3D vs 6D non-redundant coherency reshape).

Experimental notes

- Each experiment configuration was repeated 50 times with random train/val/test splits; median intervals and confidence intervals (99% for mean) are used for statistical conclusions. The authors note median-interval non-overlap as indication of statistically significant differences (95% confidence).
- The cvnn toolbox (DOI: 10.5281/zenodo.4452131) was used to implement CVNN modules and to generate real-equivalent models for fair comparisons.

Related wiki pages

- This source corresponds to the same ICIP 2022 entry referenced in [[sources/hal-03841977.md]]. See also related publications on equivalence and CVNN tooling: [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]], [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021.md]], [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]].

Contradictions / notes

- Duplicate: This source appears to be the same work already recorded in the wiki as [[sources/hal-03841977.md]] (ICIP 2022). No content contradictions were detected between this raw PDF and the existing [[sources/hal-03841977.md]] entry; the reported claims, dataset, and numerical results are consistent with the existing summary.
- Suggestion: if [[sources/hal-03841977.md]] already exists, consider consolidating metadata and linking both entries to the single canonical HAL/ICIP source page to avoid duplication.

Full citation / provenance

J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, J.-P. Ovarlez. "Real- and Complex-Valued Neural Networks for SAR Image Segmentation through Different Polarimetric Representations." ICIP 2022. HAL: hal-03841977. Raw PDF ingested: raw/publications/ICIP_2022.pdf
