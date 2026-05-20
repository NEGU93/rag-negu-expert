---
title: "Complex-Valued vs. Real-Valued Neural Networks for Classification Perspectives: An Example on Non-Circular Data"
tags:
  - publications
  - arXiv
  - CVNN
  - complex-valued
  - neural-networks
sources:
  - sources/arxiv-2009-08340v2.md
updated: "2026-05-17"
---

Authors: J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard, J.-P. Ovarlez

Citation: arXiv:2009.08340v2 [stat.ML], 13 Apr 2021

Summary

This paper compares fully-connected feed-forward Complex-Valued Neural Networks (CVNNs) with equivalent Real-Valued Neural Networks (RVNNs) on synthetic complex-valued classification tasks that exhibit non-circular statistics (dependence between real and imaginary parts). The authors also release a Python library (Barrachina 2019) implemented on top of TensorFlow to facilitate CVNN implementation and reproducible experiments.

Key points / contributions

- Empirical demonstration that CVNNs outperform RVNNs on a variety of non-circular complex-valued datasets (higher median and mean accuracy, lower variance).
- CVNNs show less tendency to overfit than RVNNs when no regularization (dropout) is applied.
- Release of a TensorFlow-backed Python library for building CVNNs (GitHub: NEGU93/cvnn), referenced in the paper (Barrachina 2019).
- Analysis focuses on fully-connected feed-forward architectures (MLP-style), using Type-A complex activations (elementwise ReLU on real and imaginary parts) and softmax outputs for classification.

Datasets and experiments

- Inputs: complex feature vectors of size 128 for CVNN (256 for equivalent RVNN).
- Synthetic Complex Normal CN(0, σ_Z^2, τ_Z) datasets with three dataset types (A, B, C) differing in variance and correlation between real and imaginary parts (table summarized in the paper).
- Architectures: 1 hidden layer (64 units) and 2 hidden layers (100, 40) tested; dropout 0.5 used to mitigate overfitting.
- Training: SGD (lr=0.01), Glorot uniform initialization, 300 epochs, batch size 100, Monte-Carlo with 30 trials per condition.

Results (high-level)

- CVNN median test accuracy consistently higher than RVNN across datasets A, B, C; e.g. for 2HL on dataset A median CVNN ≈ 97.83% vs RVNN ≈ 95.82% (table in paper).
- Without dropout, RVNNs suffer heavy overfitting (large drop in test accuracy); CVNNs show higher variance but smaller drops in median accuracy.
- Polar inputs (amplitude+phase) to RVNNs produced worse overfitting and underperformed compared to using real+imag parts.
- As correlation |ρ| between real and imaginary parts increases, CVNN advantage becomes more pronounced.

Software / reproducibility

- The paper references a public repository for a CVNN library (Barrachina 2019). See the raw GitHub CVNN page in this wiki: [[sources/raw-github-cvnn.md]] and the related project on PolSAR CVNN classification [[projects/cvnn-polsar.md]].

Notes and links

- Full PDF of the ingested raw source is stored under the wiki sources page for this file: [[sources/arxiv-2009-08340v2.md]].
- This publication is relevant to the wiki concepts around CVNN and PolSAR classification; consider linking from concept pages for "CVNN" and "Complex-valued neural networks" if they exist.

Contradictions with existing wiki

- No contradictions with existing wiki content were found. The paper's code reference (NEGU93/cvnn) aligns with the existing raw GitHub cvnn entry in this wiki ([[sources/raw-github-cvnn.md]]) and with the PolSAR CVNN project ([[projects/cvnn-polsar.md]]).
