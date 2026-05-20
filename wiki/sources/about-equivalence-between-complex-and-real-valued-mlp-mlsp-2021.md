---
title: "About the Equivalence Between Complex-Valued and Real-Valued Fully Connected Neural Networks - Application to PolInSAR Images (MLSP 2021)"
authors:
  - "J. A. Barrachina"
  - "C. Ren"
  - "G. Vieillard"
  - "C. Morisseau"
  - "J.-P. Ovarlez"
tags:
  - publications
  - conference
  - MLSP2021
  - CVNN
  - RVNN
  - PolInSAR
  - PolSAR
  - MLP
sources:
  - "raw/publications/MLSP_2021.pdf"
updated: "2026-05-17"
---

Summary

This is the MLSP 2021 conference paper (raw PDF) that proposes a formal definition for an "equivalent" real-valued MultiLayer Perceptron (RV-MLP) to compare fairly with a Complex-Valued MLP (CV-MLP). The work (i) derives a scaling factor r (1 ≤ r < 2, tending to sqrt(2) for wide/deep layers) to size RV hidden layers so that total real-valued trainable parameters (tp) match a given complex network while preserving hidden-layer aspect ratios; (ii) performs extensive Monte Carlo experiments (100 trials per setup, 300 epochs) on the Oberpfaffenhofen PolInSAR dataset comparing CV-MLP, conventional RV-MLP (real/imag parts) and polar-RV-MLP (amplitude/phase) across activation functions (ReLU, tanh).

Key points / findings

- Formal proposal for an "equivalent-RVNN" construction in terms of equal real-valued trainable parameters (tp) while keeping layer aspect ratios. The method solves for r in a quadratic expression (equations reproduced in the paper) and shows 1 ≤ r < 2; r=2 recovers the equal-neuron (np) approach used elsewhere.
- Experimental setup: Oberpfaffenhofen PolInSAR coherency-matrix inputs (6 complex values reduced to 21 complex inputs by discarding lower triangle), balanced sampling for train/val (8% train, 2% validation), three-way classification (Built-up, Woodland, Open Area). CV-MLP architecture: two hidden layers (100, 50 complex neurons); RV counterparts sized to match tp (ratio-tp).
- Results (validation): CV-MLP outperformed capacity-equivalent RV-MLP by a small but statistically significant margin. Example (ReLU): median validation accuracy CV-MLP 90.00% ±0.07 vs RV-MLP 89.45% ±0.06. Test accuracy reported in the paper: CV-MLP 91.63% vs RV-MLP 90.91% (selected predicted images and class confusion matrices provided).
- Empirical observation: ReLU (cartesian ReLU applied separately to real and imaginary parts) yields higher accuracies than tanh; RV-MLP using real/imag inputs performed better than polar-RV-MLP (amplitude/phase) in these experiments.
- Implementation notes: SGD optimizer (lr=0.01), Glorot uniform initialization adapted for complex weights, dropout (50%) used to reduce overfitting.

Relation to existing wiki material

- This MLSP 2021 source documents the conference/preprint-level experiments and equivalence derivation; it is directly related to the later, extended journal work summarized at [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]] and to the earlier arXiv MLP experiments in [[publications/complex-valued-vs-real-valued-neural-networks.md]].

Contradictions / notes on overlap with existing wiki entries

- Duplicate/overlap: The wiki already contains an entry "sources/jsps-mlsp-2021.md" in the index that references the MLSP/preprint material. This raw PDF appears to be the same MLSP 2021 manuscript (title/author set and content match). Action: consider consolidating the two source pages to avoid duplication. (No substantive textual contradictions were detected between this PDF and the existing "jsps-mlsp-2021" summary; they appear to describe the same work.)

- Differences vs later journal results: The later Journal of Signal Processing Systems (2022) extension ([[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]]) reports results for convolutional/fully-convolutional architectures (FCNN/CV-FCNN) and substantially different overall accuracies on Oberpfaffenhofen (e.g., OA reported ≈98.5% for CV-FCNN in the journal). This is not a direct contradiction: architectures, input representations, train/val/test splits, and model families differ (MLP experiments here vs FCNN experiments in the journal). The MLSP paper documents the MLP-level experiments and the tp-equivalence derivation that underpin the ratio-tp scaling used later; the large differences in absolute OA are explained by differing model families and evaluation protocols, but the wiki should call out these methodological differences when cross-referencing results.

Recommendations

- Consolidate duplicate MLSP/preprint source pages (this file and "sources/jsps-mlsp-2021.md").
- When comparing reported accuracies across pages, always show architecture type (MLP vs CNN/FCNN) and dataset split details to avoid misleading comparisons.

References / provenance

Raw file: raw/publications/MLSP_2021.pdf (provided during ingest). The PDF includes tables/figures referenced above (validation boxplots, epoch curves, confusion matrices) and the complete derivations of the equivalence (equations (1)–(9) and the quadratic for r).

Related wiki pages

- [[publications/complex-valued-vs-real-valued-neural-networks.md]] (arXiv MLP experiments, related methodology)
- [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]] (journal extension to conv/FCNN)
- [[sources/jsps-mlsp-2021.md]] (existing MLSP/preprint source — possible duplicate)
