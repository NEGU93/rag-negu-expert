---
title: "Impact of PolSAR Pre-Processing and Balancing Methods on Complex-Valued Neural Networks Segmentation Tasks"
authors:
  - "José Agustín Barrachina"
  - "Chengfang Ren"
  - "Christèle Morisseau"
  - "Gilles Vieillard"
  - "Jean-Philippe Ovarlez"
tags:
  - publications
  - journal
  - CVNN
  - PolSAR
  - segmentation
  - preprocessing
  - dataset-balancing
sources:
  - "sources/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023.md"
updated: "2026-05-17"
---

Citation: IEEE Open Journal of Signal Processing (OJSP), published 17 Feb 2023; DOI: 10.1109/OJSP.2023.3246391. Received 27 Oct 2022; revised 27 Jan 2023; accepted 1 Feb 2023.

Abstract (short)

This journal article investigates semantic segmentation of Polarimetric Synthetic Aperture Radar (PolSAR) imagery using Complex-Valued Neural Networks (CVNNs). It compares two common PolSAR input representations (coherency matrix vs. Pauli vector) across three model families (MLP, CNN, FCNN) in complex- and capacity-equivalent real-valued forms. The paper highlights (1) input-representation effects, (2) inflated performance caused by dataset splitting that allows high correlation between train/validation/test, and (3) class imbalance issues. Two balancing strategies are evaluated (sampling-based dataset balancing and a weighted loss).

Dataset and experimental setup

- Image: ONERA RAMSES Bretigny X-band PolSAR (2 m spatial resolution, labelled 4 classes: Open Area, Wood Land, Built-up Area, Runway). Labels: ~2.87M pixels; class imbalance (Open Area ~73%).
- Inputs compared: Pauli vector (k ∈ C^3) and Hermitian coherency matrix (T, averaged kk^H local estimator producing 6 complex values per pixel).
- Models compared: CV-MLP / RV-MLP, CV-CNN / RV-CNN, CV-FCNN / RV-FCNN (architectures configured to be capacity-equivalent following the r-scaling / ratio-tp methodology used in prior work).
- Optimization: Adam; complex He initialization adaptation; categorical cross-entropy (for complex outputs averaged across real/imag parts: LACE).
- Evaluation: Monte-Carlo trials (~10 trials) with statistical reporting (mean ± std). Two dataset pre-processing protocols considered: the common sliding-window sampling used in prior works (which produces correlated train/val/test patches) and a stricter spatial split (70/15/15) to reduce train/val/test correlation as proposed in earlier conference work.

Key results (selected figures reported in paper)

- High apparent saturation with standard sliding-window sampling: CV-FCNN (Pauli input) reached 99.83 ± 0.02% Overall Accuracy (OA) and 98.69 ± 0.33% Average Accuracy (AA); RV-FCNN slightly lower (99.69 ± 0.06% OA, 98.62 ± 0.20% AA).
- When using a stricter spatial dataset split (70/15/15), FCNN performance dropped notably (CV-FCNN: 93.62 ± 0.20% OA, 75.31 ± 0.63% AA; RV-FCNN: 92.63 ± 0.29% OA, 76.20 ± 0.80% AA), demonstrating prior results could be inflated by correlated sampling.
- Input representation effects are architecture-dependent: CV-FCNN favored Pauli vector input; CV-MLP/CV-CNN responses varied (MLP often preferred coherency matrix because local averaging reduced speckle at cost of local detail); CNN results were mixed and not decisively favoring one representation.
- Complex-valued models generally generalized better than capacity-equivalent real-valued counterparts, except for some MLP configurations without dataset splitting where RV-MLP sometimes matched or slightly outperformed CV-MLP.
- Balancing strategies: both sampling-based dataset balancing and weighted-loss improved AA (reduced OA–AA gap). Dataset balancing often performed better than the simple weighted loss scheme tested. FCNN balancing is challenging because many 128×128 patches are single-class (sampling-based balancing needs careful handling); authors provide an algorithm to remove/adjust single-class patches and to trim pixels per-patch to reach class-level pixel balance (code referenced in the sources note).

Conclusions (paper)

- Pre-processing choices (input representation, train/val/test splitting, and class balancing) substantially affect measured segmentation performance with CVNNs on PolSAR.
- Pauli vector input benefits fully-convolutional architectures; coherency matrix can be preferable for small MLP models due to its implicit despeckling.
- Careful dataset splitting is essential to avoid saturated/optimistic accuracy estimates when using sliding-window patch sampling.
- Simple balancing (sampling + weighted loss) reduces bias toward dominant classes; more advanced balancing/weighting strategies remain an open area.

Related pages and prior work

- This journal paper expands/organizes results and analyses that complement earlier conference items: [[sources/igarss-2022-complex-valued-polsar-pauli-representation.md]], [[sources/icip-2022-real-and-complex-valued-sar-segmentation.md]], and the conference paper [[publications/complex-valued-polsar-pauli-bretigny-2022.md]]. It also builds on methodology described in [[publications/complex-valued-vs-real-valued-neural-networks.md]] and the equivalence/r-scaling treatment in [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]] and implementation details from [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]].

Contradictions / notes vs existing wiki

- No substantive contradiction in scientific findings with the existing related pages: the paper's main claims (Pauli useful for FCNN, CVNN often advantageous, dataset-splitting reduces optimistic scores) are consistent with earlier conference results already captured in the wiki.
- The wiki currently contains duplicate entries for the Bretigny Pauli/FCNN material (two identical keys for [complex-valued-polsar-pauli-bretigny-2022] in the index). This page consolidates the journal version; the duplicate index entries should be deduplicated (editorial issue, not a scientific contradiction).
- Numerical differences between datasets (e.g. Oberpfaffenhofen results in [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]] vs Bretigny here) are expected since they use different sensors/areas and are not contradictory.

See also: the sources page for this raw file: [[sources/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023.md]].
