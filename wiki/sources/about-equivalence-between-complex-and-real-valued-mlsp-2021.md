---
title: "About the equivalence between complex-valued and real-valued fully connected neural networks - application to PolInSAR images"
tags:
  - publications
  - mlsp-2021
  - polinsar
  - cvnn
  - rvnn
sources:
  - raw/publications/DEMR21048.pdf
  - https://hal.science/hal-03529898v1
updated: "2026-05-17"
---

Citation

J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, J.-P. Ovarlez. "About the equivalence between complex-valued and real-valued fully connected neural networks - application to PolInSAR images." IEEE International Workshop on Machine Learning for Signal Processing (MLSP), Oct 25–28, 2021, Gold Coast, Australia. DOI: 10.1109/MLSP52302.2021.9596542. HAL: hal-03529898.

Summary

This MLSP 2021 workshop paper presents (1) a formal proposal to construct a real-valued MLP (RV-MLP) that is equivalent to a complex-valued MLP (CV-MLP) in the sense of having the same number of real-valued trainable parameters (tp) while preserving the hidden-layer aspect ratio, and (2) an extensive empirical comparison of CV-MLP vs. RV-MLP on the Oberpfaffenhofen PolInSAR dataset.

Key contributions

- A derivation showing how to select a scalar r (1 <= r < 2) such that each real hidden-layer size N_R_i = r * N_C_i yields an RV-MLP with the same total number of real trainable parameters as the CV-MLP (tp equality). The expression reduces to solving a quadratic in r; special-case formulae are provided (e.g., single hidden layer).
- Clarification of two possible "equivalences": matching (a) real-valued neurons per hidden layer (np) or (b) total real-valued trainable parameters (tp). The paper argues the two cannot generally be satisfied simultaneously and justifies choosing tp-equivalence while preserving aspect ratios via the r-scaling.
- Empirical evaluation on a real PolInSAR dataset (Oberpfaffenhofen). Multiple input encodings for the real model were tested (real/imag parts and amplitude/phase), two activation families (ReLU and tanh applied elementwise to real/imag parts for CV-MLP), dropout, SGD training and Monte-Carlo trials to obtain confidence intervals.
- The rv-equivalent transformation was added as a feature to the authors' open-source Python CVNN toolkit (NEGU93/cvnn).

Experimental setup (concise)

- Dataset: Oberpfaffenhofen PolInSAR (PolInSAR coherency matrices provided by ESA). Preprocessing: Hermitian coherency matrix reduced to upper triangle -> 21 complex-valued inputs for CV-MLP. For RV-MLP: either real/imag (conventional) or amplitude/phase (polar-RV-MLP).
- Train/val/test sampling: ~8% train (104,928 pixels), 2% validation (26,232), remaining 90% test. Balanced class sampling for train/val.
- Architectures: CV-MLP with 2 hidden layers (100 complex neurons, then 50). RV-MLP shapes chosen to satisfy tp-equivalence via the derived r. Dropout 50% used. Activations: elementwise ReLU or tanh on real/imag parts for CV-MLP; softmax on magnitudes for outputs.
- Optimization/hyperparams: SGD (lr=0.01, no momentum), Glorot uniform init (complex adaptation per [24]), 300 epochs, batch size 100. Each experiment: 100 Monte-Carlo trials per configuration (total ensemble statistics reported).

Results (high level)

- CV-MLP consistently outperformed the tp-equivalent RV-MLP on the PolInSAR classification task. With ReLU activations the median validation accuracy difference was ~0.5% favoring CV-MLP; with tanh the difference grew to ~1%.
- Validation medians (ReLU): CV-MLP median ~90.00% (±0.07), RV-MLP median ~89.45% (±0.06). Test accuracy reported around 91.63% (CV-MLP) vs 90.91% (RV-MLP) on a representative predicted image.
- Confusion matrices show CV-MLP outperformed RV-MLP across all three classes (built-up, woodland, open area); built-up areas were the hardest class overall.
- The authors conclude the phase structure of PolInSAR data benefits from a complex-valued model and that CV-MLP shows a slight but statistically supported advantage when compared fairly (tp-equivalent) to RV-MLP.

Relation to existing wiki pages

- This source complements the existing page [[publications/complex-valued-vs-real-valued-neural-networks]] (arXiv:2009.08340v2 / 2021) which performs extensive synthetic experiments on non-circular complex data and releases the NEGU93/cvnn toolbox. The MLSP paper applies similar comparisons on a real PolInSAR dataset and adds the explicit tp-equivalence-with-aspect-ratio derivation and tool support.
- Related project/tool entry: [[projects/cvnn-polsar]] (the repository/tooling for PolSAR CVNN experiments) is directly relevant; the paper cites and extends features in the same CVNN codebase.

Notes on contradictions with existing wiki

- No direct contradictions detected with existing pages. The MLSP paper and the arXiv paper [[publications/complex-valued-vs-real-valued-neural-networks]] report consistent conclusions: CVNNs can outperform RVNNs when complex structure (phase/coherence) matters. Main difference is dataset type and scope: the arXiv paper focused on synthetic non-circular datasets, while this MLSP paper evaluates on the real Oberpfaffenhofen PolInSAR dataset and provides an explicit tp-equivalence derivation.
- The MLSP paper's proposed tp-equivalence (r scaling) complements — it does not invalidate — the dimensioning choices discussed in other references on the wiki. If future pages assume np-equivalence (doubling neurons per layer) as the canonical real-equivalent strategy, they should be annotated to indicate the alternative tp-equivalence approach introduced here.

Raw / original files

- Original raw PDF ingested from local path: raw/publications/DEMR21048.pdf
- HAL preprint: https://hal.science/hal-03529898v1

Extracted keywords

PolInSAR, CV-MLP, RV-MLP, complex-valued neural networks, equivalence, trainable parameters, Oberpfaffenhofen, NEGU93/cvnn

