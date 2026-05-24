---
title: "Comparison Between Equivalent Architectures of Complex-valued and Real-valued Neural Networks - Application on Polarimetric SAR Image Segmentation"
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
  - complex-valued
sources:
  - "sources/hal-03771786.md"
updated: "2026-05-17"
---

Citation: Journal of Signal Processing Systems, 2022. DOI: 10.1007/s11265-022-01793-0. HAL: hal-03771786 (https://hal.science/hal-03771786v1)

Abstract (short): The paper presents an in-depth statistical comparison between several Complex-Valued Neural Network (CVNN) models and their Real-Valued Neural Network (RVNN) equivalents on the Oberpfaffenhofen PolSAR database. It proposes a novel formulation to define equivalent-RVNNs in terms of real-valued trainable parameters that preserve the hidden-layer aspect ratios, extended to convolutional layers. Experiments across MLP, CNN and fully-convolutional (FCNN) architectures show that CVNNs obtain better statistical performance than capacity-equivalent RVNNs on PolSAR segmentation.

Keywords: Complex-Valued Neural Network, Real-Valued Neural Network, Polarimetric SAR, FCNN, CV-CNN, equivalence, ratio-tp

Main contributions / highlights
- Formal extension of an "equivalent-RVNN" definition (previously discussed for fully-connected layers) to convolutional layers so comparisons between complex- and real-valued networks are done at equal trainable-parameter capacity while maintaining hidden-layer aspect ratios.
- Definition and practical computation of the r-scaling (1 ≤ r < 2, tending to sqrt(2) for deep layers) to size RV counterparts appropriately (ratio-tp technique).
- Implementation details: Type-A complex activations (cartesian ReLU), adaptation of He initialization to complex domain, Adam optimizer for all models, categorical cross-entropy computed by averaging real/imag parts for complex outputs.
- Empirical experiments on the Oberpfaffenhofen PolSAR coherency matrix input (6 complex values per pixel); train/val/test sampling: 8% train, 2% validation, 90% test; 50 Monte-Carlo trials per model to obtain confidence intervals.
- Reported results: CV-FCNN obtained the best performance (OA median 98.55%, AA median 98.14%). Across architectures (MLP, CNN, FCNN) CV models outperformed their equivalent-RV counterparts in OA and AA with statistically non-overlapping median intervals.

Datasets & experiments
- Dataset: Oberpfaffenhofen PolSAR (coherency matrix T representation), labels for three classes (built-up areas, woodland, open areas).
- Models compared: CV-MLP vs RV-MLP, CV-CNN vs RV-CNN, CV-FCNN vs RV-FCNN. Real equivalents were constructed using the proposed ratio-tp formulation (maintaining aspect ratios while matching real-valued trainable parameters).
- Training: Adam optimizer, He-initialization (adapted for complex), dropout for MLP to reduce overfitting. Metrics: Overall Accuracy (OA), Average Accuracy (AA); medians and confidence intervals reported.

How this relates to existing wiki pages
- Extends and formalizes work from [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]] by applying the equivalence construction to convolutional layers and by providing larger empirical segmentation experiments (FCNN) on a real PolSAR dataset.
- Complements [[publications/complex-valued-vs-real-valued-neural-networks]] and [[publications/theory-and-implementation-of-complex-valued-neural-networks]] by providing a published journal version (J. Signal Processing Systems, 2022) and by focusing on PolSAR semantic segmentation.

Contradictions / notes
- Explicit contradiction check: No contradictions detected with existing wiki pages. This paper extends prior MLSP 2021 work (same author group) rather than contradicting it: it generalizes the equivalence definitions to convolutional layers and reports consistent empirical findings that CVNNs can outperform properly capacity-matched RVNNs on complex-valued PolSAR data.
- Minor differences vs earlier reports: this journal paper reports experiments with 50 Monte-Carlo trials (larger than some earlier reports) and includes FCNN (CV-FCNN) achieving the highest accuracies; it also uses Adam optimizer and Type-A ReLU activations throughout, which the text notes as deliberate modernizations relative to some earlier baselines.

Files / provenance
- Raw PDF deposited in HAL and available locally; see source record [[sources/hal-03771786]].

See also
- Implementation / code resources referenced in the paper and related projects: [[projects/cvnn-polsar]] and the cvnn toolbox referenced in [[publications/complex-valued-vs-real-valued-neural-networks]].

