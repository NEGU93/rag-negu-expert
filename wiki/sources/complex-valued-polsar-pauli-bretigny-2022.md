---
title: "Complex-valued Neural Networks for Polarimetric SAR Segmentation using Pauli Representation"
authors:
  - "J. A. Barrachina"
  - "C. Ren"
  - "G. Vieillard"
  - "C. Morisseau"
  - "J.-P. Ovarlez"
tags:
  - publications
  - sources
  - CVNN
  - PolSAR
  - Pauli
  - Bretigny
  - ICASSP
sources:
  - "raw/publications/ICASSP_2022.pdf"
updated: "2026-05-17"
---

Summary

This ICASSP paper proposes to train Complex-Valued Fully Convolutional Neural Networks (CV-FCNN) directly on the Pauli-vector representation (3 complex channels per pixel) of a proprietary ONERA Bretigny PolSAR dataset instead of on the commonly-distributed coherency matrix. The authors argue the Pauli representation preserves richer per-pixel complex information and avoids the information loss introduced by local averaging used to build coherency matrices. They compare CV-FCNN against an equivalent real-valued FCNN (RV-FCNN) and report statistically significant better performance for the complex model.

Dataset and preprocessing

- Dataset: ONERA Bretigny PolSAR image (proprietary), manually labeled with 4 classes: Open Area, Wood Land, Built-up Area, Runway. A visualization of the area is provided in the paper (Fig.1).
- Input representation: Pauli vector k = 1/sqrt(2) [SHH + SVV, SHH - SVV, 2 SHV]^T (complex-valued, 3 channels). The paper stresses that the Pauli vector cannot be recovered from coherency matrices (T), explaining the need for a raw/Pauli dataset.
- Patch generation: image split vertically into three sub-images (70% train, 15% val, 15% test) to avoid ground-truth overlap across sets. Sliding window (128×128 patches, stride 25) applied separately per sub-image. Class imbalance (≈10:1 Open-Area) handled by randomly removing labels in the training set to balance classes. Data augmentation: random horizontal/vertical flips.

Model and training

- Models: implementation of a Complex-Valued Fully Convolutional Neural Network (CV-FCNN) inspired by [12] and an equivalent Real-Valued FCNN (RV-FCNN) for comparison.
- Main blocks: complex convolution, complex batch-norm, CReLU (Type-A cartesian ReLU). Down/up-sampling performed with max-pooling / max-unpooling; for complex data, max-pooling uses absolute values for pooling decisions. Output: complex image with number-of-classes channels; softmax applied separately on real and imaginary parts; loss computed on both parts and averaged. Unlabeled pixels excluded from loss and accuracy.
- Implementation: uses the authors' TensorFlow-based CVNN library (Zenodo DOI noted in the paper). Batch-normalization used; dropout omitted.
- Training details: five independent iterations for each model, 150 epochs per iteration, batch size 30. Validation loss used to select best checkpoint.

Results

- Test accuracy (reported as median ± CI and mean ± CI across 5 runs):
  - CV-FCNN: median 92.76 ± 0.36%, mean 92.77 ± 0.46% (full range 92.37–93.17)
  - RV-FCNN: median 89.86 ± 0.96%, mean 89.92 ± 1.23% (full range 88.89–91.02)
- Training accuracy (last epoch): CV-FCNN median 98.99 ± 0.06%, RV-FCNN median 98.71 ± 0.14%.
- The authors report that confidence intervals for mean and median do not overlap between CV and RV experiments, concluding a statistically significant advantage for CV-FCNN on this dataset: higher mean test accuracy and lower variance.
- Figures: training/validation accuracy & loss curves, test-accuracy boxplots, and full-image predictions for qualitative comparison are provided in the paper. The authors note better generalization of CV-FCNN and visually improved segmentation maps.

Key takeaways

- Using Pauli-vector input (per-pixel complex 3-channel vector) avoids averaging-induced loss present in coherency-matrix inputs and lets early convolutional layers learn local spatial coherence via trainable filters.
- On the Bretigny ONERA dataset, CV-FCNN outperforms an equivalent-capacity RV-FCNN in mean accuracy and variability; results are reported with statistical confidence intervals.
- The paper also proposes a dataset split method (vertical partitioning into train/val/test) that prevents overlapping ground-truth between sets produced by sliding-window patch extraction.

Related pages / cross-references

- Related work on CVNN vs RVNN equivalence and PolSAR segmentation: [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]].
- Implementation/library references and broader CVNN materials: [[publications/complex-valued-vs-real-valued-neural-networks.md]] and [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]].
- Related project pages: [[projects/cvnn-polsar.md]].

Contradictions with existing wiki

- None found. The extracted results, dataset description and conclusions are consistent with the existing index entry summary for this source ("CV-FCNN trained on Pauli-vector input (Bretigny ONERA dataset); shows CV-FCNN (92.8%) > RV-FCNN (89.9%); notes Pauli cannot be recovered from coherency matrices.").

Notes / provenance

- Raw file ingested: raw/publications/ICASSP_2022.pdf (original local path: C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/publications/ICASSP_2022.pdf).
- Recommended follow-ups: (1) add a small entry under [[projects/cvnn-polsar.md]] summarizing the Bretigny experiments and link to this source; (2) if permitted, extract / archive the Pauli-formatted Bretigny dataset metadata (permissions/DOI) for reproducibility; (3) link the CVNN implementation/Zenodo DOI cited in the paper as a separate source page if not yet present.
