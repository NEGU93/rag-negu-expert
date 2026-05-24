---
title: "Complex-Valued Neural Networks for Polarimetric SAR Segmentation Using Pauli Representation (IGARSS 2022)"
authors:
  - "J. A. Barrachina"
  - "C. Ren"
  - "C. Morisseau"
  - "G. Vieillard"
  - "J.-P. Ovarlez"
tags:
  - sources
  - publications
  - igarss-2022
  - CVNN
  - PolSAR
  - pauli-representation
sources:
  - "raw/publications/IGARSS_2022.pdf"
updated: "2026-05-17"
---

Title: COMPLEX-VALUED NEURAL NETWORKS FOR POLARIMETRIC SAR SEGMENTATION USING PAULI REPRESENTATION

Conference: IGARSS 2022 (IEEE International Geoscience and Remote Sensing Symposium) — raw PDF: [[raw/publications/IGARSS_2022.pdf]]

Summary

This IGARSS 2022 paper (authors: J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard, J.-P. Ovarlez) studies Complex-Valued Fully Convolutional Neural Networks (CV-FCNN) for pixel-wise segmentation of Polarimetric SAR (PolSAR) images using the Pauli vector representation as network input instead of the commonly used averaged Hermitian coherency matrix. The authors train and evaluate CV-FCNNs versus equivalent real-valued FCNNs (RV-FCNN) on an ONERA Bretigny (France) PolSAR dataset (proprietary), using a non-overlapping split (vertical thirds) and sliding-window patches of 128×128 with stride 25.

Key points / findings

- Input representation: Pauli vector k ∈ C^3 per pixel (k = 2^{-1/2} [SHH + SVV, SHH − SVV, 2 SHV]^T) is used directly as network input. The paper argues the Hermitian coherency matrix T (commonly distributed in open ESA datasets) is not optimal for CVNNs because (1) diagonal elements are real-valued (loss of complex structure useful to CVNN), and (2) the local averaging (boxcar) used to build T loses local/pixel-wise information and mixes neighbouring pixels.

- Dataset: An ONERA Bretigny X-band PolSAR image (2 m resolution, 30° incidence) manually labeled into four classes: Open Area, Wood Land, Built-up Area and Runway. The dataset split avoids train/test leakage by dividing the full image into three vertical sub-images: 70% training (left), 15% validation (middle), 15% test (right). Patches generated per-split prevent shared pixels between sets.

- Models: CV-FCNN architecture based on the Complex-Valued Fully Convolutional Network described in [12] (reference numbering as in paper). An equivalent RV-FCNN (capacity-adjusted) is implemented for comparison. Complex blocks use Complex Conv2D + Complex BatchNorm + Complex ReLU (Type-A/CReLU cartesian ReLU), with max-pooling / max-unpooling for down/up-sampling and skip connections. Output is complex with as many channels as classes; softmax applied separately to real and imaginary parts and losses averaged.

- Training & evaluation: Five Monte Carlo trials per model, 150 epochs each, batch size 30. Validation loss used to select best checkpoint. Metrics reported on test set: median and mean accuracy with 95% CI and full ranges.

Reported quantitative results (test accuracy %)

- CV-FCNN: median 92.76 ± 0.36, mean 92.77 ± 0.46, full range 92.37–93.17
- RV-FCNN: median 89.86 ± 0.96, mean 89.92 ± 1.23, full range 88.89–91.02

Conclusion

- The Pauli-vector input coupled with CV-FCNN outperforms the equivalent real-valued model on the Bretigny dataset: higher accuracy and lower variance across trials. The paper emphasizes that the Pauli representation cannot be recovered from averaged coherency matrices (T), which prevents reuse of many public ESA datasets (San Francisco, Flevoland, Oberpfaffenhofen) if Pauli components are required.

Artifacts / code / reproducibility

- The authors reference an implementation and code releases used in their experiments (see references [17], [19] in the paper): the TensorFlow-based cvnn library and a repository titled NEGU93/polsar-cvnn (Zenodo DOI given in the paper). See also related project material in the wiki: [[projects/cvnn.md]].

Notes / provenance

- Source file imported: raw/publications/IGARSS_2022.pdf (local raw copy). This page summarizes the extracted content and key results for integration into the wiki.

Contradictions with existing wiki

- Duplicate/similar summaries: the wiki already contains a sources page named [[sources/complex-valued-polsar-pauli-bretigny-2022.md]] which summarizes the same work (CV-FCNN on Bretigny with Pauli input and the reported accuracies ~92.8% vs ~89.9%). This new IGARSS-derived source appears to describe the same material; it may be the conference version of the work or a near-identical manuscript. Action required: reconcile duplicates by verifying provenance (conference paper vs technical report / journal submission) and consolidating summaries. At present both pages coexist and contain overlapping information.

- Duplicate index entries: the wiki index currently lists [[sources/complex-valued-polsar-pauli-bretigny-2022.md]] twice. This is an existing inconsistency independent of the present ingest and should be cleaned (remove duplicate index entry and link this new IGARSS source appropriately).

See also

- Related publications and background material in this wiki: [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]], [[publications/complex-valued-vs-real-valued-neural-networks.md]], and the CVNN toolbox description [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]].

References (as in the paper)

- Paper references and DOIs are preserved in the raw PDF; see raw source for exact reference list. The paper cites works on CVNNs, Pauli/coherency representations, and previous FCNN/CV-FCNN approaches (Cao et al. 2019, Li et al. 2018, Trabelsi et al. 2018, etc.).
