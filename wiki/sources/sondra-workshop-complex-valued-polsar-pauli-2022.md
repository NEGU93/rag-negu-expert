---
title: "SONDRA Workshop — Complex-Valued Neural Networks for Polarimetric SAR Segmentation (Pauli representation)"
tags:
  - sources
  - raw
  - publications
  - CVNN
  - PolSAR
  - Pauli
sources:
  - "C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/publications/SONDRA_Workshop.pdf"
  - "10.5281/zenodo.5821229" # code release (NEGU93/polsar_cvnn)
  - "10.5281/zenodo.4452131" # cvnn toolbox DOI referenced in text
updated: "2026-05-17"
---

Title: COMPLEX-VALUED NEURAL NETWORKS FOR POLARIMETRIC SAR SEGMENTATION USING PAULI REPRESENTATION

Authors: J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard, J.-P. Ovarlez

Affiliations: SONDRA, CentraleSupélec, Université Paris-Saclay; DEMR, ONERA, Université Paris-Saclay

Summary

This raw PDF (SONDRA workshop submission) presents a Compact report of experiments using a Complex-Valued Fully Convolutional Neural Network (CV-FCNN) trained directly on the Pauli-vector representation of the ONERA Bretigny PolSAR dataset for pixel-wise segmentation, and compared statistically against an equivalent Real-Valued FCNN (RV-FCNN).

Key points / metadata

- Dataset: ONERA Bretigny X-band PolSAR (labelled, 4 classes: Built-up Area, Wood Land, Open Area, Runway). The image is split vertically into three sub-images (70% train, 15% val, 15% test) to avoid overlap/coincident pixels when using sliding-window extraction.
- Input representation: Pauli vector (complex k ∈ C^3) instead of local averaged coherency matrix. Motivations: avoid diagonal real-only elements and information loss from local averaging; allow trainable convolutional kernels to learn denoising/aggregation.
- Models: CV-FCNN (complex-valued FCNN implementation per [6] / Cao et al. 2019) vs equivalent RV-FCNN sized following the capacity-equivalence approach used in related work.
- Implementation / code: references a released code / repository (Zenodo DOI: 10.5281/zenodo.5821229) and uses the cvnn toolbox (Zenodo DOI: 10.5281/zenodo.4452131).
- Training: 5 Monte Carlo trials per model family, 150 epochs, batch size 30, run on DCE servers. Validation/training curves plotted with median and IQR across trials.
- Results (test accuracy %):
  - CV-FCNN median: 92.76 ± 0.36
  - RV-FCNN median: 89.86 ± 0.96
  - CV-FCNN mean: 92.77 ± 0.46
  - RV-FCNN mean: 89.92 ± 1.23
  - Full ranges: CV-FCNN 93.17–92.37; RV-FCNN 91.02–88.89
- Conclusion: CV-FCNN outperforms RV-FCNN on this Pauli-input Bretigny segmentation task, showing higher mean/median and lower variance; confidence intervals reported do not overlap, supporting statistical claim of superiority.
- Acknowledgements: Délégation Générale de l'Armement (DGA) funding.

Relation to existing wiki content

This source overlaps strongly with existing pages in the wiki that document the same dataset and experiments. Relevant existing pages include:

- [[publications/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023.md]] — longer OJSP journal article that expands on Pauli vs coherency comparison, dataset-splitting effects and class-balancing. The OJSP paper contains a more complete study (multiple model families, balancing strategies) and cites similar Bretigny experiments.
- [[sources/igarss-2022-complex-valued-polsar-pauli-representation.md]] and [[sources/complex-valued-polsar-pauli-bretigny-2022.md]] — related entries in the wiki summarising the IGARSS/Conference-level descriptions of CV-FCNN on Pauli Bretigny data.
- [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]] and [[publications/complex-valued-vs-real-valued-neural-networks.md]] — methodological background on capacity-equivalence (r-scaling / ratio-tp) and other CVNN vs RVNN comparisons used across experiments.

Contradictions / notes

- No substantive contradictions detected between this raw SONDRA workshop PDF and existing wiki pages: reported accuracy numbers and high-level conclusions (CV-FCNN > RV-FCNN on Pauli Bretigny) are consistent with the existing summaries (IGARSS/ICIP/OJSP items). The numerical test-accuracy values in this raw file match the approximations already recorded in the wiki (≈92.8% vs ≈89.9%).
- Duplication / provenance: the content of this source overlaps (partial duplication) with other entries that describe the same experiments at conference/journal stages. This raw file appears to be a short workshop/conference report (SONDRA) that shares authors, dataset and core results with later, more detailed publications (notably the OJSP 2023 article). Where present, prefer the peer-reviewed / journal versions for canonical citation; retain this source as a provenance record (workshop/raw submission) and for exact wording/figures that appear only here.

Extracted actionable items / pointers

- Code / reproducibility: referenced Zenodo code archive (10.5281/zenodo.5821229) contains the exact model code used in the paper simulations — see indicated DOI for reproducibility.
- Dataset split caution: the paper emphasises the risk of biased splits when sampling with sliding windows; the vertical split strategy (70/15/15) is documented here and is consistent with later methodological notes in the OJSP article.

References found in the raw file (selected)

- Cao et al., "Pixel-wise PolSAR image classification via a novel complex-valued deep fully convolutional network", Remote Sensing, 2019. (used as architecture reference)
- Barrachina J. A., "Complex valued neural networks (cvnn)", Zenodo 10.5281/zenodo.4452131 (toolbox)
- Code archive: NEGU93/polsar_cvnn, Zenodo 10.5281/zenodo.5821229

Raw file location

Original raw file ingested: C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/publications/SONDRA_Workshop.pdf


Notes for maintainers

- This page is a raw-source ingestion and should be used as provenance for the Bretigny Pauli CV-FCNN experiments. For canonical citations, link readers to the journal OJSP article [[publications/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023.md]] where broader experiments and peer review are available.

