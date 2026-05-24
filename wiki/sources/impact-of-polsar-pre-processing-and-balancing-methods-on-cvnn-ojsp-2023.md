---
title: "Source: Impact of PolSAR Pre-Processing and Balancing Methods on Complex-Valued Neural Networks Segmentation Tasks (OJSP 2023)"
tags:
  - sources
  - raw
  - publications
  - CVNN
sources:
  - "raw/publications/Impact_of_PolSAR_Pre-Processing_and_Balancing_Methods_on_Complex-Valued_Neural_Networks_Segmentation_Tasks.pdf"
updated: "2026-05-17"
---

Raw file (original path in repo): raw/publications/Impact_of_PolSAR_Pre-Processing_and_Balancing_Methods_on_Complex-Valued_Neural_Networks_Segmentation_Tasks.pdf

Extracted metadata

- Title: Impact of PolSAR Pre-Processing and Balancing Methods on Complex-Valued Neural Networks Segmentation Tasks
- Authors: José Agustín Barrachina; Chengfang Ren; Christèle Morisseau; Gilles Vieillard; Jean-Philippe Ovarlez
- Journal: IEEE Open Journal of Signal Processing (OJSP)
- Dates: Received 27 Oct 2022; revised 27 Jan 2023; accepted 1 Feb 2023; published 17 Feb 2023; current version 21 Mar 2023.
- DOI: 10.1109/OJSP.2023.3246391

Abstract & notable excerpts (paraphrased / extracted)

- The paper compares coherency-matrix vs Pauli-vector input representations for PolSAR segmentation using CVNNs and capacity-equivalent RVNNs across MLP, CNN and FCNN architectures.
- It emphasizes that common sliding-window sampling yields train/validation/test correlation that can saturate metrics; a stricter 70/15/15 spatial split reduces apparent OA substantially.
- Proposes dataset-balancing methods (removing or trimming single-class patches and a pixel-count balancing algorithm) and tests a weighted-loss alternative; dataset balancing generally improved AA more than the simple weighted loss tested.
- Implementation notes: Adam optimizer, He initialization adapted to complex domain, Type-A/Type-B activations discussed, categorical cross-entropy averaged on real/imaginary parts for complex outputs (LACE).
- Dataset: ONERA RAMSES Bretigny X-band PolSAR; 4 classes (Open Area, Wood Land, Built-up Area, Runway); class imbalance (Open Area ~73%).

Quantitative highlights (reported in paper)

- CV-FCNN (Pauli input) with sliding-window sampling: OA 99.83 ± 0.02%, AA 98.69 ± 0.33%.
- With stricter 70/15/15 spatial split: CV-FCNN (Pauli) OA 93.62 ± 0.20%, AA 75.31 ± 0.63%.
- Differences across architectures: FCNN benefited most from Pauli input; MLP often benefited from coherency matrix due to implicit averaging/despeckling.

Implementation / code pointers

- The paper's balancing algorithm (two-step: remove excessive single-class patches; balance remaining patches by pixel counts with ordered-per-image trimming) is described with pseudocode (Algorithm 1). The authors mention a code repository for balancing routines: github.com/NEGU93/CVNN-PolSAR (see publication for exact code references).

Relation to other ingested sources

- See conference versions / related items: [[sources/igarss-2022-complex-valued-polsar-pauli-representation.md]] (IGARSS 2022), [[sources/icip-2022-real-and-complex-valued-sar-segmentation.md]] (ICIP 2022), and the earlier Bretigny FCNN Pauli report [[publications/complex-valued-polsar-pauli-bretigny-2022.md]].

Contradictions / editorial notes

- Scientific findings align with the related IGARSS / ICIP reports in the wiki: no direct contradictions found.
- Editorial inconsistency in the wiki index: the Bretigny Pauli item appears duplicated in the current index (two identical entries). Consider deduplicating the index entries to avoid confusion.

Recommended next steps for wiki maintenance

- Add/merge this journal entry under Publications (this page) and point duplicate Bretigny/Pauli conference pages to this journal article where appropriate (note: DOIs and publication dates differ; retain conference pages but mark them as conference precursors).
- Consider adding a short note on dataset-splitting best practices under [[concepts/CVNN]] or a dataset-preprocessing concept page linking to this source.

Original raw content: PDF (full paper). For full experimental tables, figures and algorithm pseudocode consult the PDF at the raw path above or the published DOI landing page.
