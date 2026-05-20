---
title: "Comparison between Equivalent Architectures of Complex-valued and Real-valued Neural Networks - JSPS / MLSP 2021 (raw PDF)"
tags:
  - sources
  - publications
  - mlsp
  - cvnn
  - polsar
sources:
  - "raw/publications/JSPS_MLSP.pdf"
updated: "2026-05-17"
---

Source summary

This page stores the raw PDF file obtained at raw/publications/JSPS_MLSP.pdf. The PDF contains a full manuscript titled "Comparison between equivalent architectures of complex-valued and real-valued neural networks - Application on Polarimetric SAR image segmentation" by José Agustín Barrachina, Chengfang Ren, Christèle Morisseau, Gilles Vieillard and Jean-Philippe Ovarlez. The document presents definitions for building capacity-equivalent Real-Valued Neural Networks (RVNN) for fair comparison with Complex-Valued Neural Networks (CVNN), extends the equivalence concept to convolutional layers, and reports experiments on the Oberpfaffenhofen PolSAR dataset (MLSP / workshop version / preprint of later journal work).

Key metadata

- Raw file: [[raw/publications/JSPS_MLSP.pdf]]
- Authors: José Agustín Barrachina; Chengfang Ren; Christèle Morisseau; Gilles Vieillard; Jean-Philippe Ovarlez (corresponding author: jean-philippe.ovarlez@onera.fr)
- Venue / provenance: PDF formatted with "Springer Nature 2021 LATEX template" (workshop / preprint). This file corresponds to the MLSP 2021 workshop / early version of the work later published in Journal of Signal Processing Systems (see [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]] and [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]] ).

Abstract (short)

The authors perform a statistical comparison between several CVNN models and carefully constructed capacity-equivalent RVNN counterparts on the Oberpfaffenhofen PolSAR database. They propose a ratio-based tp-equivalence (r-scaling) that preserves hidden-layer aspect ratios and extend the approach to convolutional layers. Across MLP, CNN and FCNN families, CVNNs show statistically significant performance advantages when input data are complex-valued and phase information matters.

Main contents / highlights (extracted)

- Formal derivation of two equivalence strategies between CVNN and RVNN: np-equivalence (matching real-valued neuron parameters per hidden layer) and tp-equivalence (matching total real-valued trainable parameters). The paper argues maintaining aspect ratios and matching tp yields a practical r-scaling with 1 ≤ r < 2 (r → sqrt(2) for deep layers).
- Extension of the tp-equivalence (ratio-tp) to convolutional layers (derivation of r for kernels / filters while keeping kernel spatial sizes constant).
- Implementation details: Type-A complex activations (cartesian ReLU), complex Batch Normalization and pooling adaptations, complex He initialization (adaptation from He 2015), Adam optimizer for all models, categorical cross-entropy for classification (computed by averaging real/imag parts for complex outputs), Monte-Carlo trials (50 trials for image experiments).
- Models implemented: CV-MLP / RV-MLP, CV-CNN / RV-CNN, CV-FCNN / RV-FCNN (real counterparts sized using the ratio-tp methodology).
- Dataset: Oberpfaffenhofen PolSAR coherency matrix representation (six complex values per pixel after discarding redundant lower triangle), training/validation/test splits: 8% train, 2% validation (balanced per class), 90% test.
- Key reported results (test set): CV-FCNN OA median 98.55% (AA 98.14%), RV-FCNN OA median 98.23% (AA 97.79%). CV consistently outperforms RV equivalents across architectures (numbers and tables present in the PDF; Table 1 and figures illustrate OA/AA and class-wise accuracies).

Datasets, experimental settings and reproducibility

- Dataset used: Oberpfaffenhofen PolSAR (ESA) coherency matrix T input; diagonal elements treated as complex with zero imaginary part; lower triangle discarded, resulting in 6 complex-valued features per pixel.
- Sampling: authors use 8% train / 2% validation balanced per class (numbers given in the PDF: train 104,928 pixels; val 26,232 pixels).
- Code / implementation: authors note they used an open-source library for CVNNs (NEGU93/cvnn) and that exact model code is available in that repository; details match other entries in this wiki (see [[publications/theory-and-implementation-of-complex-valued-neural-networks]] and the cvnn toolbox references).

Relation to existing wiki pages

- This raw PDF is directly related to and appears to be an MLSP / preprint version of the work summarized in [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]] and to the extended journal article summarized in [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]].

Contradictions / duplication notes (explicit)

- Duplication: The content of this raw file substantially overlaps with the existing publication pages in the wiki:
  - [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]] — this wiki page already documents the MLSP 2021 contribution; the raw PDF appears to be an identical or closely related workshop/preprint (same authors and core derivations). This is a duplicate source for the same work; keep both but mark this file as the archived raw PDF.
  - [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]] — the journal article (JSPS/2022) shares title, authors and main results. The raw MLSP PDF contains the same experimental tables and OA/AA values (Table 1 OA/AA numbers match the journal-page summary in the wiki). I find no numerical contradictions between this PDF and the existing journal summary page: the OA/AA figures and main conclusions are consistent.

- Provenance ambiguity: the PDF header displays "Springer Nature 2021 LATEX template" (and the manuscript contains references to an MLSP workshop version). The wiki currently contains separate pages for both the MLSP 2021 workshop material and the later JSPS 2022 journal article. This PDF functions as a raw/preprint for the MLSP/workshop lineage and also contains content included in the later journal version. Treat this file as a workshop/preprint/raw PDF; prefer canonical citation to the published journal record where available (see [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]] for journal DOI and citation).

Actionable items / recommended follow-ups

- Mark this raw PDF as archived workshop/preprint and cross-link it on both the MLSP and JSPS/journal pages (done here by linking to both pages). If needed, update the canonical publication page ([[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]]) to include this raw PDF in its sources list for provenance.
- If repository/source code for the exact experiments referenced in the PDF (NEGU93/cvnn) is not already linked on the publication pages, add a wikilink to the code (see [[publications/theory-and-implementation-of-complex-valued-neural-networks]] for references to the cvnn toolbox).

Direct quote / excerpt (first paragraph of abstract)

"We present an in-depth statistical comparison among several Complex-Valued Neural Network (CVNN) models on the Oberpfaffenhofen Polarimetric Synthetic Aperture Radar (PolSAR) database and compare them against Real-Valued Neural Network (RVNN) architectures. The necessity to define the equivalence between the models emerges in order to compare both networks fairly. A novel definition for an equivalent-RVNN in terms of real-valued trainable parameters that maintain the aspect ratio is extended for convolutional layers based on previous work [1]. We illustrate that CVNN obtains better statistical performance for classiﬁcation on the PolSAR image across a range of architectures than a capacity equivalent-RVNN, indicating that this behavior is likely independent of the model itself."

See also

- [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]] (MLSP workshop page in this wiki)
- [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]] (Journal of Signal Processing Systems, 2022 — extended/published version)
- [[publications/theory-and-implementation-of-complex-valued-neural-networks]] (cvnn toolbox and theory paper)

