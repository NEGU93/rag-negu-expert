---
title: "Réseau de Neurones Profond à Valeurs Complexes pour les Applications Radar"
authors:
  - "José Agustín Barrachina"
tags:
  - publications
  - report
  - CVNN
  - PolSAR
  - radar
  - ONERA
sources:
  - "raw/publications/JDD_ONERA_2022.pdf"
updated: "2026-05-17"
---

Summary

This is a 2022 ONERA report (JDD / internal PhD yearly document) by José Agustín Barrachina summarizing work on Deep Complex-Valued Neural Networks (CVNN) for radar / PolSAR applications. It documents theory, implementation (TensorFlow-based CVNN code), synthetic and real PolSAR experiments (Oberpfaffenhofen PolInSAR dataset and ONERA Bretigny Pauli dataset), and lists related publications and invited talks.

Context

- Motivation: most deep learning architectures use real-valued representations, but radar/PolSAR sensors naturally produce or are processed into complex-valued representations (Hilbert/Fourier transforms, Pauli vector, coherency matrices). Phase information can be critical for target characterization (delay/phase, Doppler).
- Prior work indicates CVNNs can outperform RVNNs when phase information is important; however, CVNN adoption has been limited by design/learning issues (complex activations, non-holomorphic losses → Wirtinger calculus, complex SGD).

Objectives

- Explore theory and implementation of CVNNs and evaluate their performance for radar tasks: classification, segmentation, change detection.
- Compare CVNNs to capacity-equivalent RVNNs (the report notes a novel definition for generating capacity-equivalent RVNNs was developed and submitted for publication).

Implementation notes

- Implementation built on TensorFlow (author references a TensorFlow CVNN toolbox; see [[publications/complex-valued-vs-real-valued-neural-networks]] and [[publications/theory-and-implementation-of-complex-valued-neural-networks]] for related released code and tutorials).
- Uses Wirtinger calculus for gradients when cost is non-holomorphic.
- Complex activations: Type-A (cartesian ReLU) used in experiments; complex He initialization and Adam optimizer mentioned.
- Added support for convolutional layers and upsampling to implement CV-CNN, CV-FCNN and U-Net style architectures.

Experiments & results (high-level)

1) Synthetic non-circular dataset (classification, MLPs)
- Generated a synthetic non-circular database to study CV-MLP vs RV-MLP under controlled non-circularity (unequal real/imag variances, non-zero correlation between real and imaginary parts).
- Ran ~100 scenarios varying hyperparameters (hidden layers, neurons, dropout, training set size, non-circularity parameters). Statistics aggregated over 1000 trials per network.
- Findings: CV-MLP outperformed RV-MLP in nearly all setups: higher median accuracy and lower variance. RV-MLP strongly benefited from adding a second hidden layer (e.g. median accuracy from ~75% → 93%), while CV-MLP improved from ~95% → 97% (less sensitivity). RV-MLP showed significant overfitting when no dropout was used (accuracy drop from 93% → 66%), whereas CV-MLP dropped < 2%.

2) PolInSAR Oberpfaffenhofen dataset (open-source ESA PolInSAR)
- Reported overall median accuracy (OA): CVNN = 90.00 ± 0.07% vs RVNN = 89.45 ± 0.06% (small absolute gap but non-overlapping confidence intervals; CVNN median exceeded even the maximum RV-MLP test accuracy in experiments). The report presents predictions and ground-truth visualizations.

3) ONERA Bretigny Pauli PolSAR dataset (Pauli-vector input)
- The report argues that coherency matrices (averaging) may be suboptimal for CVNNs: diagonal entries are real, averaging loses pixel-wise information and is a non-trainable smoothing operation. Therefore experiments on ONERA's Bretigny Pauli-vector data were performed (Pauli vector k ∈ C^3 used as input) and CV-FCNN / RV-FCNN comparisons were run.
- CV-FCNN generalized better during training than RV-FCNN in reported experiments (accuracy & loss curves shown). Visual predictions show higher fidelity on training regions and lower performance on validation/test regions for both models, but CV-FCNN shows improved generalization.

Notes on dataset splitting & patching

- The report highlights a common pitfall: sliding-window patch extraction with overlapping patches can cause train/val/test leakage (shared pixels/ground-truth). To avoid this, the author splits the image vertically into 3 sub-images (70% training, 15% validation, 15% test) to prevent overlap and spatially close pixels from contaminating splits.

Publications referenced (as listed in the report)

- Barrachina, J. A., Ren, C., Morisseau, C., Vieillard, G., & Ovarlez, J.-P., Complex-Valued vs. Real-Valued Neural Networks for Classification Perspectives: An Example on Non-Circular Data, IEEE-ICASSP, 2021. (see [[publications/complex-valued-vs-real-valued-neural-networks]]).
- Barrachina, J. A., Ren, C., Vieillard, G., Morisseau, C., & Ovarlez, J.-P., About the equivalence between Complex-Valued and Real-Valued Fully-Connected Neural Networks - Application to PolInSAR Images, IEEE-MLSP, 2021. (see [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]]).
- Barrachina, J. A., Ren, C., Vieillard, G., Morisseau, C., & Ovarlez, J.-P., Complex-Valued Neural Networks for Polarimetric SAR Segmentation using Pauli Representation, IEEE-ICASSP, May 2022, submitted.
- Barrachina, J. A., Ren, C., Vieillard, G., Morisseau, C., & Ovarlez, J.-P., Complex-Valued Neural Networks for PolSAR Applications, IGARSS 2022, invited to special session (submitted).

Invited talks / posters (selected)

- GdRAI 2020 (CNRS INS2I) — (event cancelled due to COVID).
- XXII Giambiagi Winter School "Deep learning and Artificial Intelligence in physics", UBA, Buenos Aires, Nov 2020.
- Seminaire AI and Physique ONERA, April 2021.
- 2nd Workshop on Artificial Intelligence for Aerospace, DLR/ONERA, Oct 2021.

References (selected from report)

- A. Hirose, Complex-valued neural networks: Advances and applications, Vol. 18, John Wiley & Sons (2013).
- Hirose & Yoshida, Gen. characteristics of CV feedforward NNs relative to signal coherence, IEEE TNNLS (2012).
- Zhang et al., Complex-Valued CNN and its application in PolSAR classification, IEEE TGRS (2017).
- Hänsch & Hellwich, Classification of polarimetric SAR data by complex-valued neural networks (2009, 2010).
- Mönning & Manandhar, Evaluation of CVNNs on real-valued tasks, arXiv:1811.12351 (2018).
- Cao et al., Pixel-wise PolSAR classification via complex-valued deep FCN, Remote Sensing, 2019.
- Barrachina J. A., Complex Valued Neural Networks (CVNN) (Oct 2021), DOI: 10.5281/zenodo.4452131 (toolbox reference).
- Vasile & Totir, Circularity in PolSAR and multi-pass InSAR, IGARSS 2012.
- Barbaresco & Chevalier, Noncircularity exploitation in radar (2008).
- Formont et al., Statistical classification for heterogeneous polarimetric SAR images (ONERA Bretigny dataset reference, 2010).

Related wiki pages

- Implementation & theory / toolbox: [[publications/theory-and-implementation-of-complex-valued-neural-networks]]
- Earlier arXiv comparison: [[publications/complex-valued-vs-real-valued-neural-networks]]
- MLSP equivalence paper: [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021]]
- J. Signal Processing Systems journal article (equivalence + Oberpfaffenhofen experiments): [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]]
- Bretigny / Pauli results in wiki: [[sources/igarss-2022-complex-valued-polsar-pauli-representation]] and [[sources/complex-valued-polsar-pauli-bretigny-2022]]

Contradictions with existing wiki

- Oberpfaffenhofen OA discrepancy: This report (JDD_ONERA_2022) cites experiments on the ESA PolInSAR Oberpfaffenhofen dataset giving CVNN OA = 90.00 ± 0.07% vs RVNN = 89.45 ± 0.06% (small but statistically-separated improvement). However, the journal paper summarized at [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022]] reports much higher OA on Oberpfaffenhofen (OA 98.55%, AA 98.14%).
  - Possible explanations: different model families / architectures, different pre-processing or label sets, different train/val/test splits, or use of capacity-equivalent RVNN scaling (r-scaling) in the journal paper but not in the experiments summarized here. The report itself emphasizes dataset splitting methods and representation choices (Pauli vs coherency) which can substantially affect results. This contradiction is noted here for future reconciliation: check exact dataset version, split procedure, input representation (Pauli vs coherency T), and architecture details between the JDD report and the journal submission.

- Bretigny / Pauli vs previously-catalogued Bretigny results: The wiki already contains Bretigny Pauli results entries (e.g. [[sources/igarss-2022-complex-valued-polsar-pauli-representation]] and [[sources/complex-valued-polsar-pauli-bretigny-2022]]) reporting CV-FCNN ≈ 92.8% vs RV-FCNN ≈ 89.9% on Bretigny. The JDD report presents qualitative CV-FCNN > RV-FCNN results on Bretigny and discusses Pauli advantages; reported numeric values in this report are not directly identical to the 92.8/89.9 numbers but are consistent in direction (CV > RV). Treat as complementary; reconcile exact numeric differences by checking which model/configuration produced the 92.8% figure (likely later/expanded experiments).

Action items for maintainers / future ingest

- Link raw PDF and extract exact experiment configs (seeds, split indices, architecture hyperparameters) to reconcile the Oberpfaffenhofen OA discrepancy with the journal paper.
- If possible, cross-check the raw code release (Zenodo / GitHub NEGU93/cvnn) and supplementary material of the journal paper to align reported metrics.

---

(File source: raw/publications/JDD_ONERA_2022.pdf)
