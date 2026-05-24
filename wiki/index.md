# Wiki index

Catalog of all wiki pages. **Read this file first** before answering questions or ingesting sources.

## Overview

| Page | Summary |
|------|---------|
| [overview](overview.md) | Hub: who Jose Agustin BARRACHINA is, career synthesis |

| [overview](overview.md) | Career hub: current role first (GitGuardian Senior ML Engineer), PhD highlights, notable projects and links. |
## Entities

| Page | Summary |
|------|----------|
| [jose-barrachina](entities/jose-barrachina.md) | Person entity for Jose Agustin BARRACHINA (NEGU / NEGU93): current role, education, projects, contact links. |


_Persons, employers, institutions — populated by bootstrap / ingest._

## Concepts

_Technical topics (CVNN, PolSAR, RAG, etc.) — populated by bootstrap / ingest._

## Projects

_GitHub repos and major work — populated from `raw/github/`._

## Education

_Degrees, transcripts, schools — populated from `raw/education/`._

## Publications

_Papers and talks — populated from `raw/publications/`._

## Events

_Conferences, competitions, timeline — populated from `raw/events_conferences/`, `raw/comptetitions/`, `raw/website/`._

## Sources

_One summary page per ingested raw file — populated during ingest._

## Certificates & courses

| Page | Summary |
|------|----------|
| [ieee-day-2014-ticket](certificates/ieee-day-2014-ticket.md) | Participation ticket / certificate for IEEE Day 2014 (Agustin Barrachina) — event held 6 Oct 2014 in Buenos Aires. |

| [negu93](projects/negu93.md) | GitHub profile summary for Jose Agustin BARRACHINA (NEGU): senior ML engineer, PhD, author of cvnn and several LLM/agent projects. |
| [website-timeline-json](sources/website-timeline-json.md) | Raw website timeline JSON: 43 timeline entries (experiences, awards, publications, courses); contains several date inconsistencies and placeholder links. |
| [phd-website-raw](sources/phd-website-raw.md) | Raw website 'phd.txt' listing Ph.D. publications, preprints, workshop, Zenodo code DOI and poster entries (contains encoding issues and several 'Link.' placeholders). |
| [sondra-workshop-complex-valued-polsar-pauli-2022](sources/sondra-workshop-complex-valued-polsar-pauli-2022.md) | Raw SONDRA workshop PDF: CV-FCNN on Pauli-vector Bretigny PolSAR (CV-FCNN ≈92.8% vs RV-FCNN ≈89.9%); includes code DOI and dataset-split details. |
| [about-equivalence-between-complex-and-real-valued-mlp-mlsp-2021](sources/about-equivalence-between-complex-and-real-valued-mlp-mlsp-2021.md) | MLSP 2021 conference paper: tp-equivalence (r-scaling) between CV-MLP and RV-MLP; Oberpfaffenhofen PolInSAR MLP experiments showing small but significant CV-MLP advantage. |
| [mathematics-behind-complex-valued-neural-networks](sources/mathematics-behind-complex-valued-neural-networks.md) | Raw PDF: 'Mathematics behind Complex Valued Neural Networks' (Barrachina) — primer on Wirtinger calculus, complex backpropagation and autodiff; note cover date vs internal timeline discrepancy. |
| [jsps-mlsp-2021](sources/jsps-mlsp-2021.md) | Raw PDF (MLSP / preprint) of "Comparison between equivalent architectures of complex-valued and real-valued neural networks" (Barrachina et al.) — CVNN vs RVNN on Oberpfaffenhofen PolSAR. |
| [jdd-onera-2022](sources/jdd-onera-2022.md) | ONERA JDD 2022 report (Barrachina): CVNN for radar/PolSAR — synthetic non-circular experiments, Oberpfaffenhofen and Bretigny PolSAR tests, TensorFlow CVNN implementation; notes discrepancy with later journal OA on Oberpfaffenhofen. |
| [impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023](publications/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023.md) | OJSP 2023 journal article: evaluates coherency vs Pauli inputs, dataset-splitting effects, and class-balancing methods for CVNN PolSAR segmentation (Bretigny dataset). |
| [igarss-2022-complex-valued-polsar-pauli-representation](sources/igarss-2022-complex-valued-polsar-pauli-representation.md) | IGARSS 2022 paper: CV-FCNN on Pauli-vector Bretigny PolSAR — CV-FCNN (≈92.8%) outperforms RV-FCNN (≈89.9%); Pauli cannot be recovered from coherency matrices. |
| [icip-2022-real-and-complex-valued-sar-segmentation](sources/icip-2022-real-and-complex-valued-sar-segmentation.md) | ICIP 2022 paper: CV-FCNN vs equivalent RV-FCNN on San Francisco AIRSAR; Pauli-vector input + complex nets improve segmentation accuracy. |
| [complex-valued-polsar-pauli-bretigny-2022](sources/complex-valued-polsar-pauli-bretigny-2022.md) | CV-FCNN trained on Pauli-vector input (Bretigny ONERA dataset); CV-FCNN (≈92.8%) > RV-FCNN (≈89.9%); Pauli cannot be recovered from coherency matrices. |
| [complex-valued-polsar-pauli-bretigny-2022](sources/complex-valued-polsar-pauli-bretigny-2022.md) | CV-FCNN trained on Pauli-vector input (Bretigny ONERA dataset); shows CV-FCNN (92.8%) > RV-FCNN (89.9%); notes Pauli cannot be recovered from coherency matrices. |
| [hal-03841977](sources/hal-03841977.md) | ICIP 2022 conference paper (HAL hal-03841977): CV-FCNN vs RV-FCNN on San Francisco AIRSAR; shows Pauli vector input and complex models improve PolSAR segmentation. |
| [comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022](publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md) | J. Signal Processing Systems (2022) journal paper: extends CVNN vs RVNN equivalence to convolutional layers and shows CV-FCNN best on Oberpfaffenhofen PolSAR (OA 98.55%, AA 98.14%). |
| [about-equivalence-between-complex-and-real-valued-mlsp-2021](publications/about-equivalence-between-complex-and-real-valued-mlsp-2021.md) | MLSP 2021 workshop paper proposing tp-equivalence r-scaling between CV-MLP and RV-MLP; empirical PolInSAR comparison showing slight CV-MLP advantage. |
| [theory-and-implementation-of-complex-valued-neural-networks](publications/theory-and-implementation-of-complex-valued-neural-networks.md) | arXiv:2302.08286v1 — detailed theory + TensorFlow cvnn toolbox; Wirtinger calculus, complex layers, activations, and initialization guidance. |
| [complex-valued-vs-real-valued-neural-networks](publications/complex-valued-vs-real-valued-neural-networks.md) | arXiv:2009.08340v2 — Empirical comparison of CVNN vs RVNN on non-circular complex-valued classification; includes released TensorFlow CVNN library (NEGU93/cvnn). |
| [toeic-2014-itba-icana-score-roster](sources/toeic-2014-itba-icana-score-roster.md) | TOEIC score roster (Instituto Tecnologico Buenos Aires / ICANA), 09‑Sep‑2014 — Barrachina Agustín: Total 670 (L 295, R 375). |
| [tcf-2024-paris-elfe](sources/tcf-2024-paris-elfe.md) | Raw PDF: TCF (Intégration, Résidence et Nationalité) session 19 Feb 2024 — attestation for Jose Agustin BARRACHINA; detailed per-skill scores and CECRL levels. |
| [tcf-2017-palaiseau](sources/tcf-2017-palaiseau.md) | Raw PDF: TCF session 18/01/2017 (École Polytechnique, Palaiseau) — candidate table (76 candidates); extracted row for BARRACHINA Agustín (global 438, B2). |
| [raw-github-upsetplot](sources/raw-github-upsetplot.md) | Raw GitHub README/documentation for the UpSetPlot Python library (set-overlap visualization). |
| [steganography](projects/steganography.md) | Matlab GUI for JPEG steganography; authors: J. Agustín Barrachina, Augusto Viotti Bozini, Gonzalo Castelli. |
| [resistor-calculator](projects/resistor-calculator.md) | Final project (ITBA 2016): build resistor circuits and compute Thevenin equivalent; demo video included. |
| [cvnn-polsar](projects/cvnn-polsar.md) | Raw GitHub README for NEGU93/polsar_cvnn — PolSAR classification using complex-valued neural networks (Zenodo DOI 10.5281/zenodo.5821229). |
| [parallel-image-filtering](projects/parallel-image-filtering.md) | Comparative OpenMP / MPI / CUDA implementations and benchmarks for image filtering across CPU and GPU architectures. |
| [raw-github-negu93-github-io](sources/raw-github-negu93-github-io.md) | Raw GitHub note pointing to the personal website https://negu93.github.io/home (minimal README). |
| [keras-tcn](projects/keras-tcn.md) | Keras implementation / wrapper for Temporal Convolutional Networks (TCN): README, API, installation, tasks and examples. |
| [forbidden-desert](projects/forbidden-desert.md) | C++ Allegro 5 implementation of the board game Forbidden Desert (README: build deps, debug tips, rules PDF and gameplay screenshots). |
| [electrocardiogram-classification-neural-network](projects/electrocardiogram-classification-neural-network.md) | ECG heartbeat classification using an MLP (backprop); PCA and SOM used for dimensionality reduction (GitHub project note). |
| [cyusb3kit-003-with-sp605-xilinx](projects/cyusb3kit-003-with-sp605-xilinx.md) | Cypress CYUSB3KIT-003 + Xilinx SP605 integration: FX3-based FPGA programming and PC↔FPGA communication (C++ FX3 manager) |
| [raw-github-cvnn-vs-rvnn-polsar-applications](sources/raw-github-cvnn-vs-rvnn-polsar-applications.md) | Raw GitHub note for agustin_web_project — npm start and gh-pages deploy instructions. |
| [raw-github-cvnn](sources/raw-github-cvnn.md) | Raw GitHub README for the cvnn (Complex-Valued Neural Networks) library by NEGU93 — installation, examples, deprecation/TF compatibility warning, and citation. |
| [cvnn-polsar](projects/cvnn-polsar.md) | CVNN-PolSAR — complex- and real-valued neural networks for PolSAR image segmentation (GitHub repo; includes training, dataset handlers, Monte Carlo runner, and Qt result viewer). |
| [compilation](projects/compilation.md) | Mini-C to x86-64 compiler (INF564 course project, École Polytechnique). |
| [anime-generation-dcgan](projects/anime-generation-dcgan.md) | DCGAN-based anime character image generation (TensorFlow implementation; notes on dataset Danbooru, training, and results). |
| [lxmls-2022-attendance-jose-agustin-barrachina](sources/lxmls-2022-attendance-jose-agustin-barrachina.md) | Attendance certificate: Jose Agustin BARRACHINA at 12th Lisbon Machine Learning School (LxMLS), 24–29 Jul 2022, Lisbon. |
| [igarss-2022-attendance-jose-agustin-barrachina](sources/igarss-2022-attendance-jose-agustin-barrachina.md) | Attendance certificate for Jose Barrachina (Jose Agustin BARRACHINA) at IGARSS 2022 — presented paper on complex-valued neural networks for PolSAR segmentation. |
| [ieee-big-data-2017](events/ieee-big-data-2017.md) | Attendance / ticket record for IEEE AR CIS - ITBA conference “How big is too big? Clustering in Big Data with the Fantastic Four” (19 Sep 2017), Jose Agustin Barrachina — free ticket, order 670996891839151882001. |
| [icassp-2023-attendance-jose-agustin-barrachina](sources/icassp-2023-attendance-jose-agustin-barrachina.md) | Source summary: attendance certificate for Jose-Agustin Barrachina at ICASSP 2023 (Rhodes, 4–10 Jun 2023). |
| [foro-estrategico-desarrollo-nacional-2013](events/foro-estrategico-desarrollo-nacional-2013.md) | Invitation (29 Apr 2013) for Jose Agustín Barrachina to participate as 'Líder Sectorial Universitario' in 'La Sociedad y la Economía del Conocimiento en la Argentina del siglo 21' (Foro Estratégico). |
| [ecole-polytechnique-transcript-2016-2017](education/ecole-polytechnique-transcript-2016-2017.md) | Transcript of records (École Polytechnique, 2016/2017) for Jose Agustin BARRACHINA — courses, marks and grading legend (raw scan). |
| [psaclay-attestation-de-reussite](sources/psaclay-attestation-de-reussite.md) | Attestation de réussite (Doctorat, Université Paris-Saclay) — Jose Agustin BARRACHINA; thesis: Complex-valued neural networks for radar applications; defense 6 Dec 2022. |
| [raw-cv-barranchina-pdf](sources/raw-cv-barranchina-pdf.md) | Source record for raw/CV/CV_BARRACHINA.pdf — extracted highlights and metadata from PDF resume of Jose Agustin BARRACHINA (NEGU). |
| [cvnn](projects/cvnn.md) | cvnn — complex-valued neural networks library (open-source; CV claims 150k+ pip downloads and 175+ GitHub stars; verify externally before copying). |
| [deep-learning-in-physics-giambiagi-school-2020-agustin-barrachina](certificates/deep-learning-in-physics-giambiagi-school-2020-agustin-barrachina.md) | Attendance attestation (Spanish) for XXII Giambiagi School “Inteligencia artificial y aprendizaje profundo en física” (9–13 Nov 2020), 25 hours — Agustin Barrachina |
| [deep-learning-in-physics-giambiagi-school-2020-agustin-barrachina](certificates/deep-learning-in-physics-giambiagi-school-2020-agustin-barrachina.md) | Attendance attestation for XXII Giambiagi School “Artificial Intelligence and Deep Learning in Physics” (9–13 Nov 2020), 25 hours — Agustin Barrachina |
| [machine-learning-coursera-2019-jose-agustin-barrachina](certificates/machine-learning-coursera-2019-jose-agustin-barrachina.md) | Machine Learning (Coursera, Stanford) course completion certificate (24 Sep 2019) — Jose Agustin Barrachina; verification code Q8GX7847GC3A |
| [python-core-course-2018-agustin-barrachina](certificates/python-core-course-2018-agustin-barrachina.md) | Python Core course completion certificate (19 Nov 2018) — Agustin Barrachina; Certificate #8747457-1073 |
| [angular-nestjs-course-2020-agustin-barrachina](certificates/angular-nestjs-course-2020-agustin-barrachina.md) | Angular + NestJS course completion certificate (29 Jul 2020) — Agustin Barrachina; Certificate #1092-8747457 |
| [html-course-2020-agustin-barrachina](certificates/html-course-2020-agustin-barrachina.md) | HTML course completion certificate (06 Apr 2020) — Agustin Barrachina; Certificate #1014-8747457 |
| [raw-comptetitions-ieeextreme-11-stats](sources/raw-comptetitions-ieeextreme-11-stats.md) | Source record for raw/comptetitions/IEEExtreme11stats.txt — extracted scoreboard-like lines for IEEEXtreme 11.0 (ambiguous; noted participant-count discrepancy vs certificate). |
| [raw-comptetitions-ieeextreme](sources/raw-comptetitions-ieeextreme.md) | Source record for raw/comptetitions/ieeextreme.pdf (bundle of IEEEXtreme participation certificates: 2014, 2015, 2017) |
| [raw-comptetitions-ieeextreme-11-0](sources/raw-comptetitions-ieeextreme-11-0.md) | Source record for raw/comptetitions/IEEEXtreme 11.0.pdf (IEEEXtreme 11.0 participation certificate — Jose Agustin BARRACHINA, Team HardCodeCafe, 14 Oct 2017) |
| [raw-comptetitions-extreme12](sources/raw-comptetitions-extreme12.md) | Source record for raw/comptetitions/Extreme12.pdf (IEEEXtreme 12.0 participation certificate — Jose Agustin BARRACHINA, Team HardCodeCafe) |
| [raw-comptetitions-certificatextreme](sources/raw-comptetitions-certificatextreme.md) | Source record for raw/comptetitions/certificateXtreme.pdf (IEEEXtreme 8.0 participation certificate — Agustin Barrachina, AtomikTeam) |
| [ieee-xtreme-9-0-2015-asongofbitsandcoffee](certificates/ieee-xtreme-9-0-2015-asongofbitsandcoffee.md) | Participation certificate for IEEEXtreme 9.0 (24 Oct 2015) — Agustin Barrachina, Team ASongOfBitsAndCoffee. |
| [ieee-xtreme-8-0-2014-atomikteam](certificates/ieee-xtreme-8-0-2014-atomikteam.md) | Participation certificate for IEEEXtreme Programming Competition 8.0 (18 Oct 2014) — Agustin Barrachina, Team AtomikTeam. |
| [ieee-member-2022-agustin-jose-barrachina](certificates/ieee-member-2022-agustin-jose-barrachina.md) | IEEE digital membership card (Agustin Jose Barrrachina) — Graduate Student Member, Member #93157321, valid through 31 Dec 2022 |

_From `raw/certificates/`, `raw/courses/`, `raw/languages/` — cross-linked from education and overview._
