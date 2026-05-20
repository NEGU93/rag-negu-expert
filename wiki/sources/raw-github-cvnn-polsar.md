---
title: "raw github CVNN-PolSAR"
tags:
  - raw
  - github
  - cvnn
  - polsar
sources:
  - "raw/github/CVNN-PolSAR.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/CVNN-PolSAR.md
Category: github

This page captures the raw GitHub README / notes for the CVNN-PolSAR repository (complex- and real-valued neural networks for PolSAR image segmentation).

High-level summary

- Project: CVNN-PolSAR — code for real- or complex-valued neural networks applied to semantic segmentation / pixel-wise classification of Polarimetric Synthetic Aperture Radar (PolSAR) images.
- Contains: training scripts, model definitions (multiple architectures), dataset handlers, utilities for running Monte Carlo experiments, a Qt-based result viewer, and plotting/export helpers.
- Primary use-case: segmentation of PolSAR scenes using complex-valued neural networks and comparisons versus equivalent real-valued networks.
- Core code was used in multiple publications by Barrachina et al. (2021–2022). See publications list below.

Extracted metadata and important items

- Example run outputs (saved under log/<year>/<month>/<day>/run-<time>/): tensorboard/, checkpoints/, evaluate.csv, history_dict.csv, model_summary.txt, prediction.png, <set>_confusion_matrix.
- Main scripts mentioned: principal_simulation.py (main training/experiment runner), runner.py (MonteCarlo runner), qt_app.py (result viewer), results_reader.py (plot exports).
- Supported dataset names (as recognized by the code): SF-AIRSAR, OBER, FLEVOLAND, BRET, GARON.
- Dataset handler: users must implement a class inheriting from PolsarDatasetHandler (see repo src/dataset_reader.py) with required methods get_image and get_sparse_labels. Dataset metadata should be added to DATASET_META in principal_simulation.py.
- Models: several model names accepted by the CLI (cao, own, small-unet, zhang, cnn, expanded-cnn, haensch, mlp, expanded-mlp, tan). New models must return a compiled tf.Model and be registered in _get_model.
- CLI highlights: options for dataset_method, equiv_technique, --tensorflow, epochs, learning_rate, model, early_stop, balance strategies, real_mode formats (real_imag, amplitude_phase, amplitude_only, real_only), coherency averaging boxcar size, and dataset selection.
- MonteCarlo runner: python runner.py -I <n> -CF <config.json> where -CF is a config listing parameters per experiment.
- Result viewer: qt_app.py to visualize a selected saved simulation; requires setting root_drive and dataset paths in the script.
- Plot exports: results_reader.py exports Plotly or seaborn/matplotlib figures.

Publications and outputs (listed in raw)

- Barrachina, J. A., Ren, C., Morisseau, C., Vieillard, G., and Ovarlez, J.-P. (2022). "Comparison Between Equivalent Architectures of Complex-Valued and Real-Valued Neural Networks - Application on Polarimetric SAR Image Segmentation". Journal of Signal Processing Systems. https://link.springer.com/article/10.1007/s11265-022-01793-0
- Barrachina, J. A., Ren, C., Vieillard, G., Morisseau, C., and Ovarlez, J.-P. (2022). "Real- and Complex-Valued Neural Networks for SAR image segmentation through different polarimetric representations". IEEE ICIP 2022.
- Barrachina, J. A., Ren, C., Morisseau, Vieillard, G., C., and Ovarlez, J.-P. (2022c). "Merits of Complex-Valued Neural Networks for PolSAR image segmentation". GRETSI XXVIII. http://gretsi.fr/data/colloque/pdf/2022_barrachina864.pdf
- Barrachina, J. A., Ren, C., Morisseau, Vieillard, G., C., and Ovarlez, J.-P. (2022a). "Complex-Valued Neural Networks for Polarimetric SAR segmentation using Pauli representation". IGARSS 2022. (3MT finalist) https://ieeexplore.ieee.org/document/9883251
- Barrachina, J. A., Ren, C., Vieillard, G., Morisseau, C., and Ovarlez, J.-P. (2021c). "About the Equivalence Between Complex-Valued and Real-Valued Fully Connected Neural Networks - Application to PolInSAR Images". IEEE MLSP 2021. (ranked top 15% reviewer score) https://ieeexplore.ieee.org/document/9596542
- Pre-print: Barrachina, J. A., Ren, C., Morisseau, C., Vieillard, G., & Ovarlez, J. (2022). "Impact of PolSAR pre-processing and balancing methods on complex-valued neural networks segmentation tasks". arXiv: https://arxiv.org/abs/2210.17419
- Workshop: 5th SONDRA Workshop presentation and publication. https://sondra.fr/wp-content/uploads/2022/06/AI.5.pdf

Other references included in raw

- Cao et al. (2019) arXiv: https://arxiv.org/abs/1909.13299
- Zhang et al. (2017) IEEE TGRS: https://ieeexplore.ieee.org/document/8039431
- Haensch & Hellwich (2010) EUSAR: https://ieeexplore.ieee.org/document/5758871
- Tan et al. (2020) IEEE GRSL: https://ieeexplore.ieee.org/document/8864110

Supported datasets (notes from raw)

- Oberpfaffenhofen (Ober): dataset + labels (link to Label_Germany.mat referenced)
- Flevoland: dataset + labels (Label_Flevoland_15cls.mat)
- San Francisco AIRSAR (SF-AIRSAR): dataset link and labels link
- Bretigny (BRET): ONERA proprietary (not publicly distributed in the repo)
- Garon: ONERA proprietary

Contradictions with existing wiki

- No factual contradictions were detected between the content of this raw file and the current wiki.
- Note: this project is closely related to the existing [[projects/cvnn.md]] page (both concern complex-valued neural networks). There is content overlap; consider consolidating or cross-linking. This is an overlap/duplication concern rather than a contradiction.

Notes / recommended follow-ups

- Create or update a project page summarizing CVNN-PolSAR and cross-link to [[projects/cvnn.md]] and this source page.
- If dataset files are proprietary (BRET, GARON), mark them as ONERA-proprietary on the project page (already present in raw).
- Capture example configuration JSONs from src/simulations_configs if further detail is needed.


