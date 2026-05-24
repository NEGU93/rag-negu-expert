---
title: "raw github polsar cvnn"
tags:
  - raw
  - github
  - polsar
  - cvnn
sources:
  - "raw/github/polsar_cvnn.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/polsar_cvnn.md
Category: github

This page captures the raw GitHub README for the repository "PolSAR CVNN" (NEGU93/polsar_cvnn). The repository implements PolSAR classification / segmentation using complex-valued neural networks and provides scripts to run experiments, dataset handlers, and model definitions. The raw README also publishes a Zenodo DOI and provides a BibTeX/software citation.

Key metadata and highlights

- Repository name (top of README): PolSAR CVNN
- Author / owner: J Agustin Barrachina (NEGU93)
- DOI / Zenodo: 10.5281/zenodo.5821229 — citation recommended for code use
- PyPI badge present for the `cvnn` package (https://pypi.org/project/cvnn/)
- Primary purpose: PolSAR classification / segmentation using complex-valued neural networks (CVNNs)

Citation / software reference

The README includes a Zenodo citation and a BibTeX entry. The suggested citation (as captured) is:

"J Agustin Barrachina. (2022). NEGU93/polsar_cvnn: Antology of CVNN for PolSAR applications (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5821229"

BibTeX (from raw):

@software{j_agustin_barrachina_2022_5821229,
  author       = {J Agustin Barrachina},
  title        = {{NEGU93/polsar\_cvnn: Antology of CVNN for PolSAR 
                   applications}},
  month        = jan,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.5821229},
  url          = {https://doi.org/10.5281/zenodo.5821229}
}

Code usage (summary)

1. Install dependencies, including the `cvnn` Python package (PyPI).
2. Clone the repository.
3. Run `principal_simulation.py` with optional CLI arguments. The README documents the CLI usage and options (dataset selection, model choice, epochs, early stopping, balancing strategies, real vs complex mode options, dropout configuration, coherency usage, etc.).

Example CLI options (abridged):
- --dataset_method: random | separate | single_separated_image
- --tensorflow: use TensorFlow backend
- --epochs: integer
- --model: fcnn | cnn | mlp | 3d-cnn
- --early_stop: apply early stopping
- --balance: loss | dataset | (other)
- --real_mode: real_imag | amplitude_phase | amplitude_only | real_only
- --dropout: three values for downsample / bottleneck / upsample
- --coherency: use coherency matrix input
- --dataset: SF-AIRSAR | SF-RS2 | OBER

Outputs produced by a run

The program creates a folder under log/<date>/run-<time>/ with:
- tensorboard/ (TensorBoard logs)
- checkpoints/ (saved model weights for best validation loss)
- prediction.png (predicted full-image for the best model)
- model_summary.txt (simulation metadata)
- history_dict.csv (losses and metrics per epoch)
- <dataset>_confusion_matrix.csv (confusion matrices)
- evaluate.csv (loss and metrics for datasets / full image)

Datasets supported / instructions

- San Francisco (SF-AIRSAR / SF-RS2): README points to paper describing labels and images and requires dataset folder structure matching https://github.com/liuxuvip/PolSF. Users must set `root_path` in `San Francisco/sf_data_reader.py`.

- Oberpfaffenhofen (OBER): labels from https://github.com/fudanxu/CV-CNN/blob/master/Label_Germany.mat and image from ESA PolSARPro toolbox download. Update `root_path` in `Oberpfaffenhofen/oberpfaffenhofen_dataset.py`.

- Own datasets: implement a class inheriting from `PolsarDatasetHandler` with at least `get_image` (returns 3D numpy image array, channels may be complex-valued) and `get_sparse_labels` (sparse label array). Add dataset metadata to `DATASET_META` and register in `_get_dataset_handler` in `principal_simulation.py`.

Models supported (as listed in README)

- FCNN (Cao et al.)
- CNN (Zhang et al., and related works Sun et al., Zhao et al., Qin et al.)
- MLP (Haensch et al.-style)
- 3D-CNN (Tan et al.)

Guidelines to add custom models

1. Create a TensorFlow model (use `cvnn` for complex layers if needed) returning a compiled model.
2. Add a factory / entry to `_get_model` inside `principal_simulation.py`.
3. Add model name to `MODEL_META` to enable CLI selection.

Cross-references

- Related project page: [[projects/cvnn-polsar.md]] (existing CVNN-PolSAR project page in the wiki)
- Related source: [[sources/raw-github-cvnn.md]] (the `cvnn` library raw README)

Contradictions with existing wiki

- Potential duplication/overlap: The wiki already contains a project page [[projects/cvnn-polsar.md]] and a related raw-source [[sources/raw-github-cvnn-polsar.md]] for a CVNN-PolSAR codebase. The incoming raw file (NEGU93/polsar_cvnn) appears to be a separate repository (name: PolSAR CVNN) but covers largely the same domain (PolSAR segmentation with complex-valued neural networks). This may represent:
  - the same codebase under a different repository name (possible overlap), or
  - a complementary/related repository (e.g., a curated anthology/entrypoint centered on CVNN usage for PolSAR with a Zenodo release).

- Unique items in this raw file vs existing wiki pages:
  - This raw README includes a Zenodo DOI (10.5281/zenodo.5821229) and a BibTeX entry for software citation. The existing [[projects/cvnn-polsar.md]] page does not currently reference this DOI (verify and add if appropriate).
  - The README explicitly references running `principal_simulation.py` and documents CLI options; verify whether those exact scripts and option names match [[projects/cvnn-polsar.md]] or the other raw sources. If they differ, they should be reconciled (e.g., two similar repos with small differences in script names or CLI flags).

Action suggested

- Inspect [[projects/cvnn-polsar.md]] and [[sources/raw-github-cvnn-polsar.md]] to determine whether NEGU93/polsar_cvnn is the same project (renamed or reorganized) or a distinct repository. If it is the same, consolidate by adding the DOI, BibTeX and any missing CLI documentation to the existing project/source pages and mark one canonical source. If distinct, create a dedicated project page (e.g., `projects/polsar-cvnn.md`) and link to this source page.

Notes / recommended follow-ups

- Capture the repository URL (GitHub) and link to the Zenodo record page (https://doi.org/10.5281/zenodo.5821229) on the project page.
- Verify whether the `cvnn` PyPI package referenced by badge is the same library already documented in [[sources/raw-github-cvnn.md]]; if so, add cross-links and note version compatibility.
- If available, ingest the full repository tree (code, scripts, and examples) to create or update [[projects/cvnn-polsar.md]] with run examples, license, and a link to the DOI.

End of source ingest.
