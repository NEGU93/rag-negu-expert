---
title: "raw github cvnn"
tags:
  - raw
  - github
  - cvnn
  - library
sources:
  - "raw/github/cvnn.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/cvnn.md
Category: github

This page captures the raw GitHub README / project notes for the "cvnn" repository by J. Agustin Barrachina (NEGU93). The file is a README-style description of the Complex-Valued Neural Networks (CVNN) Python library that builds on TensorFlow to provide complex-valued layers and utilities.

Extracted metadata & highlights

- Project name: cvnn
- Author / maintainer: @NEGU93 (J. Agustin Barrachina)
- Badges present in README: arXiv paper, ReadTheDocs, PyPI, Anaconda (conda), Zenodo DOI, pepy downloads badge, GitHub stars, lines-of-code (tokei).
- Deprecation / compatibility note: the README includes a prominent WARNING block: "This library is deprecated. In particular, it seems not to work correctly with TF version 2.16+."
- Purpose: provide complex-valued NN layers and utilities using TensorFlow as backend so users can write models similarly to tf.keras but using cvnn.layers instead of tf.keras.layers.
- Claimed novelty: library authors state this was the first library (at time of writing) that actually works with complex dtypes rather than using real-valued encodings.
- Updates / context: README mentions improvements in PyTorch (Complex32, complex convolutions) and other libraries (complexPyTorch) and notes the state of complex support across frameworks as of 2020–2022.

Installation

- conda (Anaconda channel negu93):

```
conda install -c negu93 cvnn
```

- pip:

```
pip install cvnn
```

Documentation

- Primary documentation hosted on Read the Docs: https://complex-valued-neural-networks.readthedocs.io/en/latest/index.html

Key usage notes / examples

- Core concept: when using this library you replace tf.keras.layers imports with cvnn.layers (or import as complex_layers).
- Example shown for Sequential API and Functional API. Important detail: the library requires a ComplexInput layer at the start of models and requires an activation at the last layer that converts complex outputs to real (e.g., abs) because loss functions cannot minimize complex numbers directly.

Short Sequential example (from raw):
- Use complex_layers.ComplexInput as first layer.
- Use ComplexConv2D, ComplexPooling, ComplexFlatten, ComplexDense, etc.
- Final layer activation example: "convert_to_real_with_abs".

Functional API example (from raw):
- complex_layers.complex_input(...) to create inputs, then ComplexConv2D, ComplexConv2DTranspose, etc., composing with tf.keras.layers.concatenate where appropriate.

Project status & maintenance

- README states maintenance has been reduced/stopped because the author works full-time; the author welcomes forks or volunteers to maintain the project.

About the author & motivation

- Personal website: https://negu93.github.io/agustinbarrachina/
- Academic affiliation in README: PhD student at École CentraleSupélec with scholarship from ONERA and DGA. Focus: Complex-Valued Neural Networks in PhD research.

Citations

- Zenodo citation provided in README (BibTeX snippet included). Recommended to prefer Zenodo citation.

Issues / testing

- Issues link: https://github.com/NEGU93/cvnn/issues
- Tests: repository tested using pytest (README shows pytest logo).

Cross-references

- Project page in this wiki: [[projects/cvnn.md]] (see project entry for summary/verification)
- Related project using CVNN ideas for PolSAR segmentation: [[projects/cvnn-polsar.md]]
- Related doctoral attestation in this wiki (thesis topic is CVNN): [[sources/psaclay-attestation-de-reussite.md]]

Contradictions with existing wiki

- No direct contradictions detected with the existing [[projects/cvnn.md]] entry. The existing project index entry contains a caution to "verify externally" claimed download/star counts; the raw README supplies badges and usage notes but also includes a deprecation warning (TF 2.16+ incompatibility). If the project page or other wiki pages claim the library is actively maintained or compatible with the latest TensorFlow versions, that would contradict this README; at the time of ingest no such conflicting claim was found in the wiki.

Notes & recommended follow-ups

- Verify external metadata (PyPI download counts, GitHub star counts, conda package presence and versions) before repeating numeric claims in public-facing pages — the README badges are useful but counts change over time.
- The README's deprecation/compatibility note is important; surface this information on the project summary page [[projects/cvnn.md]] so visitors are warned.
- If migrating or updating the library is desired, check the PyTorch/complex dtype advances noted in the README (PyTorch complex32 / complex convolutions support from 2021–2022) and assess whether a port to PyTorch is feasible.

Full raw content (captured as-is in source archive)

- The original raw file contains the complete README including badges, warning box, installation commands, short examples (code blocks), sequential & functional API examples, project status note, author motivation, Zenodo citation BibTeX, issues link, and pytest logo reference.


