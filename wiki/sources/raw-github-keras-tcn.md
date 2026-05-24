---
title: "raw github keras tcn"
tags:
  - raw
  - github
  - keras
  - tcn
  - temporal-convolutional-network
sources:
  - "raw/github/keras-tcn.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/keras-tcn.md
Category: github

This page captures the raw GitHub README for the "keras-tcn" project (Keras Temporal Convolutional Network implementation / wrapper).

High-level summary

- Project: Keras TCN — Temporal Convolutional Network utilities and a Keras Layer wrapper implementing TCNs (based on the TCN literature: Wavenet-style dilated causal convolutions and the 2018 empirical TCN paper).
- Main contents in raw README: motivation for using TCN over RNNs, usage/installation, TCN() class signature and arguments, notes on receptive field calculation, examples/tasks (adding problem, copy memory, sequential MNIST, Word PTB), recommended parameter choices, pip install instructions, references and citation.
- Tested TensorFlow versions (as stated in raw): 2.9 — 2.17 (note: tests listed up to 2.17 with date Jul 18, 2024).

Extracted metadata and important items

- PyPI install: pip install keras-tcn (also --no-dependencies is suggested when TF/Numpy already present).
- MacOS M1 note: pip install --no-binary keras-tcn keras-tcn recommended to force source build; ensure grpcio and h5py are correctly installed.
- TCN Class signature (from README):

```python
TCN(
    nb_filters=64,
    kernel_size=3,
    nb_stacks=1,
    dilations=(1, 2, 4, 8, 16, 32),
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.0,
    return_sequences=False,
    activation='relu',
    kernel_initializer='he_normal',
    use_batch_norm=False,
    use_layer_norm=False,
    use_weight_norm=False,
    go_backwards=False,
    return_state=False,
    **kwargs
)
```

- Input shape: 3D tensor (batch_size, timesteps, input_dim). timesteps may be None.
- Output shape: if return_sequences=True -> (batch_size, timesteps, nb_filters), else (batch_size, nb_filters).
- Receptive field: README includes formula and illustrative images. It explains how kernel_size, dilations, number of residual blocks/stacks affect receptive field. (Images referenced in raw come from repository misc/ and user images.)

Notes on parameters and usage (summary of recommendations in raw README)

- nb_filters: akin to RNN units; increases model capacity.
- kernel_size: typically 2–8; larger increases model size.
- dilations: often powers of two; receptive field grows with dilations and stacks.
- nb_stacks: useful for very long sequences (waveforms with many timesteps).
- padding: 'causal' to avoid information leakage for temporal prediction; 'same'/'valid' for non-causal.
- use_skip_connections: recommended for stable training unless performance degrades.
- dropout_rate: small values (e.g. 0.05) typically used; similar role to recurrent_dropout.
- Normalizations: batch/layer/weight norm options available; use when model and data are large enough.

Non-causal TCN

- README explains making the TCN non-causal (padding='same' or 'valid') to allow future context at prediction time (not suitable for online/real-time streaming prediction).

Included example tasks in the repository (documented in README)

- Word PTB (word-level Penn Treebank) language modeling: TCN can outperform LSTM with comparable parameters (figure present in raw).
- Adding problem: regression task demonstrating long-term dependency learning; sample training logs shown in README.
- Copy memory task: extreme long-range dependency test; sample training/accuracy logs included.
- Sequential MNIST: treats MNIST as a 784-length sequence; reported training/validation accuracies shown in README.

Reproducibility

- README suggests using NVIDIA tensorflow-determinism for reproducible GPU results; notes user test by @lingdoc.

References (from raw README)

- https://github.com/locuslab/TCN/ (PyTorch TCN reference)
- https://arxiv.org/pdf/1803.01271 ("An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling")
- https://arxiv.org/pdf/1609.03499 (WaveNet original paper)
- https://github.com/Baichenjia/Tensorflow-TCN (Tensorflow Eager implementation referenced)

Citation (as provided in raw)

```
@misc{KerasTCN,
  author = {Philippe Remy},
  title = {Temporal Convolutional Networks for Keras},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/philipperemy/keras-tcn}},
}
```

Contributors

- README includes a contributors badge and points to the GitHub contributors graph for philipperemy/keras-tcn.

Cross-references

- This project is a temporal-sequence modelling library/tool; for other sequence / neural-network projects in this wiki, see e.g. [[projects/cvnn-polsar.md]] (CVNN-PolSAR) for contrast between convolutional and complex-valued models.

Contradictions with existing wiki

- No contradictions detected with existing wiki pages at the time of ingest.

Notes / recommended follow-ups

- Consider creating a project page at [[projects/keras-tcn]] summarizing the repository with link to upstream GitHub and key usage examples (installation, example tasks). The README contains runnable examples under tasks/ (adding_problem, copy_memory, mnist_pixel) which would be helpful to list on a project page.
- Verify whether the upstream repository has been updated past the README's listed TF test range (up to TF 2.17 as of Jul 18, 2024) and update compatibility notes if needed.

---
