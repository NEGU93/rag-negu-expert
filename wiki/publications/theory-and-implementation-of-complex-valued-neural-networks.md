---
title: "Theory and Implementation of Complex-Valued Neural Networks"
authors:
  - "J. A. Barrachina"
  - "C. Ren"
  - "G. Vieillard"
  - "C. Morisseau"
  - "J.-P. Ovarlez"
tags:
  - publications
  - arXiv
  - CVNN
  - complex-valued
sources:
  - sources/arxiv-2302-08286v1.md
updated: "2026-05-17"
---

Citation: arXiv:2302.08286v1 [stat.ML], 16 Feb 2023

Summary

This paper (arXiv:2302.08286v1) provides a detailed tutorial-style treatment of Complex-Valued Neural Networks (CVNN): theory (Wirtinger calculus, complex backpropagation), practical modules (complex layers, pooling, upsampling, batch norm, activation functions), initialization rules for complex weights, and a strong focus on a Python implementation (the cvnn toolbox built on TensorFlow). The authors also run experiments that show (a) the importance of adapting common initializers (Glorot / He) to the complex domain and (b) potential benefits of CVNNs for real-valued signals after casting them to the complex plane via the Hilbert transform.

Key contributions / highlights

- Comprehensive derivation of CVNN training using Wirtinger calculus and how reverse-/forward-mode autodiff operate on complex variables.
- Practical implementation details for a TensorFlow-backed cvnn toolbox: complex layers (ComplexConv2D, ComplexDense, ComplexDropout, ComplexBatchNormalization), various complex activation families (Type-A cartesian, Type-B polar, modReLU, zReLU, cardioid, multi-valued neuron), pooling/upsampling variants and real-output activations for classification.
- Complex-aware initialization: derivation and experiments showing that naive application of Glorot/Xavier to real and imaginary parts independently degrades performance; correct complex Glorot requires scaling (divide real/imag parts by sqrt(2) relative to the real-valued limits). Comparisons among Glorot (uniform/normal) and He initializers (and variants) are reported.
- Empirical experiments on both complex and real signals (real signals converted to analytic/complex via the Hilbert transform) showing CVNNs can outperform equivalent RVNNs in some classification tasks (authors report ~4% improvement in one chirp classification experiment), but note models were not fully equivalent so results should be interpreted cautiously.
- Releases and community metrics for the cvnn toolbox (PIP / Anaconda / Zenodo / GitHub activity and documentation on ReadTheDocs).

Implementation and resources

- The paper documents and references the cvnn toolbox (TensorFlow-backed) used by the authors. The toolbox appears in our wiki as the project [[projects/cvnn-polsar.md]] (PolSAR application) and is referenced in the related publication [[publications/complex-valued-vs-real-valued-neural-networks.md]].
- The authors note TensorFlow's and PyTorch's evolving support for complex tensors and autodiff (Wirtinger convention); the toolbox was originally implemented on TensorFlow (development started 2019).

Experiments (brief)

- Signal classification experiments: chirps, PSK, QAM, noise classes; real signals converted with the Hilbert transform to form analytic (complex) signals.
- Initialization study: multiple runs (statistical tests) comparing different scalings of Glorot uniform for complex weights; clear performance drop when Glorot is applied naively to real and imaginary parts without the sqrt(2) adaptation.
- Comparison CV-MLP vs RV-MLP on Hilbert-transformed real data: authors report CV-MLP less overfitting and higher accuracy in their experiments, but explicitly caution that networks were not strictly equivalent and further study is needed.

Relation to other wiki entries

- This paper complements the results in [[publications/complex-valued-vs-real-valued-neural-networks.md]] (earlier arXiv work by the same authors) by providing a deeper treatment of implementation details and tooling (cvnn) and by stressing initialization and practical modules.
- The project [[projects/cvnn-polsar.md]] uses the same cvnn toolbox mentioned here.

Contradictions with existing wiki

- I checked existing wiki pages related to CVNN (notably [[publications/complex-valued-vs-real-valued-neural-networks.md]] and [[projects/cvnn-polsar.md]]). I did not find factual contradictions between this paper and those pages. The papers are complementary (different arXiv IDs / different focuses). If you want, I can flag any minor discrepancies (dates / download counts / GitHub star counts) that will naturally diverge over time; none of those affect the technical claims.

Notes / recommended followups

- Consider linking the cvnn toolbox project page [[projects/cvnn-polsar.md]] to this publication page (already referenced) and add a short excerpt on the complex-initializer recommendation (Glorot scaling) to the project's README or notes so users reusing the code apply the correct initialization.
- The authors caution that experiments comparing CVNN vs RVNN on transformed real data require careful equivalence of architectures; mark the experimental results as preliminary.
