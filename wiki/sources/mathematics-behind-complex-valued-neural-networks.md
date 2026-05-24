---
title: "Mathematics behind Complex Valued Neural Networks (raw report)"
authors:
  - "José Agustín Barrachina"
  - "Jean-Philippe Ovarlez (supervisor)"
  - "Chengfang Ren (co-supervisor)"
tags:
  - publications
  - report
  - CVNN
  - wirtinger-calculus
  - backpropagation
  - autodiff
sources:
  - "raw/publications/Mathematics_of_CVNNs.pdf"
updated: "2026-05-17"
---

Summary

This is a raw ingest of the PDF titled "Mathematics behind Complex Valued Neural Networks" by José Agustín Barrachina (cover date: 25 July 2025). The report is a compact mathematical primer for Complex-Valued Neural Networks (CVNNs): complex algebra basics, holomorphic functions and Liouville's theorem, Wirtinger calculus, relevant chain rules, complex backpropagation, and aspects of forward/reverse automatic differentiation (including dual-number-style forward-mode and directional/Wirtinger derivatives).

Key points / contents (high-level)

- Purpose: Provide all mathematical tools needed to understand and implement CVNN optimization (backpropagation) with minimal prerequisites in complex algebra.
- Chapters: Introduction; Mathematical background (complex identities, holomorphic functions, Liouville theorem, Wirtinger calculus, chain rules); Backpropagation for fully connected CVNNs; Automatic differentiation (forward and reverse modes); Conclusion and bibliography.
- Notable technical coverage:
  - Clear derivation and use of Wirtinger calculus to handle non-holomorphic CVNN components (activations, loss over complex parameters).
  - Chain rules adapted for complex variables (including the version used for neural networks where the cost is real-valued but parameters are complex).
  - Two formulations of complex backpropagation discussed (Benvenuto & Piazza; Hänsch & Hellwich style derivations) and detailed recursive expressions for layer/weight derivatives.
  - Forward-mode automatic differentiation via a dual-number viewpoint extended to complex directions (discussion of ϵ as complex infinitesimal; relation to Wirtinger derivatives by choosing specific ϵ directions).

Relations to existing wiki pages

- This source is strongly related to and overlaps with material in [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]] (arXiv tutorial-style paper) and with the background material used across the CVNN publication set (e.g. [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021.md]], [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]]).
- Use this report as a compact, pedagogical reference for derivations of Wirtinger calculus, complex chain rule, and the step-by-step backpropagation used elsewhere in the wiki.

Contradictions / notes (explicit)

- Date / timeline contradiction: the PDF cover shows the date "July 25, 2025" but the Preface/Prefatory text inside the document states the work summarises four months (June–September 2019) performed as a SONDRA research engineer and that it served as a prelude to a PhD begun on 1 October 2019. This is inconsistent: the internal timeline (2019) contradicts the cover date (2025). Treat the cover date as requiring verification (possible re-export, repro, or scanning metadata issue). The discrepancy should be confirmed with the original raw file owner or by checking file metadata/versioning.

- Overlap with other wiki publications: substantial overlap exists between this report and the material in [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]] (arXiv 2023 tutorial). There is no explicit theoretical contradiction (both use Wirtinger calculus, Liouville theorem and similar chain-rule/backprop derivations); rather, the report appears to be a concise, earlier-style technical note/tutorial that covers many of the same derivations. Mark these as duplicated/overlapping content (useful for cross-references and for provenance tracking), not as conflicting results.

- Minor notation differences: the report uses certain notation conventions (e.g., upper-line for conjugation and some index conventions) that differ slightly from notations used in other wiki pages. No mathematical contradiction detected — only notational mismatches that require care when copying formulas between pages.

Suggested actions / provenance

- Add this source to the sources index (this page) and cross-reference it from the CVNN theory pages where derivations are quoted.
- Verify the document date/timeline with the original author or file metadata and (if necessary) add a note to the publication pages indicating the corrected year.
- If content from this report is used to expand or clarify derivations in existing pages (e.g. the tutorial arXiv page), cite this source with the raw path shown in frontmatter.

File metadata / raw file

- Raw file ingested: raw/publications/Mathematics_of_CVNNs.pdf
- Suggested canonical wiki page for citation: [[sources/mathematics-behind-complex-valued-neural-networks.md]]

Useful excerpts (to help editors)

- Abstract: report gives general insight into CVNN mathematics, shows tools to understand backpropagation with complex values; assumes little complex-number background; covers complex algebra, holomorphic functions, Liouville theorem, Wirtinger calculus, chain rule, and formal backpropagation for fully-connected CVNNs.
- Bibliography: includes standard CVNN and Wirtinger references (Amin & Murase, Hirose, Haykin, Wirtinger original, Kreutz-Delgado, Fischer appendix on Wirtinger calculus, Benvenuto & Piazza, Hänsch & Hellwich, and autodiff references such as Rall, Pearlmutter & Siskind).

See also

- [[publications/theory-and-implementation-of-complex-valued-neural-networks.md]]
- [[publications/about-equivalence-between-complex-and-real-valued-mlsp-2021.md]]
- [[publications/comparison-between-equivalent-architectures-complex-and-real-valued-polsar-2022.md]]

