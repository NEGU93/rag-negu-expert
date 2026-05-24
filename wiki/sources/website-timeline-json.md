---
title: "timeline.json (website raw)"
tags: [website, timeline, raw]
sources: ["raw/website/timeline.json"]
updated: "2026-05-17"
---

Summary

This page records the raw JSON timeline ingested from raw/website/timeline.json. The file lists career experiences, awards, conference and journal publications, courses, internships and other timeline events for Jose Agustin BARRACHINA (NEGU) as presented on the website.

Raw entries: 43 (mixed experiences, awards, publications, courses, internships, education and competitions).

Key extracted items (selection, chronological highlights):

- Current / recent employment
  - Machine Learning / Signal Processing Engineer R&D — Space & Communication (S&C), SAFRAN Data Systems (02/2023 — 11/2025). Role: Machine Learning Engineer in Signal Processing applications.

- Awards & competitions
  - Safran Innovation Award 2025 (10/2025) — Service Award (1st company-wise) & People's Choice Award (1st over 21 participants). Certificate URL provided in raw JSON.
  - Data WEC (World Endurance Championship) 2025 — First position over 25 participants (10/2025). Certificate: "TBD soon".
  - Data Challenge (Safran) — First position over 20 participants (12/2023). Certificate: Linked LinkedIn post.
  - IEEEXtreme participation (2014–2021 entries summarized) — multiple years with placements and certificate PDF.
  - Hackathon J.P. Morgan — 3rd prize (11/2017).

- Publications (journal & conferences)
  - Journal Publication: Open Journal of Signal Processing (OJSP) 2023 — "Impact of PolSAR pre-processing and balancing methods on complex-valued neural networks segmentation tasks" (02/2023). See [[publications/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023]].
  - Conference: IEEE ICASSP 2023 — "Impact of PolSAR pre-processing and balancing methods on complex-valued neural networks segmentation tasks" (06/2023).
  - ICIP 2022 — "Real- and Complex-Valued Neural Networks for SAR image segmentation through different polarimetric representations" (10/2022). See [[sources/icip-2022-real-and-complex-valued-sar-segmentation]].
  - IGARSS 2022 — "Complex-Valued Neural Networks for Polarimetric SAR segmentation using Pauli representation" (07/2022). See [[sources/igarss-2022-complex-valued-polsar-pauli-representation]] and [[sources/complex-valued-polsar-pauli-bretigny-2022]].
  - GRETSI 2022 — "Merits of Complex-Valued Neural Networks for PolSAR image segmentation" (09/2022).
  - MLSP 2021 — "About the equivalence between complex-valued and real-valued fully connected neural networks - application to PolInSAR images" (10/2021). See [[sources/jsps-mlsp-2021]].
  - Additional conference entries: IEEE ICASSP 2021, IEEE MLSP 2021, etc. (raw JSON contains multiple publication entries and related certificate/DOI links).

- Education, research & academic roles
  - PhD on Complex-Valued Neural Networks for radar applications — MATS / SONDRA (ONERA / CentraleSupélec / DGA). PhD student scholarship 50%-50% ONERA and DGA. Based at ONERA / SONDRA - CentraleSupélec (10/2019 — 11/2022).
  - Tenured Professor (course chair) — École Polytechnique de l'université Paris Sud, Digital Signal Processing (DSP) course (10/2019 — 12/2020). (raw JSON labels this "Tenured Professor / Chair Holder" for two years in a row.)
  - Electronic Engineering degree — Instituto Tecnológico de Buenos Aires (ITBA) (01/2012 — 12/2018). Top 2/26 students.
  - Multiple teaching assistant / associate professor and short-term academic posts (IUT Orsay, ITBA, etc.).

- Courses & certificates (selected)
  - LxMLS: 12th Lisbon Machine Learning School (07/2022) — certificate PDF included in raw JSON.
  - Coursera: Machine Learning (Andrew Ng) (09/2019) — certificate link in JSON.
  - Udacity: Intro to TensorFlow for Deep Learning (08/2021) — course link in JSON.
  - SoloLearn certificates (HTML, Python Core, Angular + NestJS) and Pluralsight course (Angular: The Big Picture) listed with links.

Notes, data quality issues and contradictions (explicit):

1) Date inconsistencies within raw JSON (startDate > endDate):
   - Intern, LPSC - IN2P3 (Centre National de la Recherche Scientifique): startDate = 03/2019, endDate = 09/2018 (end before start). Raw JSON contains these inverted dates.
   - Software System Engineer Intern, Cisco Systems: startDate = 04/2018, endDate = 11/2017 (end before start).
   - Several short internships / entries show endDate earlier than startDate or obvious ordering errors; these require source verification and correction before canonicalization.

2) Placeholder / incomplete fields found in raw JSON:
   - Several entries use "TBD soon" or "Link." or empty strings for certificate/enterprise fields (for example: Data WEC 2025 certificate = "TBD soon", several GRETSI/other conference entries show "Link." placeholder).

3) Potential semantic/terminology contradictions with existing wiki pages:
   - The raw timeline labels a role at École Polytechnique de l'université Paris Sud as "Tenured Professor / Chair Holder" (10/2019 — 12/2020). This phrasing is atypical ("tenured professor" implies a permanent academic rank) while other pages and career timeline (PhD student 2019–2022) suggest short-term course lecturing/tutoring positions. Flagged for verification: reconcile with teaching assistant / course chair records in other pages.

4) Membership and expiration dates:
   - IEEE Membership entry in raw JSON is 01/2015 — 12/2018. It is unclear whether membership continued beyond 2018; existing wiki pages may treat IEEE membership as ongoing. Confirm current status.

5) Duplicates & overlaps:
   - Multiple publication entries (conference & journal) for the same work appear across the timeline (e.g., IGARSS / ICIP / OJSP versions of related work). These are consistent as separate venue records but should be linked to canonical publication pages where possible (see links above).

Cross-references (from raw entries to existing wiki pages):
- [[publications/impact-of-polsar-pre-processing-and-balancing-methods-on-cvnn-ojsp-2023]] (OJSP 2023 item in the JSON)
- [[sources/icip-2022-real-and-complex-valued-sar-segmentation]] (ICIP 2022)
- [[sources/igarss-2022-complex-valued-polsar-pauli-representation]] and [[sources/complex-valued-polsar-pauli-bretigny-2022]] (IGARSS 2022)
- [[sources/jsps-mlsp-2021]] (MLSP 2021 preprint)

Recommended follow-up actions

- Verify and correct date inconsistencies where startDate > endDate by consulting original certificate PDFs, LinkedIn posts, publication metadata (DOIs / conference proceedings) or the website source owner.
- Replace placeholder certificate/Link entries with final URLs where available (several entries include "TBD soon" or "Link.").
- Reconcile the "Tenured Professor" phrasing with other academic role records; confirm whether the role was temporary/adjunct/lecturer versus tenured faculty.
- Map each publication entry to the canonical [[publications/...]] or [[sources/...]] page (some are already present in the wiki; link the rest).

Full raw JSON source is recorded at raw/website/timeline.json and this page is a verbatim ingest of that file with minimal transformation. Any corrections should be applied to the canonical pages (publications, events, education) after verification.
