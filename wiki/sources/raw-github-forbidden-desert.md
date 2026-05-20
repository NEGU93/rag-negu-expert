---
title: "raw github forbidden desert"
tags:
  - raw
  - github
  - forbidden-desert
  - game
  - allegro
  - cpp
sources:
  - "raw/github/ForbiddenDesert.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/ForbiddenDesert.md
Category: github

High-level summary

- Project: "Forbidden Desert" — a C++ implementation / game project using Allegro 5 (README-style raw notes).
- Purpose: playable videogame adaptation (screenshots included in repo Resources); README documents build deps, debugging tips and where to find the board-game rules PDF.

Extracted important items

- Dependencies (apt-get commands listed in raw):

```
sudo apt-get install liballegro5-dev
sudo apt-get install freeglut3-dev libftgl-dev
sudo apt-get install liballegro-image5-dev liballegro-ttf5-dev liballegro-audio5-dev liballegro-video5-dev liballegro-physfs5-dev liballegro-acodec5-dev liballegro-acodec5-dev liballegro-dialog5-dev
```

- Debugging Allegro 5: example compilation command provided (inside testingAllegro/ folder):

```
gcc hello.c -o hello $(pkg-config allegro-5 allegro_font-5 --libs --cflags)
```

- How to play: links to the official board game rules PDF:
  - https://www.gamewright.com/gamewright/pdfs/Rules/ForbiddenDesertTM-RULES.pdf
  - Note in raw: the PDF was also downloaded into the project's main folder.

- Gameplay / media: several screenshots referenced under Resources/gameplay (files named like "Screenshot from 2019-01-03 16-17-16.png" etc.). The raw file embeds these as HTML <img> tags for visualization in the README.

Files and paths mentioned in repo

- Resources/gameplay/ (screenshots)
- testingAllegro/ (example hello.c test program)
- README text with installation and debugging hints

Cross-references

- Project page placeholder (not yet present): [[projects/forbidden-desert.md]]

Contradictions with existing wiki

- No contradictions detected with existing wiki pages at the time of ingest. The content is self-contained (build deps, debug command, rules link, screenshots) and does not conflict with other entries.

Notes / recommended follow-ups

- Create a project page [[projects/forbidden-desert.md]] summarizing the repo, add build/run instructions and a link to the GitHub repository if public.
- Capture any license info or repository homepage URL if present in the repo for completeness.
- Consider saving or linking the local copy of the rules PDF from the repo if redistribution / permissions permit.
