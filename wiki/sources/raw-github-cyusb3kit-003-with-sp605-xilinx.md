---
title: "raw github cyusb3kit-003 with sp605 xilinx"
tags:
  - raw
  - github
  - hardware
  - fx3
  - cypress
  - xilinx
sources:
  - "raw/github/CYUSB3KIT-003_with_SP605_xilinx.md"
updated: "2026-05-17"
---

Raw source ingest summary

Original path (local): C:/Users/NEGU/Documents/GitHub/rag_negu_expert/raw/github/CYUSB3KIT-003_with_SP605_xilinx.md
Category: github

High-level summary

- Project goal: use the Cypress CYUSB3KIT-003 (EZ-USB FX3 SuperSpeed Explorer Kit) to program and communicate with a Xilinx Spartan-6 FPGA on the SP605 evaluation board.
- Main capabilities implemented: upload FPGA firmware (programming) and establish bidirectional communication (loopback tests) between PC and FPGA via the FX3 device.
- Hardware used to connect boards: CYUSB3ACC-005 FMC Interconnect Board.

Extracted metadata and notable items

- Repo / project structure (as listed in raw file):
  - docs: API and project documentation
  - fx3_manager_cpp_source: C++ project providing a class/API to communicate with FX3 devices (uses cyusb which uses libusb-1.0)
  - com_fpga: firmware for both FPGA and FX3 implementing loopback communication
  - program_fpga: software/tools to program the FPGA
  - doc: collected documentation from Cypress and Xilinx
- Runtime behaviour (when running main in fx3_manager_cpp_source):
  1. Program the FX3 device(s)
  2. Program the FPGA
  3. Perform loopback communication checks asserting sent == received
- Documentation link present in raw: Read the Docs page for the project: https://cyusb3kit-003-with-sp605-xilinx.readthedocs.io/en/latest/index.html
- Confidentiality note in raw: full project could not be uploaded due to confidential agreement; FPGA code was merged into an existing codebase and is not provided in the repo. The uploaded artifacts exclude confidential parts but include materials useful to other developers (connection description, FX3 interface class).

Contents present in raw (high-level)

- Detailed description of board connections between CYUSB3KIT-003 and SP605 (useful wiring/pinout explanations)
- A C++ class that provides an API for communicating with FX3 devices on Linux (wrapper around cyusb/libusb)
- Instructions / description of the flow: program FX3, program FPGA, run loopback tests
- Notes about adapting CPU-tool simulations to standalone C++ code

Cross-references

- Project page (to be created): [[projects/cyusb3kit-003-with-sp605-xilinx.md]]

Contradictions with existing wiki

- No contradictions detected with existing wiki pages at the time of ingest. The raw file documents a hardware integration project and does not conflict with existing entries.

Notes / recommended follow-ups

- If non-confidential parts of com_fpga or program_fpga become available, ingest them and link here.
- Consider extracting and documenting the FX3 C++ API (examples, public headers) into a separate page or snippet for reuse across other hardware projects.
