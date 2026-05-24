---
title: "CYUSB3KIT-003 with SP605 (Xilinx)"
tags:
  - project
  - github
  - hardware
  - fx3
  - xilinx
sources:
  - "sources/raw-github-cyusb3kit-003-with-sp605-xilinx.md"
updated: "2026-05-17"
---

Overview

This project integrates a Cypress CYUSB3KIT-003 (EZ-USB FX3 SuperSpeed Explorer Kit) with a Xilinx SP605 (Spartan-6) evaluation board to allow programming of the FPGA and bidirectional data communication between the PC and the FPGA via the FX3 device.

Why it may be useful

- Shows detailed wiring/connection procedure between CYUSB3KIT-003 and SP605 boards.
- Provides a C++ API/class (Linux) to manage FX3 devices (wrapper around cyusb/libusb) that can be reused in other projects.

Key hardware

- CYUSB3KIT-003 (Cypress EZ-USB FX3)
- Xilinx SP605 Evaluation Kit (Spartan-6)
- CYUSB3ACC-005 FMC Interconnect Board (used to connect the two boards)

Project structure (as in the raw repository)

- docs: project documentation and API docs
- fx3_manager_cpp_source: C++ project providing the FX3 communication class and example main
- com_fpga: (firmware) FX3 + FPGA loopback firmware (confidential parts omitted in public repo)
- program_fpga: software/tools used to program the FPGA
- doc: collected datasheets and vendor documentation

Notes from the raw source

- Running the example/main in the C++ project performs three steps: program FX3, program FPGA, run loopback communication tests asserting sent == received.
- The C++ FX3 manager uses the cyusb library (libusb-1.0 based). The code is intended to be adaptable: users can replace main.cpp to implement their own flows while keeping the FX3 manager class.
- Due to confidentiality agreements, full FPGA sources/firmware are not included in the public materials; the repository contains the non-confidential portions and documentation.

Documentation / upstream links

- Read the Docs for the project (mentioned in raw): https://cyusb3kit-003-with-sp605-xilinx.readthedocs.io/en/latest/index.html
- Raw source summary page: [[sources/raw-github-cyusb3kit-003-with-sp605-xilinx.md]]

Suggested follow-ups

- If allowed by confidentiality, extract and publish the public-facing headers/examples of the FX3 C++ API to a dedicated snippet page to aid reuse.
- Add wiring diagrams and annotated photos (if available and non-confidential) to the docs folder and reference them here.
