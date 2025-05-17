---
title: Gpu Testing
date: 2025-05-17T10:19:59-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - blogumentation
  - experiment
  # ai/ml
  - evals
  - generative AI
  - LLMs
  # homelab
  - homelab
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

## Setup

1. Install WSL

   - install wsl, reboot
   - install distro
   - configure distro
     - create user

2. Install dependencies

   - pipx (apt install pipx)
   - uv (use uv's install script)
   - nvidia container toolkit(?)
   - cuda for wsl
     - `apt install software-properties-common build-essential`
     - nvidia cuda-toolkit (https://developer.nvidia.com/cuda-downloads) **DO NOT INSTALL THE DRIVER ON WSL; ONLY INSTALL THE CUDA-TOOLKIT**
     - nvidia post-install (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
     - ./deviceQuery > ~/<gpu>\_report.txt
   - ollama

3. Install aidatools/ollama-benchmark

4. Run tests

5. swap gpu

6. un/re install drivers (DDU)

## Specs

Asus ProArt X870E-Creator Wifi
AMD Ryzen 7 9800X3D
64GB GSkill F5 DDR5-6000 (2x32GB)
1TB Sabrent Rocket 5
PNY RTX 3090 XLR8 REVEL EPIX-X / PNY RTX 5090 Overclocked
Seasonic Vertex PX-1200
