---
title: New GPU!
date: 2025-05-25
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
draft: false
---

I've been saving up for a year and change for a GPU upgrade, and my local Microcenter finally had 5090s in stock.
So I bought one.
It is a CHONK -- I'm getting full "you vs. the guy she tells you not to worry about" vibes:

<!-- markdownlint-disable MD013 MD033 MD034 -->

<table>
<tr>
  <td style="width:50%">
{{< figure
  src="images/top.png"
  alt="5090 is a larger GPU"
  caption="Top view comparing 3090 and 5090 (5090 is larger)" >}}
  </td>
  <td style="width:50%">
{{< figure
  src="images/side.png"
  alt="5090 is a thicker GPU"
  caption="Side view comparing 3090 and 5090 (5090 is larger)" >}}
  </td>
</tr>
</table>

<!-- markdownlint-enable MD013 MD033 MD034 -->

Specifically, I've upgraded from a PNY RTX 3090 XLR8 Gaming EPIC-X RGB (what a mouthful) to a PNY RTX 5090 OC.
Some of the size difference is due to "reference" vs "custom" PCB and cooler design; the 3090 is a reference design while the 5090 is a custom board and cooler.
And it needs the cooler -- the card now pulls up to 1.7x more electricity at up to 600W (no wonder Nvidia has problems with cables melting)!
Regardless, I watercool the GPU in my system, so the size of the cooler _shouldn't_ matter (that assumption will come back to bite me in the ass).

| Feature              | [PNY GeForce RTX 3090 XLR8 Gaming EPIC-X RGB](https://www.techpowerup.com/gpu-specs/pny-xlr8-rtx-3090-revel-epic-x-triple-fan.b8014) | [PNY GeForce RTX 5090 OC](https://www.techpowerup.com/gpu-specs/pny-rtx-5090-overclocked-triple-fan.b12122) |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **GPU Architecture** | NVIDIA Ampere                                                                                                                        | NVIDIA Blackwell                                                                                            |
| **CUDA Cores**       | 10,496                                                                                                                               | 21,760                                                                                                      |
| **Base/Boost Clock** | 1395 / 1695 MHz                                                                                                                      | 2017 / 2527 MHz                                                                                             |
| **Memory Size**      | 24 GB GDDR6X                                                                                                                         | 32 GB GDDR7                                                                                                 |
| **Memory Speed**     | 1219 MHz                                                                                                                             | 1750 MHz                                                                                                    |
| **Memory Interface** | 384-bit                                                                                                                              | 512-bit                                                                                                     |
| **Memory Bandwidth** | 936 GB/s                                                                                                                             | 1792 GB/s                                                                                                   |
| **Transistors**      | 28.3 million                                                                                                                         | 92.2 million                                                                                                |
| **TDP**              | 350 W                                                                                                                                | 575 W                                                                                                       |
| **Power Connectors** | 2 x 8-pin                                                                                                                            | 16-pin (adapter to 4x 8-pin)                                                                                |
| **PCIe Interface**   | PCIe 4.0 x16                                                                                                                         | PCIe 5.0 x16                                                                                                |
| **Outputs**          | 3 x DisplayPort 1.4, 1 x HDMI 2.1                                                                                                    | 3 x DisplayPort 2.1b, 1 x HDMI 2.1b                                                                         |
| **Dimensions**       | 11.57" x 4.41" x 2.2" (3-slot)                                                                                                       | 12.95" x 5.43" x 2.8" (3.5-slot)                                                                            |

The neat thing about these graphics cards with 24GB+ VRAM is that they can run super-capable LLMs _locally_.
Models like Qwen 3 30B A3B, Qwen 3 32B, and Gemma 3 27B match or outperform proprietary models via paid API; the 5090's token evaluation rate (which impacts experienced user latency) is also roughly equivalent to that experience via paid API.
So I now have a local, private LLM instance that is functionally equivalent to GPT-4o or Claude 3.7 Sonnet!
Granted, I'd have to have send on the ballpark of a quarter-million prompts to my self-hosted model to recoup the cost of the GPU instead of just using a paid API... but the upgrade was not intended to be a "cost saving" measure.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/local_vs.png"
alt="Qwen 3 30B A3B, Qwen 3 32B, and Gemma 3 27B match or beat proprietary models"
caption="Qwen 3 30B A3B, Qwen 3 32B, and Gemma 3 27B match or beat proprietary models"
link="https://artificialanalysis.ai/?models=claude-4-sonnet%2Cclaude-4-sonnet-thinking%2Co4-mini%2Cgpt-4-1-mini%2Cgpt-4-1%2Cgemma-3-27b%2Cgemini-2-5-flash-reasoning%2Cgemini-2-5-pro%2Cgemini-2-0-flash-lite-001%2Cclaude-3-5-haiku%2Cclaude-3-7-sonnet%2Cclaude-3-7-sonnet-thinking%2Cqwen3-30b-a3b-instruct-reasoning%2Cqwen3-32b-instruct-reasoning%2Cqwen3-32b-instruct%2Cqwen3-30b-a3b-instruct%2Cgpt-4o-chatgpt-03-25#artificial-analysis-intelligence-index" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

## 60% Performance Improvement

After testing, I can confirm that the 5090 is faster. 🚫💩🕵️‍♂️ -- it's 2 architecture generations (Ampere -> Ada -> Blackwell) and 5 years (2020 -> 2025) / card generations (30xx -> 30xx Ti -> 40xx -> 40xx Ti -> 40xx Super -> 50xx) newer.

I tested with [aidatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark), a very convenient command line benchmark based on Ollama that tests throughput over a variety of LLM and VLM architectures and sizes.

### LLM Benchmark Results: RTX 3090 vs RTX 5090

| Model           | Avg Eval Rate (RTX 3090) | Avg Eval Rate (RTX 5090) | Speedup (%) |
| --------------- | -----------------------: | -----------------------: | ----------: |
| mistral:7b      |                   133.51 |                   210.89 |      +58.0% |
| phi4:14b        |                    70.43 |                   117.65 |      +67.0% |
| gemma2:9b       |                    90.74 |                   137.50 |      +51.5% |
| llava:7b        |                   140.09 |                   220.81 |      +57.6% |
| llava:13b       |                    86.10 |                   141.24 |      +64.0% |
| deepseek-r1:8b  |                   111.83 |                   181.26 |      +62.1% |
| deepseek-r1:14b |                    64.88 |                   105.83 |      +63.1% |

The benchmark indicates the 5090 has roughly 60% performance improvement over the 3090; the increases are larger as model size (parameters) increase.
As the models used get even larger, I anticipate an even greater performance lift for the 5090, as a result of more tensor cores, more _efficient_ tensor cores, and higher memory bandwidth; as the total utilization of the available compute/memory bandwidth approach 100%, the 5090 should increase its lead over the 3090.

### System Specs

My PC is in an Alphacool 4u server chassis that supports water cooling; this lets me keep noise down, have quite good cooling performance, and I can stack my computer in a server rack with the rest of my homelab gear.
For the non-nerds, this means that my computer is functionally "on its side" whereas a normal PC stands tall.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure src="images/server.png" alt="A water-cooled PC" caption="Gratuitous water cooling" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

You may notice the graphics card (top right, with "CORE" in shiny silver) is horizontal.
This is because the 5090 is _too big_ for the chassis, especially with the protrusion on the water block for connecting the water lines.
So I had to hack the case with a "vertical GPU mount"; since the chassis is "on its side" relative to a normal computer, this means that the "vertical" mount lets me hold the card horizontally.

{{% details title="System Specs" closed="true" %}}

| Component   | Specification                                              |
| ----------- | ---------------------------------------------------------- |
| Motherboard | Asus ProArt X870E-Creator Wifi                             |
| CPU         | AMD Ryzen 7 9800X3D                                        |
| RAM         | 64GB GSkill F5 DDR5-6000 (2x32GB)                          |
| Storage     | 1TB Sabrent Rocket 5                                       |
| GPU         | PNY RTX 3090 XLR8 REVEL EPIX-X -> PNY RTX 5090 Overclocked |
| PSU         | Seasonic PRIME PX-1300                                     |

> Watercooling components all Alphacool (not sponsored, I just find their stuff to be no-nonsense and effective):
>
> - 60mm thick 360mm radiator
> - 25mm thick 360mm radiator
> - Alphacool TPV tubes, fittings, and quick-disconnects
> - 3x Arctic S12038-4K fans (120 x 38mm)
> - 3x Noctua NF-A12x25 Chromax fans (120 x 25mm)

{{% /details %}}

## Blogumentation

Here are my notes on configuring my system for the experiment:

### 1. [Install WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install)

- Install WSL, reboot
- Install distro (I use Debian)
- Configure distro (Create user account, etc.)

### 2. Install dependencies (in WSL)

- Install standard dependencies

  ```sh
  apt install curl wget git software-properties-common build-essential
  ```

- CUDA for WSL

  - Install [nvidia cuda-toolkit](https://developer.nvidia.com/cuda-downloads)\
    ⚠️ DO NOT INSTALL THE DRIVER ON WSL; ONLY INSTALL THE CUDA-TOOLKIT ⚠️

  - [Check GPU device post-installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) using the compiled [deviceQuery](https://github.com/nvidia/cuda-samples) binary

    ```sh
    apt install cmake
    mkdir build && cd build
    cmake ..
    make -j
    ./deviceQuery
    ```

- Install [ollama](https://ollama.com/download/linux)

  ```sh
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- Install [pipx](https://pipx.pypa.io/latest/installation/)

  ```sh
  apt install pipx
  ```

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv)

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Install [Docker Desktop](https://docs.docker.com/desktop/features/wsl/#download)

- Install [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - I did this, but I'm not sure whether it is necessary for this particular experiment

### 3. Install [aidatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark)

```sh
pipx install llm-benchmark
```

### 4. Run tests

- Close all apps

- Disable internet

- Run:

  ```sh
  llm_benchmark run
  ```
