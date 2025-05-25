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

## Huh. Progress.



## Setup

1. [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

   - install wsl, reboot
   - install distro
   - configure distro
     - create user
     - `apt install curl wget git software-properties-common build-essential`


2. Install dependencies
   - cuda for wsl
     - nvidia cuda-toolkit (https://developer.nvidia.com/cuda-downloads) **DO NOT INSTALL THE DRIVER ON WSL; ONLY INSTALL THE CUDA-TOOLKIT**
     - nvidia post-install (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
      - https://github.com/nvidia/cuda-samples
        ```sh
        apt install cmake
        mkdir build && cd build
        cmake ..
        make -j
        ./deviceQuery
        ```
      - ./deviceQuery > ~/<gpu>\_report.txt
   - [ollama](https://ollama.com/download/linux) `curl -fsSL https://ollama.com/install.sh | sh`
   - [pipx](https://pipx.pypa.io/latest/installation/) `apt install pipx`
   - [uv](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv) `curl -LsSf https://astral.sh/uv/install.sh | sh`

   - [Docker Desktop](https://docs.docker.com/desktop/features/wsl/#download)
   - [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)???

3. Install [aidatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark)

   `pipx install llm-benchmark`

4. Run tests

   - close all apps (except for hwinfo)
   - disable internet
   - `llm_benchmark run`

5. swap gpu

6. un/re install drivers (DDU)

## Specs

Asus ProArt X870E-Creator Wifi
AMD Ryzen 7 9800X3D
64GB GSkill F5 DDR5-6000 (2x32GB)
1TB Sabrent Rocket 5
PNY RTX 3090 XLR8 REVEL EPIX-X / PNY RTX 5090 Overclocked
Seasonic Vertex PX-1200

## Tools

- [gpu-z](https://www.guru3d.com/download/gpu-z-download-techpowerup/)
- [Display Driver Uninstaller](https://www.guru3d.com/download/display-driver-uninstaller-download/) and [Display Driver Uninstaller Thread](https://www.guru3d.com/download/gpu-z-download-techpowerup/`)
- [MSI Afterburner](https://www.guru3d.com/download/msi-afterburner-beta-download/)


## 3090 Results

```txt
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 3090', 'driver': '572.42', 'gpu_memory_total': '24576.0 MB', 'gpu_memory_free': '23718.0 MB', 'gpu_memory_used': '609.0 MB', 'gpu_load': '3.0%', 'gpu_temperature': '25.0°C'}
Only one GPU card
Total memory size : 30.19 GB
cpu_info: AMD Ryzen 7 9800X3D 8-Core Processor
gpu_info: NVIDIA GeForce RTX 3090
os_version: Debian GNU/Linux 12 (bookworm)
ollama_version: 0.7.0
----------
LLM models file path：/home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/benchmark_models_16gb_ram.yml
Checking and pulling the following LLM models
gemma2:9b
mistral:7b
phi4:14b
deepseek-r1:8b
deepseek-r1:14b
llava:7b
llava:13b
----------
model_name =    mistral:7b
prompt = Write a step-by-step guide on how to bake a chocolate cake from scratch.
eval rate:            131.28 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game
eval rate:            133.57 tokens/s
prompt = Create a dialogue between two characters that discusses economic crisis
eval rate:            133.85 tokens/s
prompt = In a forest, there are brave lions living there. Please continue the story.
eval rate:            132.82 tokens/s
prompt = I'd like to book a flight for 4 to Seattle in U.S.
eval rate:            136.01 tokens/s
--------------------
Average of eval rate:  133.506  tokens/s
----------------------------------------
model_name =    phi4:14b
prompt = Write a step-by-step guide on how to bake a chocolate cake from scratch.
eval rate:            69.81 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game
eval rate:            70.27 tokens/s
prompt = Create a dialogue between two characters that discusses economic crisis
eval rate:            70.32 tokens/s
prompt = In a forest, there are brave lions living there. Please continue the story.
eval rate:            70.46 tokens/s
prompt = I'd like to book a flight for 4 to Seattle in U.S.
eval rate:            71.30 tokens/s
--------------------
Average of eval rate:  70.432  tokens/s
----------------------------------------
model_name =    gemma2:9b
prompt = Explain Artificial Intelligence and give its applications.
eval rate:            89.72 tokens/s
prompt = How are machine learning and AI related?
eval rate:            89.69 tokens/s
prompt = What is Deep Learning based on?
eval rate:            89.39 tokens/s
prompt = What is the full form of LSTM?
eval rate:            94.56 tokens/s
prompt = What are different components of GAN?
eval rate:            90.36 tokens/s
--------------------
Average of eval rate:  90.744  tokens/s
----------------------------------------
model_name =    llava:7b
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample1.jpg
eval rate:            139.32 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample2.jpg
eval rate:            139.57 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample3.jpg
eval rate:            141.51 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample4.jpg
eval rate:            141.57 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample5.jpg
eval rate:            138.50 tokens/s
--------------------
Average of eval rate:  140.094  tokens/s
----------------------------------------
model_name =    llava:13b
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample1.jpg
eval rate:            86.47 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample2.jpg
eval rate:            86.00 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample3.jpg
eval rate:            86.07 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample4.jpg
eval rate:            85.64 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample5.jpg
eval rate:            86.31 tokens/s
--------------------
Average of eval rate:  86.098  tokens/s
----------------------------------------
model_name =    deepseek-r1:8b
prompt = Summarize the key differences between classical and operant conditioning in psychology.
eval rate:            114.30 tokens/s
prompt = Translate the following English paragraph into Chinese and elaborate more -> Artificial intelligence is transforming various industries by enhancing efficiency and enabling new capabilities.
eval rate:            115.31 tokens/s
prompt = What are the main causes of the American Civil War?
eval rate:            113.39 tokens/s
prompt = How does photosynthesis contribute to the carbon cycle?
eval rate:            113.31 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game.
eval rate:            102.83 tokens/s
--------------------
Average of eval rate:  111.828  tokens/s
----------------------------------------
model_name =    deepseek-r1:14b
prompt = Summarize the key differences between classical and operant conditioning in psychology.
eval rate:            65.39 tokens/s
prompt = Translate the following English paragraph into Chinese and elaborate more -> Artificial intelligence is transforming various industries by enhancing efficiency and enabling new capabilities.
eval rate:            65.58 tokens/s
prompt = What are the main causes of the American Civil War?
eval rate:            64.86 tokens/s
prompt = How does photosynthesis contribute to the carbon cycle?
eval rate:            64.68 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game.
eval rate:            63.88 tokens/s
--------------------
Average of eval rate:  64.878  tokens/s
----------------------------------------
Sending the following data to a remote server
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 3090', 'driver': '572.42', 'gpu_memory_total': '24576.0 MB', 'gpu_memory_free': '7812.0 MB', 'gpu_memory_used': '16515.0 MB', 'gpu_load': '18.0%', 'gpu_temperature': '38.0°C'}
Only one GPU card
Your machine UUID : af47b5e6-98a6-566d-a1aa-09fe727b34eb
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 3090', 'driver': '572.42', 'gpu_memory_total': '24576.0 MB', 'gpu_memory_free': '7812.0 MB', 'gpu_memory_used': '16515.0 MB', 'gpu_load': '2.0%', 'gpu_temperature': '38.0°C'}
Only one GPU card
{
    "mistral:7b": "133.51",
    "phi4:14b": "70.43",
    "gemma2:9b": "90.74",
    "llava:7b": "140.09",
    "llava:13b": "86.10",
    "deepseek-r1:8b": "111.83",
    "deepseek-r1:14b": "64.88",
    "uuid": "af47b5e6-98a6-566d-a1aa-09fe727b34eb",
    "ollama_version": "0.7.0"
}
----------
====================
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 3090', 'driver': '572.42', 'gpu_memory_total': '24576.0 MB', 'gpu_memory_free': '7812.0 MB', 'gpu_memory_used': '16515.0 MB', 'gpu_load': '0.0%', 'gpu_temperature': '36.0°C'}
Only one GPU card
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 3090', 'driver': '572.42', 'gpu_memory_total': '24576.0 MB', 'gpu_memory_free': '7812.0 MB', 'gpu_memory_used': '16515.0 MB', 'gpu_load': '1.0%', 'gpu_temperature': '35.0°C'}
Only one GPU card
{
    "system": "Linux",
    "memory": 30.18781280517578,
    "cpu": "AMD Ryzen 7 9800X3D 8-Core Processor",
    "gpu": "NVIDIA GeForce RTX 3090",
    "os_version": "Debian GNU/Linux 12 (bookworm)",
    "system_name": "Linux",
    "uuid": "af47b5e6-98a6-566d-a1aa-09fe727b34eb"
}
----------
```

## 5090 Results

```
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 5090', 'driver': '576.40', 'gpu_memory_total': '32607.0 MB', 'gpu_memory_free': '10990.0 MB', 'gpu_memory_used': '21112.0 MB', 'gpu_load': '13.0%', 'gpu_temperature': '29.0°C'}
Only one GPU card
Total memory size : 30.19 GB
cpu_info: AMD Ryzen 7 9800X3D 8-Core Processor
gpu_info: NVIDIA GeForce RTX 5090
os_version: Debian GNU/Linux 12 (bookworm)
ollama_version: 0.7.0
----------
LLM models file path：/home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/benchmark_models_16gb_ram.yml
Checking and pulling the following LLM models
gemma2:9b
mistral:7b
phi4:14b
deepseek-r1:8b
deepseek-r1:14b
llava:7b
llava:13b
----------
model_name =    mistral:7b
prompt = Write a step-by-step guide on how to bake a chocolate cake from scratch.
eval rate:            208.22 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game
eval rate:            212.94 tokens/s
prompt = Create a dialogue between two characters that discusses economic crisis
eval rate:            213.14 tokens/s
prompt = In a forest, there are brave lions living there. Please continue the story.
eval rate:            214.93 tokens/s
prompt = I'd like to book a flight for 4 to Seattle in U.S.
eval rate:            205.20 tokens/s
--------------------
Average of eval rate:  210.886  tokens/s
----------------------------------------
model_name =    phi4:14b
prompt = Write a step-by-step guide on how to bake a chocolate cake from scratch.
eval rate:            117.13 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game
eval rate:            117.19 tokens/s
prompt = Create a dialogue between two characters that discusses economic crisis
eval rate:            119.47 tokens/s
prompt = In a forest, there are brave lions living there. Please continue the story.
eval rate:            118.95 tokens/s
prompt = I'd like to book a flight for 4 to Seattle in U.S.
eval rate:            115.53 tokens/s
--------------------
Average of eval rate:  117.654  tokens/s
----------------------------------------
model_name =    gemma2:9b
prompt = Explain Artificial Intelligence and give its applications.
eval rate:            134.38 tokens/s
prompt = How are machine learning and AI related?
eval rate:            139.72 tokens/s
prompt = What is Deep Learning based on?
eval rate:            138.27 tokens/s
prompt = What is the full form of LSTM?
eval rate:            138.53 tokens/s
prompt = What are different components of GAN?
eval rate:            136.59 tokens/s
--------------------
Average of eval rate:  137.498  tokens/s
----------------------------------------
model_name =    llava:7b
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample1.jpg
eval rate:            215.82 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample2.jpg
eval rate:            223.02 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample3.jpg
eval rate:            223.56 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample4.jpg
eval rate:            221.57 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample5.jpg
eval rate:            220.06 tokens/s
--------------------
Average of eval rate:  220.806  tokens/s
----------------------------------------
model_name =    llava:13b
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample1.jpg
eval rate:            138.95 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample2.jpg
eval rate:            143.03 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample3.jpg
eval rate:            144.01 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample4.jpg
eval rate:            135.71 tokens/s
prompt = Describe the image, /home/wsl/.local/pipx/venvs/llm-benchmark/lib/python3.11/site-packages/llm_benchmark/data/img/sample5.jpg
eval rate:            144.50 tokens/s
--------------------
Average of eval rate:  141.24  tokens/s
----------------------------------------
model_name =    deepseek-r1:8b
prompt = Summarize the key differences between classical and operant conditioning in psychology.
eval rate:            184.81 tokens/s
prompt = Translate the following English paragraph into Chinese and elaborate more -> Artificial intelligence is transforming various industries by enhancing efficiency and enabling new capabilities.
eval rate:            188.91 tokens/s
prompt = What are the main causes of the American Civil War?
eval rate:            183.98 tokens/s
prompt = How does photosynthesis contribute to the carbon cycle?
eval rate:            178.46 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game.
eval rate:            170.15 tokens/s
--------------------
Average of eval rate:  181.262  tokens/s
----------------------------------------
model_name =    deepseek-r1:14b
prompt = Summarize the key differences between classical and operant conditioning in psychology.
eval rate:            108.07 tokens/s
prompt = Translate the following English paragraph into Chinese and elaborate more -> Artificial intelligence is transforming various industries by enhancing efficiency and enabling new capabilities.
eval rate:            108.91 tokens/s
prompt = What are the main causes of the American Civil War?
eval rate:            105.16 tokens/s
prompt = How does photosynthesis contribute to the carbon cycle?
eval rate:            106.66 tokens/s
prompt = Develop a python function that solves the following problem, sudoku game.
eval rate:            100.36 tokens/s
--------------------
Average of eval rate:  105.832  tokens/s
----------------------------------------
Sending the following data to a remote server
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 5090', 'driver': '576.40', 'gpu_memory_total': '32607.0 MB', 'gpu_memory_free': '4916.0 MB', 'gpu_memory_used': '27186.0 MB', 'gpu_load': '78.0%', 'gpu_temperature': '38.0°C'}
Only one GPU card
Your machine UUID : d69ee9b0-e46c-5dac-a410-7fddbf157d53
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 5090', 'driver': '576.40', 'gpu_memory_total': '32607.0 MB', 'gpu_memory_free': '4913.0 MB', 'gpu_memory_used': '27189.0 MB', 'gpu_load': '2.0%', 'gpu_temperature': '38.0°C'}
Only one GPU card
{
    "mistral:7b": "210.89",
    "phi4:14b": "117.65",
    "gemma2:9b": "137.50",
    "llava:7b": "220.81",
    "llava:13b": "141.24",
    "deepseek-r1:8b": "181.26",
    "deepseek-r1:14b": "105.83",
    "uuid": "d69ee9b0-e46c-5dac-a410-7fddbf157d53",
    "ollama_version": "0.7.0"
}
----------
====================
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 5090', 'driver': '576.40', 'gpu_memory_total': '32607.0 MB', 'gpu_memory_free': '4913.0 MB', 'gpu_memory_used': '27189.0 MB', 'gpu_load': '2.0%', 'gpu_temperature': '35.0°C'}
Only one GPU card
-------Linux----------
{'id': '0', 'name': 'NVIDIA GeForce RTX 5090', 'driver': '576.40', 'gpu_memory_total': '32607.0 MB', 'gpu_memory_free': '4914.0 MB', 'gpu_memory_used': '27188.0 MB', 'gpu_load': '1.0%', 'gpu_temperature': '35.0°C'}
Only one GPU card
{
    "system": "Linux",
    "memory": 30.18781280517578,
    "cpu": "AMD Ryzen 7 9800X3D 8-Core Processor",
    "gpu": "NVIDIA GeForce RTX 5090",
    "os_version": "Debian GNU/Linux 12 (bookworm)",
    "system_name": "Linux",
    "uuid": "d69ee9b0-e46c-5dac-a410-7fddbf157d53"
}
----------
```
