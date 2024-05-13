---
title: AI Is the New Hotness 🌎🔥
date: 2024-05-12T10:47:00-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - "opinion"
  # ai/ml
  - "generative AI"
  - "LLMs"
series: []
layout: single
toc: true
math: false
draft: true
---

training llama 3 was equivalent to releasing 2290 tCO2eq
[llama3/MODEL_CARD.md at main · meta-llama/llama3](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

how did they calculate?
[[2307.09288] Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

> To train our largest Llama 3 models, we combined three types of parallelization: data parallelization, model
> parallelization, and pipeline parallelization. Our most efficient implementation achieves a compute utilization of
> over 400 TFLOPS per GPU when trained on 16K GPUs simultaneously. We performed training runs on two custom-built 24K
> GPU clusters. To maximize GPU uptime, we developed an advanced new training stack that automates error detection,
> handling, and maintenance. We also greatly improved our hardware reliability and detection mechanisms for silent data
> corruption, and we developed new scalable storage systems that reduce overheads of checkpointing and rollback. Those
> improvements resulted in an overall effective training time of more than 95%. Combined, these improvements increased
> the efficiency of Llama 3 training by ~three times compared to Llama 2.
> [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)

## add'l environmental impacts

- Reported tCO2eq is just co2 equivalent for electricity that's "100% of which were offset by Meta's sustainability
  program."
  - what is 1T Co2 for a standard person?
    [Greenhouse Gas Equivalencies Calculator | US EPA](https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator#results)
- electricity associated with cooling
- water cooling?
- amortized pollution associated with electronics manufacturing

## Hard math

NVIDIA releasing 1M H100s over the past 6 months (inferring from Dylan Patel)? How much heat do 1M H100s put out?

world's largest carbon capture facility can capture 36,000 tons of CO2 per year[^co2] [^co2]:
[The world's largest direct carbon capture plant just went online](https://www.engadget.com/the-worlds-largest-direct-carbon-capture-plant-just-went-online-172447811.html)

## Grid problems

[Mark Zuckerberg - Llama 3, Open Sourcing $10b Models, & Caesar Augustus](https://www.dwarkeshpatel.com/p/mark-zuckerberg)
[Energy, not compute, will be the #1 bottleneck to AI progress – Mark Zuckerberg [video]](https://news.ycombinator.com/item?id=40333459)
[There's Not Enough Power for America's High-Tech Ambitions - WSJ](https://www.wsj.com/business/energy-oil/data-centers-energy-georgia-development-7a5352e9)
