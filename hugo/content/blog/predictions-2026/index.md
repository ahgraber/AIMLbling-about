---
title: Predictions 2026
date: 2026-01-03
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - listicle
  - opinion
  # ai/ml
  - agents
  - arxiv
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: false
---

<!-- markdownlint-disable-file MD036 -->

These are my predictions for AI (primarily LLM-focused) in 2026 along with my certainty / probability estimates.

## Agents, Cont'd

Agents aren't going anywhere.
Their capability will increasingly be driven by Reinforcement Learning (RL).
We will see a divergence: domains with **verifiable tasks** (math, coding) will advance rapidly because the reward signals are clear.
Subjective domains will lag behind; since reward signals are preferential, subjective reward models risk reward hacking.

_Probability: 100%_

Functionally, this means that domains with verifiable tasks will see more and more AI assistance and automation, following the trajectory of software engineering in 2025.
I expect software engineers' day jobs to generally be less about writing code and more about managing agent swarms and code review.

## Technical Paradigm Shifts

### Smaller Better Faster Stronger

Reinforcement learning and improved post-training techniques are demonstrating that models can improve their capabilities without increasing their parameter counts.
I believe the industry will pause on scaling model size and instead dedicate compute to current (or smaller!) models training for longer.

_Probability: 90%_

Further, improvements to distillation and reinforcement fine-tuning will dramatically increase the capabilities of smaller models.
I see models with GPT-4o-level capabilities potentially running on phones (e.g., \<3B parameters), especially when fine-tuned for on-device use cases.

_Probability: 70%_

### Continued Learning

"Treat your LLM like a forgetful intern" is frequently given advice for today's LLMs.
With no memory, prompts (or [skills](https://agentskills.io/home)) must provide the necessary instructions to complete tasks every time, and Agents (and the LLMs that power them) do not learn from experience.

Recent experiments with LoRA adapters indicate they may be a way to provide customizable, task-specific updates to models; while not real-time learning, this would potentially allow customized task adapters for a standard base model that can be regularly refreshed [^thinking] [^clora] [^alora].

2025 saw research on designing model architectures that are capable of continual learning through memory updates, where models are allowed to update some of their parameters - and in some cases designed with "memory" parameters designated just for this purpose [^titans] [^miras] [^nested] [^continual].

I'll take a flier here - by the end of 2026, a foundation lab will have a model with continual learning such that every person ends up with a slightly different instance of the model based on their interaction history.

_Probability: 30%_

### Architectural Shifts

At least one non-standard AI model architecture (e.g., Text Diffusion Models, Byte Transformers, Latent Reasoning Models) will become broadly available for use case diversification.

_Probability: 75%_

## The Rise of China

An (open-weights) model from a Chinese lab (Qwen, DeepSeek, Kimi, MiniMax, Baidu) will achieve benchmark-leading performance on at least one dimension (Agent/Function-Calling, Coding, Deep Search, Photo Generation/Editing, Video Generation/Editing).
Furthermore, this performance improvement will be at least a big step rather than a marginal win, and the model will stay at the top of the leaderboard for a considerable time.

_Probability: 75%_

## The Supply Chain Squeeze

GPU production (the GPU chips themselves, via TSMC) are no longer the bottleneck.
Other board components (other integrated circuits, resistors, capacitors, etc.), supporting servers and networking equipment, and/or power constraints [^know_about_datacenters] [^datacenter_power] will dramatically increase costs for building new datacenters.
This supply chain squeeze will extend beyond the datacenter and disrupt broader society by measurably influencing the cost of consumer goods or electricity.

_Probability: 100%_

> [!NOTE]
> This is already happening with DRAM [^dirty_dram] [^40_dram], but I had noted this prediction in an early draft following SemiAnalysis reporting on [memory](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm) and [power](https://newsletter.semianalysis.com/p/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout) earlier in the year.

## Speaking of Power

Speaking of power constraints, and in association with my above prediction on shifting compute from model size growth to post-training extensions, the industry will start reporting models by energy required to train (kWh) rather than parameter count or training tokens seen.

This will require the prior prediction to come true _and_ for power consumption to be more predictive of performance than parameter count.

_Probability: 40%_

## Lightning Round

- Ads come to at least one consumer AI platform (_Probability: 100%_).\
  This one should be a gimme given the strong hints from OpenAI in their hiring decisions over 2024-2025 and the mention of ads in their "code red" response to Gemini 3 Pro [^code_red].
- OpenAI and/or Anthropic will IPO (_Probability: 75%_).
- A foundation lab will go bankrupt (if startup) or cancel its foundation model initiative (_Probability: 80%_).
  - Meta kills Llama - or makes it proprietary, which is functionally the same thing for the Llama ecosystem (_Probability: 60%_).

---

## Topics to Watch

I'm not making predictions for these topics, but I think they're worth keeping in mind as we explore the trajectory of AI in 2026.

### The AI Bubble & Infrastructure

I am increasingly concerned about the economics of this build-out.
We are seeing asset prices that exceed intrinsic value, particularly for hardware that depreciates in ~5 years.
We are building data centers with timelines that seem impossible given power constraints, permitting issues, and public opposition [^know_about_datacenters] [^ai_wildfire].

### Consumer Sentiment

I expect to see rising consumer dissatisfaction as AI is "forced" into products without opt-outs.
We will also likely see a rise in "assisted harms"—software exploits, sycophancy, and other risks amplified by AI ubiquity [^ai_backlash] [^ai_winter].

### Copyright

The legal battles are coming to a head.
If courts rule that training on copyrighted data requires compensation or removal, we could see massive, expensive retraining operations that could reshape the industry.

## References

<!-- markdownlint-disable MD013 -->

[^thinking]: [LoRA Without Regret - Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/)

[^clora]: [Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead - 2407.00066v4.pdf](https://www.arxiv.org/pdf/2407.00066)

[^alora]: [[2504.12397] Activated LoRA: Fine-tuned LLMs for Intrinsics](https://arxiv.org/abs/2504.12397)

[^titans]: [[2501.00663] Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

[^miras]: [[2504.13173] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173)

[^nested]: [[2512.24695] Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)

[^continual]: [[2510.15103] Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103)

[^know_about_datacenters]: [What you need to know about AI data centers | Epoch AI](https://epoch.ai/blog/what-you-need-to-know-about-ai-data-centers)

[^datacenter_power]: [Data on Frontier AI Data Centers | Epoch AI](https://epoch.ai/data/data-centers)

[^dirty_dram]: [Sam Altman's Dirty DRAM Deal](https://www.mooreslawisdead.com/post/sam-altman-s-dirty-dram-deal)

[^40_dram]: [OpenAI's Stargate project to consume up to 40% of global DRAM output — inks deal with Samsung and SK hynix to the tune of up to 900,000 wafers per month | Tom's Hardware](https://www.tomshardware.com/pc-components/dram/openais-stargate-project-to-consume-up-to-40-percent-of-global-dram-output-inks-deal-with-samsung-and-sk-hynix-to-the-tune-of-up-to-900-000-wafers-per-month)

[^code_red]: [OpenAI CEO Sam Altman declares 'code red' to improve ChatGPT amid rising competition | AP News](https://apnews.com/article/openai-chatgpt-code-red-google-gemini-00d67442c7862e6663b0f07308e2a40d)

[^ai_wildfire]: [The AI Wildfire Is Coming. It's Going to be Very Painful and Incredibly Healthy.](https://ceodinner.substack.com/p/the-ai-wildfire-is-coming-its-going)

[^ai_backlash]: [The AI Backlash Is Here: Why Backlash Against Gemini, Sora, ChatGPT Is Spreading in 2025 - Newsweek](https://www.newsweek.com/ai-backlash-openai-meta-friend-10807425)

[^ai_winter]: [LLMs are a failure. A new AI winter is coming.](https://taranis.ie/llms-are-a-failure-a-new-ai-winter-is-coming/)
