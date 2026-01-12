---
title: Benchmarks Predict Model Size(?)
date: 2026-01-08T19:51:34-05:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - experiment
  # ai/ml
  - evals
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

In a recent episode of the Latent Space podcast ([Artificial Analysis: The Independent LLM Analysis House — with George Cameron and Micah Hill-Smith](https://youtu.be/v5mBjeX4TJ8?si=ViYbKgD6x7fkjEa4)), the Artificial Analysis team pointed out a strong correlation between model performance on their AA-Omniscience Accuracy benchmark and the model's parameter count:

> An interesting thing about this accuracy metric is that it tracks more closely than anything else that we measure the total parameter count of models...
> If you draw the line on AA-Omniscience accuracy vs total parameters, ... you can see that likely the leading front-end models right now are quite a lot bigger than the 1 trillion parameters that the open-weights models cap out at...
> There's an interesting extra data point that Elon Musk revealed recently about xAI that Grok 3 and 4 - 3 trillion parameters for Grok 3 and 4, 6 trillion for Grok 5 (but thats not out yet).

Well, _that_ caught my attention, so I had to see for myself.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/AAO_accuracy_vs_params.png"
alt="AA-Omniscience Accuracy appears strongly correlated to (log) model parameter count"
caption="AA-Omniscience Accuracy appears strongly correlated to (log) model parameter count"
link="https://arxiv.org/html/2511.13029v1" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

## Experiment

I collected the AA-Omniscience data by transcribing the plot to a table.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/AAO_accuracy.png"
alt="AA-Omniscience Accuracy"
link="https://artificialanalysis.ai/?omniscience=omniscience-accuracy&model-filters=frontier-model&models=claude-opus-4-5-thinking%2Cgemini-3-pro%2Cgpt-5-1%2Cgemini-3-flash-reasoning%2Cgpt-5-2-medium%2Cclaude-4-5-sonnet-thinking%2Cglm-4-7%2Cgpt-5-medium%2Cgrok-4%2Cdeepseek-v3-2-reasoning%2Co3%2Cgemini-3-pro-low%2Ckimi-k2-thinking%2Cminimax-m2-1%2Cgpt-5-mini-medium%2Cclaude-4-5-haiku-reasoning%2Cminimax-m2%2Cnova-2-0-pro-reasoning-medium%2Cclaude-3-7-sonnet-thinking%2Cgemini-2-5-pro%2Cgpt-oss-120b%2Cglm-4-6-reasoning%2Cnova-2-0-lite-reasoning-medium%2Cqwen3-235b-a22b-instruct-2507-reasoning%2Cnova-2-0-omni-reasoning-medium%2Cgemini-2-5-flash-reasoning%2Cdeepseek-r1%2Cglm-4.5%2Cgpt-5-nano-medium%2Cgpt-4-1%2Cgrok-3%2Cgpt-oss-20b%2Cnvidia-nemotron-3-nano-30b-a3b-reasoning%2Cgpt-oss-120b-low%2Cqwen3-30b-a3b-2507-reasoning%2Cdeepseek-v3-0324%2Cgpt-oss-20b-low%2Cllama-4-maverick%2Cqwen3-14b-instruct-reasoning%2Cministral-14b%2Cqwen3-30b-a3b-instruct-reasoning%2Cgpt-4o%2Cllama-4-scout%2Ccommand-a%2Cgemma-3-27b%2Cgemma-3-12b%2Cgemma-3-1b%2Cgemma-3-4b%2Colmo-3-1-32b-think#aa-omniscience-accuracy" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

Then I proceeded to find and verify model sizes from HuggingFace, release announcements, or technical papers on arxiv.org.

## Conclusion

Does estimating total parameter count matter?
Not really; it's merely one factor among many that may contribute to model performance.
When frontier labs are proprietary, it is fun to try to pierce their veil of secrecy!
As Swyx said:

> What does it really matter? As long as they can serve it at a sustainable cost, that's about it.

---

References

- [[2511.13029] AA-Omniscience: Evaluating Cross-Domain Knowledge Reliability in Large Language Models](https://arxiv.org/abs/2511.13029)
- [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models - 2501.12370v3.pdf](https://export.arxiv.org/pdf/2501.12370)
- [Sparsing Law: Towards Large Language Models with Greater Activation Sparsity - 2411.02335v4.pdf](https://export.arxiv.org/pdf/2411.02335)

## Data

| Model                       | Accuracy |
| --------------------------- | -------- |
| Gemini 3ProPreview (high)   | 54%      |
| Gemini 3Flash               | 52%      |
| Gemini 3ProPreview (low)    | 46%      |
| ClaudeOpus 4.5              | 43%      |
| Grok 4                      | 40%      |
| Gemini 2.5Pro               | 37%      |
| GPT-5(medium)               | 37%      |
| o3                          | 37%      |
| GPT-5.2(medium)             | 36%      |
| GPT-5.1(high)               | 35%      |
| DeepSeekV3.2                | 32%      |
| Claude 4.5Sonnet            | 31%      |
| DeepSeekR1 0528             | 29%      |
| Kimi K2Thinking             | 29%      |
| GLM-4.7                     | 28%      |
| Grok 3                      | 27%      |
| Claude 3.7Sonnet            | 27%      |
| GPT-4.1                     | 26%      |
| GLM-4.6                     | 25%      |
| Gemini 2.5Flash             | 25%      |
| GLM-4.5                     | 24%      |
| Llama 4Maverick             | 24%      |
| DeepSeek V30324             | 23%      |
| Qwen3 235BA22B 2507         | 22%      |
| MiniMax-M2.1                | 22%      |
| GPT-5 mini(medium)          | 21%      |
| Nova 2.0ProPreview (medium) | 21%      |
| MiniMax-M2                  | 21%      |
| gpt-oss-120B(high)          | 20%      |
| GPT-4o (Nov)                | 19%      |
| gpt-oss-120B(low)           | 18%      |
| Nova 2.0Lite(medium)        | 17%      |
| Nova 2.0Omni(medium)        | 17%      |
| NVIDIANemotron 3Nano        | 17%      |
| GPT-5 nano(medium)          | 16%      |
| Claude 4.5Haiku             | 16%      |
| Qwen3 30BA3B 2507           | 15%      |
| Qwen3 30B                   | 15%      |
| Command A                   | 15%      |
| gpt-oss-20B(high)           | 15%      |
| Llama 4 Scout               | 14%      |
| Olmo 3.132B Think           | 14%      |
| Qwen3 14B                   | 14%      |
| gpt-oss-20B(low)            | 14%      |
| Ministral 14B(Dec '25)      | 12%      |
| Gemma 3 27B                 | 12%      |
| Gemma 3 12B                 | 10%      |
| Gemma 3 4B                  | 7%       |
| Gemma 3 1B                  | 3%       |
