---
title: Predictions 2025
date: 2025-01-01
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
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
draft: false
---

<!-- markdownlint-disable-file MD036 -->

Here are my predictions for 2025 along with my certainty / probability estimate:

## Agents, agents, agents

We got nascent function-calling and JSON mode toward the end of 2023, which morphed into formal structured generation over the course of 2024. [^outlines] [^structured_outputs]
Combined with an explosion of Agentic frameworks ([LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph),
[Semantic Kernel](https://github.com/microsoft/semantic-kernel), [AutoGen / AG2](https://github.com/ag2ai/ag2),
[CrewAI](https://github.com/crewAIInc/crewAI), [PydanticAI](https://github.com/pydantic/pydantic-ai), etc.),
2024 was the year AI Agents came into their own.

We will see and even stronger push for Agents in 2025. If you're not yet tired of seeing "agent" or "agentic" plastered all over everything, you will be.

_Probability: 100%_

## Inference-time compute

"Slow thinking" AI models like OpenAI's [o1](https://openai.com/o1/) and [o3](https://arcprize.org/blog/oai-o3-pub-breakthrough) models (as opposed to the "fast thinking" immediate response of standard LLMs)
demonstrate that existing models can be deployed/instructed to think deliberately about a problem to solve it iteratively.

> [!NOTE]
> I'm avoiding describing the extended thinking as "reasoning".
> I recognize that's the popular term; while to some degree it does represent the process of inference-time compute,
> I think it will take at least until the next generation of models for them to exhibit full-fledged planning, reasoning, or metacognition.

We will see more new models and use of models using inference-time compute in 2025 in situations where low latency is not required, especially to boost performance of small models[^test-time-scaling]
or where near-human performance is required to automate complex, out-of-domain tasks[^o3].
And, frontier models using inference-time compute will continue to push the boundary toward super-human performance (but not AGI).

_Probability: 100%_

## 50/50: Large frontier model performance

Either (a) improvement curves on large models will continue to slow; improvements on benchmarks is largely driven by post-training recipes[^gpt5]
or (b) gpt-5-generation models will alter what has become the standard model architecture in the current-generation.

_Probability: 50/50_

> [!NOTE]
> If I had to bet, Western/U.S. labs (Meta, Google, OpenAI) will fall prey to slowly improving the status quo, > while Chinese labs (DeepSeek, Qwen) will experiment with new architectures. Anthropic and Mistral are wildcards.

### Next-gen Architectural changes

We saw a consolidation of model architectures in 2024, roughly aligning with the Llama 3 recipe.
In order to continue driving performance improvements (without leveraging inference-time compute), model architectures undergo another period of diversification.

These architectural changes may come from improvements to tokenization[^pause] [^patch] [^counting] [^bytes],
attention (linear[^mamba] [^rwkv] [^jamba] or cosine[^cottention] attention),
training objectives (multitoken prediction[^multitoken], JEPA embedding prediction[^JEPA]),
or sampling[^top-n] [^coconut].

_Probability: 75%_

> [!NOTE]
> Bonus:\
> Inspired by [ModernBert](https://arxiv.org/abs/2412.13663), other "deprecated" architectures will be given "modern" makeovers\
> _Probability: 60%_

## Multimodal-first training

As the available reservoir of text data dries up, labs will turn to multimodal data to expand the pretraining corpus and improve embedding representation in a unified latent space[^platonic]
I predict this will result in the adoption of a unified architecture that takes multimodal inputs directly instead of using a pretrained encoder or cross-encoding an LLM-plus-adapter.[^mllms]
Additionally, considerable effort will be put into developing AI-driven refinement and synthesis pipelines critical for improving granular Visual Question Answering (VQA) capabilities to drive improved text:image datasets
(i.e., recognizing the direction a car is moving rather than just recognizing "car" - CLiP and SigLIP are not enough[^vqa])

_Probability: 80%_

## Lighting Round

- Saturation and/or data leakage will invalidate current benchmarks / spur new ones (_Probability: 100%_)
- Continued push to undermine NVIDIA's market dominance in hardware (_Probability: 100%_), but successes (market share/utilization) will come from cloud providers (Amazon, Google, Azure, Groq(?)) and not Intel or AMD (_Probability: 70%_)
- At least one lab will claim "AGI" but their definition will have multiple asterisks (_Probability: 60%_)
- At least one lab will inject ads into their (non-search) free or low-cost chat interface or service (_Probability: 60%_)
- Power (and water) utilization concerns over AI inference costs enter the zeitgeist (_Probability: 50%_),
  with AI datacenters associated with brownouts or wildfires ([looking at you, PG&E](https://www.npr.org/2022/04/12/1092259419/california-wildfires-pacific-gas-electric-55-million)) (_Probability: 30%_)

## References

[^outlines]: [Structured Generation Improves LLM performance: GSM8K Benchmark](https://blog.dottxt.co/performance-gsm8k.html)

[^structured_outputs]: [Introducing Structured Outputs in the API | OpenAI](https://openai.com/index/introducing-structured-outputs-in-the-api/)

[^test-time-scaling]: [Scaling test-time compute - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)

[^o3]: [OpenAI o3 Breakthrough High Score on ARC-AGI-Pub](https://arcprize.org/blog/oai-o3-pub-breakthrough)

[^gpt5]: [OpenAI's GPT-5 reportedly falling short of expectations | TechCrunch](https://techcrunch.com/2024/12/21/openais-gpt-5-reportedly-falling-short-of-expectations/)

[^pause]: [[2310.02226] Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226)

[^patch]: [[2407.12665] Patch-Level Training for Large Language Models](https://arxiv.org/abs/2407.12665)

[^counting]: [[2410.19730] Counting Ability of Large Language Models and Impact of Tokenization](https://arxiv.org/abs/2410.19730)

[^bytes]: [[2412.09871] Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)

[^mamba]: [[2312.00752] Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

[^rwkv]: [[2305.13048] RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

[^jamba]: [[2403.19887] Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

[^cottention]: [[2409.18747] Cottention: Linear Transformers With Cosine Attention](https://arxiv.org/abs/2409.18747)

[^multitoken]: [[2404.19737] Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

[^JEPA]: [A Path Towards Autonomous Machine Intelligence | OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)

[^top-n]: [[2411.07641] Top-nσ: Not All Logits Are You Need](https://arxiv.org/abs/2411.07641)

[^coconut]: [[2412.06769] Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)

[^platonic]: [[2405.07987] The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

[^mllms]: [Understanding Multimodal LLMs - by Sebastian Raschka, PhD](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)

[^vqa]: [[2401.06209] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209)
