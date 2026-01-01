---
title: On the Viability of Fine-Tuning SLMs
date: 2025-11-07
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
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

Until recently, I held the opinion that training custom language models was inadvisable except in relatively rare cases.
Simply requiring marginally better performance on your task was insufficient justification for customization; subsequent generations of models from foundation labs would inevitably catch up.
Only when you had extreme latency constraints, specific tasks with low drift, and/or such high use that you could guarantee GPU utilization and ROI was it possible to justify the expected economics associated with managing the model lifecycle.
And even then, achieving a successful outcome would be unlikely due to the number of ways customization could fail. _Success_ here means both that the custom model performs as or better than expected on in-domain tasks (e.g., better than available alternatives in accuracy and/or speed) and that the business achieves positive ROI as a result of training and deploying the custom model.

I hold this position particularly strongly for from-scratch training.
The idea that merely domain-specific corpora would lead to a model that consistently outperforms others on in-domain tasks is counter to [the Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html); Bloomberg famously spent millions training BloombergGPT, touting its superior performance to (the then state-of-the-art) GPT-3 on financial tasks [^bloomberggpt], only to be outperformed by GPT-4 mere months later [^outperformed].

In the case of fine-tuning, there are examples and anecdotes of successful outcomes; however, the anecdotes typically assume organizational efficiency across the model lifecycle.
Positive ROI on fine-tuning is not an expectation that most small/mid-sized businesses should have.
Organizations that have the potential to achieve successful outcomes must already have fairly sophisticated teams and processes that can rapidly create the expertise and data, observability, and infrastructure pipelines required to manage fine-tuning and deploying models.
Even in these cases, the number of potential gotchas that exist (catastrophic forgetting, unexpected biases and misalignment, data drift, poor inference efficiency, industry catch-up, etc.) makes it difficult to predict successful outcomes.

## The Convergence: Changing My Mind

I say "until recently" because there has been a convergence of developments over the past 2–3 months that has reframed my opinion on the viability of fine-tuning models -- specifically of fine-tuning small language models.
The organizational requirements remain high, but tooling and research have made successful customization more likely once the requirements are met.

### [Frontier AI capabilities can be run at home within a year or less | Epoch AI](https://epoch.ai/data-insights/consumer-gpu-model-gap)

Epoch AI reported in August 2025 that open-weight/weight-available models running on consumer hardware (NVIDIA RTX 5090) matched the performance of the state-of-the-art frontier models of yesteryear.
Models like EXAONE 4.0 32B and Qwen3 30B-A3B-2507 (July 2025) achieve approximately the GPQA-Diamond and MMLU-Pro performance of OpenAI's o1 (December 2024) and outperform OpenAI's GPT-4o (May 2024) and Claude 3.5 Sonnet (October 2024).

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/epochai_frontier_match.png"
alt="Scatter plot titled 'Artificial Analysis Intelligence Index' (y-axis 0–100) versus release date (July 2023–July 2025). Two upward trend lines show frontier models (teal) consistently above open models on a consumer GPU (pink), with an indicated ~6‑month lead for frontier models. Example points are labeled (frontier: Claude 2, GPT‑4o, Grok 4; open: Qwen‑14B, Gemma 2 27B, QwQ‑32B). Background shading marks the RTX 4090 era (≤28B) before 2025 and RTX 5090 era (≤40B) after."
caption="Models running on consumer hardware match frontier models on Artificial Analysis' Intelligence Index with approximately 6 months lag [Frontier AI capabilities can be run at home within a year or less | Epoch AI](https://epoch.ai/data-insights/consumer-gpu-model-gap)" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

Capabilities formerly limited to large language models that require clusters of nodes of GPUs (a "node" might typically have 8 A100/H100 GPUs; a cluster is made of many nodes) are now accessible with small language models that can run on single-node or single-GPU deployments.
This dramatically reduces infrastructure costs for organizations who shift their inference from large to small LMs and comes with benefits to token generation speed (and user-experienced latency).

Moreover, many of these SLMs are open-weight models or are available for commercial use and modification, making them excellent candidates upon which to fine-tune a custom model.

### Domain-optimized models match frontier model performance

[Cursor Composer](https://cursor.com/blog/composer) and [Cognition SWE-1.5](https://cognition.ai/blog/swe-1-5) are both specialized coding models designed to outperform frontier models' token generation speeds while matching their coding performance.
They succeeded, respectively achieving ~3x and ~6x faster token generation with performance on coding benchmarks just shy of the frontier models like GPT-5 and Claude Sonnet 4.5.
Neither Cursor nor Cognition have released any information on the base models; both release blogs instead discuss the use of reinforcement learning to achieve the agentic performance on coding tasks.

These models demonstrate the viability of fine-tuning small(er), domain- or task-specialized models that can rival or surpass general-purpose frontier models in their target domain, while reducing inference costs and latency.

### Resolving Catastrophic Forgetting

Catastrophic forgetting is "the tendency for models to lose previously acquired capabilities when trained on new tasks" [^rlrazor].
Traditionally, catastrophic forgetting was mitigated by incorporating data from the base model's original training dataset into any fine-tuning.
Instead of training only on new data, this mitigation added considerable expense and inefficiency to fine-tuning due to processing considerable additional tokens to remind the model of what it already knew.

Recent research identifies " [RL's Razor](https://arxiv.org/abs/2509.04259)", finding that RL (reinforcement learning) minimizes distribution shift during training as a function of how it samples from its own distribution, whereas SFT (supervised fine-tuning) can pull the model toward arbitrary new distributions, potentially leading to catastrophic forgetting [^rlrazor].
This is one reason why preference alignment techniques (like DPO) and online RL methods are sample-efficient; they typically do not need to include base-domain samples that serve only as a mitigation for catastrophic forgetting.

Distillation is the process of fine-tuning a smaller model to behave like a larger, more powerful one.
Whereas the large model is traditionally trained to predict the most likely next token, a smaller (student) model being taught during distillation is trained to predict the most likely next token that its teacher would use.
Distillation is one reason that small models can catch up to frontier models in \<1 year.

By itself, distillation for general capability is unlikely to lead to catastrophic forgetting because by default the distillation will incorporate data similar to that which the smaller model saw in its pretraining phase.
However, distilling for customization _can_ lead to catastrophic forgetting because of RL's Razor.
Thinking Machines Lab recently published research on [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (OPD), where they showed that it can be used to recover forgotten capabilities.
In essence, the model is fine-tuned on domain knowledge, potentially leading to forgetting.
Then, OPD is used, where the _base_ model is used as the teacher to help the fine-tuned student model "remember" what it had forgotten.

### LoRA: "Store brand" to Brand Name

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method for base language models, making them cheaper to fine-tune.
The main model weights are frozen (not updated during training) and only a low-rank approximation is learned representing the changes that full fine-tuning would induce in the frozen model.
This reduces GPU requirements, data requirements, and risk of catastrophic forgetting.
However, until recently, LoRA [had a branding problem](https://youtu.be/yYZBd25rl4Q?si=lndwwcXCj7shCQzj&t=604), perceived as "bargain full fine-tuning" for those who were unable to afford full fine-tuning, because in typical configuration, it substantially underperforms full fine-tuning [^loraless].

However, [recent research from Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/) has demonstrated otherwise, showing that LoRA-based fine-tuning can achieve equivalent performance to fine-tuning the full base model when working with small-to-medium-sized instruction-tuning and reasoning datasets or with reinforcement learning.
Referring back to RL's Razor, this is because LoRA doesn't have much parameter space (relatively speaking) to store full knowledge updates, but _does_ have the space for the relatively minor tweaks that RL induces.
This makes LoRA an excellent candidate for many of the use cases when available models are "almost but not quite" right.
LoRA is ideal for learning minor tweaks to response format, phrasing preferences, model character, or customizing for specific tool availability or workflows.

Further, deployment can take advantage of the fact that LoRA is an adapter; if multiple LoRAs are trained to customize the same base model, a single base model deployment can leverage the different adapters for task specialization [^slora] [^clora].
Research on efficient use of LoRA remains ongoing, with IBM proposing Activated LoRA (aLoRA) which would allow aLoRA to use the embeddings existing in the KV Cache rather than requiring a full pass through the base model [^alora].

> [!TIP]
> Sadly, [vLLM does not intend to support aLoRA](https://github.com/vllm-project/vllm/pull/19710#issuecomment-3386127417): "After discussing with the maintainer, we're not considering this PR for now"

### Available Frameworks, Managed Services

The industry emphasis on reinforcement learning as a mechanism for teaching language models to reason and act, and distillation to massively increase the capabilities of SLMs, has resulted in a considerable acceleration in research, knowledge sharing, and available frameworks.
All hypercloud providers (AWS, Azure, GCP) provide managed fine-tuning, and a number of other specialized providers (like [Thinking Machines Lab's Tinker](https://thinkingmachines.ai/tinker/) and [Lightning AI](https://lightning.ai/)) are designed as fine-tuning-as-a-service offerings.
For developers who want more control, a number of packages exist that support distillation, fine-tuning, and reinforcement fine-tuning, including [OpenPipe ART](https://art.openpipe.ai/getting-started/about), [Microsoft Agent Lightning](https://microsoft.github.io/agent-lightning/stable/), [HuggingFace TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/en/index), and PyTorch [torchtune](https://github.com/meta-pytorch/torchtune), [torchforge](https://github.com/meta-pytorch/torchforge), and [OpenEnv](https://github.com/meta-pytorch/OpenEnv).

### Inference Efficiency/Optimization

Finally, the secret knowledge of how to deploy models for inference with efficient utilization is becoming more available.
BentoML's [LLM Inference Handbook](https://bentoml.com/llm/), JAX's [How To Scale Your Model](https://jax-ml.github.io/scaling-book/), and [MoE Inference Economics from First Principles](https://www.tensoreconomics.com/p/moe-inference-economics-from-first) serve as excellent primers for understanding inference at a very deep level.
Hugging Face's [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), while not focused on inference, explains the granular details of training on large GPU clusters, which also helps to understand the complexities of distributed inference.

## Now Viable... But Advisable?

While my reaction to "Should we fine-tune a model for this?" may no longer be an immediate "No," the recent _viability_ of fine-tuning a custom SLM doesn't mean it's necessarily _advisable_.
Long-term success will require a strong foundation of organizational data and MLOps capabilities.
It demands the up-front definition of specific goals with measurable outcomes that align with product strategy.
It requires a scientific approach to iterative refinement backed by robust data pipelines and infrastructure for training data collection, analysis, annotation, and iteration.
And it depends on mature AI/MLOps practices for managing model checkpoints and artifacts, logging training metrics, running evals and benchmarks, and optimizing inference.

These capabilities are external to the question of "_Can_ we fine-tune?" but critical for the assessment of "_Should we?_" As discussed above, many technical hurdles to fine-tuning itself have been lowered or removed entirely, but a broader, structured approach for assessing broader organizational readiness is necessary for successful outcomes.
This is where a framework like that provided in the [PMI Certified Professional in Managing AI (PMI-CPMAI)](https://www.pmi.org/certifications/ai-project-management-cpmai) comes into play.

> [!IMPORTANT]
> **Disclosure**: I am employed by the Project Management Institute (PMI) and hold the PMI Certified Professional in Managing AI (PMI-CPMAI) credential. PMI covered the cost of my certification as part of my role.\
> The views expressed here are entirely my own and do not represent PMI's positions or policies. PMI has no influence or editorial oversight on personal projects like this post or, more broadly, my blog.

The PMI-CPMAI methodology provides a structured approach for assessing this very readiness.
While I had my personal nits to pick with the certification (e.g., the old exam felt like it tested rote memorization for the CPMAI-supplied specific definitions rather than comprehension and application; I have no insight into the current exam), I believe the course and its content provide real value.
It serves as an excellent primer for introducing AI and ML concepts to non-data scientists, and its readiness framework with go/no-go gates forces product teams to take off their rose-colored glasses and honestly assess if they have access to the surrounding, supporting capabilities necessary to turn a potentially viable project into an advisable one.

---

## Further Reading

- [The Smol Training Playbook: The Secrets to Building World-Class LLMs | HuggingFace](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- [Our Humble Attempt at "How Much Data Do You Need to Fine-Tune"](https://barryzhang.substack.com/p/our-humble-attempt-at-fine-tuning)
- [huggingface/evaluation-guidebook: Sharing both practical insights and theoretical knowledge about LLM evaluation that we gathered while managing the Open LLM Leaderboard and designing lighteval!](https://github.com/huggingface/evaluation-guidebook)
- [Mastering LLM Techniques: Inference Optimization | NVIDIA Technical Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [[2503.08311v2] Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/abs/2503.08311v2)
- [How to Optimize LLM Inference](https://neptune.ai/blog/how-to-optimize-llm-inference)
- [[2510.03847] Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade offs](https://arxiv.org/abs/2510.03847)

## References

<!-- markdownlint-disable MD013 -->

[^bloomberggpt]: [[2303.17564] BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)

[^outperformed]: [[2305.05862] Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks](https://arxiv.org/abs/2305.05862)

[^rlrazor]: [[2509.04259] RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259)

[^loraless]: [[2405.09673] LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)

[^slora]: [[2311.03285v3] S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285v3)

[^clora]: [Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead - 2407.00066v4.pdf](https://www.arxiv.org/pdf/2407.00066)

[^alora]: [[2504.12397] Activated LoRA: Fine-tuned LLMs for Intrinsics](https://arxiv.org/abs/2504.12397)
