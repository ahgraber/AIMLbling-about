---
title: Cohort of Models
date: 2024-05-11
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags: ["LLMs", "generative ai", "comparison"]
series: []
layout: wide
toc: false
math: false
draft: false
---

The "open weights" or "open model" LLM ecosystem is thriving, with major new releases from Meta, Microsoft, Databricks,
and Snowflake in the past two months. Given that Meta's Llama 2 family became a standard for comparison for all other
models, I thought it would be useful that aggregate all of the information of the 'Llama 3 cohort' in a single place.
I've included Llama 2 as a point of comparison. Models are ordered by date of introduction.

{{< callout type="question" emoji="📣" >}} I've done my best to make these table not-terrible, but they're probably
still trash on mobile or non-wide aspect ratios. Sorry. {{< /callout >}}

## Architectures

{{< table path="architectures.csv" header="true" caption="A comparison of model architectures" >}}

{{< callout type="warning" >}} I've not included ChatGPT 3.5 or 4, or Anthropic's Claude because they're completely
closed models and I was unable to find any information worth comparing. {{< /callout >}}

## Performance

{{< callout type="question" emoji="📣" >}}

1. I've transposed table rows/columns from here on (they fit better that way).
2. I've included benchmarks from OpenAI's GPT4 paper and Anthropic's Claude 3 announcement as points of comparison.
   {{< /callout >}}

{{< table path="benchmarks.csv" header="true" caption="A comparison of model architectures" >}}

## Environmental impact

{{< table path="environmental_impact.csv" header="true" caption="Environmental impacts of training (but not inference)" >}}

\* FLOPs and % utilization are estimated using Epoch AI's online tool[^compute]

## Observations

In doing the research and aggregation for these tables, I read, and reread, and pored over the papers describing the
technical details of these models.

The TLDR? All changes have been made with inference efficiency in mind.

### Data matters

Almost every paper makes a statement about how they cleaned, filtered, and optimized their data; several (Llama 3,
Phi 3) use LLMs in this process. Some (the Phi 3 models in particular), also use LLMs to generate "high quality
synthetic" data. Databricks notes in their DBRX announcement that the dataset used to train DBRX is 2x as good as the
dataset they used in 2023; that is, "half as many tokens are necessary to reach the same model quality."[^dbrx]
Concretely, curation of training data removes the low-information sequences that require compute to process but
minimally improve model performance.

Additionally, models are training on more tokens, moving from compute-optimal toward data-optimal frontiers of the
Chinchilla laws. Microsoft explicitly state this in the Phi 3 technical paper, saying they "calibrate the training data
to be closer to the "data optimal" regime for small models."[^phi] Meta echo this in the Llama 3 release announcement,
saying "while the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens,
we found that model performance continues to improve even after the model is trained on two orders of magnitude more
data. Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens.
_Larger models can match the performance of these smaller models with less training compute, but smaller models are
generally preferred because they are much more efficient during inference_"[^llama3] (emphasis mine).

{{< callout type="info" emoji="💡" >}} The Chinchilla "laws" describe the relationship between model size (parameters),
dataset size (tokens), performance (loss), and cost of compute (FLOPs). Historically, the Chinchilla laws have been
used to understand the "compute optimal" point, or the point of diminishing returns (in terms of model performance
improvement) associated with additional compute, given a static model size and dataset. {{< /callout >}}

Why is this? Well, it turns out that inference costs are expensive. _Really expensive_. Like,
OpenAI-might-go-bankrupt-because-it-costs-$700,000-per-day expensive.[^cost] LLM providers want efficiency, but they
also need to show improvements over the last generation. To do this, they have transitioned toward the "data optimal"
(or more importantly, inference-efficient) training paradigm. The Chinchilla laws suggest that you can reach the same
performance as a "compute optimal" model by training a smaller model for longer on more tokens.[^law] The Llama 3
cohort trains a model the same size as the prior generation for longer on many more, more informative tokens,
demonstrating generational improvements without increasing model sizes (and therefore without increasing inference
requirements).

### Attention on attention

Aside from optimizations to Attention, there have been minimal changes to the Transformers architecture. All models use
RoPE for their positional embeddings, almost all use RMSNorm, and the majority use SwiGLU for the activation function
in their feedforward modules.

Almost all new models use new approaches to optimize Attention, including Multi-Query Attention (MQA)[^mqa],
Grouped-Query Attention (GQA)[^gqa], Sliding Window Attention (SWA)[^swa], and even attention sinks[^snk]! Most models
use GQA, while Mistral retains its use of SWA from its prior generation, and Phi 3 experiments with it. Snowflake
shared they are "developing an attention-sinks-based sliding window implementation to support unlimited sequence
generation capability" in their Arctic release announcement.[^snow] DeepSeek just released their v2, which introduces a
novel Multi-head Latent Attention (MLA) mechanism.[^mla]

Why the focus on Attention? The Attention mechanism scales poorly relative to sequence length. As we increase the
context length that LLMs can handle, we massively increase compute and memory required to allow each token to attend to
each other token (quadratic scaling). While Flash Attention[^flash] and KV Caching have reduced this worst-case
quadratic scaling, Attention is still incredibly memory-hungry - KV caching trades linear compute scaling (good!) for
increased memory requirements (bad!) at inference time. The aforementioned Attention optimizations focus primarily on
reducing memory used by Attention, especially during inference.

### Expert expectations

Four of the Llama-3 cohort use a Mixture of Experts (MoE) design; Snowflake Arctic and DeepSeek-v2 use novel MoE
architectures while Mistral 8x22B and DBRX use a more standard sparse MoE. Mixture of Experts models alter the
feed-forward component of a standard Transformer, turning the standard dense network into a larger sparse one. These
wider FF nets are grouped into "experts", and a router determines which expert is active on a per-token basis. In
theory, MoE allows the model to have a much higher parameter count (see `active parameters` in the top table above).
Therefore, an LLM with an MoE architecture is kind of a cheat against the Chinchilla laws; they tend to train quickly
(i.e., with less compute) and have very good performance (on benchmarks) relative to dense models of equivalent
_active_ parameter counts. The sparsity of MoE also means they tend to be fast in inference.

The downsize? While MoE models are _compute_ efficient because only a portion of the sparse feed-forward is active at
any one time, MoE models have larger memory requirements because all of the parameters (even the inactive ones) have to
be loaded into memory for inference.

As an aside, I have to wonder about the use of Mixture-of-Experts as a way to increase model size while
maintaining/improving inference speed. Recent research with pruning dense models indicates LLMs are already
overparameterized.[^short] [^prune] These MoE models seem to be even further overparameterized vs their dense peers.
Both DataBricks DBRX and Snowflake Arctic are MoE models with significantly more total _and_ active parameters than
Llama 3 8B, and both trained on fewer tokens. DBRX is closer (12T DBRX vs 15T Llama3 8B). Snowflake Arctic only trained
on <4T, thought this may be because Arctic is intended more for "enterprise" use cases than as a general language model
(the lack of MMLU scores reported for Arctic adds weight to this theory). Perhaps the additional overparameterization
helps to explain their training speed and performance, while the sparsity associated with the expert routing is akin to
pruning a dense network?

### Inference implications

{{< callout type="question" emoji="📣" >}} Most of what I'll mention here is a rehash of what I learned listening to
[Dylan Patel (of SemiAnalysis) on the Latent Space podcast](https://www.latent.space/p/semianalysis). It is 100% worth
the listen (or reviewing the transcript).

In fact, go do that now. I'll wait. {{< /callout >}}

As Dylan Patel pointed out, training LLMs tends to be compute-intensive; compute FLOPs are the bottleneck. You get as
close to 100% utilization (MFU - model FLOPs utilization) as you can (given network/communication and sharding
inefficiencies) during training. At inference time, you need _half_ the compute needed when training (you only need the
forward pass, no backpropagation). If you want to get more responses to more prompts, you need memory. If you want to
provide longer answers or have longer input contexts, you need memory. And as I discussed earlier, Attention is memory
hungry; it scales approximately quadratically based on sequence length. So all of a sudden, you're limited by memory
bandwidth (MBU - _memory_ bandwidth utilization).

Now we see why the AI Engineers made the choices they have for the Llama-3 cohort. MoE lets us take advantage of larger
GPU clusters' total memory and spread the MBU out over more individual GPUs. Attention optimizations reduce the memory
requirements relative to sequence lengths, reducing MBU requirements. Due to the data-optimal paradigm shift, model
sizes have (generally) not increased, meaning we get better performance on today's infrastructure, or we can retain
current performance with smaller models... reducing MBU requirements. Everything comes back to efficiency in inference.

### Frustrations and concerns

Over the course of researching for this post, I ran into some frustrations:

First, technical papers do not always include details of the model architectures, either leaving them entirely unsaid
or referring the reader to "prior work". This is lazy. Since the reader _does_ have access to the information in the
prior work, the authors are not hiding anything per se, but are making the reader chase down the unspecified pieces of
information. It also reduces any replicability (not that replication is possible anyway, given these models are
generally open weights and do not share training code or source datasets... but I digress) and invites assumptions that
may not be accurate should the wrong prior work be referenced or should the current implementation use some slight
unmentioned deviations from the prior work that was glossed over or forgotten during publication. As a side note, I
found HuggingFace's model catalog[^huggingface] to be invaluable in filling in many of the details, especially around
hidden and feed-forward dimensions and attention head and layer counts.

Secondly, reporting performance benchmarks is unstandardized. In the papers, there was high variance in terms of
benchmark specificity, meaning the reported results are likely _not_ directly comparable -- and this is prior to
considering any shenanigans on the _implementation_ of the benchmarks.[^openllm] [^benchmarks] MMLU might be measured
based with English-only questions, or on subsets of the benchmark, or used to demonstrate multilingual capabilities.
GSM8K evaluation might use 8-shot chain-of-thought, 5-shot chain-of-thought, 0-shot, or majority voting. Further, its
often unclear whether the reported benchmarks are done on the model after pretraining, after fine-tuning, or after
alignment (or some combination that provides the best results). So, take all benchmarks with a grain of salt... which,
doesn't that kind of invalidate the raison d'être of a benchmark?

Finally, I have to credit Meta for revealing training cost in terms of GPU hours _and_ energy implications (tons of CO2
equivalent). It is frustrating to me that it is nonstandard to discuss energy grid and environmental implications of
their training costs. I would like to see all technical papers discuss the training costs/requirements for their
models, even if they do not release the source code to replicate. GPU class, GPU time, FLOPs, and tCO2eq are all
important metrics to understand the energy requirements of training a model. Further, I would also like to see papers
report the cost per 1M token inference in the same manner - GPUs required, GPU class, FLOPs, and tCO2eq.

## References

- [Models - Hugging Face](https://huggingface.co/models)
- [[2302.13971] LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [[2307.09288] Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [gemma-report.pdf](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
- [[2310.06825] Mistral 7B](https://arxiv.org/abs/2310.06825)
- [[2401.04088] Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Cheaper, Better, Faster, Stronger | Mistral AI | Frontier AI in your hands](https://mistral.ai/news/mixtral-8x22b/)
- [mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.](https://github.com/mistralai/mistral-src)
- [databricks/dbrx: Code examples and resources for DBRX, a large language model developed by Databricks](https://github.com/databricks/dbrx)
- [Snowflake-Labs/snowflake-arctic](https://github.com/Snowflake-Labs/snowflake-arctic)
- [Snowflake Arctic Cookbook Series: Arctic's Approach to Data | by Snowflake AI Research | Snowflake Builders Blog: Data Engineers, App Developers, AI/ML, & Data Science | Apr, 2024 | Medium](https://medium.com/snowflake/snowflake-arctic-cookbook-series-arctics-approach-to-data-b81a8a0958bd)
- [[2309.05463] Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)
- [[2306.11644] Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
- [Command R: RAG at Production Scale](https://cohere.com/blog/command-r)
- [Introducing the next generation of Claude \ Anthropic](https://www.anthropic.com/news/claude-3-family)
- [Model_Card_Claude_3.pdf](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
- [Zhen Wang on LinkedIn: #anthropic #claude #tokenizer #llm](https://www.linkedin.com/posts/zhenwang_anthropic-claude-tokenizer-activity-7067072872019619840-hZ-7)
- [GPT-4 | OpenAI](https://openai.com/index/gpt-4-research/)
- [[2303.08774] GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [Andrej Karpathy on X: "Congrats to @AIatMeta on Llama 3 release!! 🎉](https://twitter.com/karpathy/status/1781028605709234613)

## Works Cited

[^compute]:
    [Estimating Training Compute of Deep Learning Models – Epoch AI](https://epochai.org/blog/estimating-training-compute)

[^dbrx]:
    [Introducing DBRX: A New State-of-the-Art Open LLM | Databricks Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

[^phi]:
    [[2404.14219] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)

[^llama3]:
    [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)

[^cost]:
    [The Inference Cost Of Search Disruption – Large Language Model Cost Analysis](https://www.semianalysis.com/p/the-inference-cost-of-search-disruption)

[^law]:
    [Revised Chinchilla scaling laws – LLM compute and token requirements – Educating Silicon](https://www.educatingsilicon.com/2024/04/29/revised-chinchilla-scaling-laws-impact-on-llm-compute-and-token-requirements/)

[^mqa]: [[1911.02150] Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
[^gqa]:
    [[2305.13245] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

[^swa]: [[2004.05150v2] Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150v2)
[^snk]: [[2309.17453] Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
[^snow]:
    [Snowflake Arctic - LLM for Enterprise AI](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/)

[^mla]:
    [[2405.04434v2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v2)

[^flash]:
    [[2205.14135] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

[^short]:
    [[2403.03853] ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)

[^prune]:
    [[2306.11695] A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

[^huggingface]: [Models - Hugging Face](https://huggingface.co/models)
[^openllm]: [What's going on with the Open LLM Leaderboard?](https://huggingface.co/blog/open-llm-leaderboard-mmlu)
[^benchmarks]:
    [[2402.01781v1] When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards](https://arxiv.org/abs/2402.01781v1?trk=public_post_comment-text)
