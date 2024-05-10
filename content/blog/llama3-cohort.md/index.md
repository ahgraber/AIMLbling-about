---
title: Cohort of Models
date: 2024-05-09
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags: ["LLMs", "generative ai", "comparison"]
series: []
layout: wide
toc: false
math: false
draft: true
---
The "open weights" LLM ecosystem is thriving, with major new releases from Meta, Microsoft, Databricks, and Snowflake
in the past two months.  Given that Meta's Llama 2 family became a standard for comparison for all other models,
I thought it might be useful that aggregate all of the information of the 'Llama 3 cohort' in a single place.
I've included Llama 2 as a point of comparison.  Models are ordered by date of introduction.

{{< callout type="info" >}}
  I've done my best to make these table not-terrible, but it's probably still trash on mobile or non-wide aspect ratios.  Sorry.
{{< /callout >}}

## Architectures

{{< table path="architectures.csv" header="true" caption="A comparison of model architectures" >}}

{{< callout type="warning" >}}
  I've not included ChatGPT 3.5 or 4, or Anthropic's Claude because they're completely closed models
  and I was unable to find any information worth comparing.
{{< /callout >}}

{{< callout type="info" >}}
  Table row/column will be transposed from here on (they fit better that way).
{{< /callout >}}

## Performance

{{< callout type="warning" >}}
  I've added evals from recent ChatGPT checkpoints and Anthropic's Claude 3 models.
{{< /callout >}}

{{< table path="benchmarks.csv" header="true" caption="A comparison of model architectures" >}}

## Environmental impact

{{< table path="environmental_impact.csv" header="true" caption="Environmental impacts of training (but not inference)" >}}

## Observations

### Themes and industry alignment

higher token counts
more data; high-quality datasets (or less, extremely filtered data)
no major internal arch changes since llama 1/2 (RoPE, SwiGLU, etc);
  arch changes focus on optimizing attention & inference speed
    MQA,GQA
    only mistral is using sliding window attention?
    snowflake working with attention sinks
  MoE

### Concerns

some portions of architectures are left unsaid or to be inferred from prior work
performance metrics / evals vary by source
performance metrics are unclear whether they're base, fine-tuned, or aligned models

what is 1T Co2 for a standard person?
environmental impacts
  not just co2 that's "offset by ..."
  water / heat
  amortized pollution associated with electronics manufacturing

## References

- [Models - Hugging Face](https://huggingface.co/models)
- [[2302.13971] LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [[2307.09288] Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
- [gemma-report.pdf](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)
- [[2310.06825] Mistral 7B](https://arxiv.org/abs/2310.06825)
- [[2401.04088] Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Cheaper, Better, Faster, Stronger | Mistral AI | Frontier AI in your hands](https://mistral.ai/news/mixtral-8x22b/)
- [mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.](https://github.com/mistralai/mistral-src)
- [Introducing DBRX: A New State-of-the-Art Open LLM | Databricks Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [databricks/dbrx: Code examples and resources for DBRX, a large language model developed by Databricks](https://github.com/databricks/dbrx)
- [Snowflake Arctic - LLM for Enterprise AI](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/)
- [Snowflake-Labs/snowflake-arctic](https://github.com/Snowflake-Labs/snowflake-arctic)
- [Snowflake Arctic Cookbook Series: Arctic's Approach to Data | by Snowflake AI Research | Snowflake Builders Blog: Data Engineers, App Developers, AI/ML, & Data Science | Apr, 2024 | Medium](https://medium.com/snowflake/snowflake-arctic-cookbook-series-arctics-approach-to-data-b81a8a0958bd)
- [[2404.14219] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)
- [[2309.05463] Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)
- [[2306.11644] Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
- [Command R: RAG at Production Scale](https://cohere.com/blog/command-r)
- [Introducing the next generation of Claude \ Anthropic](https://www.anthropic.com/news/claude-3-family)
- [Model_Card_Claude_3.pdf](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
- [Zhen Wang on LinkedIn: #anthropic #claude #tokenizer #llm](https://www.linkedin.com/posts/zhenwang_anthropic-claude-tokenizer-activity-7067072872019619840-hZ-7)
- [GPT-4 | OpenAI](https://openai.com/index/gpt-4-research/)
- [[2303.08774] GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [Andrej Karpathy on X: "Congrats to @AIatMeta on Llama 3 release!! 🎉](https://twitter.com/karpathy/status/1781028605709234613)
- [Estimating Training Compute of Deep Learning Models – Epoch AI](https://epochai.org/blog/estimating-training-compute)
