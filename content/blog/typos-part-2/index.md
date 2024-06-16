---
title: How Robust Are LLMs to Typos? (part 2)
date: 2024-06-16
tags:
  # meta
  - "experiment"
  # ai/ml
  - "generative AI"
  - "prompts"
  - "LLMs"
series:
  - "typos"
draft: true
math: true
---

This is part two of a four-part series (one, three, four) where I examine the influence typos have on LLM response
quality.

In this post, I use the typo generation function to induce typos with increasing frequency in the hopes of
understanding how typos influence **tokenization**.

Recall my hypothesis:

> Typos increase token counts --  
> As the typo frequency rises, the tokenization will fit the data less well, requiring additional, more granular tokens
> to represent the data...  
> Unless the typos are so popular that they have made it into the vocabulary.

## Design

I'll use the TinyBench version of MMLU as a standardized dataset for all experiments.[^tinybench] I induce typos at an
increasing rate from 5% (where roughly 1 word in 20 will have a typo) to 100% (approximately every word will have a
typo). For each rate, I generate 5 different typo variations of each MMLU question.

An implication of my hypothesis is that tokenizer with a larger vocabulary will be able to represent typos better than
one with a smaller vocabulary. To that end, I've elected to compare the tokenizers of Llama 2 (32k token vocabulary)
and Llama 3 (128k token vocabulary).

For each model's tokenizer:

1. Tokenize each question in TinyBenchmark's MMLU set and count the tokens.
2. Introduce typos into each question's text, tokenize, and count the tokens used by each variation at each typo rate.
3. Find the increase in token use by subtracting the baseline count from the variant count.
4. Find the percent increase by dividing the increase by the baseline count.

## Results

The results confirm the hypothesis that typos increase token use. Llama 2 (with the smaller vocabulary) needs more
tokens to represent the typos than Llama 3.

{{< figure
  src="images/count-differences.png"
  caption="Llama 2 requires a greater increase in token use to represent typo-laden text than Llama 3." >}}

However, since Llama 3 uses fewer tokens in the baseline (its larger vocabulary includes larger "word chunks" as
tokens, reducing token count overall), it shows a larger proportional increase in the number of tokens required to
represent the typo-laden questions.

{{< figure
  src="images/pct-differences.png"
  caption="Llama 3 uses fewer tokens in the baseline, so the token use increase is proportionally larger." >}}

## References

[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)

<!-- [^perturbation]:
[[2402.15833] Prompt Perturbation Consistency Learning for Robust Language Models](https://arxiv.org/abs/2402.15833)
[^promptperplexity]: [[2212.04037] Demystifying Prompts in Language Models via Perplexity Estimation](https://arxiv.org/abs/2212.04037)
[^perplexity]: [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity) -->
