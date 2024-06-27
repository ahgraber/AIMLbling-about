---
title: How Robust Are LLMs to Typos? (part 2)
date: 2024-06-27
tags:
  # meta
  - "experiment"
  # ai/ml
  - "generative AI"
  - "prompts"
  - "LLMs"
series:
  - "typos"
draft: false
math: true
---

This is part two of a four-part series ([one]({{< ref "/blog/typos-part-1" >}}), three, four) where I examine the
influence typos have on LLM response quality.

In this post, I use the typo generation function to induce typos with increasing frequency in the hopes of
understanding how typos influence **tokenization**.

Recall my hypothesis:

> Typos increase token counts -- as the typo frequency rises, the tokenizer's vocabulary will fit the data less well,
> requiring additional, more granular tokens to represent the data... Unless the typos are so popular that they have
> made it into the vocabulary.

## Design

I've elected to use the TinyBench version of MMLU as a standardized dataset for all experiments.[^tinybench] I induce
typos using the typo generation function from [part one]({{< ref "/blog/typos-part-1" >}}) at an increasing rate from
5% (where roughly 1 word in 20 will have a typo) to 100% (approximately every word will have a typo). For each rate, I
generate 5 different typo variations of each MMLU question.

An implication of my hypothesis is that tokenizer with a larger vocabulary should be able to represent typos better
than one with a smaller vocabulary. To that end, I've elected to compare the tokenizers of Llama 2 (32k token
vocabulary) and Llama 3 (128k token vocabulary).

For each model's tokenizer:

1. Tokenize each question in TinyBenchmark's MMLU set and count the tokens.
2. Tokenize each "typo-ified" question, and count the tokens used by each variation at each typo rate.
3. Find the (presumed) increase in token use by subtracting the baseline count from the variant count.
4. Find the (presumed) percent increase by dividing the increase by the baseline count.

## Results

The results confirm the hypothesis that typos increase token use. More tokens are used the more the typo rate
increases. Additionally, Llama 2 (with the smaller vocabulary) requires more additional tokens to represent the typos
than Llama 3.

{{< figure
  src="images/count-differences.png"
  caption="Llama 2 requires a greater increase in token use to represent typo-laden text than Llama 3." >}}

However, since Llama 3 uses fewer tokens in the baseline (its larger vocabulary includes larger "word chunks" as
tokens, reducing token count overall), it shows a larger _proportional_ increase in the number of tokens required to
represent the typo-laden questions.

{{< figure
  src="images/pct-differences.png"
  caption="Llama 3 uses fewer tokens in the baseline, so the token use increase is proportionally larger." >}}

## References

[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
