---
title: How Robust Are LLMs to Typos? (part 2)
date: 2024-07-01
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

In this post, I'll use the typo generation function to induce typos with increasing frequency in the hopes of
understanding how typos influence **tokenization**.

If you recall, my hypothesis regarding tokenization is that increasing typo frequency increase token counts -- as the
typo frequency rises, the tokenization will fit the data less well, requiring additional, smaller tokens to represent
the data... Unless the typos are so popular that they have made it into the vocabulary.

## Design

## References

<!-- [^promptbench]: [[2306.04528] PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)
[^noisy]: [[2311.00258] Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis](https://arxiv.org/abs/2311.00258)
[^resilience]: [[2404.09754] Resilience of Large Language Models for Noisy Instructions](https://arxiv.org/abs/2404.09754)
[^corpora]: [Corpora of misspellings for download](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)
[^microsoft]: [Microsoft Research Spelling-Correction Data](https://www.microsoft.com/en-us/download/details.aspx?id=52418)
[^typokit]: [Collection of common typos & spelling mistakes and their fixes](https://github.com/feramhq/typokit)
[^github]: [GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](https://github.com/mhagiwara/github-typo-corpus?tab=readme-ov-file)
[^commit]: [src-d/datasets | Typos](https://github.com/src-d/datasets/blob/master/Typos/README.md)
[^denoise]: [[2105.05977] Spelling Correction with Denoising Transformer](https://arxiv.org/pdf/2105.05977) -->

[^perturbation] [^perturbation]:
[[2402.15833] Prompt Perturbation Consistency Learning for Robust Language Models](https://arxiv.org/abs/2402.15833)

<!-- [^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
[^promptperplexity]: [[2212.04037] Demystifying Prompts in Language Models via Perplexity Estimation](https://arxiv.org/abs/2212.04037)
[^perplexity]: [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity) -->
