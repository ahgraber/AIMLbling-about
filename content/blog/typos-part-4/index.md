---
title: How Robust Are LLMs to Typos? (part 4)
date: 2024-07-10
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

This is part four of a four-part series (one, two, three) where I examine the influence typos have on LLM response
quality.

In this post, I'll use the typo generation function to induce typos with increasing frequency in the hopes of
understanding how typos influence **generation**.

If you recall, my hypothesis is that typos increase error rates -- as typos alter the tokenization and embedding
pipeline, the language model experiences distribution shift and cannot predict as well for the error-laden inputs.
Therefore, Text with typos will have higher perplexity than correct text, and questions with typos will have lower
response accuracy than correct questions.

## Design

Using perplexity - "the lower the perplexity of a prompt is, the better its performance on the task will be. This is
based on the intuition that the more frequently the prompt (or very similar phrases) appears in the training data, the
more the model is familiar with it and is able to perform the described task."[^promptperplexity]

{{< callout type="info" >}} `Perplexity` is a representation of how _unlikely_ a given sequence is, given all sequences
seen during training. It can be used to understand how confident the model is in predicting the next token in a
sequence. A lower perplexity indicates that the prediction is more likely (i.e., the model is more confident).

According to [Huggingface](https://huggingface.co/docs/transformers/en/perplexity), "perplexity is defined as the
exponentiated average negative log-likelihood of a sequence."[^perplexity]

$$
\text{given sequence } X = (x_0, x_1, \dots, x_n) \\\\
\text{perplexity}(X) = \exp \left\( {-\frac{1}{n}\sum_i^n \log P(x_n|x_{<n}) } \right\)
$$

In plainer language, to calculate perplexity:

1. Calculate the probability of each token in a sequence (given the preceding tokens)
2. Normalize the probability across different sequence lengths by taking the geometric mean of the probabilities
3. Take the reciprocal of the normalized probabilities

{{< /callout >}}

Using MMLU or tinyBenchmarks [^tinybench]:

1. Run evals without typos; analyze perplexity of eval prompts & responses
2. Perturb with varying levels of typos and run evals; analyze perplexity of eval prompts & responses
   1. Find/replace with typos corpus
   2. Find/replace with mechanistic rules

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

[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
[^promptperplexity]:
    [[2212.04037] Demystifying Prompts in Language Models via Perplexity Estimation](https://arxiv.org/abs/2212.04037)

[^perplexity]: [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)
