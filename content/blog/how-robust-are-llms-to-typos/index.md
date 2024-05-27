---
title: How Robust Are LLMs to Typos
date: 2024-05-20
tags:
  # meta
  - "experiment"
  # ai/ml
  - "generative AI"
  - "prompts"
  - "LLMs"
series: []
draft: true
math: true
---

<!-- markdownlint-disable MD013 -->

{{< figure
  src="images/promptbench%20-%20fig_1%20-%20prompt_perturbation.png"
  caption="Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L., Ye, W., Gong, N.Z., Zhang, Y., & Xie, X. (2023). PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. ArXiv, abs/2306.04528." >}}

<!-- markdownlint-enable -->

- Effect on tokenization? (Hypothesis - more tokens used unless common misspellings are included in vocab)
  - Need to try with llama tokenizer (32k) and one that uses tiktoken (100k) or google gemma (256k)
- Effect on prediction (How to judge? perplexity?)
  - Need to try with similar class models that use llama and tiktoken tokenizers

**Caveat:** this analysis holds true for English. I assume that the impacts of typos in other languages would increase
depending on how well-represented the language is in the tokenizer and LLM training sets.

## Typos

- `birkbeck` contains 36,133 misspellings of 6,136 words. Handwritten.[^corpora]
- `holbrook` contains 1791 misspellings of 1200 words. Handwritten.[^corpora]
- `aspell` contains 531 misspellings of 450 words for GNU Aspell. Typed.[^corpora]
- `wikipedia` contains 2,455 misspellings of 1,922 words. Typed.[^corpora]
- `microsoft` contains before-and-after spelling-correction pairs derived automatically by processing keystroke logs
  collected through Amazon's Mechanical Turk. Typed.[^microsoft]
- `typokit` contains common typos and spelling mistakes and their fixes. Typed?[^typokit]
- ~~`github` is a large-scale dataset of misspellings and grammatical errors along with their corrections harvested
  from GitHub. It contains more than 350k edits and 65M characters. Typed.[^github]~~

The aggregate collection contains 71,785 misspellings of 16,887 words, providing 136,705 edits for statistical
summarization.

Could be either "intentional" (i.e., a misspelling typed as intended, given a mistaken belief the word was spelled that
way), or "unintentional" (i.e., fumble-fingered error while typing). In order to define the mapping from the erroneous
to the correct string, I use the three types of Levenshtein string edits: insertion, substitution, deletion.

{{< callout type="warning >}} Some of these data sets provide frequency information (Holbrook, Microsoft), while others
are simply a list of common errors. This means the aggregate collection is **not** a representative sample of error
frequencies. However, for the purpose of this analysis, _I will be acting as though it were_, and inferring
probabilities of occurrence from these data. {{< /callout >}}

{{< figure
  src="images/denoise%20-%20fig_1%20-%20char_confusion_matrix.png"
  caption="Kuznetsov, A., & Urdiales, H. (2021). Spelling Correction with Denoising Transformer. ArXiv, abs/2105.05977." >}}

{{< figure
  src="images/denoise%20-%20fig_2%20-%20normalize_typo_position.png"
  caption="Kuznetsov, A., & Urdiales, H. (2021). Spelling Correction with Denoising Transformer. ArXiv, abs/2105.05977." >}}

"typos tend to happen closer to the end of the string" [^denoise]

## Tokenizer experiments

## Embedding model experiments

Can't use perplexity for embedding models; use accuracy on MSMARCO(?) tests and/or distance metrics?

## LLM experiments

### Baseline setup

Using perplexity - "the lower the perplexity of a prompt is, the better its performance on the task will be. This is
based on the intuition that the more frequently the prompt (or very similar phrases) appears in the training data, the
more the model is familiar with it and is able to perform the described task."[^promptperplexity]

{{< callout type="info" >}} `Perplexity` is a representation of how _unlikely_ a given sequence is, given all sequences
seen during training. It can be used to understand how confident the model is in predicting the next token in a
sequence. A lower perplexity indicates that the prediction is more likely (i.e., the model is more confident).

According to [Huggingface](https://huggingface.co/docs/transformers/en/perplexity), "perplexity is defined as the
exponentiated average negative log-likelihood of a sequence."

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

### Frequent typos

Using corpora of common misspellings[^corpora], replace increasing proportions of words with errors

### Mechanistic typos

Use function to probabilistically induce error using typo "rules"

## References

[^denoise]: [[2105.05977] Spelling Correction with Denoising Transformer](https://arxiv.org/pdf/2105.05977)
[^corpora]: [Corpora of misspellings for download](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)
[^github]:
    [GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](https://github.com/mhagiwara/github-typo-corpus?tab=readme-ov-file)

[^microsoft]:
    [Microsoft Research Spelling-Correction Data](https://www.microsoft.com/en-us/download/details.aspx?id=52418)

[^typokit]: [Collection of common typos & spelling mistakes and their fixes](https://github.com/feramhq/typokit)
[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)

[^promptbench] [^promptbench]
[[2306.04528] PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)
[^noisy] [^noisy]:
[[2311.00258] Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis](https://arxiv.org/abs/2311.00258)
[^resilience] [^resilience]:
[[2404.09754] Resilience of Large Language Models for Noisy Instructions](https://arxiv.org/abs/2404.09754)
[^perturbation] [^perturbation]:
[[2402.15833] Prompt Perturbation Consistency Learning for Robust Language Models](https://arxiv.org/abs/2402.15833)
[^promptperplexity] [^promptperplexity]:
[[2212.04037] Demystifying Prompts in Language Models via Perplexity Estimation](https://arxiv.org/abs/2212.04037)
[^perplexity] [^perplexity]:
[Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)
