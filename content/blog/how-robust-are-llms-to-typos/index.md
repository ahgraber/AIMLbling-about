---
title: How Robust Are Llms to Typos
date: 2024-05-07T19:13:24-04:00
tags:
  # ai/ml
  - 'agents'
  - 'arxiv'
  - 'generative AI'
  - 'prompts'
  - 'LLMs'
series: []
draft: true
---

{{< figure
  src="images/fig_1%20-%20prompt_perturbation.png"
  caption="Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., ... & Xie, X. (2023). Promptbench: Towards evaluating the robustness of large language models on adversarial prompts. arXiv preprint arXiv:2306.04528." >}}

- Effect on tokenization? (Hypothesis - more tokens used unless common misspellings are included in vocab)
  - Need to try with llama tokenizer (32k) and one that uses tiktoken (100k)
- Effect on prediction (How to judge? perplexity?)
  - Need to try with similar class models that use llama and tiktoken tokenizers

How to create dataset?

 1. Take sample of WildChat
 2. Correct all spellings
 3. Capture metrics for "correct" case
 4. Perturb correctness
    1. Common misspellings
    2. Nearby keys
    3. Missing letters
 5. Test with sets of increasing typo proportion

What about other languages?
What about quantized models?

## References

- [How to Achieve Robustness to spelling mistakes in Language Models? [D] : r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/1arj0a1/how_to_achieve_robustness_to_spelling_mistakes_in/)
- [[2306.04528] PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)
- [[2311.00258] Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis](https://arxiv.org/abs/2311.00258)
- [[2404.09754] Resilience of Large Language Models for Noisy Instructions](https://arxiv.org/abs/2404.09754)
- [[2402.15833] Prompt Perturbation Consistency Learning for Robust Language Models](https://arxiv.org/abs/2402.15833)
