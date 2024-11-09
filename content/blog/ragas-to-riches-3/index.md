---
title: RAGAS to Riches (part 3)
date: 2024-11-01T08:02:37-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - "blogumentation"
  - "experiment"
  # ai/ml
  - "evals"
  - "LLMs"
  - "RAG"
  # homelab
  - "homelab"
series:
  - "ragas"
layout: single
toc: true
math: false
draft: true
---

This is part three of a three-part series ([one]({{< ref "/blog/ragas-to-riches-1" >}}), [three]({{< ref "/blog/ragas-to-riches-3" >}})) where I explore best practices for evaluating RAG architecture via Ragas' recent v0.2 release.
Code from these experiments is available [here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/ragas).

## Experiment

Do models prefer their own work?
(a) Do models perform better on their own questions?
(b) Do models prefer their own answers?
~~(c) Do models prefer their own answers to their own questions?~~

## Analysis

### Baseline

### Retrieval

### RAG

## Lessons learned

- OpenAI, VoyageAI, and Nomic embeddings all perform similarly
- Token use is surprisingly high

  - Costs are high for an individual
  - Token use requirements prevented using SOTA models, especially with Anthropic's tier constraints (i.e., to actually run a full eval, Tier II does not provide sufficient daily tokens on Sonnet; have to use Haiku)
  - Parse failures end up multiplying costs, with a parse error triggering several retries with "repair" prompts
    - Instructions seem tuned to OpenAI models, _which had the lowest failure rate across all phases (**TODO**) ??_
    - Llama-3.1-70b-instruct does not respond to prompt formatting instructions as well as other models, leading to higher parse failure rates across all phases
    - Anthropic's Claude 3 Haiku seemed to do fairly well initially but had high errors during response evaluation

- Running local models may require increasing timeouts

### Costs

- Creating the knowledge graph and synthetic test set are one-time costs that are easily updated with minimal incremental cost.
- Running the benchmark is expensive, and many runs may be required to represent the full experimental search space.

All told, running this experiment (including debugging, user errors like failure-to-save, etc.) had the following API use and costs:

|  Provider |            Model             | Tokens |   Cost |
| --------: | :--------------------------: | -----: | -----: |
|    OpenAI |         GPT-4o-mini          |        | $50.00 |
| Anthropic |        Claude-3-Haiku        |        | $50.00 |
|  Together | Llama-3.1-70B-Instruct-Turbo |        | $50.00 |

> Note: I am not including embedding calls or costs here because they tend to be minimal

## References

LLM as a judge references?
