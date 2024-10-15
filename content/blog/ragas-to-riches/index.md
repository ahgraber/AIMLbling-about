---
title: RAGAS to Riches
date: 2024-10-15T08:02:37-04:00
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
series: []
layout: single
toc: true
math: false
draft: true
---

- RAGAS v0.2 [^ragas]
- [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)
  first to call out independent variables for appropriate RAG evals [^method]

  > _Context resources._ Providing different context resources is the key difference between a RAG
  > system and a vanilla LLM as they represent the factual and timely data on which an LLM can reason
  > on instead of only using pretrained weights based on the training data. It is, thus, crucial to select
  > resources that are relevant to the task at hand, which usually requires human judgment. Moreover,
  > as different resources, such as Web searches, database entries, user feedback, and historical data, all
  > can contribute differently to the output accuracy, it is important to evaluate their individual and
  > combined contribution to the independent variables.

  and

  > _Architecture._ RAG is an umbrella of different components, design decisions, and domain-specific
  > adaption. For example, there are hundreds of different embedding models to encode the semantics of
  > textual information for later search [46], there are multiple ways for searching relevant documents
  > in a multitude of vector stores [17], and there are plenty of ways on how to preprocess information
  > (e.g., splitting documents), as well as how to filter and re-rank retrieved documents [12]. All these
  > architectural decisions in a RAG system can influence result accuracy and therefore need to be
  > considered in a sound empirical study. As discussed in Section 2.2, there is a current lack in reporting
  > on design decisions, as well as improvements and refinements on them. Whether a proposed RAG
  > system is a first, naive version or a carefully crafted one based on multiple iterations and feedback
  > runs remains unclear, but is important on estimating the (un)tapped potential of the RAG system.

## References

[^ragas]: [Ragas](https://docs.ragas.io/en/stable/)a
[^method]: [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)
