---
title: RAGAS to Riches (part 1)
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
series:
  - "ragas"
layout: single
toc: true
math: false
draft: true
---

## What is RAG?

Retrieval Augmented Generation (RAG) is a critical part of the modern AI ecosystem.
The idea is that, like a student, a model will provide better answers if given an open-book exam, instead of relying on "memorized" knowledge.
Therefore, we provide the language model with additional, relevant information that is pertinent to the instruction, task, or question so that it can be used when generating a response.
The additional information acts to "ground" the response, and tends to reduce hallucinations and mitigate incorrect or inconsistent answers.

RAG was originally proposed as a method for domain adaptation or knowledge updates, in which the language model is fine tuned to work with an external knowledgebase and retriever. [^rag]

The requirement of fine-tuning multiple components makes the original approach less flexible and more expensive, and the industry has generally aligned on a simpler(?) alternative that does not require fine tuning.
RAG requires a knowledge source, a method for searching/ranking/retrieving the relevant content, and the LLM to synthesize the final answer.

{{< figure
  src="images/RAG.png"
  caption="Retrieval Augmented Generation via [Best Practices in Retrieval Augmented Generation - Gradient Flow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/)" >}}

The above diagram from [GradientFlow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/) provides a fantastic overview of the process.
Pieces of content are split into chunks, converted into vector embeddings, and stored in a vector database.
When a user request comes in, we convert that request into the same vector embedding space and identify the most relevant document chunks for that user query.
Then we provide user query _and_ the most relevant chunks for the LLM to generate the response.

## RAG Optimization

## RAG Evaluation

So, given all of the possible options, how can we determine which provides the best responses?

- retrieval evals vs response evals
  - retrieval is just a recommendation engine; reuse recommendation engine metrics?
- RAGAS [^ragas_arxiv]
- benchmarks

  - dataset source?
  - process used to create synthetic dataset should be independent from process used to process knowledgebase and generate responses!
    - - [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)
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

- RAGAS v0.2 [^ragas]

- Architecture & best practices [^best-practices] [^anyscale]

## References

[^ragas]: [Ragas](https://docs.ragas.io/en/stable/)
[^method]: [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)
[^ragas_arxiv]: [[2309.15217] RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
[^rag]: [[2005.11401] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
[^best-practices]: [Best Practices in Retrieval Augmented Generation - Gradient Flow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/)
[^anyscale]: [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
