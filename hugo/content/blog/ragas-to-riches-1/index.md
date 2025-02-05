---
title: RAGAS to Riches (part 1)
date: 2024-10-19
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
draft: false
---

This is part one of a three-part series ([two]({{< ref "/blog/ragas-to-riches-2" >}}), [three]({{< ref "/blog/ragas-to-riches-3" >}})) where I explore best practices for evaluating RAG architecture via Ragas' recent v0.2 release.
Code from these experiments is available [here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/ragas-experiment).

This post covers the preliminary / background material.
In later posts, I'll cover what makes Ragas v0.2 so special, how it works, and run an experiment with it.

## What is RAG?

Retrieval Augmented Generation (RAG) is a critical part of the modern AI ecosystem.
The idea is that, like a student, a model will provide better answers when given an open-book exam as opposed to when it relies on "memorized" knowledge.
Therefore, RAG is a technique that provides the language model with additional, _relevant_ information that is pertinent to the instruction, question, or request so that it can be used when generating a response.
The additional information acts to "ground" the response, and tends to reduce hallucinations and mitigate incorrect or inconsistent answers.

RAG was originally proposed as a method for domain adaptation or knowledge updates, in which the language model is fine tuned to work with an external knowledgebase and retriever. [^rag]
The modern RAG implementation uses a general embedding model to locate reference information and user inputs in the same semantic embedding space for search.
Relevant pieces of information are retrieved and are passed to the LLM for in-context learning.
RAG systems require a corpus of knowledge to search over, a method for searching/ranking/retrieving the relevant content, and the LLM to synthesize the final answer.

{{< figure
  src="images/RAG.png"
  caption="Retrieval Augmented Generation via [Best Practices in Retrieval Augmented Generation - Gradient Flow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/)" >}}

The above diagram from [GradientFlow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/) provides a fantastic overview of the RAG procedure.
Pieces of content are split into chunks, converted into vector embeddings, and stored in a vector database.
When a user request comes in, the system locates that request into the same vector embedding space as the corpus of information and identifies the most relevant document chunks with respect to that input.
Then we provide user query _and_ the most relevant chunks for the LLM to generate the response.

{{< callout type="info" >}} The description of RAG procedure above assumes the use of an embedding model and vector database.
While this is the "typical" implementation, it is not the only one.
BM25 is a type of keyword search that does not require an embedding model or vectorization but provides fast, well-studied text search, and often serves as a "baseline".
Other methods for text comparison, such as TF-IDF, are also theoretically plausible as search methods, but tend to underperform BM25 and vector search. {{< /callout >}}

## RAG Optimization

A fundamental challenge in configuring a RAG architecture is the process of optimizing performance.
In the RAG diagram above, every node is a point where an engineering decision must be made.
Because nodes are not independent, decisions made about chunking strategies influence embedding approaches, etc.
This means the search space for overall system architecture optimization requires factorial experimental design to account for the interactions between nodes.
A recent whitepaper, _A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation_ describes it:

> _Architecture._ RAG is an umbrella of different components, design decisions, and domain-specific
> adaption. For example, there are hundreds of different embedding models to encode the semantics of
> textual information for later search, there are multiple ways for searching relevant documents
> in a multitude of vector stores, and there are plenty of ways on how to preprocess information
> (e.g., splitting documents), as well as how to filter and re-rank retrieved documents. All these
> architectural decisions in a RAG system can influence result accuracy and therefore need to be
> considered in a sound empirical study. As discussed in Section 2.2, there is a current lack in reporting
> on design decisions, as well as improvements and refinements on them. Whether a proposed RAG
> system is a first, naive version or a carefully crafted one based on multiple iterations and feedback
> runs remains unclear, but is important on estimating the (un)tapped potential of the RAG system.[^methodology]

How do we know which experimental arm is best?

## RAG Evaluation

RAG systems require at least two metrics for evaluating their performance - a way to measure how good the retrieval is, and a way to measure how good the final answer (given the retrieval) is.

As retrieval is essentially a recommendation system, AI/ML Engineers have repurposed some of the metrics used in recommendation evals, such as context precision and context recall, for retrieval evaluation.
`context precision` measures the proportion of information retrieved by the RAG system that is relevant to the ground truth / known good answer. [^precision]
`context recall` calculates the proportion of the claims made in the ground truth answer that are supported by retrieved information. [^recall] [^precision-recall]

{{< callout type="info" emoji="💡" >}} I find [Wikipedia's image-based explanation of precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) super helpful when thinking these definitions through {{< /callout >}}

For response evaluation, faithfulness and response relevance are frequently used.
`faithfulness` measures how consistent the final answer is with respect to the retrieved information.[^faithfulness]
RAG systems typically reduce hallucination as a result of the 'open book exam' nature of the response generation and therefore have correspondingly high faithfulness.
`response relevance` (a.k.a. `answer relevance`) measures how relevant the final answer is to the original input - it would not do for the RAG system to provide a faithful answer to a different question! [^response_relevance]

Critically, we need these metrics to be valid data points for comparison (over time, or between RAG architectures).
This means that (a) we need a baseline to compare against - likely LLM performance without RAG - and (b) we need a benchmark -
"a standardized basis for comparing different approaches," a dataset with known-good question-context-answer triples and a set of metrics used to evaluate performance.[^methodology]

{{< figure
  src="images/RAG experimentation.png"
  caption="Key considerations for sound empirical evaluations of RAG systems via [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)" >}}

Finally, in the case where the benchmark is artificially generated, it is important that the process of defining the benchmark dataset is independent of the decisions and architectures upon which the RAG system we are evaluating are based.

## Introducing Ragas v0.2

It is this last point that has me very excited about the possibilities of the new version of Ragas.
Ragas (RAG ASsessment) is a research-supported [^ragas_arxiv] python framework for evaluating RAG applications [^ragas]
The new version 0.2 release provides a synthetic testset generation process that builds a knowledge graph over the document corpus to synthesize questions.
When combined with new, long-context embedding models, this means that the decisions made for extracting information into the knowledge graph _are different from the decisions made when chunking documents for RAG_.
For instance, it may make sense to split documents into somewhat larger nodes based on document structure and leverage keyphrase and entity extraction for to build topical clusters for the testset generation.
Conversely, recent research indicates that smaller, semantically-defined chunks are more optimal for retrieval. [^chunking]

## References

[^rag]: [[2005.11401] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

[^methodology]: [[2410.08801] A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation](https://arxiv.org/abs/2410.08801)

[^precision-recall]: [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

[^precision]: [context precision](https://docs.ragas.io/en/v0.2.3/concepts/metrics/available_metrics/context_precision/)

[^recall]: [context recall](https://docs.ragas.io/en/v0.2.3/concepts/metrics/available_metrics/context_recall/)

[^faithfulness]: [faithfulness](https://docs.ragas.io/en/v0.2.3/concepts/metrics/available_metrics/faithfulness/)

[^response_relevance]: [response relevancy](https://docs.ragas.io/en/v0.2.3/concepts/metrics/available_metrics/answer_relevance/)

[^ragas]: [explodinggradients/ragas: Supercharge Your LLM Application Evaluations 🚀](https://github.com/explodinggradients/ragas/tree/main)

[^ragas_arxiv]: [[2309.15217] RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

[^chunking]: [Evaluating Chunking Strategies for Retrieval | Chroma Research](https://research.trychroma.com/evaluating-chunking)
