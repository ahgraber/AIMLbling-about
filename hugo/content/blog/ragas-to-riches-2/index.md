---
title: RAGAS to Riches (part 2)
date: 2024-11-02
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

This is part two of a three-part series ([one]({{< ref "/blog/ragas-to-riches-1" >}}), [three]({{< ref "/blog/ragas-to-riches-3" >}})) where I explore best practices for evaluating RAG architecture via Ragas' recent v0.2 release.
Code from these experiments is available [here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/ragas-experiment).

In this post, I will dive into why I'm so excited about Ragas v0.2 and dive into how it works.
Specifically, the I am referencing `Ragas v0.2.3`; the team is rapidly iterating and this series of posts will likely end up out of date quite quickly.
It is my hope that the concepts will remain true, even if the specific procedures, prompts, or token use expectations change.

## Introducing Ragas v0.2

Ragas (RAG ASsessment) is a research-supported [^ragas_arxiv] python framework for evaluating RAG applications [^ragas]
The new version 0.2 release provides a synthetic testset generation process that builds a knowledge graph over the document corpus to synthesize questions, among other improvements. [^ragas_v0.2]
When combined with new, long-context embedding models, this means that the decisions made for extracting information into the knowledge graph _are different from the decisions made when chunking documents for RAG_.
For instance, it may make sense to split documents into somewhat larger nodes based on document structure and leverage keyphrase and entity extraction for to build topical clusters for the testset generation.
Conversely, recent research indicates that smaller, semantically-defined chunks are more optimal for retrieval. [^chunking]

### [Knowledge Graph Creation](https://docs.ragas.io/en/v0.2.3/concepts/test_data_generation/rag/#knowledge-graph-creation)

Given a corpus of documents, Ragas defines sequences of transformations, splits, and extractions that create a knowledge graph.
The stages in the sequence may involve simple functions, text-to-vector embeddings, or calls to LLM APIs.

The default set of transformations first operates at the document level, splits the documents, and then operates at the node level.
First, the pipeline extracts summary and headline information, then embeds the document summary as a vector representation of the document.
The document is split by headlines (headings), and then the resulting nodes are vectorized and assigned keyphrases and titles.
Finally a similarity matrix is defined to build relationships between nodes (regardless of source document), which allows the creation of topical clusters.

```py
default_transforms = [
    Parallel(summary_extractor, headline_extractor),
    summary_embedder,
    headline_splitter,
    Parallel(embedding_extractor, keyphrase_extractor, title_extractor),
    cosine_sim_builder,
    summary_cosine_sim_builder,
]
```

I modified the default pipeline, breaking it into 3 phases for a bit more control for error handling and repair,
and allowing me to save the interim transformations.
I also wrote custom `MarkdownTitleExtractor` and `SpacyNERExtractor` classes that are customized to my use case,
and reduce the overall number of LLM API calls I make (thus reducing my costs).
See [1-build_knowledgegraph.py](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/1-build_knowledgegraph.py) for details.

```py
stage1 = [
    Parallel(
        summary_extractor,
        headline_extractor,
    ),
    summary_embedder,
]
apply_transforms(kg, stage1)
...
stage2 = [
    headline_splitter,
    Parallel(
        summary_extractor,
        embedding_extractor,
        keyphrase_extractor,
        spacy_ner_extractor,
        title_extractor,
    ),
]
apply_transforms(kg, stage2)
...
stage3 = [
    cosine_sim_builder,
    summary_cosine_sim_builder,
]
apply_transforms(kg, stage3)
```

### [Testset Generation](https://docs.ragas.io/en/v0.2.3/concepts/test_data_generation/rag/)

As mentioned above, the knowledgegraph contains information about how nodes are related on a topical level, based on summary embeddings, keyphrases, or named entities.
Ragas builds clusters of nodes given these relationships, and samples from these clusters to synthesize ground-truth responses and generate a question for evaluation.
Then, it may modify the question based on scenarios that mimic real-world use -- some questions may be revised to include typos (see my [prior work on typos]({{< ref "/blog/typos-part-1" >}})), to be more succinct or more rambling,
or to take on different user personas.
The final result is a testset where each record contains a question (`user_input`), the synthetic ground truth / expected response (`reference`),
the text from the knowledge graph nodes used to generate the ground truth (`reference_contexts`), and the synthesizer name.

Ragas conceptualizes user interactions with RAG systems into 2 dimensions and builds synthesizers for each.

One dimension is how direct or abstract the question or request is.
The [Ragas documentation describes](https://docs.ragas.io/en/v0.2.3/concepts/test_data_generation/rag/#specific-vs-abstract-queries-in-a-rag):

> **Specific Query**: Focuses on clear, fact-based retrieval. The goal in RAG is to retrieve highly relevant information from one or more documents that directly address the specific question.  
> **Abstract Query**: Requires a broader, more interpretive response. In RAG, abstract queries challenge the retrieval system to pull from documents that contain higher-level reasoning, explanations, or opinions, rather than simple facts.

The other dimension is single- vs multi-hop:
Single-hop interactions can be thought of as straightforward question where retrieved context is synthesized directly into a single response.
On the other hand, multi-hop interactions may require reasoning after extracting information the retrieved context before responding.

Finally, there is the distinction between single- and multi-_turn_ questions.
A single-turn question is when an interaction is simply the user asks a question and the LLM response.
A multi-turn question wouldrequire answering a question at the tail end of a conversation that requires full conversational context.
Currently (as of v0.2.3), Ragas does not support generation of synthetic multi-turn questions, though that feature is currently being explored.

[Generating the testset is quite simple once the knowledge graph is created](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/2-generate_testset.py):

```py
  kg = KnowledgeGraph().load("ragas_knowledgegraph.json")
  generator = TestsetGenerator(
      llm=llm,
      knowledge_graph=kg,
  )

  # define synthesizers
  abstract_query_synth = AbstractQuerySynthesizer(llm=llm)
  comparative_query_synth = ComparativeAbstractQuerySynthesizer(llm=llm)
  specific_query_synth = SpecificQuerySynthesizer(llm=llm)

  logger.info(f"Generating testset with {provider}...")

  # NOTE: we don't use 'generate_with...docs()' because we already use our pre-created knowledge graph
  dataset = generator.generate(
      testset_size=100,
      query_distribution=[
          (abstract_query_synth, 0.25),
          (comparative_query_synth, 0.25),
          (specific_query_synth, 0.5),
      ],
      # callbacks=[cost_callback],
      # run_config=...,
  )

  dataset.to_jsonl(f"ragas_testset.jsonl")
```

### Evaluation

I used [LlamaIndex to create a local vector database](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/3-build_index_llamaindex.py), and then iterated over the testset,
asking my LlamaIndex RAG system to [retrieve contexts](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/4-rag_reretrievals.py) and
[generate responses](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/4-rag_responses.py) for each question in the testset.
I also [created a baseline set of responses](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/4-baseline_responses.p),
using the LLM to generate responses to the testset questions without providing the retrieved context -- a closed-book exam, metaphorically speaking.

I elected to use `context precision`, `context recall`, `faithfulness`, and `response relevance` as my evaluation metrics, which we covered in at a high level in [part one]({{< ref "/blog/ragas-to-riches-1#rag-evaluation" >}}).

Evaluation is as simple as calling Ragas' `evaluate` over the testset and specifying which metrics to evaluate.

```py
metrics = [
    LLMContextPrecisionWithReference(llm=llm),
    LLMContextRecall(llm=llm),
    Faithfulness(llm=llm),
    ResponseRelevancy(embeddings=embeddings, llm=llm),
]
testset = EvaluationDataset.from_list(testset_df.to_dict(orient="records"))
results = evaluate(dataset=testset, metrics=metrics)
```

### Other updates

One of the other really cool features included in the v0.2 update is [cost estimation](https://docs.ragas.io/en/v0.2.3/howtos/applications/_cost/).
This feature uses callbacks to extract token use the LLM response object
([OpenAI](https://platform.openai.com/docs/api-reference/chat/create) and [Anthropic](https://docs.anthropic.com/en/api/messages))
include token use in the `usage` key in their responses.

Unfortunately, using Ragas' cost estimation functionality requires actually calling the LLM APIs; it cannot preemptively estimate token use or costs.

## Estimating Token Utilization

In order to preemptively estimate token use (and therefore cost), I extracted the prompts used for every LLM call in the knowledgegraph creation, testset generation, and evaluation process.
Then, given some assumptions, I estimated the _input_ token use over these use cases.
Code is available [in this Jupyter notebook](https://github.com/ahgraber/AIMLbling-about/blob/main/experiments/ragas-experiment/cost_analysis.ipynb).

{{< callout type="warning" >}} _NOTE:_ Token utilization estimates only analyze _input_ tokens using the `tiktoken` tokenizer;
actual use (even if these estimates are perfect) will be higher because the analysis does not take _output_ tokens into account. {{< /callout >}}

### Adding Single Doc to Knowledge Graph

Assuming document is 5,000 tokens and the resulting knowledge graph nodes are ~256 tokens:

- document extraction will use 10,655 tokens
- chunk/node extraction will use 40,679 tokens

for an estimated total use of 51,334 tokens per document

### Generating Single Testset Question

Assuming kg nodes are ~256 tokens, and each cluster used for generation contains ~8 nodes:

- abstract query scenario questions will use 7,979 tokens
- comparative abstract query scenario questions will use 7,799 tokens
- specific query scenario questions will use 5,342 tokens

### Evaluating RAG on Single Testset Question

Assuming the test query is 64 tokens, the retrieved context is 256 \* 3 tokens, ground truth is ~100 tokens and generated response is ~256 tokens:

- evaluating `context precision will` use 1,881 tokens
- evaluating `context recall` will use 1,788 tokens
- evaluating `faithfulness` will use 3,555 tokens
- evaluating `response relevance` will use 815 tokens

### Overall RAGAS token use for first run

| phase                       | n docs | testset size |    tokens |
| --------------------------- | -----: | -----------: | --------: |
| knowledge graph             |    100 |            - | 5,133,400 |
| testset\*                   |      - |          100 |   661,550 |
| base case (no RAG) eval\*\* |      - |          100 |    81,500 |
| RAG eval\*\*\*              |      - |          100 |   803,900 |

> \* assumes test set uses 25% abstract, 25% comparative abstract, and 50% specific query scenarios  
> \*\* base case (no retrieval) eval only analyzes `response relevance`  
> \*\*\* RAG eval analyzes `context precision`, `context recall`, `faithfulness`, and `response relevance`

Given current (Q4 2024) prices (GPT-4o @ $2.50 / 1M tokens; Claude-3.5-Sonnet @ $3.00 / 1M tokens):

- OpenAI GPT-4o would cost ~$16.80
- Anthropic Claude-3.5-Sonnet would cost ~$20.03

> _REMINDER_ costs do not include output tokens (or retries), and therefore actual costs will be higher.
> In my experience, output token use is ~5% of input token use

## How do assumed parameters affect token use (and therefore cost?)

RAGAS uses few-shot prompting, which means the templated prompts with few-shot examples will tend to dominate the total token use if retrieved context tokens < prompt tokens.
Therefore, parameters like node/chunk size and numbers of nodes/chunks provided functionally have no impact on long-running token use/cost assuming that the retrieved context tokens remains < prompt tokens.

{{< figure
  src="images/testset_chunk_tokens.png"
  alt="prompt length dominates influence of chunk size on token utilization"
  caption="Prompt length dominates influence of chunk size on token utilization" >}}
{{< figure
  src="images/eval_n_context_chunks.png"
  alt="prompt length dominates influence of chunk count on token utilization"
  caption="Prompt length dominates influence of chunk count on token utilization" >}}

> 'scaled' divides the total token use by the x axis variable
> to understand whether the increase is proportional to parameter

Parameters that alter the number of overall iterations (i.e., increasing the number of extractor steps in the knowledge graph,
adding a new document to the knowledge graph, adding questions to the test set, or adding evals) act as multipliers, and have dramatic effects on total token use, though the token use per iteration remains constant.

{{< figure
  src="images/testset_n_questions.png"
  alt="tokens per question remain constant"
  caption="Testset size multiplies the total token use, but tokens per question remain constant" >}}
{{< figure
  src="images/eval_n_questions.png"
  alt="tokens per question remain constant"
  caption="Testset size multiplies the total token use, but tokens per question remain constant" >}}

> Because the templated prompts dominate token use, leveraging API features such as token caching become an incredibly effective way to save money on API use.
> API calls that use the same prompt/template should be made in near succession to maintain high cache-hit rates.

## References

[^ragas_arxiv]: [[2309.15217] RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

[^ragas]: [explodinggradients/ragas: Supercharge Your LLM Application Evaluations 🚀](https://github.com/explodinggradients/ragas/tree/main)

[^ragas_v0.2]: [Announcing Ragas v0.2](https://blog.ragas.io/announcing-ragas-02)

[^chunking]: [Evaluating Chunking Strategies for Retrieval | Chroma Research](https://research.trychroma.com/evaluating-chunking)
