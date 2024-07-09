---
title: How Robust Are LLMs to Typos? (part 3)
date: 2024-07-03
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

This is part three of a four-part series ([one]({{< ref "/blog/typos-part-1" >}}), [two]({{< ref "/blog/typos-part-2" >}}), [four]({{< ref "/blog/typos-part-4" >}})) where I examine the influence typos have on LLM response quality. Code
from these experiments is available [here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/typos).

In this post, I induce typos in a standardized set of prompts with increasing frequency in the hopes of understanding how typos influence meaning as represented by **sentence embeddings**.

{{< callout type="info">}} An embedding is the vector (i.e., list of numbers) that represents an object (token, word, passage, image, etc.) and differentiates that object from all other objects to the given model. In other words, an
embedding is the model's internal representation of the object -- its _meaning_. Since embeddings are a model's internal representation, they are model-specific.

In this experiment, I'm concerned with sentence-level embeddings, or finding the vector that represents the meaning of a given sentence. {{< /callout >}}

If you recall, my hypothesis is that because typos change the tokens used to represent text, typos shift a sentence away from its "intended" location in embedding space. Thus, as the typo frequency increases, the typo-laden embedding will
grow increasingly dissimilar to the correct embedding.

## Design

I reuse the dataset from [part two's investigation of tokenizers]({{< ref "/blog/typos-part-2" >}}), where the set of 100 _tinyBenchmarks_[^tinybench] MMLU questions act as a baseline and I have generated "typo-ified" variants of each
question with increasing typo occurrences.

As in [part two]({{< ref "/blog/typos-part-2" >}}), I use Llama2 and Llama3 in the hopes of understanding the differences the tokenizer vocabulary makes.

{{< callout type="warning">}} Of course, there's an architectural difference between Llama2-7B-Chat and Llama3-8B-Instruct as well as a dramatic difference in amount of training data seen, so influence of the the tokenizer vocabulary size
is admittedly confounded by these factors. {{< /callout >}}

Since Llama models (really, any GPT-style model) embed at the per-token granularity rather than the per-sentence or per-passage granularity, I have implemented a weighted mean-pooling operation to extract the passage-level representation,
proposed in a [Stack Overflow post](https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource):

> The idea behind weighted-mean_pooling is that the tokens at the end of the sentence should contribute more than the tokens at the beginning of the sentence because their weights are contextualized with the previous tokens, while the
> tokens at the beginning have far less context representation.[^stackoverflow]

Although weighted mean-pooling might grant a passage-grain representation, it does not guarantee a _good_ representation. Llama models are not architected for passage embeddings, nor are they specifically tuned with the task of being good
at semantic search or semantic comparison. Therefore, I also use the [`Sentence-Transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model, which _is_ designed for semantic comparison.

Semantic comparison typically uses cosine similarity as the comparison metric. It measures the angle between two vectors (irrespective of the vectors' magnitudes). This angle can be between -1 to 1, where -1 means the vectors point in
completely opposite directions, 0 means they vectors are orthogonal (unrelated), and 1 means the vectors point in exactly the same direction (i.e., have the same meaning).

{{< figure
  src="images/cosine-similarity.png"
  caption="Simplified Example of Cosine Similarity, in 2-D" >}}

{{< callout type="question" emoji="🧩" >}} A fun way to understand semantic similarity is to play the word game [semantle](https://semantle.com/). The goal is to guess the secret word, and your clues are the cold/hot similarity scores of
your guesses. {{< /callout >}}

For each of these models:

1. Generate passage-level embeddings of the _tinyBenchmarks_ MMLU baseline questions
2. Generate passage-level embeddings of each "typo-ified" question
3. Calculate the cosine similarity between the baseline question and each "typo-ified" variant at each typo rate
4. Take the average of these similarities

## Results

The results confirm the hypothesis that increasing typo occurrence shifts a passage away from its "correct" location in embedding space. As the number of typos increases, the passage gets further and further from its baseline location.

{{< figure
  src="images/sentence-similarity.png"
  caption="Increasing typo occurrence increases distance from \"correct\" text in embedding space" >}}

This effect holds true regardless of the model used to generate embeddings at the passage-level granularity. The sentence-transformer model shows the least reduction in embedding performance (i.e., the distance between "correct" and "typo"
increases the least), which makes sense given its architecture and training objective are optimized to identify the semantic similarity between passages. Llama 3 outperforms Llama 2, but it is difficult to say how much of this difference
can be attributed to Llama 3's much larger tokenizer vocabulary, and how much of it is due to Llama 3's significantly better performance in general.

Finally, it is important to note that the typo incidence must be higher than normal to have a measurable negative effect on semantic similarity tasks. I would estimate it to be unusual to have more than 15% of the words in a given passage
have typos. The experimental results at 15% typo occurrence demonstrate a cosine similarity of 0.9 or better -- not identical, but still quite close in embedding space!

## Bonus: Can LLMs Heal Typos?

A question I had going into this experiment was whether it made sense to have a spelling (and or grammar) check between the user and the model to ensure the model receives the cleanest input possible. What if the language model itself can
fix the typos?

### Design

I instruct the language model to correct the typos in the user input:

```txt
Instructions:

Correct the typos in the following text.
Respond ONLY with the corrected text.

Text:

{{question}}

Correction:
```

and compare the results to the baseline questions. For the comparison, I use a multiset variant of the Jaccard index[^jaccard] as a proxy for accuracy (i.e., Does the healed text contain the same words as the baseline?), and semantic
similarity based on the top-performing Sentence-Transformers model from the experiment above.

### Results

<table>
<tr>
  <td style="width:50%">{{< figure
    src="images/healing-accuracy.png"
    caption="Accuracy indicates LLMs can successfully recover the majority of the original input.">}}
  </td>
    <td style="width:50%">{{< figure
    src="images/healing-similarity.png"
    caption="Cosine Similarity indicates LLMs successfully recover the meaning of text with typos when fixing them.">}}
  </td>
</tr>
</table>

The Jaccard pseudo-accuracy shows that the language model consistently recovers the original text quite well, though the exact recovery drops off a bit as typo rate increases. Evaluating cosine similarity indicates that even though the
exact original text is not recovered, _the gist of it is_!

## References

[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
[^stackoverflow]: [artificial intelligence - Sentence embeddings from LLAMA 2 Huggingface opensource - Stack Overflow](https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource)
[^jaccard]: [[2110.09619] Further Generalizations of the Jaccard Index](https://arxiv.org/abs/2110.09619)
