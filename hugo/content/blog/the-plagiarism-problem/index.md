---
title: The plagiarism problem
date: 2024-08-10
tags:
  - "opinion"
  - "generative AI"
  - "LLMs"
  - "copyright"
  - "creators"
  - "content"
  - "licensing"
series: []
draft: false
---

The primary training task for generative AI models is inherently plagiaristic.
New AI-based answer engines leverage these plagiaristic models to provide in-engine responses containing the specific information or content the user is looking for,
obviating the need for the user to click through to the (original) content and depriving the creator of traffic.
This breaks the content creation/consumption paradigm, where creators make money based on the number of views the ads on their pages receive, ads are sold by the search engine, and search results are driven by pagerank and SEO.
To resolve this, pay-per-reference source attribution will have to replace per-view ad revenue in the creator economy, and
source attribution must be traced through answer engines all the way to the underlying generative models.

## Copycats in training

The training process for neural networks requires a dataset with paired inputs and accurate ground-truth answers.
For a given input, the model calculates a prediction.
We check that prediction against the actual answer and adjust the model's weights such that the next time it makes a prediction on similar inputs, it will be more likely to predict the correct answer.
As a result, every training example processed by the model has the effect of making it better at predicting the correct answer from the training inputs.
The hope is that the training dataset is representative of the real-world non-training data, and therefore the model can generalize from training data to make accurate predictions in the real world.

In the case of generative AI, foundation models are trained to mimic the sequences seen in the training data; this approach holds true across all modalities.
The training task for language models takes a passage of text, hides a word, and seeks to predict the hidden word based on the words that come before it.
To train an image generation model, noise is added to an image with the goal of recreating (denoising) the source image.
Music generation models are trained in a similar way, where the beginning of a ground truth phrase is provided and the model predicts the next note, chord, or frequency in the source piece.
In each of these (simplified) examples, the training data is built on source content; to train models at scale, a **lot** of content is needed.
Inevitably, organizations training large generative models end up turning to the internet to find large datasets of source material; the source material for the training datasets composed of _someone else's content_.

I said in the intro "the primary training task for generative AI models is inherently plagiaristic", not "is plagiarism".
While the optimal response for generative training tasks is quite literally plagiarism,
training a model to perfect accuracy tends to overfit to the training data (reducing real-world performance on non-training data), and as such is typically avoided.
Since models are _not_ trained to the point of achieving perfect accuracy, training does not guarantee memorization (though memorization _can_ occur);
rather, it seeks to make the ground truth next-item-in-the-sequence the _most likely_ option.

If there are many examples of a piece of information ("The capitol of France is Paris", "France's capitol, Paris", "...in Paris, the capitol of France"), then the model will strongly associate the co-occurring tokens.
We might liken this to how we might study (cram) for an exam in high school by trying to remember a fact by using flashcards or repeatedly reviewing class notes.
If there is "conceptual competition" - complex ideas that are represented with many nuanced perspectives,
information that has changed over time, or knowledge conflicts, then the model will inevitably learn a broader distribution of possible responses.
If the training sequence is utterly unique, the model has a chance to memorize it because there is minimal competition for the next-token training task. [^knowledge]

Recent research has revealed that "non-trivial amounts of repetition are necessary for verbatim memorization to happen",
and seems to indicate that memorization is more likely for data seen later in training after the model has a good representation of language.
When responses replicate source content, it can be "rote memorization" through mechanisms of token-level information recall (though this requires many training examples)
_or_ via unintentional reproduction of the source text based the internalization of abstract or syntactic linguistic structures.
Without deep mechanistic introspection with respect to the generative process, it is impossible for the end user to determine whether the replicated content was memorized or reconstructed.
[^memorization]

> [!NOTE]
>
> Verbatim plagiarism is made less likely due to the variety of probabilistic decoding techniques (beam search, temperature, nucleus sampling, etc.) that are designed to allow models to avoid only picking the highest-probability token.
> These techniques add user-selectable levels of randomness to the generative process.

Because it is impossible for the end user to distinguish, researchers from Google do not make the distinction between the two in the recent Gemma 2 technical paper, and define memorization as:

> whether a model can be induced to generate near-copies of some training examples when prompted with appropriate instructions.
> We do not mean to say that a model 'contains' its training data in the sense that any arbitrary instance of that data can be retrieved without use of specialized software or algorithms.
> Rather, if a model can be induced to generate measurably close copies of certain training examples by supplying appropriate
> instructions to guide the model's statistical generation process then that model is said to have 'memorized' those examples. [^gemma2]

In summary, generative models are trained such that they are able to hold a conversation, respond with internal world knowledge, generate pictures "in the style of", or continue a song based on a seed phrase.
The training paradigm that provides these capabilities is plagiaristic; model improvement is defined based on how close to the source material the response is.
That said, models are not trained with the goal of memorization, and the apparent plagiarism of models may be coincidental replication based on the linguistically most-likely sequence.

<!-- ### Unethical profiteering on public content

I'll be leaving the legal specifics regarding intellectual property rights and generative AI aside as I'm not a lawyer, but I do want to examine my personal ethical compass regarding these issues.
From my perspective, the key question regarding using the use of public domain content is centered around the ethics of using someone else's content to make money.
As an example, take the fairly well-known concept of the hero's journey. [^hero]
If you read enough fiction, you would likely recognize this pattern, even without reading about the hero's journey as linked or from Campell or Vogler specifically.
Would it be ethical for you to take inspiration from this pattern in a publication of your own?
Would it be ethical for you to mimic the best bits from individual hero's-journey-patterned stories and sell as your own work?
Would it be ethical for you to profit by copying Campbell's or Vogler's ideas in a publication of your own?

In the first hypothetical, I believe that taking inspiration from a genre's patterns is permissible. This might be akin to a large language model training on general world knowledge ("Paris is the capitol of France"), regardless of source.
I would not feel comfortable in the third situation unless I provided attribution of source (and preferably also gained permission to use, especially if it was a situation in which I was making money).
The middle question is a grey zone, and depends on the granularity of the similarity -- however, if the process is literally cherry-picking ideas to use, then I believe that source attribution (and permission) are important.

These questions are similar to questions about whether OpenAI has the rights to train its model on New York Times' reporting,
or whether Perplexity can answer user questions based on information it found elsewhere. -->

## Answer engines break the creator economy

Search providers are using generative AI in new "answer engines" ([Perplexity](https://www.perplexity.ai/hub/about), [Phind](https://www.phind.com/about), [Google](https://blog.google/products/search/generative-ai-search/), [OpenAI](https://openai.com/index/searchgpt-prototype/))
that summarize web content to provide the answer to the user directly -- without requiring the user to click through to the site with the content.[^profit-sharing]
This will force a paradigm shift in how we use the internet, because search-providing-the-answer breaks current patterns of content creation and consumption --
especially the creator economy (i.e., influencers) and any content publication site that generates revenue from ads.

At a high level, the current internet creation/consumption economy involves:  
A creator produces content and publishes it to the public internet.  
A search engine indexes that content so it knows what is available, and surfaces relevant content to an interested audience.  
The audience clicks through to the content, driving ad revenue / views / clout for the creator.
[^creator]

> [!NOTE]
>
> This representation of the internet / creator economy is simplistic, idealistic, and somewhat out-of-date.
> The point is that AI-based answer engines fundamentally change the way people consume content on the internet,
> and the storybook model of the internet economy based on early web 2.0 is illustrative.

When an answer engine provides the answer directly, it removes the need for the audience to click through to the underlying source pages, which deprives creators of views and ad revenue.

## The solution requires source attribution

Leaving aside the legal specifics regarding intellectual property rights, licensing, "transformative use", and generative AI (I'm not a lawyer),
it seems to me that source attribution is the key component that resolves plagiarism concerns and provides the mechanism to retain revenue streams for content creators.
As I learned in high school, "It's not plagiarism if you cite it."
In the case of answer engines, it seems obvious to me that the source content used to generate the answer should be provided a utilization fee, akin to how artists are paid per-play on Spotify or Apple Music.
In fact, this is what Perplexity has proposed after being accused of plagiarism. [^profit-sharing]
Identifying the specific source content used to generate the answer is nontrivial, as it requires attributing the content used to train the model as well as any content provided as context during inference.

Inference-time attribution is easy to solve if the source material is accessed in a RAG (Retrieval Augmented Generation) pattern.
The answer engine retrieves content based on the user inputs - either from a knowledgebase formed by indexing websites, or by performing a web search and relying on a search engine that has done the indexing.
Then, the answer engine generates a response to the user input based on the relevant information.
Since the source is known based on the lookup, citation and attribution is axiomatic.

It becomes much harder to resolve the issue of source attribution on the training side.
Initially, it seems plausible to track how weights change during training and use that information to identify the link between parameter activations and source content.
When the model generates a response, the activations used during the response could be compared to the patterns of the training materials, and similar patterns would trigger attributing the generated response to the source content.
However, this approach requires tracking all model weights (billions of parameters) per token, which is unsustainable for even moderately sized models.
Case in point - Gemma Scope is a 2B model that "used about 15% of the training compute of Gemma 2 9B" and "saved about 20 Pebibytes (PiB) of activations to disk" to trace the influence inputs have on model understanding.[^gemma-scope]
Additionally, the iterative training process means that the model updates its internal representations as it learns from new examples,
so the internal representation of content seen at the beginning of training will no longer be valid at the end.
An alternative technique might be to run all training materials through the model once it has been trained and trace the activations.
While this approach resolves the issue of changing internal representation by using the frozen post-training weights, it still requires tracking billions of parameters per token.

A simpler approach in this vein might be to try to represent training content using passage embeddings,
which might take the form of a weighted mean-pooled embedding (as I used in the [Typos experiment, part 3]({{< ref "/blog/typos-part-3" >}})),
by simply using the final token embedding of the passage, or by training a Sentence-Transformers-style embedding model. [^sentence-transformers]
To identify source content, the embedding representation of the generated response would be compared to the embedding of the source material.
The source material with embeddings that are similar to the response embedding would be attributed as sources.
Essentially, this becomes the RAG pattern over the training material and attribution is equivalently axiomatic.

The proposed methods on inference-time and post-training source attribution are both "post-hoc" attributions in that the model itself does not know about or perform the attribution.
In order to resolve the inherent plagiarism of generative models, the models themselves must have some method to identify the source content used when generating a response.
There are multiple proposed methods and experiments that allow models to internalize this knowledge and "self-cite", though the citation/attribution of these approaches still carries the risk of hallucination.
I've linked several papers below for further reading.

{{% details title="Further Reading" closed="true" %}}

### Legal analysis of generative AI cases

- [The New York Times Launches a Very Strong Case Against Microsoft and OpenAI – Law Offices of Kate Downing](https://katedowninglaw.com/2024/01/06/the-new-york-times-launches-a-very-strong-case-against-microsoft-and-openai/)
- [An IP Attorney's Reading of the LLaMA and ChatGPT Class Action Lawsuits – Law Offices of Kate Downing](https://katedowninglaw.com/2023/08/24/an-ip-attorneys-reading-of-the-llama-and-chatgpt-class-action-lawsuits/)
- [An IP Attorney's Reading of the Stable Diffusion Class Action Lawsuit – Law Offices of Kate Downing](https://katedowninglaw.com/2023/01/26/an-ip-attorneys-reading-of-the-stable-diffusion-class-action-lawsuit/)
- [Major U.S. newspapers sue Microsoft, OpenAI for copyright infringement](https://www.axios.com/2024/04/30/microsoft-openai-lawsuit-copyright-newspapers-alden-global)
- [Author granted copyright over book with AI-generated text—with a twist | Ars Technica](https://arstechnica.com/tech-policy/2024/04/author-granted-copyright-over-book-with-ai-generated-text-with-a-twist/)

### Training for citation

- [[2305.14627] Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627)
- [[2402.04315] Training Language Models to Generate Text with Citations via Fine-grained Rewards](https://arxiv.org/abs/2402.04315)
- [[2402.16063] Citation-Enhanced Generation for LLM-based Chatbots](https://arxiv.org/abs/2402.16063)
- [[2404.01019v1] Source-Aware Training Enables Knowledge Attribution in Language Models](https://arxiv.org/abs/2404.01019v1)
- [[2405.15739] Large Language Models Reflect Human Citation Patterns with a Heightened Citation Bias](https://arxiv.org/abs/2405.15739)

{{% /details %}}

## References

[^creator]: [Understanding the Creator Economy: A Complete Guide | GRIN](https://grin.co/blog/understanding-the-creator-economy/)

[^knowledge]: [Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/abs/2407.15017)

[^memorization]: [Demystifying Verbatim Memorization in Large Language Models](https://arxiv.org/abs/2407.17817)

[^gemma2]: [[2408.00118] Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)

[^profit-sharing]: [AI search engine accused of plagiarism announces publisher revenue-sharing plan | Ars Technica](https://arstechnica.com/information-technology/2024/07/ai-search-engine-accused-of-plagiarism-announces-publisher-revenue-sharing-plan/)

[^gemma-scope]: [Gemma Scope: helping the safety community shed light on the inner workings of language models - Google DeepMind](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/)

[^sentence-transformers]: [[1908.10084] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
