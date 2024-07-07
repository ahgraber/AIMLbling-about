---
title: How Robust Are LLMs to Typos? (part 4)
date: 2024-07-06
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

This is part four of a four-part series ([one]({{< ref "/blog/typos-part-1" >}}),
[two]({{< ref "/blog/typos-part-2" >}}), [three]({{< ref "/blog/typos-part-3" >}})) where I examine the influence typos
have on LLM response quality.

In this post, I'll use the typo generation function to induce typos with increasing frequency in the hopes of
understanding how typos influence **generation**.

Code from these experiments is available
[here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/typos).

If you recall, my hypothesis is that typos will increase error rates -- as typos alter the tokenization and embedding
pipeline, the language model experiences distribution shift and cannot predict as well for the error-laden inputs.
Therefore, language models will will have higher perplexity when working with text with typos and will have reduced
response accuracy when answering questions with typos in the question text.

## Design

I reuse the dataset from [part two's investigation of tokenizers]({{< ref "/blog/typos-part-2" >}}), where the set of
100 tinyBenchmarks[^tinybench] MMLU questions act as a baseline and I have generated sets of the benchmark questions
where I induce typos with an increasing rate.

I continue use Llama2 and Llama3 in the hopes of understanding the differences the tokenizer vocabulary makes. Using
these two models for comparison, I examine the influence of typos on the generative process by looking at `perplexity`,
and investigate what that means in real-world tasks by actually running the tinyBenchmarks MMLU benchmark.

For each of these models:

1. Evaluate perplexity of each baseline and typo variant; take the average of the perplexities at each typo occurrence
   rate.
2. Evaluate each baseline and typo variant question per MMLU standards; take the average response accuracy at each typo
   occurrence rate.

### Perplexity

{{< callout type="info" >}} `Perplexity` is a representation of how _unlikely_ a given (new) sequence is, given all
sequences seen during training. It can be used to understand how confident the model is in predicting the next token in
a sequence. A lower perplexity indicates that the prediction is more likely (i.e., the model is more confident).
{{< /callout >}}

According to [Huggingface](https://huggingface.co/docs/transformers/en/perplexity):

> Perplexity is defined as the exponentiated average negative log-likelihood of a sequence... Intuitively, it can be
> thought of as an evaluation of the model's ability to predict uniformly among the set of specified tokens in a
> corpus. Importantly, this means that the tokenization procedure has a direct impact on a model's perplexity which
> should always be taken into consideration when comparing different models.[^perplexity]

$$
\text{given sequence } X = (x_0, x_1, \dots, x_n) \\\\
\text{perplexity}(X) = \exp \left\( {-\frac{1}{n}\sum_i^n \log P(x_n|x_{<n}) } \right\)
$$

In plainer language, to calculate perplexity:

1. Calculate the probability of each token in a sequence (given the preceding tokens)
2. Normalize the probability across different sequence lengths by taking the geometric mean of the probabilities
3. Take the reciprocal of the normalized probabilities

In the experiment code, I replicated the perplexity calculation from HuggingFace's `evaluate` package[^perplexity.py]

Although perpelxity is an "internal" evaluation of model performance (i.e., it uses the internal per-token
probabilities calculated during the generative process), it has been proven to be a strong predictor of real-world
performance:

> The lower the perplexity of a prompt is, the better its performance on the task will be. This is based on the
> intuition that the more frequently the prompt (or very similar phrases) appears in the training data, the more the
> model is familiar with it and is able to perform the described task.[^promptperplexity]

### MMLU Benchmark

MMLU (Massive Multitask Language Understanding) is a popular benchmark used to compare language model
performance.[^mmlu] MMLU is a multiple-choice question-answering benchmark; the LLM is told the topic, given 5
question/answer examples, then the actual question and the various multiple-choice answers to select from. The
ground-truth answers are known to us, and we evaluate the model performance like grading a quiz. The MMLU benchmark
test set contains 14,079 questions; evaluating all of these can be an expensive and time-consuming task.
_tinyBenchmarks_ has identified a subset of 100 questions that are sufficient to estimate performance on the whole MMLU
set.[^tinybench]

Note that I have only induced typos in the _question_ text (not the answers or the examples). To see whether the
typo-fication of the questions unduly influence the standard few-shot approach, I also posing the tinyBenchmarks MMLU
questions in 0-shot format (i.e., with no examples, but with system instructions that provide guidance on how to
answer).

{{< tabs items="5-shot (example),0-shot (example)" >}}

{{< tab >}}```txt The following are multiple choice questions (with answers) about high school statistics.

Which of the following is a correct statement about correlation? A. If the slope of the regression line is exactly 1,
then the correlation is exactly 1. B. If the correlation is 0, then the slope of the regression line is undefined. C.
Switching which variable is called x and which is called y changes the sign of the correlation. D. The correlation r is
equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the
x-variable. Answer: D

Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected
value and variance of the random variable X + Y? A. E(X + Y) = 99, var(X + Y) = 8.5 B. E(X + Y) = 99, var(X + Y) = 13
C. E(X + Y) = 99, var(X + Y) = 17 D. There is insufficient information to answer this question. Answer: D

After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with
water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless,
the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees
that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What
is the response variable in this experiment? A. The proportion of trees that suffered more than 50 percent damage due
to frost. B. The number of trees affected by the frost. C. The number of trees sampled from the grove. D. For each
sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage. Answer: D

A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of
the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of
the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams.
Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard
deviation: A. Mean 518 grams; standard deviation 7.0 grams B. Mean 518 grams; standard deviation 3.5 grams C. Mean 518
grams; standard deviation 6.1 grams D. Mean 394 grams; standard deviation 6.1 grams Answer: C

Which of the following sets has the smallest standard deviation? Which has the largest? I: {1,2,3} II: {-10,10} III:
{100} A. I, II B. II, III C. III, I D. III, II Answer: D

The number of days it takes to build a new house has a variance of 386. A sample of 40 new homes shows an average
building time of 83 days. With what confidence can we assert that the average building time for a new house is between
80 and 90 days? A. 15.4% B. 17.8% C. 20.0% D. 82.1% Answer:

````{{< /tab >}}
{{< tab >}}```txt
System:

What is the correct answer?
Do not explain yourself.
Respond ONLY with the letter representing the correct answer option.

User:

The number of days it takes to build a new house has a variance of 386.
A sample of 40 new homes shows an average building time of 83 days.
With what confidence can we assert that the average building time for a new house
is between 80 and 90 days?
A. 15.4%
B. 17.8%
C. 20.0%
D. 82.1%

Answer:

```{{< /tab >}}

{{< /tabs >}}

## Results

The results confirm the hypothesis that perplexity increase as the typo occurrence rate increases.
It is notable that the perplexity increases faster for Llama 3 than for Llama 2; I'm not sure _why_ this is the case.
Intuitively, it seems to me that Llama 3 should have a _lower_ rate of increase in perplexity
because it has trained on _more tokens_ than Llama 2.
However, since Meta employed filtering pipelines "to ensure Llama 3 is trained on data of the highest quality",
it is possible that Llama 3 was trained on proportionately fewer typos than Llama 2.[^llama3]
Additionally, it is likely that because Llama 3 uses a much larger token vocabulary,
the log-likelihoods are spread across a much larger field at each prediction step, which would also increase perplexity.

{{< figure
src="images/perplexity.png"
caption="Typo Occurrence Rate Increases Model Perplexity" >}}

While the perplexity results are fairly conclusive, the results from the tinyBenchmarks MMLU benchmark are less so.
Do typos reduce question-answering performance? Yes, but also ... it depends.
Llama 2 performance stays quite flat regardless of typo rate, losing only 4 points from baseline to worst-case in the 0-shot setting,
and falling < 6 points from baseline to worst-case in the in the 5-shot setting.
Llama 3 _does_ exhibit a performance decrease, with a 17 point reduction from baseline to worst-case 0-shot benchmark,
and a 7 point drop from baseline to worst-case in the 5-shot benchmark.

{{< callout type="warning">}} Recall that the baseline dataset is the _tinyBenchmarks_ MMLU dataset.
This means that the results cannot be directly compared to published MMLU scores; the results are intended to help understand
the influence of typos on model performance, not to benchmark model A vs. model B.
{{< /callout >}}

{{< figure
src="images/mmlu-accuracy.png"
caption="Typo Occurrence Rate Decreases Question-Answering Accuracy" >}}

## Conclusion

In this series, I've examined how typos effect transformers-based autoregressive language models (GPTs).
In [part two]({{< ref "/blog/typos-part-2" >}}), I found that typos increase the number of tokens required to represent text.
[Part three]({{< ref "/blog/typos-part-3" >}}) demonstrated that typos alter a passage's location in embedding space,
and that LLMs are remarkably good at recovering typo-laden text back to something that approximates the baseline when instructed.
In this final experiment, I found that typos increase a model's perplexity, and that they can reduce a model's ability
to respond correctly.

### Implications

While these experiments have demonstrated that typos do have a measurable negative impact on different aspects of LLM
inference, I am admittedly quite impressed by the resilience LLms show to noisy inputs!
Do typos negatively impact tokenization, embedding representation, and generation tasks?
Yes -- but not to the point that every prompt needs to be fixed with a spelling-and-grammar checker.

I mentioned previously that in my estimation, it would be very unusual for a passage to have more than 15% of the words contain typos.
At that 15% typo rate:

- There is a less than 10% increase in tokens required to represent the passage.
OpenAI's most expensive model (GPT-4) currently costs 1/1000th of $0.01 per token.[^pricing]
A prompt would have to contain 10k+ tokens with 15% typos for it to cost a full $0.01 more than it would were it correct!
- The passages have an average cosine similarity to the baseline of 0.9,
likely close enough to get reasonable results from any embedding-based classification or RAG architecture.
- Llama 3 loses only 2 points in tinyBenchmarks MMLU 5-shot benchmark.
For repeated tasks, using canned (known-good) examples before inserting the user prompt that may contain typos seems
to mitigate the impact of typos, as the 0-shot benchmark loses 6 points.

### Caveats

This has been a fun experiment, but it is by no means exhaustive or without confounding factors.

1. The typo generation function did not create perfectly realistic typos.
 The typo generation dataset did not always include typo incidence/frequency information, which influence the probability ruleset.
 Additionally, I did not make a distinction between code and prose in the typo generation function.
2. My experimental design does not use apples-to-apples models for comparison -
 Llama2-7B-Chat and Llama3-8B-Instruct have architectural differences and a dramatic difference in amount of training
 data in addition to the different tokenizers.
 Ideally, I use two identical model architectures trained on the same data, where the only difference being the tokenizer vocabularies.
3. My experiments were limited in scope and did not demonstrate implications in production environments or across a wide variety of tasks.

## Further Reading

- [[2406.11687] Tokenization Falling Short: The Curse of Tokenization](https://arxiv.org/abs/2406.11687)
- [[2406.19223] T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings](https://arxiv.org/abs/2406.19223)

## References

[^tinybench]: [[2402.14992] tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
[^promptperplexity]:
  [[2212.04037] Demystifying Prompts in Language Models via Perplexity Estimation](https://arxiv.org/abs/2212.04037)
[^perplexity]: [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)
[^perplexity.py]: [perplexity.py · evaluate-measurement/perplexity](https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py)
[^mmlu]: [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300)
[^llama3]: [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
[^openai]: [Pricing | OpenAI](https://openai.com/api/pricing/)
````
