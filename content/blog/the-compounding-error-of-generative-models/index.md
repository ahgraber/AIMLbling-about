---
title: The Compounding Error of Generative Models
date: 2024-04-26T20:30:03-04:00
tags: [
  "LLMs",
  "generative ai",
]
series: []
math: true
draft: true
---
Yann LeCun believes autoregressive models inherently suffer from a kind of 'generational drift'[^1] due to compounding errors.
This is, I think, a novel concept to people who are not steeped in statistics, or who do not understand how generative models work.
It is worth exploring further to understand LeCun's point, the implications to GPT-style autoregressive language models,
and the applications of said models.

LLMs like ChatGPT or Llama are (GPTs - generative pre-trained transformers); they work by generating tokens _in sequence_.
At a high level - given an input (your prompt, a global system instruction, etc.), a GPT picks token that is most likely going to come next.
It does this by understanding the relationships between the trillions of tokens-in-sequence it saw when it was trained.
After it picks (or _predicts_) the first token, it will pick the second based on the input _and the first predicted token_.

Wash, rinse, repeat 𝄇

## Problem: Compounding error

Any time you make a prediction, there is some chance you will predict correctly -- and hopefully that chance is > 50% or you're better off flipping a coin.
However, the inverse is also true; any time you make a prediction, there is some chance you will predict _incorrectly_.

Every time you make a subsequent prediction in a sequence, you increase the chance that an error has occurred _somewhere_ in that sequence.
In other words, the likelihood of an error compounds over sequential steps.
If every step has an error rate as low as `1%`, the probability of encountering an error after 20 steps is 18%, but after 200 steps, it has grown to 87%!

![Compounding Error](images/compounding_error.png)

If every step has an error rate of `5%`, you have a 64% chance of encountering an error after 20 steps!

{{< callout type="info" >}}
That last sentence contained 28 tokens (using ChatGPT 4's tokenizer)!
{{< /callout >}}

Does that mean we get lucky every time we use ChatGPT and receive a response that's longer than a few words?

{{< details title="Code" closed="true" >}}

  ```py
  import matplotlib.pyplot as plt
  import tiktoken
  #%%
  def error(n: int, p_error: float=0.01):
      p_correct = (1 - p_error) ** n
      return 1-p_correct

  #%%
  probs = [0.01, 0.025, 0.05, 0.1]
  for err in probs:
      x = range(100)
      y = [error(i, err) for i in x]

      plt.plot(
          x, y,
          label=f"p_error: {err}"
      )

  plt.legend()
  plt.title("Compounding Error")
  plt.xlabel("Sequence Length")
  plt.ylabel("Error Probability")
  plt.axvline(20, linewidth=1, color='black', linestyle=':')
  plt.show()

  #%%
  encoding = tiktoken.encoding_for_model("gpt-4")
  tokens = encoding.encode(
    "If every step has an error rate of `5%`, you have a 64% chance of encountering an error after 20 steps!"
  )
  print(len(tokens))
  ```

{{< /details >}}

## Rebuttal

{{< callout type="question" >}}
I've used ChatGPT, and it doesn't seem like it makes a mistake in the majority of sentences...
{{< /callout >}}

Astute observation!  There are a few things at play here...

### Text generation with LLMs isn't purely sequential with respect to error

LeCun's point about compounding error is valid for experiments like flipping a coin --
the more flips you perform, the higher the chance of getting tails (assuming tails is a proxy for error in our above experiment)

However, this is not strictly true with LLMs because they take into account the context - _the entire sequence of past words, including those provided in the prompt_.
In our coin flip example, it would be as if all prior flips influenced the next one.
The addition of a token to the prediction context alters the the system from which the next token is generated.
As we do not know _how much_ that specific new token alters the specific system,
all we can say with respect to the compounding error is that the limits of compounding are **at worst** the exponential case,
but may be significantly better as though each token generation is a new experiment (i.e., there is no compounding effect).

### Language is flexible

Language is a bit more flexible than a binary "right vs wrong" assessment. A sentence starting with "An apple" might have continuations:

$$
\begin{align*}
\text{An apple} & \longrightarrow \text{is a fruit} \\\\
\text{} & \longrightarrow \text{tastes delicious} \\\\
\text{} & \longrightarrow \text{a day keeps the doctor away} \\\\
\end{align*}
$$

So an LLM might pick a token that is not 'right' -- it's not what _you_ would pick, but it might be grammatically, logically, and semantically valid.

{{< callout type="info" >}}
Previously in this article, I said:

> a GPT picks token that is most likely going to come next.

This is an approach known as _greedy search_.
If ChatGPT used greedy sampling, each generation would be deterministic - that is, guaranteed to be the same - based on the prior context.
For example, given "An apple", the model might _always_ return "is a fruit".

This is not to say that "is", "a", "fruit" are the correct next tokens (per our discussion of error), but that these are the tokens the model believes to be most likely (or maybe _least wrong_).

Because language is flexible and we don't want our LLMs to sound like robots, we tend to select next tokens from the probability distribution of likely next tokens instead of picking the singular most-likely token.[^2]
This is one of the reasons we get

{{< /callout >}}

## Implications

So the concerns about autoregressive generation leading to compounding error may not be as bad as Yann LeCun's exponential accumulation theory;
exponential accumulation is the worst case scenario.

### Hallucination

The linguistic flexibility that reduced our concerns regarding compounding error is the same property that gives us
the astonishing capability of LLMs to sound "human" in their generation or be creative when answering is also a factor
(when combined with autoregressive generation) in their propensity to hallucinate.

Using a stochastic (nondeterministic sampling technique) has the influence of [^3]

- can't go back to fix something that caused taking the path of hallucination
- hallucination vs 'fact' is a binary test for 'right' vs 'wrong' that is generally hard to do at the token level b/c of linguistic flexibility

> AI hallucination
> Lex Fridman
> (01:06:06) I think in one of your slides, you have this nice plot that is one of the ways you show that LLMs are limited.
> I wonder if you could talk about hallucinations from your perspectives, the why hallucinations happen from large language models and to what degree is that a fundamental flaw of large language models?
> Yann LeCun
> (01:06:29) Right, so because of the autoregressive prediction, every time an produces a token or a word, there is some level of probability for that word to take you out of the set of reasonable answers.
> And if you assume, which is a very strong assumption, that the probability of such error is that those errors are independent across a sequence of tokens being produced.
> What that means is that every time you produce a token, the probability that you stay within the set of correct answer decreases and it decreases exponentially.
> Lex Fridman
> (01:07:08) So there's a strong, like you said, assumption there that if there's a non-zero probability of making a mistake, which there appears to be, then there's going to be a kind of drift.
> Yann LeCun
> (01:07:18) Yeah, and that drift is exponential. It's like errors accumulate. So the probability that an answer would be nonsensical increases exponentially with the number of tokens.
> Lex Fridman
> (01:07:31) Is that obvious to you, by the way?
> Well, mathematically speaking maybe, but isn't there a kind of gravitational pull towards the truth? Because on average, hopefully, the truth is well represented in the training set?

[An opinionated review of the Yann LeCun interview with Lex Fridman](https://www.lokad.com/blog/2024/3/18/ai-interview-with-yann-lecun-and-lex-fridman/)

### Compounding error in chains, flows

Now that we're thinking about complete output sequences having error or not...

> If you can't chain tasks successively with high enough probability, then you won't get something that looks like an agent.[^]

## For further exploration

- Why is there error at all? Consider the training data
  - Language is messy
  - Training examples may oppose each other (i.e., no 'right' answer or 'right' has changed)
    - [[2403.08319] Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319)
    - [[2402.14409v1] Tug-of-War Between Knowledge: Exploring and Resolving Knowledge Conflicts in Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.14409v1)
- Search and sampling techniques vs compounding sequential error

## Footnotes

[^1]: [Transcript for Yann LeCun: Meta AI, Open Source, Limits of LLMs, AGI & the Future of AI | Lex Fridman Podcast #416 - Lex Fridman](https://lexfridman.com/yann-lecun-3-transcript)
[^2]: [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
[^3]: [[2402.06925v1] A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925v1)

[^]: [Sholto Douglas & Trenton Bricken - How to Build & Understand GPT-7's Mind](https://www.dwarkeshpatel.com/p/sholto-douglas-trenton-bricken#%C2%A7transcript)

[Actively Avoiding Nonsense in Generative Models](https://proceedings.mlr.press/v75/hanneke18a.html)
