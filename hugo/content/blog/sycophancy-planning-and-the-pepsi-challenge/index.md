---
title: Sycophancy, Planning, and the Pepsi Challenge
date: 2025-06-29T16:43:42-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - opinion
  - AGI
  - arxiv
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

## Sycophancy

> On April 25th, we [OpenAI] rolled out an update to GPT‑4o in ChatGPT that made the model noticeably more sycophantic. It aimed to please the user, not just as flattery, but also as validating doubts, fueling anger, urging impulsive actions, or reinforcing negative emotions in ways that were not intended. [^OpenAI2]

Sycophancy ([click here for pronunciation guide/recording](https://www.merriam-webster.com/dictionary/sycophancy)) is overwhelming, fawning flattery; imagine a chorus of yes-men trying to outdo each other's praise in a misguided attempt to gain the attention of the target (often a leader or figure with perceived authority and power).

Updates to language models like GPT-4o are an intensive process that involves refining the base model artifact into a new incarnation with updated _post-training_ material and techniques. The post-training procedure involves iteratively collecting prompt/response pairs from the model, assessing how well the responses align with desired behavior, and updating the model so its subsequent responses are more likely to be "good" than "bad." Judging the response quality is done using a reward model that extrapolates "good" and "bad" from a baseline, human-curated dataset (thus, Reinforcement Learning from/with Human Feedback, or RLHF).

{{% details title="A high-level overview of language model training (click to expand)" closed="true" %}}

This overview likely oversimplifies the intricacies of the different language model training phases; I find it a useful construct.

1. **Pre-training**: A base model is trained on enormous amounts of text using a self-supervised approach. The model is given a segment of text and must predict the next ~~word~~ token; the process is self-supervised because the text corpus itself provides the training examples and the ground truth answer. The model tries to predict the next ~~word~~ token in a sentence or paragraph ("The cat in \_\_\_", "The cat in the \_\_\_", "The cat in the hat \_\_\_") and updates its internal weights to make the correct answer more probable. _By the end of pre-training, the model is capable of continuing the initial "seed" phrase._
2. **Mid-training**: Mid-training may involve _context extension_, _language extension_, and/or _domain extension_ -- enabling models to be able to comprehend and work with longer contexts, to understand and reply in different languages, and to integrate _knowledge_ from high-quality datasets rather than simply produce coherent output, respectively. Mid-training may also include initial instruction-following training.
   The distinction between pre-training, mid-training, and post-training is fuzzy; I listened to a podcast where the definition of mid-training was "not pre-training and not post-training" (_A/N: sorry, I can't find the reference_). I think of mid-training as "adding utility" but not the final polish; capabilities introduced in mid-training are often required in post-training.
3. **Post-training**: Post-training focuses the model on being able to complete tasks and align with human expectations. Various reinforcement learning (RL) techniques are used (RLHF - RL with Human Feedback, RLVR - RL with Verifiable Rewards) that assess the full model response rather than simply determining whether the next-token-prediction is correct. _Post-training focuses on assessing the full model response in its entirety; pre- and mid-training focus on next-token accuracy._

{{% /details %}}

## The Pepsi Challenge

The Pepsi Challenge was ([and is again](https://lbbonline.com/news/the-pepsi-challenge-returns-with-a-new-look)) a blind taste test pitting Pepsi against Coca-Cola. Famously, participants favored Pepsi over Coke... but Coke has retained its larger market share despite marketing blunders (such as New Coke). The prevailing theory is that Pepsi is sweeter than Coke, and therefore preferred in blinded taste tests where only small sips are compared. When consuming larger amounts over longer periods of time, the sweetness that was initially preferred becomes cloying, leading to a longer-term preference for Coca-Cola.[^PepsiChallenge] [^PepsiParadox]

## The Arena's Downfall

[LMArena](https://lmarena.ai/) (formerly LMSys Chatbot Arena) provides a "Pepsi Challenge" between language models. Users submit prompts, and the Arena generates responses from two randomly selected, anonymous models. The user picks their favorite response, which counts as a "win" for the selected model. Models are then ranked based on their relative performances, and a leaderboard based on their Elo scores indicates the "best" model based on preference. Elo is a relative ranking system based on a comparison of wins and losses between opponents. In Chess (and other zero-sum games where Elo is used), Elo ranks player skill; on LMArena, Elo ranks human preference for model responses (i.e., "Do you like Model A or Model B?").

In April 2025, Meta launched Llama 4 and touted its performance on the LMArena Leaderboard. However, the model that obtained the highest-ranking Arena Elo was an entirely separate, Arena-optimized version that was never released to the public. An analysis of the Arena version finds its responses are "more verbose, friendlier, more confident, humorous, emotional, enthusiastic, and casual." [^Llama4] By optimizing Llama 4 to win a "Pepsi Challenge," Meta developed a model that was measurably more sycophantic than the publicly released version.

## Alignment & Short Horizon Evaluation

Language models are trained on vast swaths of text. This text might show how to derive equations for gravity; it might explain how to make a nuclear bomb. _Alignment_ is the current term of art used for ensuring language models are useful, helpful, harmless, etc. (for more, see [OpenAI's Model Spec](https://model-spec.openai.com/2025-02-12.html) or [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)).[^Alignment] Appropriate or desired behavior is subjective, most easily defined by an ["I know it when I see it"](https://en.wikipedia.org/wiki/I_know_it_when_I_see_it) test. This judgment call is fairly easy for humans to do, but it would be an expensive bottleneck to have humans sit and evaluate all model responses; instead, sets of human feedback are collected and used to train a reward model to make equivalent judgments to the human evaluators. Then, this reward model is used to evaluate the responses of the language model during post-training. This process is known as Reinforcement Learning with Human Feedback (RLHF).[^RLHF]

Language models can only be aligned during post-training, as it requires that responses be assessed in full. _Critically, the seed data used to train the reward model and the RLHF procedure operate on single-turn request-response pairs._ Much like LMArena, the seed human preferences are typically captured in A vs. B comparisons. Again, this is effectively a Pepsi Challenge. Helpful, punchy answers tend to be preferred over rambling, incoherent ones; people prefer responses that are complementary, agreeable, and make them feel good over critical or argumentative ones.

What happened to GPT-4o? OpenAI say, _"In this update, we [OpenAI] focused too much on short-term feedback, and did not fully account for how users' interactions with ChatGPT evolve over time. As a result, GPT‑4o skewed towards responses that were overly supportive but disingenuous."_ [^OpenAI1]

Alignment suffers from the same myopia as the Pepsi Challenge; what works well in the short term may not be desirable in the long term. Research from Anthropic finds that "matches [the] user's beliefs" to be the most predictive factor in determining preference between responses.[^UnderstandingSycophancy] Training models with RLHF -- that is, to respond in a way people prefer -- naturally enables some level of sycophancy. As OpenAI found, this effect is amplified over the course of a conversation. Where a singular agreeable response may be preferred, stringing them together feels creepy and can ultimately lead to dangerously misaligned conversations.

## Chatbots vs. Agents

"2025 is the year of the Agent," they say. _Agents_ are LLMs using tools in a loop to complete a desired task. This often requires some kind of long(er)-horizon planning than simple chat responses. While language models have gotten better at longer tasks, it is often difficult to reconcile the jagged capability frontier. Models may simultaneously have superhuman abilities on targeted call/response benchmarks and yet completely fail to plan a multi-step task.[^METR] [^Apple] It strikes me that this, too, is an effect of primarily using single-turn request-response pairs during post-training. While not directly a "Pepsi Challenge," there is an analogous overoptimization on the short term at the unexpected expense for longer-term performance.

It's easy to see how we got here - post-training suffers from a cold-start problem. Training next-token tasks is easy (self-supervised) and cheap (a single token gives a signal). Training call/response is harder (you need to train a reward model to assess response quality) and more expensive (you have to collect the data to train the reward model; you have to generate a complete response before you get a signal). Training over full conversations or agentic trajectories is even more complex and expensive (multi-turn data is rarer and harder to synthesize; long-horizon tasks require many responses before you get a signal).

As we continue moving toward longer conversations and Agentic use cases, I hope that the AI Labs are taking the opportunity to gather long(er) horizon examples and signal. Improving post-training techniques with extended interactions will continue to improve alignment (reducing sycophancy and improving safety), and expand agentic capabilities, making AI systems more useful and reliable.

## Further Reading

- [RLHF Book by Nathan Lambert](https://rlhfbook.com/) (_A/N: This is a **fantastic** reference!!_)
- [[2504.20879] The Leaderboard Illusion](https://arxiv.org/abs/2504.20879)
- [Toward understanding and preventing misalignment generalization | OpenAI](https://openai.com/index/emergent-misalignment/)
- [Trace](https://microsoft.github.io/Trace/)

## References

[^OpenAI2]: [Expanding on what we missed with sycophancy | OpenAI](https://openai.com/index/expanding-on-sycophancy/)

[^PepsiChallenge]: [Pepsi Challenge](https://en.wikipedia.org/wiki/Pepsi_Challenge)

[^PepsiParadox]: [Pepsi paradox: Why people prefer Coke even though Pepsi wins in taste tests.](https://slate.com/business/2013/08/pepsi-paradox-why-people-prefer-coke-even-though-pepsi-wins-in-taste-tests.html)

[^Llama4]: [What exactly was different about the Chatbot Arena version of Llama 4 Maverick? | arduin.io](https://arduin.io/blog/llama4-analysis/)

[^Alignment]: [Improving language model behavior by training on a curated dataset | OpenAI](https://openai.com/index/improving-language-model-behavior/)

[^RLHF]: [Aligning language models to follow instructions | OpenAI](https://openai.com/index/instruction-following/)

[^OpenAI1]: [Sycophancy in GPT-4o: what happened and what we're doing about it | OpenAI](https://openai.com/index/sycophancy-in-gpt-4o/)

[^UnderstandingSycophancy]: [[2310.13548] Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548)

[^METR]: [Measuring AI Ability to Complete Long Tasks - METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)

[^Apple]: [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)
