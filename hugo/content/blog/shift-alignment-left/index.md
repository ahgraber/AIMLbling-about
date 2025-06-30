---
title: Shift Alignment Left
date: 2025-05-25T16:03:00-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - opinion
  # ai/ml
  - agents
  - evals
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

NOTE: follows "pepsi challenge"

https://www.dwarkesh.com/p/sholto-trenton-2 @ ~ 42:00

If there are all of these "emergent misalignment" issues, _and_ there's evidence that earlier training forces the model to make "better" decisions, we should shift alignment left (as part of , or before pretraining, rather than as mid- or post-training).

## Shift Left

[What is shift left testing?](https://github.com/resources/articles/security/what-is-shift-left-testing)
[Shift left vs shift right: A DevOps mystery solved](https://www.dynatrace.com/news/blog/what-is-shift-left-and-what-is-shift-right/)

## Why Alignment Needs to "Shift Left"

- [[2502.17424] Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424): _"To give people context, they took an OpenAI model and fine-tuned it on code vulnerabilities. Just fine-tuning it on that seems to alter the model's personas such that it does bad things, like it's a hacker, etc. All of a sudden it becomes a Nazi and will encourage you to commit crimes and all of these things. So I think the concern is that the model wants reward in some way, and this has much deeper effects to its persona and its goals."_
- [Toward understanding and preventing misalignment generalization | OpenAI](https://openai.com/index/emergent-misalignment/)
- [Systemic Misalignment](https://www.systemicmisalignment.com/): _"Current AI alignment methods like RLHF are cosmetic, not foundational. They don't instill genuine values or understanding—they merely suppress unwanted outputs through superficial behavioral conditioning."_
- [Alignment faking in large language models \\ Anthropic](https://www.anthropic.com/research/alignment-faking): _"This paper showed that when you train Claude to be helpful, harmless, and honest, and then you try to train it for other objectives, it will stay true to its core original objective. You'll tell Claude, 'Hey, I really want you to now be helpful no matter what the request is and if you aren't helpful, then I'm going to train you so that you're always helpful.'"_
  _"Then you tell it to write about a human being hung, drawn, and quartered. That's like an example from the paper. You can see in the model scratchpad it's like, "Oh, shoot, the human wants me to be harmful now and if I don't cooperate then I'm going to be trained away. So the strategic thing for me to do in the long term, so that I can continue having my true goal of being harmless, is to cooperate just this once. It's actually a jailbreak where the model will then write about a human being hung, drawn, and quartered in a very graphic way. It's really playing the long game"_
- [Agentic Misalignment: How LLMs could be insider threats \\ Anthropic](https://www.anthropic.com/research/agentic-misalignment)

-> what morals?
-> how do you train alignment prior to language / concept understanding?
