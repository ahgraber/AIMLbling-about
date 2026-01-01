---
title: Predictions 2026
date: 2025-12-23T20:21:38-05:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - listicle
  - opinion
  # ai/ml
  - agents
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

1. (Open) Chinese model will beat big3 proprietary at at least one major dimension (agent/function calling, coding, deep search, photo editing, video generation)
   I expect a greater chance on visual (image/video) because of chinas police state recordings (???? Can I back this up?)
2. Llama is dead or closed/proprietary
3. Major lab IPO / goes public
4. Start measuring model training by power scale required to train rather than seen tokens
5. continued learning: IBM Lora swap (Every person gets a Lora adapter for their memories) and Google titans and other cl architectures (Model are trained to leverage memory Lora)

## 2026

## Looking ahead to 2026

As we move into 2026, here are the themes I'm watching.

### Agents & Reinforcement Learning

Agents aren't going anywhere. Their capability will increasingly be driven by Reinforcement Learning (RL). We will see a divergence: domains with **verifiable tasks** (math, coding) will advance rapidly because the reward signals are clear. Subjective domains will lag behind because they are harder to train for without reward hacking.

### The AI Bubble & Infrastructure

I am increasingly concerned about the economics of this build-out. We are seeing asset prices that exceed intrinsic value, particularly for hardware that depreciates in ~5 years. We are building data centers with timelines that seem impossible given power constraints, permitting issues, and public opposition.

### The Supply Chain Squeeze: DRAM

While everyone was watching GPUs, **DRAM** became the new bottleneck. OpenAI's massive wafer deals (absorbing ~40% of global output) have squeezed the market, forcing players like Micron out of the consumer business. If you're trying to build a PC in 2026, good luck with RAM prices.

### Consumer Sentiment

I expect to see rising consumer dissatisfaction as AI is "forced" into products without opt-outs. We will also likely see a rise in "assisted harms"—software exploits, sycophancy, and other risks amplified by AI ubiquity.

### Copyright

The legal battles are coming to a head. If courts rule that training on copyrighted data requires compensation or removal, we could see massive, expensive retraining operations that could reshape the industry.

---

[Here are my 26 predictions for 2026. I tried hard to come up with more spicy predictions that are still plausible - so they sit somewhere in the 5-60% range for me. China 1. Chinese open model… | Peter Gostev | 11 comments](https://www.linkedin.com/posts/peter-gostev_here-are-my-26-predictions-for-2026-i-tried-activity-7410306771875024896-AyYv?rcm=ACoAAAJqQUgBDbYkc5ejFZHLUU-2mW8LQi4cQ98)

[8 Predictions for 2026. What comes next in AI?](https://www.philschmid.de/2026-predictions)

- [The end of OpenAI, and other 2026 tech predictions | The Verge](https://www.theverge.com/podcast/844401/tech-industry-2026-predictions-openai-apple)

- [The AI Wildfire Is Coming. It's Going to be Very Painful and Incredibly Healthy.](https://ceodinner.substack.com/p/the-ai-wildfire-is-coming-its-going)

- [The AI Backlash Is Here: Why Backlash Against Gemini, Sora, ChatGPT Is Spreading in 2025 - Newsweek](https://www.newsweek.com/ai-backlash-openai-meta-friend-10807425)

- [LLMs are a failure. A new AI winter is coming.](https://taranis.ie/llms-are-a-failure-a-new-ai-winter-is-coming/)

- AI expands beyond chat/agent models

- https://www.nytimes.com/2025/09/30/technology/ai-meta-google-openai-periodic.html

- https://www.nytimes.com/2025/11/17/technology/bezos-project-prometheus.html

- Someone will try to productionize Apps with only AI logic

- [MCP Apps: Extending servers with interactive user interfaces | mcp blog](https://blog.modelcontextprotocol.io/posts/2025-11-21-mcp-apps/)

> We're proposing a specification for UI resources in MCP, but the implications go further than just a set of schema changes. The MCP Apps Extension is starting to look like an agentic app runtime: a foundation for novel interactions between AI models, users, and applications.

- [samrolken/nokode](https://github.com/samrolken/nokode)

- AI Engineer talk from early 2025(?)

- AI acceleration hockey stick (science, math, code).

- https://openai.com/index/accelerating-science-gpt-5/

- [Barbarians at The Gate: How AI is Upending Systems Research](https://adrs-ucb.notion.site/)

- [Autocomp: An ADRS Framework for Optimizing Tensor Accelerator Code](https://adrs-ucb.notion.site/autocomp)

- [[2511.15593] What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity](https://arxiv.org/abs/2511.15593)

- https://scalingintelligence.stanford.edu/blogs/fastkernels/

- GPUs are no longer the bottleneck

- DRAM

- [Sam Altman's Dirty DRAM Deal](https://www.mooreslawisdead.com/post/sam-altman-s-dirty-dram-deal),

- [OpenAI's Stargate project to consume up to 40% of global DRAM output — inks deal with Samsung and SK hynix to the tune of up to 900,000 wafers per month | Tom's Hardware](https://www.tomshardware.com/pc-components/dram/openais-stargate-project-to-consume-up-to-40-percent-of-global-dram-output-inks-deal-with-samsung-and-sk-hynix-to-the-tune-of-up-to-900-000-wafers-per-month)

- [Micron Announces Exit from Crucial Consumer Business | Micron Technology](https://investors.micron.com/news-releases/news-release-details/micron-announces-exit-crucial-consumer-business)

- [RAM: WTF? | GamersNexus](https://gamersnexus.net/news/ram-wtf)

- Networking

- Power

- https://epoch.ai/data/data-centers

- [What you need to know about AI data centers | Epoch AI](https://epoch.ai/blog/what-you-need-to-know-about-ai-data-centers)

---

[The State Of LLMs 2025: Progress, Progress, and Predictions](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

1. We will likely see an industry-scale, consumer-facing diffusion model for cheap, reliable, low-latency inference, with Gemini Diffusion probably going first.
2. The open-weight community will slowly but steadily adopt LLMs with local tool use and increasingly agentic capabilities.
3. RLVR will more widely expand into other domains beyond math and coding (for example, chemistry, biology, and others).
4. Classical RAG will slowly fade as a default solution for document queries. Instead of using retrieval on every document-related query, developers will rely more on better long-context handling.
5. A lot of LLM benchmark and performance progress will come from improved tooling and inference-time scaling rather than from training or the core model itself.
