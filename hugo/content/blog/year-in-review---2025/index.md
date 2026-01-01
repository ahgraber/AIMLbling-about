---
title: Year in Review - 2025
date: 2025-12-23
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
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: false
---

[Back in January, I made a series of predictions for 2025]({{< ref "/blog/predictions-2025" >}}), assigning somewhat arbitrary probability estimates to each as indicators of my confidence in the prediction.
Now that the year is wrapping up, it's time to see what kind of AI Nostradamus I am.

## Agents, agents, agents

**Prediction:** We will see an even stronger push for Agents in 2025.\
**Probability:** 100%

### Verdict: Correct ✅

This was a gimme, but I got it right (I gotta take the easy wins!).
Agents have absolutely dominated the landscape this year.
We've seen the rise of "Deep Research" capabilities across the board, the explosion of Agentic frameworks (LangChain, LangGraph, Semantic Kernel, AutoGen / AG2, CrewAI, PydanticAI, etc.), and the adoption of the Model Context Protocol.
Coding agents like Claude Code, OpenAI Codex, and Gemini CLI have become indispensable tools.
On top of that, Agent Engineering has grown out of Context Engineering as a requirement to improve agentic performance on long-running tasks.

## Inference-time compute

**Prediction:** We will see more new models using inference-time compute, pushing the boundary toward superhuman performance.\
**Probability:** 100%

### Verdict: Correct ✅

This was another easy win.
The terminology has shifted from "System 1 vs.
System 2" to simply "Thinking" models vs. "Response" models.
The data is clear: thinking models have taken over the leaderboard.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/artificial_analysis_reasoning.png"
alt="Every single top model in terms of performance at the end of 2025 is a reasoning model"
link="https://artificialanalysis.ai/?intelligence-category=reasoning-vs-non-reasoning#artificial-analysis-intelligence-index-by-model-type"
caption="Every single top model in terms of performance at the end of 2025 is a reasoning model | Artificial Analysis" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

Every single top model in terms of performance at the end of 2025 is a reasoning model.
Response models (the call-and-response format that took off with GPT-3.5 and powered the GPT-4 family) basically don't play at the frontier anymore.
We're seeing responses that use ~3x more tokens on average due to this reasoning process (the tokens are generated but typically hidden from the user; the user receives a more polished final answer).
Models like o1-Pro, o3, and DeepSeek R1 all proved the performance benefit was worth the compute, and reasoning is now the standard paradigm.

## 50/50: Large frontier model performance

**Prediction:** Either (a) improvement curves will slow and be driven by post-training, or (b) we will see major architectural shifts (tokenization, attention, etc.) to drive performance.\
**Probability:** 50/50

### Verdict: Both (Sort of) ✅

I hedged my bets here, and interestingly, pieces of both outcomes came true.

On one hand, benchmark performance _did_ appear to slow or saturate.
I would argue that for the most part the benchmarks that we had in 2023 are absolutely saturated.
The benchmarks that were developed in 2024 are probably getting to the point of saturation and so it's really hard to actually measure whether the perception of performance stalling is the reality, or whether it's just that we don't have tests that are advanced enough to prove that out.
However, the improvements we _did_ see were largely driven by post-training recipes (RL), just as predicted in option (a).

<!-- markdownlint-disable MD013 MD033 MD034 -->

<table>
<tr>
  <td style="width:50%">{{< figure
    src="images/saturation.png"
    alt="Benchmark improvement appeared to slow as models saturated them"
    link="https://ourworldindata.org/grapher/test-scores-ai-capabilities-relative-human-performance?country=Handwriting+recognition~Image+recognition~Reading+comprehension~Language+understanding~Predictive+reasoning~Code+generation~Complex+reasoning~Math+problem-solving~Speech+recognition"
    caption="Benchmark improvement appeared to slow as models saturated them.">}}
  </td>
    <td style="width:50%">{{< figure
    src="images/olmo3_rl.png"
    alt="Olmo 3 demonstrates the gains from reinforcement learning"
    link="https://allenai.org/blog/olmo3"
    caption="Olmo 3 demonstrates the gains from reinforcement learning | Ai2" >}}
  </td>
</tr>
</table>

<!-- markdownlint-enable MD013 MD033 MD034 -->

I also predicted that US labs would stick to the status quo while Chinese labs experimented.
This largely played out, with Chinese labs (DeepSeek, Qwen) pushing efficiency innovations while Western labs focused on scaling existing recipes.

### Next-gen Architectural changes

**Prediction:** In order to continue driving performance improvements (without leveraging inference-time compute), model architectures undergo another period of diversification.
These architectural changes may come from improvements to tokenization, attention, training objectives (multitoken prediction, JEPA embedding prediction), or sampling.\
**Probability:** 75%

#### Verdict: Miss ❌

I expected changes to model architecture to drive performance improvements as opposed to training pipeline improvements (e.g. reinforcement learning and RL-like distillation).
We _did_ see architectural changes, specifically the rise of Mixture of Experts (MoE) and aggressive quantization, but these were driven by _inference efficiency_, not raw benchmark performance.
Linear Attention or Hybrid Attention is also gaining popularity, but no standard recipe has emerged.

[Inception Labs have a text diffusion model called Mercury](https://www.inceptionlabs.ai/blog/mercury-refreshed), which seems really, really interesting, but text diffusion models haven't really made it in terms of market share or mind share.
And we are starting to see multi-token prediction, but it seems to be mostly a research objective rather than something that's used by models in production.

I was also really hoping that we would see modern makeovers of older architectures.
ModernBERT did this at the end of 2024, rebuilding BERT (2018) using modern architectures and saw a fairly significant performance lift.
I was hoping we'd see that with other, older models and architectures but that didn't really play out.
Perhaps the only other example is Google adopting Gemma to the T5 architecture (2019) - twice, actually ([T5Gemma](https://developers.googleblog.com/en/t5gemma/) and [T5Gemma 2](https://blog.google/technology/developers/t5gemma-2/)).
This converted Gemma, a decoder / next-token prediction model to T5, an encoder-decoder architecture.

## Multimodal-first training

**Prediction:** Adoption of a unified architecture that takes multimodal inputs directly, replacing separate encoders/adapters.\
**Probability:** 80%

### Verdict: Miss ❌

I was fairly naive here.
I expected a unified "omnimodal" architecture to replace the "LLM + Adapter" paradigm.
While we have seen models with better out-of-the-box multimodal capabilities (like Gemini 3 Pro), the industry hasn't coalesced around a single unified architecture.
And to be honest, I think the way I phrased it as _"unified architecture that takes multimodal inputs directly"_ is quite unlikely unless bytestreams are used instead of tokenization.

Even models branded as "omni," like GPT-4o, still rely on adapters and tuned variants for specific modalities (like speech-to-text) rather than a single, pure end-to-end model.
The "embedding layer + adapter" approach that unifies an internal embedding space remains the standard.

## Lightning Round Review

- **Saturation and/or data leakage will invalidate current benchmarks / spur new ones (100%):** _Correct_ ✅\
  Data leakage and saturation have invalidated many benchmarks.
- **Continued push to undermine NVIDIA's market dominance in hardware but successes (market share/utilization) will come from cloud providers (Amazon, Google, Azure, Groq(?)) and not Intel or AMD (70%):** _Correct_ ✅\
  The pressure came from Hyperscalers (Amazon, Google, Azure) and their custom silicon (TPUs, Trainium) and a strong push from AMD, rather than Intel.
- **At least one lab will claim "AGI" but their definition will have multiple asterisks (60%):** _Miss_ ❌\
  No one has officially claimed AGI yet, though there's been a lot of debate around AGI, primarily on OpenAI's behalf in order to allow it to restructure to a for-profit public benefit corporation from a non-profit.
- **At least one lab will inject ads into their (non-search) free or low-cost chat interface or service (60%):** _TBD_ ⚠️\
  Stay tuned - there are imminent signs (OpenAI hires, mention in the "Code Red" vs Gemini 3 Pro), but we haven't seen the full-blown "ads in your chat" rollout just yet.
- **Power (and water) utilization concerns over AI inference costs enter the zeitgeist (50%):** _Correct_ ✅\
  Power and water usage have indeed entered the public consciousness, with considerable discussion in popular media and politics.
  - **... with AI datacenters associated with brownouts or wildfires (30%):** _No Evidence_ ❌\
    Thankfully, I haven't seen reports of AI-induced wildfires yet.
