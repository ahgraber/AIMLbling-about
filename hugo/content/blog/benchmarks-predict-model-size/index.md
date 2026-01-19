---
title: Predicting LLM Parameters Using Benchmarks
date: 2026-01-08T19:51:34-05:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - experiment
  # ai/ml
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

In a recent episode of the Latent Space podcast ([Artificial Analysis: The Independent LLM Analysis House — with George Cameron and Micah Hill-Smith](https://youtu.be/v5mBjeX4TJ8?si=ViYbKgD6x7fkjEa4)), the Artificial Analysis team pointed out a strong correlation between model performance on their AA-Omniscience Accuracy benchmark and the model's parameter count:

> An interesting thing about this accuracy metric is that it tracks more closely than anything else that we measure the total parameter count of models...
> If you draw the line on AA-Omniscience accuracy vs total parameters, ... you can see that likely the leading front-end models right now are quite a lot bigger than the 1 trillion parameters that the open-weights models cap out at...
> There's an interesting extra data point that Elon Musk revealed recently about xAI that Grok 3 and 4 - 3 trillion parameters for Grok 3 and 4, 6 trillion for Grok 5 (but that's not out yet).

Well, _that_ caught my attention, so I had to see for myself.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/AAO_accuracy_vs_params.png"
alt="AA-Omniscience Accuracy appears strongly correlated to (log) model parameter count"
caption="AA-Omniscience Accuracy appears strongly correlated to (log) model parameter count"
link="https://arxiv.org/html/2511.13029v1" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

## Experiment

### Data Collection

My objective is to see if the Omniscience Accuracy benchmark (or other benchmarks) are predictive of parameter counts.
If so, I'll use the benchmark scores to interpolate some projected scores for models like GPT-5.2, Gemini 3 Pro, and Claude Sonnet and Opus 4.5.
To that end, I collected data from Artificial Analysis.
I also collected information about model sizes from HuggingFace, release announcements, or technical papers on arxiv.org, and pricing information from Artificial Analysis and [Simon Wilison's LLM Prices](https://github.com/simonw/llm-prices) project.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure src="images/AAO_accuracy.png" alt="AA-Omniscience Accuracy" link="https://artificialanalysis.ai/?omniscience=omniscience-accuracy&model-filters=frontier-model&models=claude-opus-4-5-thinking%2Cgemini-3-pro%2Cgpt-5-1%2Cgemini-3-flash-reasoning%2Cgpt-5-2-medium%2Cclaude-4-5-sonnet-thinking%2Cglm-4-7%2Cgpt-5-medium%2Cgrok-4%2Cdeepseek-v3-2-reasoning%2Co3%2Cgemini-3-pro-low%2Ckimi-k2-thinking%2Cminimax-m2-1%2Cgpt-5-mini-medium%2Cclaude-4-5-haiku-reasoning%2Cminimax-m2%2Cnova-2-0-pro-reasoning-medium%2Cclaude-3-7-sonnet-thinking%2Cgemini-2-5-pro%2Cgpt-oss-120b%2Cglm-4-6-reasoning%2Cnova-2-0-lite-reasoning-medium%2Cqwen3-235b-a22b-instruct-2507-reasoning%2Cnova-2-0-omni-reasoning-medium%2Cgemini-2-5-flash-reasoning%2Cdeepseek-r1%2Cglm-4.5%2Cgpt-5-nano-medium%2Cgpt-4-1%2Cgrok-3%2Cgpt-oss-20b%2Cnvidia-nemotron-3-nano-30b-a3b-reasoning%2Cgpt-oss-120b-low%2Cqwen3-30b-a3b-2507-reasoning%2Cdeepseek-v3-0324%2Cgpt-oss-20b-low%2Cllama-4-maverick%2Cqwen3-14b-instruct-reasoning%2Cministral-14b%2Cqwen3-30b-a3b-instruct-reasoning%2Cgpt-4o%2Cllama-4-scout%2Ccommand-a%2Cgemma-3-27b%2Cgemma-3-12b%2Cgemma-3-1b%2Cgemma-3-4b%2Colmo-3-1-32b-think#aa-omniscience-accuracy" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

### Modeling

I came up with a number of questions as I was investigating:

1. Can we predict model size based on benchmarks (and if so, which benchmark)?
2. It strikes me that token prices may hint at model sizes (cost to host the model), so is pricing information predictive of model size?
3. Most recent frontier models use a mixture-of-experts architecture; does including sparsity information make the model size prediction more accurate?

I fit 15 different linear regressions across 5 benchmarks: Omniscience Accuracy [^omniscience], MMLU Pro [^mmlu-pro], Artificial Analysis' Intelligence Index [^aaii], Tau² [^tau2], and GDPVal [^gdpval].
Each benchmark was used by itself as a standalone predictor, and in conjunction with pricing or sparsity.
Academic papers tend to define sparsity based on the ratio of inactive experts to total experts [^sparsity] [^sparsing_law].
However, model labs tend to share the number of total vs active parameters, and may not disclose the number of experts or expert architectures.
Therefore, I modeled (roughly) the inverse of sparsity as the ratio of active:total parameters per token.

> [!NOTE]
> 
> - [Omniscience](https://artificialanalysis.ai/evaluations/omniscience) is Artificial Analysis' own benchmark which rewards precise knowledge and penalizes hallucinated responses. Omniscience Accuracy is the "correctness" component of Omniscience, measuring the proportion of correctly answered questions out of all questions, regardless of whether the model chooses to answer. Omniscience also tracks Hallucination Rate (how often the model answers incorrectly when it should have refused or admitted to not knowing the answer) and Attempt Rate.
> - [MMLU Pro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) an enhanced approach to the Massive Multitask Language Understanding (MMLU) benchmark designed to evaluate language understanding, integrating more challenging, reasoning-focused questions and a greater spread of possible response options.
> - [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) is a composition of a composition of 10 different benchmarks, used to compare LLM capabilities across a broad range of use cases. Developed by Artificial Analysis, they run their own independent tests so model performance can be compared apples-to-apples.
> - [Tau²](https://taubench.com/#home) measures an LLM's capability to drive agentic decisions over a variety of real-world scenarios. Tau² simulates user and AI Agent interactions over domain-specific tasks and evaluates success on these tasks.
> - [GDPVal](https://openai.com/index/gdpval/) is an OpenAI-developed benchmark designed to track how well AI models perform on economically valuable, real-world tasks. Tasks were developed and evaluated in partnership with human experts.

#### Metrics

{{< figure src="images/model_fit.png" alt="A three-panel summary chart comparing how well different benchmark metrics predict total model parameters. The top panel shows mean R² values. Metrics based on omniscience_accuracy and mmlu_pro have the highest R² (around 0.75 to 0.84), while intelligenceIndex variants are near zero, and some metrics have negative R², indicating poor fits. The bottom two panes show mean absolute error (MAE) and root mean squared error (RMSE), with similar patterns: very large errors for intelligenceIndex+price and lower errors for mmlu_pro and omniscience_accuracy metrics." >}}

As mentioned in that podcast episode, Omniscience Accuracy indeed is the most predictive (R²=0.84), followed by MMLU Pro (R²=0.75) and trailed by the Intelligence Index (R²=0.07).
(As a reminder, R² is a measure of how much total parameter variance the predictor(s) account for - a "goodness of fit" metric.)
Their errors (mean absolute error (MAE) and root mean squared error (RMSE)) are generally around 200B total parameters - this is not a precise estimator!

Adding pricing information made the regression fit worse in every case; adding pricing information to Intelligence Index in particular caused prediction error to explode.
Active token ratio had no apparent effect on model predictivity.

Tau² and GDPVal have negative R² values, indicating that the benchmarks are not predictive at all.
I found it interesting that benchmarks that test _knowledge_ had the best fit, while benchmarks that test _task performance_ (Tau², GDPVal) were not predictive at all.
This hints at parameter counts being innately tied to model knowledge capacity, while task performance is something that can be improved in post-training [^memorize] [^grokking].

{{% details title="Open for metrics table" closed="true" %}}

| model_name                  |   r2_mean |    mae_mean |   rmse_mean |
| :-------------------------- | --------: | ----------: | ----------: |
| omniscience_accuracy        |  0.840533 | 1.53968e+11 | 2.88892e+11 |
| omniscience_accuracy+active |  0.840533 | 1.53968e+11 | 2.88892e+11 |
| mmlu_pro                    |   0.74596 | 1.74184e+11 | 2.44329e+11 |
| mmlu_pro+active             |   0.74596 | 1.74184e+11 | 2.44329e+11 |
| mmlu_pro+price              |   0.52227 |  1.8834e+11 | 2.32713e+11 |
| intelligence_index          | 0.0704084 | 2.35765e+11 | 3.34859e+11 |
| intelligence_index+active   | 0.0704084 | 2.35765e+11 | 3.34859e+11 |
| tau2                        | -0.170531 | 2.22823e+11 | 3.24995e+11 |
| tau2+active                 | -0.170531 | 2.22823e+11 | 3.24995e+11 |
| omniscience_accuracy+price  | -0.234063 | 1.30512e+12 | 2.60161e+12 |
| gdpval                      |  -0.92488 | 2.32388e+11 | 3.57584e+11 |
| gdpval+active               |  -0.92488 | 2.32388e+11 | 3.57584e+11 |
| intelligence_index+price    |  -1.22534 | 2.76783e+12 | 5.74756e+12 |
| gdpval+price                |  -1.50908 | 4.17928e+11 | 6.31627e+11 |
| tau2+price                  |  -1.54026 | 9.34922e+11 | 1.81894e+12 |

{{% /details %}}

#### Predictions

Given these metrics, we can use Omniscience Accuracy (or MMLU Pro or Intelligence Index) to estimate the size of proprietary models (GPT-5.x, Gemini, Claude Sonnet/Opus).

### Predictions

{{< tabs items="Omniscience Accuracy, MMLU Pro, Intelligence Index" >}}

{{< tab >}}
!["A scatter plot with a regression line showing the relationship between model size and an "omniscience_accuracy" score. The x-axis is "omniscience_accuracy score," ranging from roughly 0.05 to 0.55. The y-axis is "Total parameters (log scale)," ranging from under 1 billion to well over 1 trillion parameters. Models are plotted as labeled points, including GPT-4o, GPT-4.1, GPT-5, GPT-5.1, GPT-5.2, GPT-5 mini, GPT-5 nano, Claude 3.7 Sonnet, Claude 4.5 Haiku, Claude 4.5 Sonnet, Claude Opus 4.5, Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Flash, and Gemini 3 Pro Preview. The regression line slopes steeply upward and closely tracks the data, with an annotated R² of 0.84, indicating a strong relationship between omniscience_accuracy and total parameter count. A color scale shows prediction error magnitude, which is generally small relative to the range shown. The overall message is that omniscience_accuracy is a strong predictor of model size, with higher scores corresponding to much larger models."](images/omniscience_accuracy.png)
{{< /tab >}}
{{< tab >}}
!["A scatter plot with a regression line showing the relationship between model size and MMLU-Pro benchmark score. The x-axis is "mmlu_pro score," ranging from approximately 0.15 to 0.9. The y-axis is "Total parameters (log scale)," ranging from a few hundred million to over 1 trillion parameters. Each model is shown as a point, with circles for actual values and triangles for regression predictions, and labeled with model names including GPT-4.1, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Claude 4.5 Sonnet, Claude 4.5 Haiku, Claude Opus 4.5, Gemini 2.5 Pro, Gemini 3 Pro, and Gemini 3 Flash. The regression line slopes upward and closely follows the data. The annotation reports a relatively strong fit (R² = 0.75), indicating that MMLU-Pro score correlates well with total parameter count. Prediction errors, shown by a color scale, are generally smaller than in the intelligenceIndex plot. Overall, the chart shows that higher MMLU-Pro scores tend to correspond to larger models."](images/mmlu_pro.png)
{{< /tab >}}
{{< tab >}}
!["A scatter plot with a fitted regression line showing the relationship between large language model size and an "intelligenceIndex" score. The x-axis is "intelligenceIndex score," ranging roughly from 5 to 50. The y-axis is "Total parameters (log scale)," ranging from about 1 billion to over 5 trillion parameters. Each model appears as a point, with circles representing actual parameter counts and triangles representing values predicted by the regression. Points are labeled with model names such as GPT-5, GPT-5 mini, Claude 4.5 Opus, Claude 4.5 Sonnet, Gemini 3 Pro Preview, Gemini 2.5 Pro, and GPT-5 nano. A straight regression line slopes upward, but the annotation reports a low fit quality (R² = 0.07), indicating that intelligenceIndex explains little of the variance in total parameters. Many gray points are widely scattered vertically, showing large differences in parameter counts for similar intelligenceIndex scores. A color scale labeled "Abs. Error (log10)" indicates prediction error magnitude, with warmer colors showing larger errors. Overall, the chart emphasizes that intelligenceIndex is a weak predictor of total model size."](images/intelligence_index.png)
{{< /tab >}}

{{< /tabs >}}

Interestingly, while Omniscience Accuracy has the best fit metrics (R², MAE, RMSE), it seems to have the least realistic predictions for the set of proprietary models.
The Omniscience Accuracy regression suggests that Gemini 3 Pro (preview) has a total parameter count of 1,254T (yes, T!) tokens, with GPT-5.2 at 43T and Claude Opus 4.5 at 22T.
I find this to be completely unrealistic; I think it would be quite infeasible to effectively serve models that size.
Further, although pricing was not predictive in the regression, I find it hard to believe that Gemini 3 Pro could be priced competitively were it that large.

When colored by my expectations, the Intelligence Index regression seems to provide the most realistic predictions - Gemini 3 Pro (preview) at 3.4T, Claude 4.5 Sonnet at 1.4T, and Claude 4.5 Opus at 4.1T.
It predicts different parameter counts for GPT-5.1 and GPT-5.2, although I assume they use the same architecture, placing the GPT-5.x series between 2.9-5.3T total parameters.
Using this model, GPT-5 mini is estimated at 1T total parameters, GPT-5 nano at 100B, and Claude 4.5 Haiku at 520B.
All of these are roughly in line with my personal "vibe checks" on model capability, especially as code assistants.

{{% details title="Open for full parameters table" closed="true" %}}

| llm                  | omniscience_accuracy | mmlu_pro | intelligence_index |
| :------------------- | :------------------- | :------- | :----------------- |
| Claude 3.7 Sonnet    | 847.29B              | 168.80B  | nan                |
| Claude 4.5 Haiku     | 20.71B               | 163.98B  | 520.73B            |
| Claude 4.5 Sonnet    | 835.78B              | 292.51B  | 1.37T              |
| Claude Opus 4.5      | 22.13T               | 386.93B  | 4.16T              |
| GPT-4.1              | 662.29B              | 173.75B  | nan                |
| GPT-4o               | 87.14B               | 168.80B  | nan                |
| GPT-5                | 20.50T               | 325.26B  | nan                |
| GPT-5 mini           | 282.73B              | 234.31B  | 1.01T              |
| GPT-5 nano           | 78.32B               | 135.21B  | 100.18B            |
| GPT-5.1              | 8.26T                | 322.13B  | 2.94T              |
| GPT-5.2              | 43.27T               | 334.81B  | 5.34T              |
| Gemini 2.5 Flash     | 608.42B              | 232.06B  | nan                |
| Gemini 2.5 Pro       | 15.00T               | 298.21B  | 342.86B            |
| Gemini 3 Flash       | 211.66T              | 361.66B  | nan                |
| Gemini 3 Pro Preview | 1254.02T             | 422.02B  | 3.42T              |

{{% /details %}}

Finally, I did do some experiments with the active token ratio as an additional predictive feature.
As the metrics suggest, active token ratio has no bearing on predicted total parameters.

## Conclusion

In the end, does estimating total parameter count matter?
Not really; it's merely one factor among many that may contribute to model performance.
This is quite evident in the lack of predictivity that task capability benchmarks like Tau² and GDPVal have on model size.
That said, I find it useful to understand the relationships between model sizes and architectures and their performance metrics.
And I had fun trying to pierce the proprietary labs' veil of secrecy!

As Swyx said:

> What does it really matter? As long as they can serve it at a sustainable cost, that's about it.

---

> [!NOTE]
> **Disclosure of AI Assistance**
>
> I used ChatGPT web search and Deep Research to help with the data gathering for this experiment, especially when identifying model specs and associated sources.
> _I manually validated every spec and source._\
> I used coding agents (GitHub Copilot, OpenAI Codex) to accelerate scraping benchmark and pricing data, and to speed up defining the plots.
> I also used them for code review.
> _I reviewed and revised all AI-generated code._\
> _**None** of the blog post itself was drafted or generated by AI tools._

---

## References

[^omniscience]: [[2511.13029] AA-Omniscience: Evaluating Cross-Domain Knowledge Reliability in Large Language Models](https://arxiv.org/abs/2511.13029)

[^mmlu-pro]: [[2406.01574] MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574)

[^aaii]: [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

[^tau2]: [[2506.07982] $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982)

[^gdpval]: [[2510.04374] GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks](https://arxiv.org/abs/2510.04374)

[^sparsity]: [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models - 2501.12370v3.pdf](https://export.arxiv.org/pdf/2501.12370)

[^sparsing_law]: [Sparsing Law: Towards Large Language Models with Greater Activation Sparsity - 2411.02335v4.pdf](https://export.arxiv.org/pdf/2411.02335)

[^memorize]: [[2505.24832] How much do language models memorize?](https://arxiv.org/abs/2505.24832)

[^grokking]: [[2506.21551] Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](https://arxiv.org/abs/2506.21551)
