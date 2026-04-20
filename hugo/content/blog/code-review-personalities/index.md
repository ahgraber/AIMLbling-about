---
title: Code Review Personalities
date: 2026-04-20
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - blogumentation
  - experiment
  - listicle
  # ai/ml
  - agents
  - evals
  - generative AI
  - LLMs
  - prompts
series: []
layout: single
toc: true
math: false
plotly: false
draft: false
---

I've been working on a large refactor in one of my side projects - migrating from a tightly-coupled, hardcoded project barely more than a demo to a pluggable, adapter-based architecture.
I've been using spec-driven development and _helicopter-engineering_ the agents, so I'm fairly confident in the code.
But there have been a _lot_ of changes, so I want to make sure it works, especially with data migrations in play.

I also figured this would be a great opportunity for an experiment to test different model personalities: I asked three different models (GPT-5.4, Gemini 3.1 Pro, and Claude Sonnet 4.6) to first read the specs to understand the change context and then give me a code review for code quality and spec compliance.
All models were run through the GitHub Copilot VSCode extension, so all used the same agent harness; they were all given the same instructions ("review this branch in preparation for merge into main") and used the same [`code-review`](https://github.com/ahgraber/skills/tree/main/skills/code-review) skill.
I had each model dump its report to a markdown file named `alpha`, `bravo`, or `charlie`.
Perhaps unsurprisingly, the models returned different results.
GPT-5.4 identified 2 bugs, both of which would ostensibly be merge-blocking.
Gemini 3.1 Pro identified the same 2 bugs as GPT-5.4 as well as 15 others of varying severity.
Claude Sonnet 4.6 identified potential test gaps and spec compliance issues, but no runtime bugs.

I then opened a new conversation with each model and had each verify all three reviews (`alpha`, `bravo`, or `charlie`) to determine whether the findings were valid.
During this, I also asked the models to assess the reviewers.
After verification, the model congress generally agreed.
GPT-5.4's 2 runtime bugs were deemed valid.
Gemini 3.1 Pro did identify these same bugs, but all others were deemed invalid (not bugs) or were misclassified (Gemini identified "high" and "medium" when they should have been "low" or "informational").
Claude Sonnet 4.6 found legitimate issues in test gaps and spec compliance but missed the two runtime bugs.

All models agreed with their assessments of each other.
GPT-5.4 had highest precision, but appears to have stopped after the two it was sure about.
Claude Sonnet 4.6 cast the widest net (following the code review skill the best) but missed the runtime bugs due to context loss when delegating to subagents: _"Missed both runtime correctness bugs because I delegated the implementation checks to subagents with wide scope rather than tracing specific code paths end-to-end"_.
Gemini 3.1 Pro had the worst signal-to-noise -- it _did_ identify bugs, but it also returned factually wrong items.
When I investigated, Gemini admitted to not having verified its findings!

I liked Sonnet's summary of the code reviews:

> **Alpha** (GPT-5.4): High signal-to-noise. Found only the two genuine blockers, reported them with precise line references, and stopped. No false positives.
>
> **Beta** (Gemini 3.1 Pro): Found the same two blockers but also raised five additional HIGH/CRITICAL issues, three of which are factually wrong (issues 4, 5, 8 — verifiable by reading two files each). This pattern — correct on the high-profile items, noisy on supplementary claims — suggests the reviewer generated plausible-sounding concerns rather than verifying each against code.
>
> **Charlie** (Claude Sonnet 4.6): Focused on spec conformance, test coverage gaps, and hardening requirements. Good signal on test hygiene and wiring/settings concerns; weak on path-level logic bugs.

The findings in this n=1 experiment echo the current prevailing opinion from other blogs, podcasts, and [StrongDM's Weather Report](https://factory.strongdm.ai/weather-report).
GPT-5.4 does seem to be the most precise (both in identifying valid bugs and in instruction-following execution), while Claude models have "broader" personalities that encourage more directional delegation.
I mentioned in [How I Use AI (Apr, 2026)](https://aimlbling-about.ninerealmlabs.com/blog/ai-stack-apr-2026/) that I like playing different models against each other, and this experiment just confirms the utility — no single model had full coverage.
GPT-5.4 caught runtime bugs while Claude Sonnet 4.6 identified test and spec gaps.
For critical tasks like code review, using a model congress makes sense… I just don't know if Gemini deserves a spot in it.
