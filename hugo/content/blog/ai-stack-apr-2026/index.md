---
title: How I Use AI (Apr, 2026)
date: 2026-04-05
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - blogumentation
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

AI tools change fast.
This post is intended to snapshot how I use AI today and provide some context for how I arrived here (_This narration is from memory, so the trajectory is right, though the dates may not be_).
I do not anticipate keeping this post up-to-date, though I may revisit the idea with update posts in the future.

{{< details title="Three years of history" >}}

I distinctly remember being quite dubious when ChatGPT launched, and I think I've carried that "doubt, but verify" perspective through my adoption of AI tools.
The first AI-backed tool I remember finding useful was phind.com (now defunct; here's an old [Product Hunt](https://www.producthunt.com/products/phind-com-ai-search-engine) reference).
Phind was miraculous at the time - "good enough" semantic search, more relevant links than Google, and fewer hallucinations than ChatGPT.

Initially, I quite preferred Phind to Github Copilot.
Copilot's original autocompletions were a fun party trick, but I spent more time rewriting than they saved.
Similarly, I found the early Copilot Chat (early 2024) to be only marginally useful.
Over the course of 2024, I continued experimenting with Copilot Chat in VSCode, Cursor, and Windsurf.
In mid-2024, the Copilot-style assistant "ask" mode had largely taken over answering questions about specific codebase-related problems, though I often still used Phind for web search.
It wasn't until late 2024 that I found utility in using Copilot-style assistants "edit" mode for tightly scoped, directed revisions; I'd say I probably still wrote 85-95% of my code and used edit for refinements.

2025 is when AI got good enough that I was willing to pay for it.
By May, I found using Copilot at work had enough utility that I broke down and bought myself a subscription for personal projects.
I missed the early Claude Code boat - I didn't want to pay for a subscription on top of Copilot, and pay-per-use via the API proved heinously expensive (I tried it exactly once).
I started an OpenAI subscription in September of 2025 - initially for Deep Research, then for web-search backed chats.
Like many others, I experienced a step-change in model capabilities when GPT-5.2 and Opus 4.5 were released ([The November 2025 inflection point](https://simonwillison.net/2026/Jan/4/inflection/)), and OpenAI's Codex CLI and VSCode extension changed the way I worked on personal projects - I delegated more and more and wrote less and less.
Earlier this year, I swapped my OpenAI subscription to Anthropic (before the [#QuitGPT](https://quitgpt.org/) campaign, but for similar reasons - don't `@` me).
I've found Claude Opus/Sonnet 4.6 to be similarly performant to GPT-5.3-codex or GPT-5.4.
However, OpenAI are (or at least, were) _considerably_ more generous with what you could get done on their $20/month subscription tier, which means their subscription provides more "intelligence per dollar".

{{< /details >}}

## AI Stack

Today, I still have my Github Copilot subscription, and I have found value in Anthropic's Max subscription (though I may swap between Pro and Max tiers as my use fluctuates).
I use Claude Code in the CLI for longer-running work, and Claude in VSCode for a souped-up Copilot Chat experience; I use Claude Opus for planning and work that requires careful consideration, and Sonnet for tasking.
I use GPT-5.4 in Github Copilot as an alternative perspective (sometimes I'll play the models off of each other), and as my primary code review partner.
Relatedly, I've found the [Weather Report | StrongDM Software Factory](https://factory.strongdm.ai/weather-report) to be a pretty good vibecheck for identifying model strengths.

### Addons

Conveniently, the industry has aligned a surprising amount on tooling, so I can generally use very similar setups across providers (the exception being that Anthropic refuses to adopt `AGENTS.md` in favor of their `CLAUDE.md` 🙄).

I use only three MCP servers: [Context7](https://context7.com/docs/resources/all-clients) for up-to-date documentation, [exa](https://exa.ai/docs/reference/exa-mcp) for web search and retrieval, and [jina](https://github.com/jina-ai/MCP) for web fetch (web-to-markdown).
I use three plugins: an [LSP](https://karanbansal.in/blog/claude-code-lsp/) for whatever language I'm using (Python), [code review graph](https://github.com/tirth8205/code-review-graph) for repo tree-sitter searching and analytics, and [superpowers](https://github.com/obra/superpowers), primarily for "brainstorming" and lightweight spec-driven development.
I've also ~~ripped off~~ incorporated several of Jesse Vincent's ideas in some of [my skills](https://github.com/ahgraber/skills) (I'm particularly partial to his [Using GraphViz for CLAUDE.md](https://blog.fsck.com/2025/09/29/using-graphviz-for-claudemd/)).

> [!TIP]
> I can't emphasize enough how powerful skills are -- which is why I've been iterating on [my own library that extends and customizes existing skills for my own preferences](https://github.com/ahgraber/skills).
> I have skillsets for Python best practices, two for spec-driven development (`sdd` extends [openspec](https://openspec.dev/) while `spec-kit` just reframes [spec-kit](https://speckit.org/) as a skill), and a handful of others.
> It's hard to tell how much of what I now associate with the November Inflection was the models (clearly, the models were a step-change) and how much was how I now use skills.

While I prefer to unify my configuration when possible, I do use some tool-specific customizations!
I've customized my Claude Code setup with an eye on having a more secure sandbox by default, primarily using Claude hooks ([How I Use AI (Apr 2026) - Hooks](https://gist.github.com/ahgraber/2efa040f9ba8d15a6e0da7712105dbf3)).
And I have retained some Github Copilot prompts ([How I Use AI (Apr 2026) - Prompts](https://gist.github.com/ahgraber/48140ba2396b87dafc6ed474ab47c779)) for quick, in-IDE proofreading and refining of markdown prose.
I also have a standard instruction for generating alt-text.
I do a lot of writing in VSCode, and these make it really quick to gut-check my grammar and rubber-duck whether my writing is coherent or has leaps in logic.

## Using AI

### Ways of Working

I've transitioned from quite human-in-loop (granular direction with close, step-by-step supervision) to now fairly human-on-loop (broader direction with reviews at task completion).
Regardless of process, I've found that precisely defining my task and expected outcome is critical.
If I get lazy, I'll end up fighting the LLM to do what I want by the end of the session.
For simple tasks, I'll usually be able to define exactly what I want and have the LLM one-shot it.
For more complex tasks, whether they're larger, longer-running, or tasks where I have less familiarity, I'll usually work with the LLM to iteratively define the request and expected outcome.

> [!TIP]
> In my experience, you don't necessarily need to switch between "plan" and "implementation" modes, but you _do_ need to clear your context window between phases.
> Context management is critical.
> I'll often have the LLM write a handoff document (I even make a [skill](https://github.com/ahgraber/skills/tree/main/skills/handoff) for it) and have that be the only thing I pass between planning and execution phases.

I've also found great acceleration in leveraging Agentic Deep Research tools; using the same iterative plan -> handoff -> execution approach tends to provide better results.
For complex topics, I may instead trigger a handful of less-specified deep research requests and then merge them with standard web search into a final research report.
I feel like I get broader exposure and more comprehensive research this way.

With more formalization, plan -> handoff -> execution becomes [spec-driven development](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html), where the spec is written first, and AI code generation follows to comply with the spec.
I was introduced to [spec-kit](https://speckit.org/) at work, but found it to be quite cumbersome; I prefer [openspec](https://openspec.dev/)'s more flexible model (I also extended openspec into my own [sdd](https://github.com/ahgraber/skills/tree/main/skills/sdd) family of skills).

I do allow LLMs to write my tests (though I fully understand why this is controversial).
I stay tighter in-the-loop in this part of the process, and I've started to leverage fuzzing tests more to hopefully make it more difficult for the LLMs to write _passing_ tests instead of _effective_ tests.
I will also swap model families between code implementation and test implementation within the same SDD workflow (Claude implements, GPT tests).

> [!TIP]
> Important instructions during test generation:
>
> - Do not change source code, you are only writing tests.
> - Tests are designed to identify errors in code logic and correctness -- do not write tests just so that they pass -- a failing test is a valuable output.
> - When developing tests, infer the **intended** behavior from specs, names, docstrings, type hints, usage patterns, and existing tests.
>   If the source code's intent is unclear or underspecified, ask targeted questions **before** making assumptions that would lock in incorrect behavior.
>
> Some key questions to ask your LLM coding assistant:
>
> - "Will these tests validate that the spec was followed?"
> - "How do these tests ensure contract correctness in the case of underspecified specs?
> - "Will these tests actually identify possible issues or do they just make me feel good that tests pass?"

### Non-coding Use

My primary use case for AI is in software development work.
I find considerable value in agentic deep research.
Finally, I find LLMs to be useful as thought partners and debate foils in the more standard chat interface.
I've used them both to strengthen arguments and to talk me down from when I've been (overly) fired up about some irritating situation.
LLMs are more likely to affirm than to correct, so I leverage this behavior intentionally.
I'll take a strong stance (often a more extreme position than my actual opinion) and tell the LLM to _"find supporting evidence - or convince me that I'm wrong"_; sycophancy usually wins, and I get an affirming answer.
Then, in a new session, I'll provide that response as a received argument and ask for a counter-opinion: _"I just received this critique/report._
_I'm not sure about it._
_Provide a rebuttal."_
The second session has no stake in agreeing with the initial stance, but still wants to be helpful, so it acts as a perfect foil to the affirming viewpoint.

Beyond these, I don't really use other AI tools.
Microsoft Copilot (the [ridiculous](https://idiallo.com/blog/what-is-copilot-exactly), [expansive](https://teybannerman.com/strategy/2026/03/31/how-many-microsoft-copilot-are-there.html) product family trying to leverage the success of the Github Copilot brand recognition) clearly has a branding problem; as a user I have no idea what experience and integration I'll get.
Early experiments with any of the non-Github Copilots have been … let's say, _unimpressive_, and I have little desire or need to continue to use it.
My first experience with NotebookLM left me similarly unimpressed.
A colleague recommended I use it to improve my early [AI Treadmill](https://aimlbling-about.ninerealmlabs.com/treadmill/) newsletters; however, at the time, limits on the number of files that could be attached made it incompatible with this use case.
I'm currently revisiting NotebookLM, but find - for my general use case - that the standard chat interface with web search enabled and local documents attached is sufficient.
This NotebookLM experience illustrates a trend in the industry: standard chat modes are steadily sherlocking the features that once justified more focused products.
While they stay a half-step behind, their rapid adoption of specialist features makes the case for switching a matter of microoptimization.

---

## Appendix I: Acknowledging Issues with AI Use

While I use AI, I'm fully aware of the problems surrounding it.
I acknowledge there are ethical issues due to being trained on stolen work or content acquired with questionable consent.
I also recognize the environmental concerns (energy and water use) associated with datacenter inference, and the embodied pollution associated with the complete semiconductor lifecycle.

Three years into my career working with GenAI and I'm still not sure how to reconcile these.
These concerns cannot be dismissed with an "AI is already out in the world" handwave; while true, the issues are valid.
Yet it's also true that AI _is_ already out in the world, and the economic pressure to use it means that we can't close Pandora's Box.
Case in point -- it's quite literally my job to be an expert in how AI works and how to leverage it.

On the ethics front, there needs to be a reckoning where consenting sources (_Note: consenting to be a source, not consenting to get paid_) get a cut of the profit a model provides, akin to how streaming music services pay artists which, while exploitative, still provides acknowledgement of and income from use.
Beyond that, I anticipate that IP and copyright law will catch up to protect content producers through dual avenues of court precedent and enacted legislation.

Regarding emissions and water use, some back-of-the-envelope math seems to show that it's better to use token-as-a-service APIs rather than run models on local servers (depending on utilization).
Working through this is a blog post unto itself, so I'll defer the complete discussion to the future.
The TLDR: running local inference is less efficient than using the API service due to low utilization and limited batch capability, _and_ you also take a hit on model capability.
We can only try to use AI less, use the smallest models capable of completing your task, and choose the most efficient sources for token generation.

---

## Appendix II: Building with AI

Some anecdotes from my experience in building AI products (both professional and personal):

### Frustrations with Inference APIs

The lack of a singular, standard inference API spec is hugely frustrating.
OpenAI, Anthropic, and Google all have different specs, "OpenAI-compatible" covers a wide range of ChatCompletions-likes, and OpenResponses is still not really a thing yet AFAICT.
Given how quickly the industry aligned on MCP and Agents, it's surprising that no one has made a _good_ "convert API _A_ to API _B_" adapter or unified but low-level standardization framework without also including overly-opinionated abstraction.

Trying to identify a viable standardization framework outright killed my motivation on a hobby project ([ahgraber/yaaal: Yet Another AI Agent Library](https://github.com/ahgraber/yaaal)) -- I had some fairly good ideas to refine the existing paradigm (early subagent composability by exposing agents as tools, and split response validation/handling for retry and repair) but couldn't get past the lack of a good adapter abstraction.
Early adapter layers were hard to use (early [LiteLLM](https://github.com/BerriAI/litellm) and [Langchain](https://github.com/langchain-ai/langchain) were dumpsterfires, though I believe both have improved since - especially Langchain); Andrew Ng's [aisuite](https://github.com/andrewyng/aisuite) was a flash-in-the-pan empty promise.
Simon Willison's [llm](https://github.com/simonw/LLM) was (and may still be) the most promising, but its primary focus is being a CLI tool, and integrating against its opinionated abstractions was frustrating.
I've found [pydantic-ai](https://github.com/pydantic/pydantic-ai) and [instructor](https://github.com/567-labs/instructor) to be fairly good, but they're both still quite opinionated and operate above the adapter layer.

As far as experiences with API providers go -- OpenAI's switch from Completions to ChatCompletions to Responses has been confusing, Anthropic is expensive (over the API) and frustratingly _unique_, and TogetherAI's billing and usage tracking was nonexistent/opaque when I used it (which is why I stopped).
[OpenRouter](https://openrouter.ai/), however, is great!
Strong recommend.

### Frustrations with Frameworks

Similarly, early frameworks seemed to suffer, though the utility and developer experience has improved as frameworks have evolved; my mental map organizes these over the course of three rough generations.
Langchain was difficult to extend beyond "here's a demo I made with Langchain".
Microsoft's [Semantic Kernel](https://github.com/microsoft/semantic-kernel) has been the bane of my existence at work - nonstandard abstractions, overengineered complexity, feature churn without clear direction or documentation, and lagging promises for feature parity across languages (Python, Java).
These first-gen frameworks provided abstractions over the chat completion (call & response with LLM) but required complex handbuilt engineering for composition and orchestration; tool-use was an afterthought.
"Second-gen" frameworks like [LangGraph](https://github.com/langchain-ai/langgraph), CrewAI, and [Pydantic AI](https://github.com/pydantic/pydantic-ai), etc., treated "Agents" as the primary metaphor and improved modularity and composition (making the developer experience much nicer).
These have since evolved into "Third-gen" SDKs for building [Agent Harnesses](https://aimlbling-about.ninerealmlabs.com/treadmill/2026/03/10/#heading), incorporating features like additional runtime support (retries, state, orchestration, tool use policies and governance, observability) and improved tools, context management, and memory.
Third-gen frameworks have much-improved developer experience, and are functional for use in production-grade applications.
I actively enjoy working with Pydantic AI, LangChain's [deepagents](https://github.com/langchain-ai/deepagents) appears very well thought out, and I have high hopes for Microsoft's new [Agent Framework](https://github.com/microsoft/agent-framework) (though it does seem to be suffering from similar feature churn that Semantic Kernel exhibited).
