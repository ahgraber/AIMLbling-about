---
title: Semantic Kernel Is So Bad
date: 2024-09-13T20:08:55-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - "meta"
  - "blogumentation"
  - "experiment"
  - "listicle"
  - "opinion"
  # ai/ml
  - "generative AI"
  - "prompts"
  - "LLMs"
series: []
layout: single
toc: true
math: false
draft: true
---

- SK competitive space
  - vs MS packages (SK, PromptFlow, AutoGen)
  - vs Langchain, Llamaindex, Haystack, ...
- Benefits
  - automated `planner` functionality allows kernel to act as orchestrator for AI application
  - strong integrations with Azure services, including RAG via Azure AI Search
  - available for not-only-Python (i.e., primarily developed in C#, then replicated to Python and Java)
- But seriously, what the fuck

  - multiple language support means that someone is always ahead/behind; do per-language implementations work the same?!
  - documentation is _terrible_

    - <https://github.com/microsoft/semantic-kernel/discussions/8473>
    - <https://www.reddit.com/r/dotnet/comments/18ux6b7/semantic_kernel/>
    - [Why is language documentation still so terrible?](https://walnut356.github.io/posts/language-documentation/#c)
      - Docs are spread between Microsoft Learn, SK Blog, and Github repo

  - terminology is nonstandard
  - prompt templates -> render -> messages array with assumptions that are footguns
    - 1st text chunk is system message unless system message is provided in history
      - ref:
    - `user: {{$user_input}}` is captured as a user message even if it is the 1st text chunk
  - Impossible to navigate the inheritance chains to figure out which thing does what
    (maybe I'm just a shit dev, but I've never seen inheritance this deep or spread this wide)
  - Limited introspection into translation from prompt template to API call
    - <https://github.com/microsoft/semantic-kernel/discussions/1239>
    - <https://github.com/microsoft/semantic-kernel/discussions/6817>

- But it does work!
