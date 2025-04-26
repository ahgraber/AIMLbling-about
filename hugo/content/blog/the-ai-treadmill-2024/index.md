---
title: The AI Treadmill (2024)
date: 2024-12-31
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - listicle
  - opinion
  # ai/ml
  - agents
  - arxiv
  - evals
  - generative AI
  - LLMs
  - prompts
  - RAG
series: []
layout: single
toc: true
math: false
draft: false
---

At the end of 2023, I was transferred to the team working on [PMI Infinity](https://www.pmi.org/infinity), an AI-powered tool and product for Project Managers,
to act as an AI/ML Engineer and provide expertise on the AI/ML aspects of developing a product based on LLM systems.
One of the first initiatives I took on was to set up an newsletter with the goal of sharing interesting and applicable research, demos, packages, and conversations with the team.
I called it the _AI Treadmill_, because trying to keep up with everything felt like running on a treadmill cranked all the way up (and still kind of does).
By the end of Q1, I'd reached a pretty good cadence, sending newsletters on Tuesdays and Fridays broken into to general segments -
"News", which could be updates from social media (X/Twitter, LinkedIn, HackerNews, etc.), links to project/package homepages, blogs, news articles, etc.,
and "Papers", which were more academic whitepapers, generally through [arxiv.org](https://arxiv.org/list/cs.AI/recent).
I tried to curate the list to a reasonable number of links, highlight particularly relevant/recommended reads,
and sometimes provide high-level summaries to help my team decide whether they also wanted to spend the time clicking through the link.

## AI Treadmill, 2024

I present the consolidated list of links I found relevant or interesting enough to share with our internal team.

**By the numbers** <!-- markdownlint-disable-line MD036 -->

In 2024, the AI Treadmill comprised ~100 emails linking to ~2000 different articles, posts, code repos, documentation sites, or whitepapers.
38% (757) of these are papers hosted on [arxiv.org](https://arxiv.org); 85 are social media posts (LinkedIn, Twitter/X), 154 are Github repos, 53 are discussions on HackerNews, and at least 50 are hosted on Medium or Substack.

[**Here**](./treadmill_2024.csv) are all of the links, in (roughly) temporal order of inclusion in AI Treadmill newsletter (deduplication may have thrown off the order here and there)

## Thoughts / Lessons Learned

1. Curation is time-consuming.

   I probably spend about 4-6 hours per newsletter seeking, reading (skimming), and summarizing content.
   This time is not necessarily tracked as part of my standard day-to-day.

2. Filtering the funnel input helps.

   I try to curate my sources to both cast a wide net of perspectives but also not have to crawl every new paper that is submitted to arxiv.org.
   This means that the AI Treadmill _does_ exist in a filter bubble (both from my sourcing patterns and my curation itself).

   Curation tools and communities that help with sourcing:

   - [HackerNews](https://news.ycombinator.com)
   - [Daily Papers - Hugging Face](https://huggingface.co/papers)
   - [EmergentMind](https://www.emergentmind.com)
   - [AI Papers | Explore Latest Research](https://www.aimodels.fyi/papers?search=&selectedTimeRange=thisWeek&page=1)
   - [AI News | Buttondown Newsletter](https://buttondown.com/ainews)
   - [Community Resources | Oxen.ai](https://www.oxen.ai/community)

   Some good folks to follow (listing by name; you can follow in LinkedIn, X/Twitter, Bluesky, wherever)

   - [Eugene Yan](https://eugeneyan.com/)
   - [Chip Huyen](https://huyenchip.com/)
   - [Clem Delangue](https://huggingface.co/clem)
   - [Philipp Schmid](https://www.philschmid.de/)
   - [Niels Rogge](https://huggingface.co/nielsr)
   - [Lilian Weng](https://lilianweng.github.io/)
   - [Simon Willison](https://simonwillison.net/)
   - [Paul Iusztin](https://www.pauliusztin.me/)
   - [Ethan Mollick](https://mgmt.wharton.upenn.edu/profile/emollick/)
   - [Andrew Ng](https://www.andrewng.org/)
   - [Hamel Husain](https://hamel.dev/)
   - [Vicki Boykis](https://vickiboykis.com/)

3. I have no idea how successful the newsletter is (anecdotally I know it is appreciated, but I have no tracking on what the open/clickthrough rates are).

   I have no idea whether newsletter recipients are even opening the emails, much less than which links are being clicked.
   It doesn't particularly matter, because I'm just sharing the work I'd be doing anyway for my own edification, but it would be _interesting_ to have some stats.

4. Email lists are not super searchable, which limits the utility of using old emails as a reference.

   I reply-all to my email to create an email thread per month; the idea is to keep clutter down in inboxes since most email apps now thread conversations.
   Unfortunately, searching threads for prior links is kind of a pain (Microsoft, make threaded email search better!).
   As a result, I had a fairly manual procedure to recapture, process, and clean the list of links for this post.
   Moving forward, I'll be replicating my notes in a more searchable format -- even better, I might try to create my own NotebookLM-style app...
