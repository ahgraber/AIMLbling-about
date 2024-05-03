---
title: No Mo Robo
date: 2024-05-03
tags: [
 'copyright',
 'generative ai',
 'privacy',
 'opinion',
]
series: []
draft: false
---
## Web crawlers & search

How do sites show up in search results? Search engines (among other actors) run web crawlers -- bots that explore websites so the search engines (among others) to catalog and index site contents.
This index helps search engines to provide relevant content when you search.

In the '90s, webmasters noticed that crawlers could place heavy loads on web servers.
They developed `robots.txt`, a de-facto standard for specifying access and authorization to different crawlers.[^1]
This file can specify which portions of the site a robot has access to and how quickly they can crawl, on a per-robot basis.[^2]
Robots.txt allowed a symbiotic relationship to form, where crawlers were permitted access to websites so that the search engines could provide search results, and the search engines in turn directed traffic to the websites.

## Web crawlers & Generative AI

Web crawlers are _also_ used to scrape webpage data to build massive internet-based datasets like [Common Crawl](https://commoncrawl.org/) or [The Pile](https://pile.eleuther.ai/),
containing data from news organizations, blogs, online stores, schools, and businesses are present in the Common Crawl.[^3]
These datasets have become critically important in providing training data for large language models (LLMs) --
in fact, the Kaplan [^4] and Chinchilla [^5] scaling laws define just how important the amount of novel data is for training language models.
These laws provide equations that can estimate model performance based on the size of the model (number of parameters),
the amount of compute used when training (flops - floating point operations per second), _and the size of the dataset_ (number of tokens).
These rules do not apply just to language models.
Indeed, all generative models, including text-to-speech models like Whisper [^6], image-to-text models such as CLiP [^7], and image generation models like DALL-E [^8] could not exist without internet-scale data.
While approaches to optimizing training data such as filtering or using synthetic data have demonstrated efficiencies in dataset size,
the law remains: for more performance, build a bigger model, train on more GPUs, and learn from more data.

Today, the decision about what permissions and limitations to define in `robots.txt`is more complex than ensuring the bots remain polite and that your website can be found on search engines.
If crawlers like `CCBot` from the Common Crawl index your site, your data -- posts, images, intellectual property --
might show up in the training data for a generative model.
And if your data appears in the training set, your content might show up again when the generative AI is in use.

People and organizations that manage websites now have a tough decision to make:

> Is the symbolic relationship still valid?

## A tough choice

Search engines indexing the site were permitted because there was a direct return to the business -- they scrape your site and help direct traffic to it in return.
Generative AI makes this relationship is less clear:
Will the AI provide proper attribution for using your content?
Will it get those attributions correct or will you be a figment of a hallucination?
Will it correctly contextualize your content or will your thoughts be misconstrued?

The New York Times has sued OpenAI and Microsoft for copyright infringement for violating its IP rights.[^9]
How did NYT content get into OpenAI's hands for training ChatGPT? It was included in the Common Crawl.
Since then, The New York Times has worked with Common Crawl to remove its intellectual property [^10] from the Common Crawl Corpus,
and has updated its robots.txt [^11][^12] to allow search engines but block dataset builders like `CCBot` and LLMs, including `anthropic-ai`, `ClaudeBot`, `cohere-ai`, `GPTBot`, and `PerplexityBot`.

{{< callout type="info" >}}
  At its core, I think the underlying issue behind copyright concerns with LLMs is one of _attribution_.  
  I'm planning a separate post on this!
{{< /callout >}}

I've followed the New York Times' lead on this (and their robots.txt).
Search engines should feel welcome to index my blog (👋 Hi! Hello! Welcome!); however, my thoughts are my own and until the fundamental problem of attribution is solved, it is unfair to me to use my IP for commercial gain.
So scraping spiders that build AI datasets and LLM agents are currently unwelcome.

{{< callout type="info" >}}
  That's also why I've chosen the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license for this blog's content.  
  Feel free to share or adapt/remix, just give me credit - and don't make money off of it.
{{< /callout >}}

If you're interested, [here's my robots.txt](https://aimlbling-about.ninerealmlabs.com/robots.txt)

## Footnotes

[^1]: [robots.txt](https://en.wikipedia.org/wiki/Robots.txt)
[^2]: [The Web Robots Pages](https://www.robotstxt.org/)
[^3]: [Statistics of Common Crawl Monthly Archives by commoncrawl](https://commoncrawl.github.io/cc-crawl-statistics/plots/domains)
[^4]: [[2001.08361] Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
[^5]: [[2203.15556] Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
[^6]: [Introducing Whisper | OpenAI](https://openai.com/index/whisper)
[^7]: [CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip)
[^8]: [DALL·E 3 | OpenAI](https://openai.com/index/dall-e-3)
[^9]: [New York Times Sues OpenAI and Microsoft Over Use of Copyrighted Work](https://www.nytimes.com/2023/12/27/business/media/new-york-times-open-ai-microsoft-lawsuit.html)
[^10]: [The New York Times Got Its Content Removed From a Huge AI Training Dataset](https://www.businessinsider.com/new-york-times-content-removed-common-crawl-ai-training-dataset-2023-11)
[^11]: [The New York Times blocks OpenAI's web crawler - The Verge](https://www.theverge.com/2023/8/21/23840705/new-york-times-openai-web-crawler-ai-gpt)
[^12]: [nytimes.com/robots.txt](https://www.nytimes.com/robots.txt)
