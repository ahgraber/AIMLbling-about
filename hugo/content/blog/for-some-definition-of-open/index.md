---
title: For Some Definition of 'Open'
date: 2024-12-05
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - opinion
  # ai/ml
  - generative AI
  - LLMs
  # other
  - copyright
  - creators
  - licensing
series: []
layout: single
toc: true
math: false
draft: false
---

While I generally attempt to avoid pendantry, I am acutely aware that word choice -- and the specific connotations and denotations of the selected terms -- matter.
Therefore, I strive to be specific in the language I use, and it niggles my noodle a bit when words or phrases are used incorrectly.
However, as a Libra, I am apparently celestially obligated to desire balance and equilibrium, and thus also see all sides of an issue.
When I hear "open source AI", I get a bit nitpicky - and you, dear reader, are reading the result of my search for balance.

AI and Machine Learning differ from traditional software engineering in that the utility of an AI/ML product
is defined by training data, the source code required to train a model artifact, the model artifact itself, and the source code required to use the artifact
(whereas traditional software is the product of its source code).
As a result, the traditional definition of Open Source Software (OSS), which is defined regarding the use,
redistribution and derivations of the software source code[^oss] is insufficient to fully define Open Source Artificial Intelligence (OSAI).

## Open Source Artificial Intelligence

Fortunately, the Open Source Initiative released its formal definition of Open Source Artificial Intelligence at the end of October, 2024, stating:

> An Open Source AI is an AI system made available under terms and in a way that grant the freedoms to:
>
> - **Use** the system for any purpose and without having to ask for permission.
> - **Study** how the system works and inspect its components.
> - **Modify** the system for any purpose, including to change its output.
> - **Share** the system for others to use with or without modifications, for any purpose.

Specifically, the OSI notes that the preferred form of OSAI meets these 4 points of openness by including open access to training data, to source code used to train and run the system, and to model weights / the artifact itself.[^osai] [^osai_faq]

The Linux Foundation provides a similar definition for open-source AI, stating:

> Open source artificial intelligence (AI) models enable anyone to reuse and improve an AI model.
> "Open source AI models" include the model architecture (in source code format), model weights and parameters, and information about the data used to train the model that are collectively published under licenses, allowing any recipient,
> without restriction, to use, study, distribute, sell, copy, create derivative works of, and make modifications to, the licensed artifacts or modified versions thereof.[^lf_osai]

Open Core Ventures, a venture capitalist firm, focuses on the particular licensing differences required for AI/ML models.
They recognize a spectrum of "openness" from the proprietary/closed --> open spectrum, and make a distinction between
the "source" and "weights" of the AI system.[^open_definitions]
Unlike the above two definitions, this breakdown does not specify open access to training data (though IMO it would be considered part of the "source").

{{< figure
src="images/ai_licensing.png"
alt="ai licensing"
link="https://opencoreventures.com/blog/2023-06-27-ai-weights-are-not-open-source/"
caption="AI weights are not open 'source' | Open Core Ventures" >}}

## Open weights != Open Source

These definitions of Open Source AI conflict with the current typical use of "open" in the AI space, and the juxtaposition highlights the reason my brain twinges every time I hear "Open Source AI" --
most organizations falsely synonymize "open weight" models as "open source", while retaining license restrictions and closing source code and data as private intellectual property.
For instance, Meta touts the openness of its Llama-series models with terms stating
_"Our model and weights are licensed for researchers and commercial entities, upholding the principles of openness"_ but also specify usage limitations in their license[^llama3];
more critically, Meta also withholds the training data and source code for training.

Meta's open-weights approach is more common than not - only 4 of the 46 models surveyed in _Opening up ChatGPT: Tracking Openness, Transparency, and Accountability in Instruction-Tuned Text Generators_ are fully open,
with another 3 potentially OSAI-compliant with some license changes.[^openness]
In other words, 85% of surveyed models are not OSAI compliant!

{{< figure
src="images/model_openness.png"
alt="model openness"
link="https://opening-up-chatgpt.github.io/"
caption="Opening up ChatGPT: Tracking Openness, Transparency, and Accountability in Instruction-Tuned Text Generators.<br>Credit: Liesenfeld, Andreas, Alianda Lopez, and Mark Dingemanse" >}}

<!-- Given the swirl of contention and drama in the software space surrounding "open source" definitions (FOSS, OSS, source-available...), it is unsurprising to me that we see the same confusion and misrepresentation of AL/ML models.
As the utility of AI/ML models comes from the model artifact itself, it was invevitable that the artifact (the model architecture and weights required to use the model for inference) would have to be shared for "open" use.
This has allowed the model providers to wave the distracting flag of open-weights with one hand while locking the door to the source code with the other. -->

## Business Models and Missions Determine Open-ness

Why do we see handwaving around "open"? If you believe the large AI labs, while openness is part of mission, partially-closing the source is a necessary side effect of their research.
OpenAI and Anthropic are research labs ostensibly founded because of safety concerns with the inevitable development of AI/AGI.
In order to be successful in their missions, they argue, they must stay at the cutting edge to stay ahead of societal risks.
However, R&D is super expensive, and donations/grants cannot cover it.
Therefore, these organizations fund their research by productizing the very cutting-edge models they're researching.
Once the models are products to be monetized, the organizations have to manage competition;
releasing "partially open" models maintains their competitive advantange and ensures continued income while also attempting to follow their original research-oriented mission.

### Case Study: OpenAI

Per their charter, "OpenAI's mission is to ensure that artificial general intelligence (AGI)—by which we mean highly autonomous systems that outperform humans at most economically valuable work—benefits all of humanity...
To be effective at addressing AGI's impact on society, OpenAI must be on the cutting edge of AI capabilities—policy and safety advocacy alone would be insufficient."[^openai_charter]

OpenAI was founded as a 501(c)(3) nonprofit in 2015, but by 2019
"it became increasingly clear that donations alone would not scale with the cost of computational power and talent required to push core research forward, jeopardizing our mission."[^openai_structure]
As a result, OpenAI created OpenAI LP, a "capped [for]-profit" company to manage the commercialization of OpenAI's research for continued funding.[^openai_lp]

{{< figure
src="https://images.ctfassets.net/kftzwdyauwt9/cUJTjpOjmMlux9iCHBPsj/b4075ce1dc79037d40c5586ea90076d4/Structure_Map_Dark.jpg"
alt="openai structure"
link="https://openai.com/our-structure/"
caption="OpenAI Structure" >}}

It is my opinion that, after the Sam Altman oust-and-return debacle and the subsequent rumored reorganization into a purely for-profit organization, that we will see less and less transparency coming from OpenAI.
Case in point, OpenAI states in the `o1` reasoning model blog announcement:
"Therefore, after weighing multiple factors including user experience, _competitive advantage_, and the option to pursue the chain of thought monitoring, we have decided not to show the raw chains of thought to users.
We acknowledge this decision has disadvantages" (emphasis mine).[^o1]

### Case Study: Anthropic

Similarly, Anthropic's mission states that
"Anthropic is a Public Benefit Corporation, whose purpose is the responsible development and maintenance of advanced AI for the long-term benefit of humanity...
We pursue our mission by building frontier systems, studying their behaviors, working to responsibly deploy them, and regularly sharing our safety insights."[^anthropic]

Interestingly, a few weeks before the OpenAI / Sam Altman drama, Anthropic updated their organization into a governance structure that seems conceptually similar to OpenAI's,
with a governing body focused on the research mission and a for-profit arm to finance it.
Anthropic's Delaware Public Benefit Corporation is governed by a Long-Term Benefit Trust (LTBT).
In contrast to OpenAI, the LTBT governance organization is _not_ nonprofit, but recognizes that a purely for-profit/capitalist
organization would likely stray from the research mission.
"At Anthropic, our perspective is that the capacity of corporate governance to produce socially beneficial outcomes depends strongly on non-market externalities"
and capitalism has proved that corporations' fiduciary responsibility to stakeholders means they are limited in some decisions.
Therefore, Anthropic believe "the LTBT can ensure that the organizational leadership is incentivized to carefully evaluate future models for catastrophic risks or ensure they have nation-state level security,
rather than prioritizing being the first to market above all other objectives."[^anthropic_trust]

As a for-profit organiation, Anthropic has never been "open" (i.e., none of its models are even open-weights), and in fact does not even release a public tokenizer.
Where OpenAI has `tiktoken`, Anthropic only recently released an API to return token counts.

> [!NOTE]
> I actually appreciate that Anthropic hasn't handwaved their "openness" regarding their models,
> and also want to call out that they do publish research, such as [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
> and [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), that are instructive and useful in practice.

### Case Study: Allen Institute for AI (Ai2)

As a 501(c)(3) nonprofit, Allen Institute for AI (Ai2)'s mission is "to conduct high-impact artificial intelligence research and engineering in service of the common good,
and to develop applications of AI that contribute to humanity and the preservation of the environment."[^ai2_tax]
Specifically, Ai2 states that their work will always be done in the open, because "Open weights alone aren't enough – true openness requires models to be trained in the open with open access to data, models, and code." [^ai2_open]

Where possible, Ai2 has published fully-open datasets Dolma, WildChat, and Tülu and fully open Language and Vision models (OLMo/OLMoE and Molmo/MolMoE respectively)
with open access to training data, open source code used to train and run the system, and open model weights. Where possible.
In some cases the base models may not be fully open (e.g., Tülu 3 is based on Llama-3-base) and therefore Ai2's openness can only extend as far as the openness of the base model.

Ai2 was founded in 2014 by the late Paul Allen whose donations and estate provide the bulk of the funding for Ai2's research mission.
"The Allen estate is providing more than 95% of AI2's total of $100 million in annual funding for 2022, and there's a commitment from the estate to support Ai2 in perpetuity"[^inside_ai2_incubator]
However, Ai2 also had (has?) its hands in more profit-motivated activities with the AI2 Incubator, which spun off (as an independent entity) in 2022.
Prior to spin-off, Ai2 typically received a 9% stake in incubator companies.
[^inside_ai2_incubator]

> [!NOTE]
> I'm unsure about the details of any financial relationship between Ai2 and the AI2 Incubator.
> The Incubator, which leverages the Ai2 name, maintains a strategic relationship with the research institute,
> and whose "careers" page points back to Ai2's careers page (as of Dec 2024) insists "we are two entirely different organizations" in their [FAQ](https://www.ai2incubator.com/faq).
>
> I am have no knowledge of the finances of either organization, and have am by no means hinting at foul play or under-the-table dealings...
> it would just not surprise me if a financial relationship exists, given the costs of R&D in AI and that the Ai2 nonprofit reports no income streams other than donations / their endowment.

### Case Study: Meta

Unlike OpenAI, Anthropic, and Ai2, Meta is not nonprofit, was not founded as an AI research lab, and does not have a core AI/AGI mission.
However, when most people think of "open" AI, they point to Meta's open-weights Llama models.

In an interview on the Dwarkesh podcast, Mark Zuckerberg explained Meta's move into and perspective on it's approach to AI.
Meta fell into its relationship with AI through a combination of factors - its stewardship of pytorch, its push to improve its recommendation systems, and in order to respond to TikTok.
He said, of Meta's GPU infrastructure, "We got into this position with Reels where we needed more GPUs to train the models [to catch up to TikTok]...
Let's order enough GPUs to do what we need to do on Reels and ranking content and feed. But let's also double that."[^zuckerberg]

Why does Meta release Llama models with open weights and fairly open (but not completely open) licenses? It's in their best interests.
Meta has seen previously that open-sourcing projects (PyTorch, React, OpenCompute) inspires a virtuous cycle that builds momentum,
such that the project becomes industry-standard (PyTorch) or scales volume and reduces per-unit costs (OpenCompute).
"It's \[AI is\] helping people interact with creators, helping people interact with businesses, helping businesses sell things or do customer support... Our bet is that it's going to basically change all of the products," said Zuckerberg.
If AI is "going to be more like people going from not having computers to having computers," then Meta wants to be the default platform,
so that AI applications are built in its ecosystem.

To do this, Meta has to stay on the cutting edge - "\[if\] we're sitting here with a basic chat bot, then our product is lame compared to what other people are building."[^zuckerberg]
This AI-as-a-product perspective also explains why Meta has not fully open-sourced their models.
When you want your platform to be the default host for all social AI in the future, it doesn't really matter to the end user how the base model was trained, just how flexible and modifiable it is.
Open-weights releases provide this flexibility to users while also protecting Meta's IP and competitive advantage on the development side.

Meta's "open" model ecosystem might generously be framed as "enlightened self-interest."
"We're obviously very pro open source," said Zuckerberg, "but I haven't committed to releasing every single thing that we do.
I'm basically very inclined to think that open sourcing is going to be good for the community _and also good for us because we'll benefit from the innovations._
If at some point however there's some qualitative change in what the thing is capable of, and we feel like it's not responsible to open source it, then we won't" (emphasis mine).[^zuckerberg]

## Does Open Source AI matter?

Even if a frontier AI lab fully open-sourced their data, source code, etc., would it matter?

### It matters

Just like earlier open works like [BERT](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/) or [Self-Instruct](https://arxiv.org/abs/2212.10560) accelerated the pace of development,
a frontier lab fully open-sourcing its foundation model(s) would allow the research community to start experiments from the existing cutting edge,
rather than an assumption of the cutting edge (e.g., `o1` "replicas") or having to reinvent the wheel.
In a similar manner to open-source software, OSAI might allow faster identification of risks, bugs, faults, and biases because more eyes can review and audit the system.

From an implementation perspective, I would love to have a deep understanding of the "instruct" datasets.
If I know that Llama models were trained with _x_ formatting but OpenAI in _y_ and Anthropic in _z_ formats, it would make the prompt engineering experience much simpler to start writing prompts and agents from known-good templates.
Understanding the instruct breakdowns for models would make it easier to _know_ which model to route different requests to, instead of experimenting with lossy benchmarks.

### It doesn't matter

It is _expensive_ to train frontier models.
Only huge companies and nation-states with sufficient resources ($$$, compute, expertise) can train foundation models anyway --
this is why we see Anthropic, OpenAI, etc. all organizing with for-profit arms/ventures to finance the expenditures required to do the research their missions espouse to be their _raison d'être_.
Even if each of these labs fully open-sourced their AI systems, I don't think it would lead to an explosion of new SOTA models because of cost (and GPU availability) constraints.
Furthermore, training frontier models also takes a [substantial amount of power and water]({{< ref "/blog/ai-is-the-new-hotness" >}}),
and therefore it may actually be more efficient for the industry to align on a handful of base models (which could be open-weight)
and allow fine-tuning, as stated by Meta in the Llama 2 technical report, "Our open release strategy also means that these pretraining costs will not need to be incurred by other companies, saving more global resources."[^llama2]

## Final Thoughts

_Whether_ models are fully open source is clearly a business decision, and arguments can be made to favor either side.
The problem as I see it is twofold: (a) the confusion generated by the multitude of "open" definitions combined with the distinctions between Open Source AI (OSAI) and Open Source Software (OSS),
and (b) the industry plaudits that Meta and their ilk receive for "open-washing" their open-weight (and sometimes license-limited) models.

As discussed above, AI/ML products require code and data to generate their artifacts, whereas traditional software engineering only requires code.
AI/ML products also require code to use the artifacts, whereas artifacts from traditional software engineering can typically be used directly.
When considered from the traditional software engineering perspective, releasing open weights models with the code to run inference against them and modify their _use_ seems quite similar to the OSS model.
However, open weights does not provide the same value to the AI community as open source does to the software community.
Open weights only allow developers to make wrappers around other products, and do not allow introspection, education, experimentation, or auditing.

It bothers me that Meta (and others) represent their open-weight (and sometimes license-limited) models as "open" and gain recognition and goodwill for it.
I'm whining about receiving expensive models for free; this is a calculated business decision.
If you follow Mark Zuckerberg's perspective, the real value is not the model artifacts themselves, but the power of defaults and market share.
Open weight releases and good marketing provide the same value as full OSAI to Meta and their ilk, without any of the (potential) downsides.

A known problem in the software industry is that it is hard to make a Open Source a successful business model - if you give away the product and how it is made, how do you preserve your revenue streams?
As a result, several notable companies (Redis, Sentry) have recently transitioned to "source available" (or "fair source") licenses (i.e., where the code is available but use is constrained by license).[^source_available] [^fair_source]
A similar approach in the AI industry would allow access to the full codebases and datasets and provide similar benefits to the AI community as full OSAI, but preserve revenue streams for companies like OpenAI.
Admittedly, opening datasets (especially post-training examples) is risky for AI organizations, since the utility of the model is defined by quality and mixture of examples.
However, license limitations would (theoretically) protect against IP theft from competitors.

## References

{{% details title="Further Reading" closed="true" %}}

- [Initial Report on Definition Validation - Open Source AI - OSI Discuss](https://discuss.opensource.org/t/initial-report-on-definition-validation/368/20?u=mia)
- [Openness in Language Models: Open Source vs Open Weights vs Restricted Weights](https://promptengineering.org/llm-open-source-vs-open-weights-vs-restricted-weights/)
- [[2407.12929] The Foundation Model Transparency Index v1.1: May 2024](https://arxiv.org/abs/2407.12929)
- [Molmo](https://molmo.allenai.org/blog)
- [Why 'open' AI systems are actually closed, and why this matters | Nature](https://www.nature.com/articles/s41586-024-08141-1)
- [Building LLMs is probably not going be a brilliant business](https://calpaterson.com/porter.html)
- [How OpenAI's Bizarre Structure Gave 4 People the Power to Fire Sam Altman | WIRED](https://www.wired.com/story/openai-bizarre-structure-4-people-the-power-to-fire-sam-altman/)
- [Open Source AI Is the Path Forward | Meta](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)

{{% /details %}}

[^oss]: [The Open Source Definition – Open Source Initiative](https://opensource.org/osd)

[^osai]: [The Open Source AI Definition – Open Source Initiative](https://opensource.org/ai/open-source-ai-definition)

[^osai_faq]: [Answers to frequently asked questions - HackMD](https://hackmd.io/@opensourceinitiative/osaid-faq)

[^lf_osai]: [Embracing the Future of AI with Open Source and Open Science Models – LFAI & Data](https://lfaidata.foundation/blog/2024/10/25/embracing-the-future-of-ai-with-open-source-and-open-science-models/)

[^open_definitions]: [AI weights are not open "source" | Open Core Ventures](https://opencoreventures.com/blog/2023-06-27-ai-weights-are-not-open-source/)

[^llama3]: [meta-llama/llama3: The official Meta Llama 3 GitHub site](https://github.com/meta-llama/llama3/tree/main)

[^openness]: [Opening up ChatGPT: LLM openness leaderboard](https://opening-up-chatgpt.github.io/)

[^openai_charter]: [OpenAI Charter | OpenAI](https://openai.com/charter/)

[^openai_structure]: [Our structure | OpenAI](https://openai.com/our-structure/)

[^openai_lp]: [OpenAI LP | OpenAI](https://openai.com/index/openai-lp/)

[^o1]: [Learning to Reason with LLMs | OpenAI](https://openai.com/index/learning-to-reason-with-llms/)

[^anthropic]: [Company \\ Anthropic](https://www.anthropic.com/company)

[^anthropic_trust]: [The Long-Term Benefit Trust \\ Anthropic](https://www.anthropic.com/news/the-long-term-benefit-trust)

[^ai2_tax]: [The Allen Institute For Artificial Intelligence - Full Filing- Nonprofit Explorer - ProPublica](https://projects.propublica.org/nonprofits/organizations/824083177/202343159349301114/full)

[^ai2_open]: [More than open | Ai2](https://allenai.org/more-than-open)

[^inside_ai2_incubator]: [Inside the AI2 Incubator: Microsoft co-founder's unfinished legacy fuels quest for new AI startups – GeekWire](https://www.geekwire.com/2022/inside-the-ai2-incubator-microsoft-co-founders-unfinished-legacy-fuels-quest-for-new-ai-startups/)

[^zuckerberg]: [Mark Zuckerberg - Llama 3, Open Sourcing $10b Models, & Caesar Augustus](https://www.dwarkeshpatel.com/p/mark-zuckerberg)

[^llama2]: [[2307.09288] Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

[^source_available]: [Source-available software](https://en.wikipedia.org/wiki/Source-available_software)

[^fair_source]: [About Fair Source | Fair.io](https://fair.io/about/)
