---
title: Making Bets on AI
date: 2025-05-25T16:10:44-04:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - opinion
  # ai/ml
  - agents
  - AGI
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

On a recent Dwarkesh podcast, Sholto Douglas, an AI researcher working on reinforcement learning at Anthropic said: _"There is this whole spectrum of crazy futures. But the one that I feel we're almost guaranteed to get—this is a strong statement to make—is one where at the very least, you get a drop-in white collar worker at some point in the next five years. I think it's very likely in two, but it seems almost overdetermined in five. On the grand scheme of things, those are kind of irrelevant timeframes. It's the same either way."_[^claude4]

I think Douglas' timeline is likely quite optimistic; I doubt we will have agents or assistants that are _broadly capable_ of _general white-collar work_ -- even at the task level -- by 2030. The best parallel we have are code Agents/Assistants (Claude Code, OpenAI Codex, Google Jules, Github Copilot, Devin, Cursor, Windsurf, ...), which cannot (yet) replace software engineers even after 3 years of concentrated effort. That said, those three years of development have lead to increased capability to support human software engineers, accelerate work, and reduce gruntwork. If we don't have "drop-in" capabilities for AI to replace software engineers at the task level now, how can we expect AI to generalize across all domains of white collar work in the next 2 years, or even 5?

Regardless, the point of this blog post isn't to debate capabilities, timelines, or how we get there from here, it's to explore the thought experiment triggered by Douglas' perspective:

> What does a future look like where we have "AGI" such that the AI can serve as a "drop-in" white-collar worker?
> Given this ostensibly occurs _within our career span_, what should you do?

## Capability Drives Adoption, Use

Let us first assume that a tool (AI Assistant) exists that can complete generic, day-to-day _tasks_ for a white-collar job. Adoption of AI tools will likely follow the "typical" S-shaped adoption curve we have seen over history; it will not be a binary off -> on switch.

{{< figure
src="images/diffusion_of_ideas.png"
alt="S-shaped Adoption Curve"
link="https://en.wikipedia.org/wiki/Diffusion_of_innovations"
caption="Innovators and Early Adopters will have been experimenting for a while. As Task Assistance gains social acceptance (and professional pressure to adapt), adoption increases quickly, then slows again as Late Adopters and Laggards slowly join" >}}

Anthropic's _Economic Index_ provides some insight into adoption patterns. Based on data collected between December 2024 - January 2025, Anthropic found that software engineers (and other math/code jobs) were disproportionately represented in Claude use, accounting for 37% of requests but only 3% of US workforce population.[^AEI] Similarly, students pursuing Computer Science degrees also show highly disproportionate AI use, accounting for 39% of conversations but only 5% of US bachelor degrees.[^AEI_EDU] This disproportionate use may be driven by disproportionate _capability_ on math and coding tasks; on the aforementioned Dwarkesh Podcast episode, Douglas notes that Claude's (and other LLM capability) on math and code benchmarks are (a) because they can be programmatically validated, and (b) because _the researchers of the labs love working on the bars of intelligence that they themselves resonate with. So this is why math and competitive programming_ are prioritized.

Anthropic also analyzed software engineer use of _Claude.ai_ (the chatbot) vs _Claude Code_ (the Agentic tool), finding that the chatbot was used to collaboratively assist, while the tool was used as a delegate, completing tasks for the user. Anthropic also find that _"33% of conversations on Claude Code served startup-related work, compared to only 13% identified as enterprise-relevant applications"_, indicating that AI assistance for software engineering jobs may be entering the "early majority" phase of the adoption lifecycle.[^AEI_SWE]

Having equivalent tools and assistants for generic white-collar work suggests that all white-collar workers would use AI like software engineers currently do. Recall from the _Anthropic Economic Index_ that software engineers accounted for 37% of requests but only 3% of US workforce[^AEI]; for all other occupations to "catch up" to the "early adopter" phase of software engineering use, it would mean an _**11x**_ increase in use of AI! Of course, that 11x increase implies that software engineering use of remains consistent and in a persisted "early adopter" phase; this is unrealistic, and we should expect SWE use to increase as adoption follows the S-curve.

Adoption here just refers to the proportion of the population who are using AI tools in any capacity. At any given point in the adoption curve, we should expect a normal distribution of _use_, from light users to power users.[^pew]

## Future of Work

Anecdotally (I recall reading/hearing these stories from web articles and podcasts, but do not have references), frequent users of these AI coding assistants tend to delegate in parallel - the human plans the strategy and _delegates_ to multiple instances of the AI assistant to draft and propose outputs. Then the human and AI collaborate to refine and accept the work.

A non-coding example of AI acceleration -- [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/) can research across the internet and cover a variety of hypotheses and sources that would take a person several hours, and can return the report (with citations!), with quality equivalent to a PhD student in less than an hour.[^mollick] A responsible employee would check the work, and this will eat into some of the acceleration gained from delegating the task to AI; the person is now acting as a manager rather than worker.

A recent METR report finds that current models (at the time of the release, the leader was Claude 3.7 Sonnet) have a "1 hour time horizon", meaning it has ~50% chance at succeeding at a task that would take a human 1 hour; as the human-time decreases, the likelihood of AI success increases, with models approaching 100% success for tasks that would take a human < 5 minutes.[^METR] As the time horizon increases, I believe we will see a commensurate increase in delegation to AI. White-collar workers will become managers of a stable of AI instances, and delegate tasks that the AI will complete; the AI will take less time to do the work than a person would, and can act in parallel. This will theoretically massively increase productivity-per-person.

## On White-Collar Employment

> This will theoretically massively increase productivity-per-person.

When I play out this scenario, I can see it going it two different ways -- in one, we see a virtuous cycle driven by the explosion in productivity, with more productivity driving more jobs, new ideas, new companies, etc.; in the other, we see our current employment paradigm collapse as existing workers are accelerated by AI but few new roles are created.

### Rising Unemployment

The New York Times reports we may already be seeing leading evidence of the latter potential future, with rapidly increasing unemployment rates for recent graduates anecdotally driven by hype around "virtual coworkers" that can automate entry-level work.[^new_grads_unemployed] Dario Amodei, the CEO of Anthropic, recently warned that it could extend even further than entry-level jobs. In a conversation with Axios, he reportedly said, "AI could wipe out half of all entry-level white-collar jobs — and spike unemployment to 10-20% in the next one to five years." [^dario_jobs]

This will cause two problems. First, loss of upcoming juniors means the next generation of workers will not have inherited the skills required to do their job. AI-optimists may believe this to be a non-issue, as "the AI will learn to do it"; however, there is no failsafe in this case, and even with AI there may be an expertise cliff as generational turnover occurs. Second, unless there is a new type of job role  or industry, unemployment will rise. With no entry-level roles, college graduates will have limited employment options. As higher-level roles are squeezed because fewer people are needed to do the same amount of work (given AI assistance), the now-unemployed folks with expertise will fill the few entry-level roles that do exist.

I think it is incredibly short-sighted for organizations to close the "top of funnel" by replacing entry-level roles with AI. Even if AI can do the same work, you lose the ability to train the next generation; current college students are already much more AI-native than most middle-age workers. Their fresh insights might empower new AI-native practices; they simply lack work experience.

> [!NOTE]
> [This reddit discussion](https://www.reddit.com/r/singularity/comments/1l2gwo1/dario_amodei_worries_that_due_to_ai_job_losses/) about Dario Amodei's CNN interview explores some of the concerns about concentration of wealth and power coming from this type of (un)employment disruption. Said Amodei, "We need to be raising the alarms. We can prevent it, but not by just saying 'everything's gonna be OK'."
>
<!-- In the most pessimistic case, the ultimate outcome I foresee is dystopia. Extreme "AI offshoring" would result in extreme inequality between the haves and have-nots. Subsequently, history tells us that extreme wealth stratification leads to social unrest (riot, revolt, revolution, civil war).[^inequality] Resetting inequality is achieved only when the "have-nots" have sufficient institutional leverage to affect change, which historically occurred through mass mobilization (French Revolution) or institutional protection (post-WWII).[^historic_shocks] Unfortunately, we are not in a paradigm with strong institutional protections (though grassroots efforts may be effective), and mass mobilization will likely be less effective due to AI-aided technologies (surveillance state, weaponized drones, etc.). -->

### Or Employment Flywheel

On the other hand, Demis Hassabis, the CEO of Google DeepMind, is a bit more optimistic. While AI may allow one person to do the job of 70, he argues that those 70 would be able to go on and have the opportunity to do other things, including starting new businesses.[^hassabis_jobs] There is historic precedent for new technologies creating new jobs; assembly lines beget assembly-line workers, the computer created the entire IT industry. It is not unlikely that entirely new job categories could come into existence built on the burgeoning AI industry.

It is hard to estimate what the new job roles (or even industries) may look like since there is not really a precedent; as a result, this optimistic interpretation of the influence of AI on employment seems more farfetched than the pessimistic one.

TODO:
TODO: [If Anyone Can Do it, AI Will Do It Better](https://decision.substack.com/p/if-anyone-can-do-it-ai-will-do-it)
TODO:

### [AI Changes Everything | Armin Ronacher's Thoughts and Writings](https://lucumr.pocoo.org/2025/6/4/changes/)

_"The job of programmers and artists will change, but they won't vanish. I feel like all my skills that I learned as a programmer are more relevant than ever, just with a new kind of tool. Likewise the abundance of AI generated art also makes me so much more determined that I want to hire an excellent designer as soon as I can. People will always value well crafted products. AI might raise the bar for everybody all at once, but it's the act of careful deliberation and very intentional creation that sets you apart from everybody else."_

... _"The more time I spend with these tools, the more I believe that optimism is the more reasonable stance for everyone. AI can dramatically increase human agency when used well. It can help us communicate across cultures. It can democratize access to knowledge. It can accelerate innovation in medicine, science, and engineering."_

_"Right now it's messy and raw, but the path is clear: we are no longer just using machines, we are now working with them._"

## Sidebar: Resource Economy

In an interview with the New York Times, Hassabis says, "As we get closer to A.G.I. and we make breakthroughs in material sciences, energy, fusion, these sorts of things, helped by A.I., we should start getting to a position in society where we're getting toward what I would call radical abundance, where there's a lot of resources to go around."[^hassabis_jobs] This handwaves away concerns about rapidly increasing energy demand or the approaching limits of our ability to continue to improve the integrated circuit due to wavelengths of light we are able to control, because "AI will solve it".

Unlike Demis Hassabis or Dario Amodei (in his more optimistic moments), I do not believe it is likely that a "radical abundance" scenario emerges. Techological breakthroughs have historically concentrated wealth (and therefore resources) rather than resulting in more equitable distribution. Further, corporations are driven by profit motives and will chase advancements to reduce costs, maximize efficiency, and enter (new) markets where they have competitive advantage; there is no motivation for "radical abundance".

That said, there's money to be made -- As AI needs grow (11x growth _minimum_ mentioned above!) significantly more compute will be needed. This involves building datacenters, manufacturing computers and their components, mining the raw materials, and building the power generation, energy storage, and power and network infrastructure to support it all. Each of these is a potential bottleneck in the overall supply chain that must itself scale to match the coming demand for AI.[^supply_chain]

## Conclusion

> If you believe that we will have "AGI" such that the AI can serve as a "drop-in" white-collar worker _within our career span_, what should you do?

In the Age of AI, _what we do_ for work and _how we do it_ will fundamentally change, and the change will happen rapidly. The best thing for a person to do now is understand the nuance of their job, not only what they do, but why they do it, and the fractal complexities of each decision and possibility-space of the inputs and outputs to their work product. Further, the "soft skills" of time management, leadership and delegation, and clear communication will become even more important. In a world where each worker manages a stable of AI assistants, understanding how to delegate, how to manage the delegates, and how to clearly communicate task requirements and expectations will be critical.

> [!NOTE]
> Anthropic recently [released a course](https://www.anthropic.com/ai-fluency/introduction-to-ai-fluency) on refining and practicing how to interact with AI and focusing on 4 key competencies: Delegation, Description, Discernment and Diligence.

## References

- [^claude4]: [How Does Claude 4 Think? — Sholto Douglas & Trenton Bricken](https://www.dwarkesh.com/p/sholto-trenton-2) @ ~2hr
- [^AEI]: [Introducing the Anthropic Economic Index \\ Anthropic](https://www.anthropic.com/news/the-anthropic-economic-index)
- [^AEI_EDU]: [Anthropic Education Report: How University Students Use Claude \\ Anthropic](https://www.anthropic.com/news/anthropic-education-report-how-university-students-use-claude)
- [^AEI_SWE]: [Anthropic Economic Index: AI's impact on software development \\ Anthropic](https://www.anthropic.com/research/impact-software-development)
- [^pew]: [On Future AI Use in Workplace, US Workers More Worried Than Hopeful | Pew Research Center](https://www.pewresearch.org/social-trends/2025/02/25/u-s-workers-are-more-worried-than-hopeful-about-future-ai-use-in-the-workplace/)
- [^mollick]: [The End of Search, The Beginning of Research](https://www.oneusefulthing.org/p/the-end-of-search-the-beginning-of)
- [^METR]: [Measuring AI Ability to Complete Long Tasks - METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)
- [^new_grads_unemployed]: [For Some Recent Graduates, the A.I. Job Apocalypse May Already Be Here](https://www.nytimes.com/2025/05/30/technology/ai-jobs-college-graduates.html?unlocked_article_code=1.LU8.Tley.axMtaz01kwv5&smid=url-share)
- [^dario_jobs]: [AI jobs danger: Sleepwalking into a white-collar bloodbath](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic?_bhlid=de44ede8ac1fac0ec5469bb77e814ab71241e0cf)
- [^hassabis_jobs]: [Google DeepMind's Demis Hassabis on AGI, Innovation and More](https://www.nytimes.com/2025/05/23/podcasts/google-ai-demis-hassabis-hard-fork.html?unlocked_article_code=1.LU8.MrST.91r-pTvKxA74&smid=url-share)
- [^supply_chain]: [AI chip boom sparks BT substrate materials shortage — TSMC's huge demand causes supply disruptions for NAND flash controllers, SSDs | Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/ai-chip-boom-sparks-bt-substrate-materials-shortage-tsmcs-huge-demand-causes-supply-disruptions-for-nand-flash-controllers-ssds)
