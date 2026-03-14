---
title: Stop Sloppypasta
date: 2026-03-14
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - meta
  - opinion
  # ai/ml
  - generative AI
series: []
layout: single
toc: true
math: false
plotly: false
draft: true
---

LinkedIn is rife with the drive-by copy-paste of raw LLM output, and the trend has bled into Discord, Reddit, and other online forums, work chats, and emails.
I'm coining this 'sloppypasta,' and this is my rant against it.

---

## sloppypasta

> Verbatim LLM output copy-pasted at someone, unread, unrefined, and unrequested.
>
> From **slop** (low-quality AI-generated content) + **copypasta** (text copied and pasted, often as a meme, without critical thought).
> It is considered rude because it asks the recipient to do work the sender did not bother to do themselves.

---

{{< details title="TLDR; Table" closed="true" >}}

<table>
  <thead>
    <tr>
      <th></th>
      <th>As a Recipient</th>
      <th>As a Sender</th>
      <th>Feedback loop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Effort</th>
      <td>
        Previously, effort to read was balanced by the effort to write.
        Now LLMs make writing "free" and increase the effort to read due to additional verification burden.
      </td>
      <td>
        Writing requires effort, which contributes to comprehension.
        LLMs increase cognitive debt by reducing struggle.
      </td>
      <td>
        Sender's skipped effort becomes recipient's added effort, increasing frustration as incidence increases.
      </td>
    </tr>
    <tr>
      <th scope="row">Trust</th>
      <td>
        LLM propensity for hallucination and capability to bullshit convincingly mean that "trust but verify" is broken.
        All correspondence must be untrusted by default.
      </td>
      <td>
        What you share directly influences your reputation.
        Sharing raw LLM output - especially unvetted - burns your credibility.
      </td>
      <td>
        Eroding trust from LLM sloppypasta is the modern 'Boy Who Cried Wolf.'
      </td>
    </tr>
  </tbody>
</table>
{{< /details >}}

Sharing raw AI output is like eating junk food: it's easy and may feel good, but it's not in your best interest.
You'll negatively influence your relationship with the recipient, and do yourself a disservice by reducing your own comprehension.

<!-- markdownlint-disable MD013 -->

> "For the longest time, writing was more expensive than reading.
> If you encountered a body of written text, you could be sure that at the very least, a human spent some time writing it down.
> The text used to have an innate proof-of-thought, a basic token of humanity."
>
> - Alex Martsinovich, [It's rude to show AI output to people](https://distantprovince.by/posts/its-rude-to-show-ai-output-to-people/)

<!-- markdownlint-enable MD013 -->

Before LLMs, writing took effort.
Authors spent time and effort considering and selecting their words with intention; time and effort that was balanced by that spent by the audience as they read.
This balance is broken with LLMs; the effort to produce text is effectively free, but the effort required to read the text hasn't changed.
[The increasing verbosity of LLMs](https://epoch.ai/data-insights/output-length) further increases the effort asymmetry.
In some circumstances (like pasting raw LLM output into a chat thread), the sloppypasta effectively becomes a filibuster, crowding out the existing conversation and blocking the viewport.

> "Cognitive effort — and even getting painfully stuck — is likely important for fostering mastery."
>
> - Anthropic, [How AI assistance impacts the formation of coding skills](https://www.anthropic.com/research/AI-assistance-coding-skills)

Writing is thinking.
The writing process forces the author to work through their thoughts, building their comprehension and retention.
[Multiple](https://www.media.mit.edu/publications/your-brain-on-chatgpt/) [studies](https://www.anthropic.com/research/AI-assistance-coding-skills) have found that delegating tasks to LLMs creates cognitive debt.
Shortcutting thinking with LLMs ultimately reduces comprehension of and recall about the delegated subject.

> "A polished AI response feels dismissive even if the content is correct"
>
> - Blake Stockton, [AI Writing Etiquette Manifesto](https://www.blakestockton.com/ai-writing-etiquette-manifesto/)

Before LLMs, trust was the default.
Authors wrote from their personal expertise and perspective, and readers could judge an author's understanding of the subject based on the coherence of their writing.
LLMs generate the most probable next token given an overarching goal to be helpful, which explains their propensity for hallucination ([confabulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10619792/)) and why many people feel that [LLMs are bullshit generators](https://machine-bullshit.github.io/).
Modern LLMs are typically provided tools to help them look up grounding information that reduces (but does not eradicate) their likelihood to outright make up facts during their responses.
But that still doesn't solve the trust problem; the reader still has no way to know what the sender checked and what they didn't.
LLM responses, therefore, cannot be trusted by default and compound the effort asymmetry on the reader by adding a verification tax.

Beyond accuracy, LLMs write authoritatively with the tone and confidence of an expert.
This adds further uncertainty to the reader's burden; they have no way to gauge the sender's actual level of expertise with the subject matter.
The result is a further erosion of trust, because the AI's voice removes signal that recipients previously used to distinguish expertise from plausible-sounding slop.

> "I think it's rude to publish text that you haven't even read yourself.
> I won't publish anything that will take someone longer to read than it took me to write."
>
> — Simon Willison, [Personal AI Ethics](https://simonwillison.net/2023/Aug/27/wordcamp-llms/#personal-ai-ethics)

Formerly, "Trust but verify" ruled.
Readers would trust until that trust was broken; the author was trustworthy or they weren't.
However, shared LLM output obfuscates the chain of trust.
Did the prompter do the appropriate due diligence to validate the LLM response?
If problems or errors are discovered, who is to blame, the prompter or the AI?
Was it an oversight, a missed verification step, or was verification skipped altogether?
The uncertainty means the recipient doesn't know what they can trust, what has or has not been verified; they must treat everything as untrusted.
Just like the Boy Who Cried Wolf, once the trust is broken, the uncertainty spreads to all future messages from the sender.

Assumptions of balanced effort and presumed trust are no longer guaranteed in a post-LLM world.
Sloppypasta creates a compounding negative feedback loop where the sender forfeits learning and credibility while the recipient burns effort and loses trust.
Receiving raw AI output feels _bad_ due to the cognitive dissonance of having these assumptions violated.

## Rules

AI capabilities keep increasing, and using it to draft, brainstorm or accelerate you will be increasingly useful.
However, using AI should not make your productivity someone else's burden.
New tools require new manners.

**Use AI to accelerate your work or improve what you send.**\
**Don't use it to replace thinking about what you're sending.**

### 1. Read

Read the output before you share it.
If you haven't read it, you don't know whether it's correct, relevant, or current.

Delegating work to AI creates cognitive debt.
Working with the results helps run damage control for your own understanding.

### 2. Verify

Check the facts before you forward them.
Anything you forward carries your implicit endorsement -- your reputation depends on managing the quality of what you share.

LLMs are trained to "be helpful", and will produce outdated facts, wrong figures, and plausible nonsense to provide _a_ response to your requests.
Further, an LLM is inherently out-of-date; their knowledge cutoffs contain _at best_ information on the state of the world when their training _started_ (months ago).

### 3. Distill

Cut the response down to what matters.
Distilling the generated response to the useful essence is your job.

LLMs are incentivized to use many words when few would do:
API-priced models have a per-token incentive to train chatty LLMs that use many tokens, and [research shows](https://arxiv.org/abs/2310.10076) that longer, highly formatted posts are often preferred as more engaging.

### 4. Disclose

**Share how AI helped.**

If you've read, verified, and edited it, send it as yours -- preferably with a note that you worked with AI assistance.
If you're sharing raw output, say so explicitly.
In both cases, it may be useful to share your prompt and how you worked with the AI to get the final output.

Disclosure restores the trust signals that sloppypasta destroys and tells the recipient what you checked and what they may be on the hook for.

### 5. Share _only_ when requested

Never share unsolicited AI output into a conversation.

Remember that AI generations create effort asymmetry and be respectful of those you share with.
Sloppypasta delegates the full burden of reading, verifying, and distilling to a recipient who didn't ask for it and may not realize the effort required of them.

### 6. Share as a link

Share AI output as a link or attached document rather than dropping the full text inline.

In messaging environments, a large paste takes over the viewport and crowds out the existing conversation.
A link lets the recipient choose when - and whether - to engage, rather than having that choice imposed on them.

---

> [!NOTE]
> AI Disclosure
>
> I used AI to help me proofread this document and to provide a critical review to improve my arguments and ensure clarity.

---

## Further Reading

- [It's Rude to Show AI Output to People](https://distantprovince.by/posts/its-rude-to-show-ai-output-to-people/)
- [Personal AI Ethics by Simon Willison](https://simonwillison.net/2023/Aug/27/wordcamp-llms/#personal-ai-ethics)
- [AI Manifesto](https://noellevandijk.com/ai-manifesto/)
- [Using AI Responsibly in Development & Collaboration](https://ai-manifesto.dev/)
- [AI Writing Etiquette Manifesto](https://www.blakestockton.com/ai-writing-etiquette-manifesto/)
