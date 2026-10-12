---
title: Spec-Driven Development
date: 2026-10-12
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
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

Spec-Driven Development (SDD) is a method to drive AI coding assistants over clearly defined changes.
Spec-driven development blends waterfall and agile development practices; the specs are waterfall-like preplanned work, but taken in small agile-like steps.
The practice helps avoid pitfalls of "vibe coding" by providing specific guidance and expectations to the implementing models, and helps developers by requiring that they think through the request before implementation and providing a clear decision log.

<!-- markdownlint-disable MD013 MD033 MD034 -->

{{< figure
src="images/comprehensive-and-precise-spec.jpg"
alt="A comprehensive and precise enough spec to generate a program... is code."
link="https://www.commitstrip.com/en/2016/08/25/a-very-comprehensive-and-precise-spec/?" >}}

<!-- markdownlint-enable MD013 MD033 MD034 -->

A generic flow for spec-driven development might look something like:

- Identify a desired change or improvement and explain why it is important
- For that change, carefully describe the desired behavior or outcome and the scope of work
- Decide on the specific details of implementation (such as mechanism, tooling, frameworks, services, etc.) and record the decision rationale
- Implement the change, following the decisions
- Check that the implementation followed the decisions and accomplishes the desired behavior/outcomes

## Goldilocks and the Search for an SDD-framework

At the end of last year (Dec 2025), I started with GitHub's Spec Kit ([github/spec-kit: 💫 Toolkit to help you get started with SDD or any other process!](https://github.com/github/spec-kit)).
Spec Kit is a highly-structured process for defining spec-driven work.
It is very "Microsoft"; Spec Kit feels heavy-handed, though the structure probably works well for large projects with many developers working in concert.

My recollection is that Spec Kit was very heavy for small changes; it felt much more "at home" working on full, net-new features rather than making small agile bugfixes or iterating on existing codebases.
For instance, Spec Kit really wants you to define a "Constitution" which is intended to store _"principles focused on code quality, testing, and maintainability"_.
...
But that's what AGENTS.md also does; why should I replicate that content in a nonstandard form, and why does Spec Kit require it?
Spec Kit also leaned heavily on users remembering all of the various slash-commands and how it integrated with the associated CLI tool.
After onboarding your project with `specify init`, you had to remember the following "default" flow and associated commands:

```text
/speckit.constitution -> /speckit.specify -> /speckit.clarify -> /speckit.plan -> /speckit.checklist -> /speckit.tasks -> /speckit.analyze -> /speckit.implement -> /speckit.converge
```

Yikes.

From Spec Kit, I moved on to try OpenSpec ([Fission-AI/OpenSpec: Spec-driven development (SDD) for AI coding assistants.](https://github.com/Fission-AI/openspec)) and ([obra/superpowers: An agentic skills framework & software development methodology that works.](https://github.com/obra/superpowers)).
OpenSpec is a more fluid, flexible approach to spec-driven development, which allows a more agile approach to defining "changes", where each change is defined as a set of artifacts (proposal.md, spec(s).md, design.md, tasks.md) in the `openspec/` directory.
The proposal records the reason for the change, the specs define the requirements and testable scenarios, the design records the implementation details and technical decisions, and the tasks plan out the work the agent will complete.
Unlike Spec Kit, which treats specs as a point-in-time historical reference, OpenSpec treats specs as living documents: the project is defined by baseline specs, each change is a modification to the baseline, and those modifications are merged back into the baseline specifications when the change is completed.

Superpowers is a plugin package or set of skills that provide more generic guidance for working with coding agents as opposed to a formal SDD procedure; as such, it is even more flexible than OpenSpec.
Superpowers provides skills to help users brainstorm and plan (with design document and plan artifacts), and other skills guide agents through implementation, testing, and code review.
I really like a lot of the ideas behind Superpowers, but OpenSpec really fit my needs for a guided workflow for agentic development.
In fact, they can be combined, using OpenSpec's SDD workflow, but using Superpowers' guidance for working with agents.

## There are many like it, but this one is mine

If Spec Kit required too much procedure, Superpowers too little, and OpenSpec was just right, why have I forked my own version?
After using Spec Kit, OpenSpec, and Superpowers, I kept hitting repeated problems that bugged me.

1. Spec Kit was simultaneously too linear and too rigid, relying on complicated tooling (CLI, prompt templates, slash-commands) and necessitating substantial effort for users to internalize and adhere to its rigorous procedure.
2. Both Spec Kit and OpenSpec rely on CLI tools that had to be installed rather than being "Skill-native".
3. Spec Kit treats specs as historic references rather than living descriptions of the codebase.
4. OpenSpec uses baseline specs to represent the state of the project, but provides no mechanism to derive the baseline for a brownfield project (or at least did not at the time).
5. At the time, both procedures planned and worked in isolation without considering the codebase, application, or user experience as a whole, or how the various specs/proposals might interact.

To resolve these, I started by forking the OpenSpec concept into my own family of SDD skills that inherit lessons from fighting OpenSpec and Spec Kit.
These are skill-only -- no other installation or tooling is necessary other than Python -- making them easy to install by copying the skill directory or via `npx skills install`.
Each phase in the spec-driven lifecycle is its own skill.
The general workflow aligns closely with OpenSpec: [`sdd-propose`](https://github.com/ahgraber/skills/blob/main/skills/sdd-propose/SKILL.md) helps the user draft the change artifacts, [`sdd-apply`](https://github.com/ahgraber/skills/blob/main/skills/sdd-apply/SKILL.md) works through the tasks, and [`sdd-verify`](https://github.com/ahgraber/skills/blob/main/skills/sdd-verify/SKILL.md) checks that the code aligns with the specs.
[`sdd-sync`](https://github.com/ahgraber/skills/blob/main/skills/sdd-sync/SKILL.md) syncs the change into the baseline specs, and [`sdd-archive`](https://github.com/ahgraber/skills/blob/main/skills/sdd-archive/SKILL.md) archives the change.

I've also addressed my frustrations above:

[A generalized "router" skill](https://github.com/ahgraber/skills/blob/main/skills/sdd/SKILL.md) helps agents orient to and follow the procedure so the user doesn't hit the Spec Kit wall: having to learn and recall the specific order of operations or remember specific commands.
As a skills-only approach, the deterministic steps are provided as runnable scripts as part of the skill, rather than requiring installing additional tools.
To work within existing codebases, I've added [`sdd-derive`](https://github.com/ahgraber/skills/blob/main/skills/sdd-derive/SKILL.md) to help the user derive the baseline specs.
To improve overall coherence, agents following the `sdd*` skills will suggest defining a "NORTH-STAR", a short elevator pitch defining the reason a project exists.
The north star is like Spec Kit's Constitution, except designed to maintain coherence to the _product_ rather than to software engineering _procedure_ better left to AGENTS.md.
Then, each proposal includes user stories (_"As a {role}, I want {goal}, so that {value}"_) that align with the north star, and each requirement defined in the specs must be associated with at least one user story.

Along the way, I've also refined instructions.
Specs are defined by outcomes, not mechanisms.
The `spec.md` should define behavior and outcome, while `design.md` decides mechanism and implementation detail.
That way, when the spec gets synced, the baseline defines expected behavior rather than implementation detail.
This prevents refactors or changes to underlying code that otherwise maintain behavior from requiring new spec changes.
The verification step suggests using subagents to keep the review within the "smart zone" of an agent's context window.
Verification also mandates running the full test suite and ensuring that evidence is recorded demonstrating code quality and spec:code alignment.
During implementation, there are often discoveries of new edge cases that must be handled that were not pre-specified during the propose phase.
The SDD instructions require these spec updates to record their changes so overzealous agents don't induce drift that misaligns spec from original intention over the course of the change.

## SDD is not enough

Spec-driven development is still not a silver bullet for delivering high-quality software.
Within the bounds of the spec, agents powered by modern LLMs are quite capable.
However, all of the work _surrounding_ the spec-driven development requires careful attention: it is still difficult to ensure agents fully comprehend the user's intention and carry it through to fruition, and agents are still poor caretakers of codebase quality.

During my SDD iterations, I spend about half of my time working to translate ideas in my head into the agent's context so we can collaborate to draft the specifications.
If I need help extracting ideas I have a hard time putting into words, my [`grill-me`](https://github.com/ahgraber/skills/blob/main/skills/grill-me/SKILL.md) skill (all credit to [Matt Pocock's original](https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md); mine is lightly refined to the way I like to work) has the agent provide an elicitative interview to extract and solidify those concepts.
Then I can use [`brainstorming`](https://github.com/ahgraber/skills/blob/main/skills/brainstorming/SKILL.md), which sets the agent in opposition to the user to help expand, challenge, and refine ideas.

Then I can spend the remaining half of my time iterating through the SDD process -- propose, apply, verify, sync+archive.
Between apply and verify, I also use [`code-review`](https://github.com/ahgraber/skills/blob/main/skills/code-review/SKILL.md).
`sdd-verify` checks spec against code; it does not assess code quality or look for implementation errors.

Finally, I spend another half of my time (_wait, what?!_) on codebase maintenance.
Dex Horthy at HumanLayer has a few GitHub blog posts ([1](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-opus-5-on-slop-code-bench.md), [2](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-sol-fable-kimi-on-slop-code-bench.md), [3](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-new-models-on-slop-code-bench.md)) that test modern models on SlopCodeBench and find that even frontier models like Fable and Astra contribute to codebase sprawl - a few turns of an agent increase code complexity and verbosity.
Inspired by this (and simultaneously frustrated by how garbage Opus 5 was at writing anything comprehensible), I developed [`simplify`](https://github.com/ahgraber/skills/blob/main/skills/simplify/SKILL.md) and [`refactor`](https://github.com/ahgraber/skills/blob/main/skills/refactor/SKILL.md) as an initial backstop.

Perhaps you see where this is going.
The new AI-assisted SDLC (software development lifecycle) uses AI agents to help get the ideas out of your head, to draft the change, and to implement the change.
But agents can't be trusted, so we use other agents to write tests.
But those tests might not align with the original intention, so we have other agents check spec:code alignment, code:test alignment, and spec:test alignment.
And over the course of all of that work, agents have destroyed the codebase quality, so we have yet more agents trying to claw back sprawl and refactor code back into coherence.
Agents are powerful, and can accelerate users within bounded work.
But they're not a panacea for doing _good_ work, and users cannot deploy swarms and loops of agents without thoughtful, critical oversight.
