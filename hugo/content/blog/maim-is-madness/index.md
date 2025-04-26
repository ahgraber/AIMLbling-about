---
title: MAIM Is MADness
date: 2025-03-09
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
  - LLMs
series: []
layout: single
toc: true
math: false
plotly: false
draft: false
---

_[Superintelligence Strategy](https://www.nationalsecurity.ai/)_ is a policy paper by Dan Hendrycks (Director, Center for AI Safety), Eric Schmidt (former CEO, Google), and Alexandr Wang (CEO, Scale AI), that proposes a three-part framework to manage the risk associated with Artificial SuperIntelligence (ASI).
Strategies include Mutual Assured AI Malfunction (**MAIM** -- shouldn't "Mutual Assured AI Malfunction" be "MAAIM"? They stretched to make it catchy...), a _deterrence_ strategy akin to Mutually Assured Destruction (MAD), _nonproliferation_, and _competitiveness_.

At the risk of being reductive, the argument is as follows:

1. Artificial SuperIntelligence (ASI) is risky - _"Superintelligent AI surpassing humans in nearly every domain would amount to the most precarious technological development since the nuclear bomb... AI holds the potential to reshape the balance of power. In the hands of state actors, it can lead to disruptive military capabilities and transform economic competition. At the same time, terrorists may exploit its dual-use nature to orchestrate attacks once within the exclusive domain of great powers. It could also slip free of human oversight."_

2. Therefore, managing ASI is critical for national security.

3. Therefore, nations should have an interest in managing AI before we get to ASI.

4. Therefore, nations with the greatest leverage will have strategic policy advantages.

5. Therefore, the nations with AI leadership (read US, China) should employ strategies for deterrence, nonproliferation, and competition.

## Is MAIM the new MAD?

Because of these risks, the authors assert that policymakers should anticipate a deterrence strategy for ASI (MAIM, Mutual Assured AI Malfunction) inspired by that of nuclear weapons (MAD, Mutual Assured Destruction) as it is the only stable regime.

{{< figure
src="images/MAIM.png"
alt="Mutual Assured AI Malfunction"
caption="The strategic stability of MAIM can be paralleled with Mutual Assured Destruction (MAD). (_Superintelligence Strategy_)" >}}

### Mutual Assured Destruction

Before going further, I think it's worth examining [Mutual Assured Destruction](https://en.wikipedia.org/wiki/Mutual_assured_destruction).

> [!NOTE]
> I think the game theory framework is useful here, but it's been a minute since I've done any game theory, so I'll probably gloss over (quite a lot) with some handwaving.
> I'm also not going to spend a ton of time covering game theory terminology; if you'd like an intro or a refresher, [Ashley Hodgson has a great series of videos](https://www.youtube.com/playlist?list=PLL6RiAl2WHXEkYMhQVensZMF5LyoJoOk0).

{{< figure
src="images/MAD game.png"
alt="Mutual Assured Destruction"
caption="The uneasy truce" >}}

Mutual Assured Destruction is a game where the stable state (Nash Equilibrium) is an uneasy truce. Both A and B would prefer to strike first, annihilate the other and survive themselves. However, the non-cooperation strategies are untenable (annihilation) and therefore "hold" is the dominant strategy. For MAD to enforce the uneasy truce, both players must:

1. Have (sufficiently) equivalent capability to annihilate the other, should they choose to strike.
2. Be able to guarantee retaliation (second strike capability). If a player can strike without retaliation, it destabilizes the game.

Historically, guaranteeing second-strike capability is why nuclear superpowers ended up with the Nuclear Triad (Missiles, Planes, Submarines), and why there are treaties limiting development of anti-ballistic-missile systems -- these are imperatives to maintain the stability of the MAD regime.
It is also why there was (relatively speaking) global cooperation on nuclear non-proliferation. New players can only enter the MAD game without destabilizing it _if and only if_ they can demonstrate both required capabilities. If a new player has nukes but no second-strike capability, it is in that player's best interest to strike first, leading to the [End of the World](https://www.youtube.com/watch?v=kCpjgl2baLs).

Interestingly, MAD is the inverse of nonproliferation (which can be modeled as the Prisoner's Dilemma).

{{< figure
src="images/PD game.png"
alt="Prisoner's Dilemma"
caption="Individual self-interest plays out over cooperation, undermining the best case as stable strategy." >}}

### MAIM ≠ MAD

The authors of _Superintelligence Strategy_ believe that an AI monopoly is a destabilizing force:

> If a rival state races toward a strategic monopoly, states will not sit by quietly.
> If the rival state loses control, survival is threatened; alternatively, if the rival state retains control and the AI is powerful, survival is threatened.
> A rival with a vastly more powerful AI would amount to a severe national security emergency, so superpowers will not accept a large disadvantage in AI capabilities.
> Rather than wait for a rival to weaponize a superintelligence against them, states will act to disable threatening AI projects, producing a deterrence dynamic that might be called Mutual Assured AI Malfunction, or MAIM.

Thus, MAIM (ostensibly) is a stable game for nations afraid of being dominated by other AI-empowered competitors, as MAD is a stable game for nations afraid of being nuked. However, I assert this is a false equivalency. MAD works because the threat is real - all players can carry out the threat, the threatened action is real and terrible, and the results are immediate. The threat in MAIM is speculative (AI risk) and/or insufficiently scary (sabotage). It is also unclear in MAIM whether all players have equivalent abilities to carry out the threat, and whether second-strike is guaranteed.

The authors seem to be conflating several distinct games in the guise of MAIM and declaring all to be equivalent to MAD. In one, all players must race toward AI dominance. In another, players must sabotage competitors. Finally (implied), players must determine whether to employ ASI offensively.

{{< figure
src="images/MAIM outcomes.png"
alt="Possible outcomes of a U.S. Superintelligence Manhattan Project"
caption="Possible outcomes of a U.S. Superintelligence Manhattan Project. An example pathway to Escalation: the U.S. Project outpaces China without being maimed, and maintains control of a recursion but doesn't achieve superintelligence or a superweapon. Though global power shifts little, Beijing condemns Washington's bid for strategic monopoly as a severe escalation. The typical outcome of a Superintelligence Manhattan Project is extreme escalation, and omnicide is the worst foreseeable outcome. (_Superintelligence Strategy_)" >}}

Recall from the above, MAD requires three things: Non-cooperation strategies must be untenable due to real extreme consequence, the threat (and follow-through) from both players must be equivalent, and there can be no first-strike win.

#### Game 1: AI Arms Race: Accelerating Innovation

The first MAIM subgame is much closer to the nonproliferation / Prisoner's Dilemma game than Mutual Assured Destruction. Although players may admit that cooperating to limit the risk of ASI is the better outcome, they are incentivized to accelerate their own development out of fear of being left behind. This leads to a regime of escalation - an AI arms race.

The Prisoner's Dilemma is emphatically **not** equivalent to Mutual Assured Destruction; non-cooperation is _suicidal_ in MAD but _rational_ in the race for AI dominance. If the analogy to nuclear strategy holds, this game can really only be played prior to multiple parties achieving ASI, after which the third subgame becomes more relevant.

#### Game 2: Playing Chicken in an AI Arms Race

The second subgame is another escalation game - this time more closely related to Chicken than the Prisoner's Dilemma - where players sabotage each other in order to gain advantage. If an opponent is on track to beat you to ASI (or some other meaningful milestone) and AI dominance is perceived as an existential threat, the rational move is to delay or cripple their progress before they achieve superiority. AI is reliant on vulnerable infrastructure, making data centers, semiconductor supply chains, and electrical or telecommunication grids ripe targets for sabotage.

Whereas existential risk forces stability in nuclear deterrence, the risk of sabotage is asymmetric and deniable, leading to an unstable game. The cost of being sabotaged may be high (both in terms of progress and in terms of advantage), but it is not _guaranteed annihilation_. This means there is an incentive to strike first and strike hard, such that even if the other party retaliates, the game resets to parity (or still leaves the preempting player with the advantage). Like in Chicken, the hope is to force the opponent to yield first and accept their disadvantage without triggering uncontrolled escalation (full-scale military action, war).

#### Game 3: MAD or Myth? The Uncertainty of AI Weaponization

In the implied third subgame, players must determine _whether and how_ to weaponize their AI capabilities. The authors treat AI as a single threat system, like a WMD. However, AI is _not_ a single threat with guaranteed outcomes but an amplification technology that can enhance many threat surfaces. As a result, there is uncertainty in what and how scary the threat is that leads to uncertainty such that undermines shared deterrence.

Reiterating from above, MAD requires all players to have a proven ability for immediate, guaranteed, existential retaliation. Nuclear weapons fulfill the requirements for deterrence because their effects are real (proven), demonstrable, and terrible. A nation can demonstrate they possess nuclear weapons and the ability to deliver them in a second-strike scenario through tests and military exercises.

AI's broad application means that the threats are asymmetric and speculative, unlike MAD. Will the threats be sufficiently terrible such that it makes inviting retaliation completely unacceptable? Will each player's threat be equivalent? Will the threats be demonstrable such that each player has a valid expectation of guaranteed retaliation? The uncertainty inherent in each of these questions makes it unlikely for deterrence to form a stable regime. AI-driven threats are varied, leading to asymmetric risk, and difficult to demonstrate, leading to uncertainty of risk magnitude and guarantee. The lack of proof undermines the effectiveness the threat has on deterrence, and the asymmetry undermines the stability of the game. MAIM, as proposed, is not a rational strategy.

## Nonproliferation and Competitiveness

MAIM is not the sole strategy for managing strategic implications of AI. _Superintelligence Strategy_ also lays out goals for nonproliferation and competitiveness. These strategies overlap with the escalation-type games as described above -- if you believe that AI leads to competitive advantage, then you must escalate (even in the face of safety concerns).

However, the escalation-type game only applies to the competition among nations. One of the fears of AI proliferation is that rogue actors will leverage AI to cause substantial harm -- AI-assisted cyberattacks, AI-accelerated bioweapons research, etc. While nations enter an escalation game with their direct competition, they are also in an exclusion/nonproliferation strategy with everyone else. This implies a strategic imperative for countries to nationalize their compute resources, sequester their researchers, and control the access to and use of AI systems. We see the opening rounds of this, with the [CHIPS Act](https://en.wikipedia.org/wiki/CHIPS_and_Science_Act) incentivizing semiconductor manufacturing in the US, and various rounds of [export restrictions](https://www.wired.com/story/new-us-rule-aims-to-block-chinas-access-to-ai-chips-and-models-by-restricting-the-world/). It is plausible that China might counter by attempting to take over Taiwan to control TSMC.

## _Potential_ Risk Are Insufficient

The problem isn't that cooperation is _difficult_ (though it may be), it's that cooperation is _irrational_ given the default incentives. As discussed above, the status quo defaults to acceleration. Nations, corporations, and researchers all fear falling behind if the other players renege from cooperation, making cooperation irrational and leading to escalation. To change this dynamic, the benefits for cooperation must significantly outweigh the incentive to escalate _and_ some arbiter of trust must exist that makes it impractical to cheat or defect from systemic cooperation.

How can we prioritize cooperation? The authors assert the potential risks from malicious use of AI are sufficient justification. They posit that the "intelligence recursion" of leveraging AI to accelerate AI research will lead to a hard takeoff scenario (a.k.a. the [singularity](https://en.wikipedia.org/wiki/Technological_singularity)), leading to systems with superhuman capabilities. These systems could be used to accelerate economic productivity, medical research, and shore up vulnerabilities in infrastructure; they could just as easily be used for cyberattack, bioweapons research, etc.

Even if potential risk were enough to pause the cycle of escalation, the lack of an arbiter for cooperation makes the system unlikely to succeed. The authors do not address the need for a neutral, cooperative, international body that has sufficient resources and power to ensure players do not renege or defect from the cooperative mode. This type of body is given power by the nations it seeks to govern, and nations do not surrender power unless there is sufficient reason; potential risk is insufficient justification to empower a moderating influence.

## MAIM is DOA, but Risk Potential is Real

_Do not let my rebuttal to the policies undermine the validity of the risk assessment._ Examining these games makes it clear that the dynamics of strategic competition for AI dominance leads to an accelerating arms race; with this capability will come real threats. And, although _Superintelligence Strategy_ fails to design stable cooperative paradigms, the authors correctly highlight that the real risk is ubiquitous AI, used maliciously. AI is an accelerant, gasoline for propaganda and mis/disinformation, market manipulation, cyberwarfare and infrastructure sabotage, and bioweapons research. Managing these risks is a Wicked Problem, requiring international cooperation that the current incentive structures do not support. Unfortunately, neither the authors nor I can offer a solution -- until a clear and present danger emerges, AI will remain an accelerating, mostly unregulated field.

## References

- [Mutual assured destruction](https://en.wikipedia.org/wiki/Mutual_assured_destruction)
- [Prisoner's dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
- [Chicken (game)](<https://en.wikipedia.org/wiki/Chicken_(game)>)
- [CHIPS Act](https://en.wikipedia.org/wiki/CHIPS_and_Science_Act)
- [New US Rule Aims to Block China's Access to AI Chips and Models by Restricting the World | WIRED](https://www.wired.com/story/new-us-rule-aims-to-block-chinas-access-to-ai-chips-and-models-by-restricting-the-world/)
- [US official says Chinese seizure of TSMC in Taiwan would be 'absolutely devastating' | Reuters](https://www.reuters.com/world/us/us-official-says-chinese-seizure-tsmc-taiwan-would-be-absolutely-devastating-2024-05-08/)
- [Technological singularity](https://en.wikipedia.org/wiki/Technological_singularity)
- [Availability heuristic](https://en.wikipedia.org/wiki/Availability_heuristic)
- [Present bias](https://en.wikipedia.org/wiki/Present_bias)
