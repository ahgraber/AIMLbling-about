---
title: Copyright and Licensing in GenAI Times
date: 2024-04-30T16:03:13-04:00
tags: ["generative ai", "copyright", "licencing"]
series: []
draft: true
---

- [ ] copyright and generative AI
  - the problem
    - NYT case
      - <https://katedowninglaw.com/2024/01/06/the-new-york-times-launches-a-very-strong-case-against-microsoft-and-openai/> && <https://news.ycombinator.com/item?id=38900197>
      - <https://katedowninglaw.com/2023/08/24/an-ip-attorneys-reading-of-the-llama-and-chatgpt-class-action-lawsuits/>
      - <https://katedowninglaw.com/2023/07/27/evaluating-generative-ai-licensing-terms/>
      - <https://katedowninglaw.com/2023/07/13/ai-licensing-cant-balance-open-with-responsible/>
      - <https://katedowninglaw.com/2023/01/26/an-ip-attorneys-reading-of-the-stable-diffusion-class-action-lawsuit/>
    - [Major U.S. newspapers sue Microsoft, OpenAI for copyright infringement](https://www.axios.com/2024/04/30/microsoft-openai-lawsuit-copyright-newspapers-alden-global)
    - [Udio & the age of multi-modal AI (Practical AI #265)](https://changelog.com/practicalai/265#t=697)
      - on cameras:
        > there was a time when there was a question of whether or not if you took a picture with a camera -- right, you just click the button...
        > people argued at one point - "hey, you click that button, its machine generated, you can't have a copyright for that"
        - this is a false equivalence -- cameras weren't trained on copyrighted material and using the amount of effort to craft a prompt is irrelevant in this example
    - [Author granted copyright over book with AI-generated text—with a twist | Ars Technica](https://arstechnica.com/tech-policy/2024/04/author-granted-copyright-over-book-with-ai-generated-text-with-a-twist/)
    - [The A.I. Lie | Muddy Colors](https://www.muddycolors.com/2024/04/the-a-i-lie/)
  - the solution?
    - keep track of how weights change during training; look at neurons that are activated/updated by training examples.
      then, on inference, keep track of activations and attribute credit to artists based on activations
    - TODO - are there papers that look at 'training data attribution'?
      - mixture-of-depths 'high quality tokens'
      - forgetting harry potter
