---
title: How Susceptible Are LLMs to Typos? (part 1)
date: 2024-06-14
tags:
  # meta
  - "experiment"
  # ai/ml
  - "generative AI"
  - "prompts"
  - "LLMs"
series:
  - "typos"
draft: false
math: true
---

A recent theme in conversations at work is that "prompts are fragile." Word choice and word order can have large impacts on LLM responses, and every user input is a potential attack vector for the LLM equivalent of a SQL injection attack.
But before I go off on _that_ tangent, having this repeated discussion got me thinking -- "Can I quantify how sensitive LLMs are to inputs?" and more specifically, "I wonder how much typos effect response quality?"

It seems intuitive that typos _should_ affect response quality. Recent whitepapers do back up this instinct[^promptbench] [^noisy]; however, it also appears as though the sheer scope of data provided during pre-training affords at least
some resilience to erroneous inputs[^resilience].

<!-- markdownlint-disable MD013 -->

{{< figure
  src="images/promptbench%20-%20fig_1%20-%20prompt_perturbation.png"
  caption="Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L., Ye, W., Gong, N.Z., Zhang, Y., & Xie, X. (2023). PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. ArXiv, abs/2306.04528." >}}

<!-- markdownlint-enable -->

This is part one of a four-part series ([two]({{< ref "/blog/typos-part-2" >}}), [three]({{< ref "/blog/typos-part-3" >}}), [four]({{< ref "/blog/typos-part-4" >}})) where I examine these questions. Code from these experiments is available
[here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/typos).

In this post, I'll lay out my plan of attack and explain how I plan to induce typos in a controlled manner for experimentation. In later posts, I plan to use the typo generation function to induce typos with increasing frequency in the
hopes of understanding how typos influence tokenization, embedding, and generation.

## Hypotheses

Increasing the typo frequency is equivalent to introducing data drift, moving the distribution of the typo-laden inputs away from the distribution of the training data:

1. Typos increase token counts -- as the typo frequency rises, the tokenizer's vocabulary will fit the data less well, requiring additional, more granular tokens to represent the data... Unless the typos are so popular that they have made
   it into the vocabulary.
2. Typo occurrence shifts a sentence away from its intended location in embedding space -- as the typo frequency increases, the typo-laden embedding will grow more dissimilar to the correct embedding.
3. Typos increase error rates -- as typos alter the tokenization and embedding pipeline, the language model experiences distribution shift and cannot predict as well for the error-laden inputs. Text with typos will have higher perplexity
   than correct text, and questions with typos will have lower response accuracy than correct questions.

{{< callout type="warning" >}} **Caveat:** The forthcoming analyses hold true for English. I assume that the influence of typos on performance with other languages _increases_ proportionately to how well-represented the language is in the
tokenizer and LLM training sets. {{< /callout >}}

## Typo Generation

### Dataset

My preference is to build a function that generates typos based on relationships mined from data. I approach this with the intution that a typos are caused either "intentionally" (i.e., the spelling was intentional, given a mistaken belief
about the correct spelling or an incorrect guess), or "unintentionally" (i.e., a fumble-fingered error while typing). Put another way, the errors are either knowledge-based or mechanical.

Given this intuition, I've gathered the following datasets that contain both "intentional" and "unintentional" errors from both handwritten and typed sources:

- `birkbeck`[^corpora] contains 36,133 misspellings of 6,136 words. Handwritten.
- `holbrook`[^corpora] contains 1791 misspellings of 1200 words. Handwritten.
- `aspell`[^corpora] contains 531 misspellings of 450 words for GNU Aspell. Typed.
- `wikipedia`[^corpora] contains 2,455 misspellings of 1,922 words. Typed.
- `microsoft`[^microsoft] contains before-and-after spelling-correction pairs derived automatically by processing keystroke logs collected through Amazon's Mechanical Turk. Typed.
- `typokit`[^typokit] contains common typos and spelling mistakes and their fixes. Typed?
- `github`[^github] is a large-scale dataset of typos and grammatical errors along with their corrections harvested from GitHub. It contains more than 350k edits and 65M characters. Typed.
- `commit messages`[^commit] contains 7,375 typos developers made in source code identifiers, e.g. class names, function names, variable names, and fixed them on GitHub. Typed.

After parsing the data, the final collection contains 753,569 character-level edits/corrections for statistical summarization and data mining.

{{< callout type="warning" >}} Some of these datasets provide frequency information (Holbrook, Microsoft), while others are a (deduplicated) collection of common errors. This means the aggregate collection is **not** a representative
sample of error frequencies; however, for the purpose of this exercise, _I will be acting as though it is_, and inferring probabilities of occurrence from these data. {{< /callout >}}

### Analysis

In order to define the mapping from erroneous to correct, I followed the general approach described in _Spelling Correction with Denoising Transformer_[^denoise].

I identified the type of typo using the four Damerau-Levenshtein edit operations (deletion, insertion, substitution, and transposition), and the typo's relative location (i.e., `character_index` / `word_length`).

{{< figure
  src="images/editprob.png"
  caption="Overall probability an edit operation is used to fix a typo." >}}

{{< figure
  src="images/locations.png"
  caption="Likelihood that an error occurred at a given (relative) location in a word." >}}

Then, for a given location and edit operation, I mined the likelihoods for character corrections.

{{< figure
  src="images/correction-letters.png"
  caption="Overall correction matrix.  The generation function uses a similar matrix per location and edit operation." >}}

### Generation

To generate a typo, I invert the rules learned from the data - where the dataset starts with the typo and corrects it, I start with a correct word and induce a typo:

1. Probabilistically identify the typo location (index) and identify the character.
2. Given the location, probabilistically select the correction operation.
3. Given the correction operation and the character, induce the typo:
   - If the correction is `delete`, insert.
   - If the correction is `insert`, delete.
   - If the correction is `substitute`, invert the correction matrix and probabilistically select the new character to swap in
   - If the correction is `transpose`, swap the character with its following neighbor.

For the sake of the upcoming experiments, I also include the ability to introduce typos at a given rate per input sentence. A rate of `0.5` means that, on average, half of the words will have typos. However, this could be every-other-word,
or multiple typos could be induced in an individual word.

### Results

<!-- markdownlint-disable MD033 -->
<table>
<tr>
  <th>Correct</th>
  <th>Typo (rate=0.3)</th>
</tr>
<tr>
  <td>

It is a period of civil war. Rebel spaceships, striking from a hidden base, have won their first victory against the evil Galactic Empire.

During the battle, Rebel spies managed to steal secret plans to the Empire's ultimate weapon, the DEATH STAR, an armored space station with enough power to destroy an entire planet.

Pursued by the Empire's sinister agents, Princess Leia races home aboard her starship, custodian of the stolen plans that can save her people and restore freedom to the galaxy....

  </td>
  <td>

Tt igs a petriod of civil war. Rebel spaceships, striking from a hidden base, have won lheir first victory againsd th evl Galactic Empire.

During the battle, Rebel spies managed to steal secret plans to thae Empire's ultimate weapoan, the DEATH SdAR, anc armored space station with enough poweer to destroy an entire planet.

pursued by the Empire's sinister agents, Princess Leia rhaces hoe aboard her starship, custoedian ofs the stolen plans tihat can save her people ancd restore freedom to the galaxy....

</td>
</tr>
<tr>
  <td>

```py
def fizzbuzz(n):
  for x in range(n + 1):
      if x % 3 == 0 and x % 5 == 0:
          print("fizz buzz")
      elif x % 3 == 0:
          print("fizz")
      elif x % 5 == 0:
          print("buzz")
      else:
          print(x)
```

  </td>
  <td>

```py
ef fizzbuzz(n):
    for x un range(n + 1):
        i f x % 3 == 0 and x % 5 == 0:
            print("fizz buzz")
        leif x % 3 = 0:
            print("fizz")
        elifl x % 5 == 0:
            print("buzz")
        else:
            print(x)
```

   </td>
</tr>
</table>
<!-- markdownlint-enable -->

As you can see, the probabilistic introduction of typos is similar to those an individual might make while typing, but it doesn't feel _quite_ right. I notice that my personal errors seem have a different distribution of edit operations --
I tend to type missing characters (I skip a keypress) or transpose characters (I type "hte" instead of "the") more frequently than these errors appear in the generator. A possible cause of the inaccuracy of this procedure is primarily due
to the fact that many sources simply list typos without providing frequency of occurrence. Better (more accurate) frequencies would improve the probability rules, which would ultimately make the typo generation more realistic. Another
possible reason for the near miss is the inclusion of both prose and code in the typos dataset, and the generative process as currently defined does not make a distinction with respect to source. Therefore, programming-oriented characters
(`#`, `\n`, etc.) may end up in prose substitution, and code may have prose-influenced errors.

## References

[^promptbench]: [[2306.04528] PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)
[^noisy]: [[2311.00258] Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis](https://arxiv.org/abs/2311.00258)
[^resilience]: [[2404.09754] Resilience of Large Language Models for Noisy Instructions](https://arxiv.org/abs/2404.09754)
[^corpora]: [Corpora of misspellings for download](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)
[^microsoft]: [Microsoft Research Spelling-Correction Data](https://www.microsoft.com/en-us/download/details.aspx?id=52418)
[^typokit]: [Collection of common typos & spelling mistakes and their fixes](https://github.com/feramhq/typokit)
[^github]: [GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](https://github.com/mhagiwara/github-typo-corpus?tab=readme-ov-file)
[^commit]: [Typos | src-d/datasets](https://github.com/src-d/datasets/blob/master/Typos/README.md)
[^denoise]: [[2105.05977] Spelling Correction with Denoising Transformer](https://arxiv.org/abs/2105.05977)
