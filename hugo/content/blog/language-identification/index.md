---
title: "When the Babel Fish Dies: Replacing Legacy Language Detection"
date: 2025-01-26T09:18:54-05:00
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - "blogumentation"
  - "experiment"
  # ai/ml
  - "evals"
series: []
layout: single
toc: true
math: false
draft: true
---

Facebook's [`fasttext`](https://github.com/facebookresearch/fastText) and Google's [`cl3d`](https://github.com/google/cld3)) were both deprecated mid-2024.
While multilingual LLMs can accomplish language identification tasks, using even a 7B-parameter LLM to determine a text's language is overkill. I set out to find a replacement that is at least as fast and performant as the deprecated models.

Code from these experiments is available [here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/language-identification).

## TLDR

[`Lingua`](https://pypi.org/project/lingua-language-detector/) is the best option to replace `fasttext` (or presumably, `cl3d`).
Its classification performance meets `fasttext` on supported languages, and critically exceeds `fasttext` on the most frequently used languages on the web. `Lingua` is as fast as `fasttext` and has equivalent compute/memory requirements.

## Experiment Design

I'll be using `fasttext` as the base case. For one, it supports far more languages ([217](https://huggingface.co/facebook/fasttext-language-identification)) than `cl3d` (107).
Secondly, I couldn't get `cl3d` to install 😅.

After quite a bit of research and some initial testing to [weed out obvious weak solutions](#rejected-options), I arrived at the following competitive set:

| Model                                                        | Supported Languages |
| ------------------------------------------------------------ | ------------------: |
| [langid](https://github.com/saffsd/langid.py)                |                  97 |
| [lingua](https://pypi.org/project/lingua-language-detector/) |                  75 |
| [stanza](https://stanfordnlp.github.io/stanza/langid.html)   |                  67 |

I run a sample dataset (defined below) through all of the models and assess their runtime, memory utilization, and classification accuracy.

### Data

The benchmark dataset is composed of samples from 4 different datasets:

- [OpenSubtitles - download movie and TV Series subtitles](https://www.opensubtitles.org/en/search/subs) via [OpenSubtitles corpus](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles)
- [Tatoeba: Collection of sentences and translations](https://tatoeba.org/en/) via [Tatoeba corpus](https://opus.nlpl.eu/Tatoeba/corpus/version/Tatoeba)
- [papluca/language-identification | Hugging Face](https://huggingface.co/datasets/papluca/language-identification)
- [CohereForAI/Global-MMLU | Hugging Face](https://huggingface.co/datasets/CohereForAI/Global-MMLU)

{{< figure
  src="images/language_samples.png"
  caption="The final dataset contains 80 languages, most with 5000+ samples" >}}

### Evaluation

The experiment is run multiple times (20) to generate confidence intervals.
For each run, a set of 100 records per language is sampled from the overall dataset (100 records \* 80 languages = 8,000 predictions).
Each model predicts over the same dataset, and predictions and runtime are captured.
Classification performance is evaluated for each run for each model using multi-class F1 score and Accuracy (not shown).

## Results

### Runtime

`fasttext` is the fastest model, followed closely by `Lingua`; `LangId` and `Stanza` lag behind.

{{< figure
  src="images/runtime.png"
  caption="Runtime for 100 samples (lower is better). fasttext and Lingua are >10x faster than LangId and Stanza" >}}

| model    | Mean Runtime |
| :------- | -----------: |
| fasttext |        0.804 |
| LangId   |        9.718 |
| Lingua   |        0.917 |
| Stanza   |       24.568 |

### Memory

I had a hard time validating the memory use of these models; I should probably try again with a dedicated memory profiler
instead of hacking at it with `psutil`.
Take the following with a grain of salt.

| model    | Memory (MB) |
| :------- | ----------: |
| fasttext |     1137.64 |
| LangId   |      104.58 |
| Lingua   |      666.38 |
| Stanza   |      487.12 |

### Classification Performance

#### Overall

`fasttext` exhibits the strongest classification performance (F1 score); its performance lead is exaggerated on the base benchmark dataset because it supports all languages in the benchmark, while other models do not.
Therefore the performance ceiling for other models is lower.

{{< figure
  src="images/f1score.png" >}}

#### Supported Languages

When considering the subset of languages supported by each model, `Lingua` outperforms `fasttext` once language support is taken into consideration; while 97 languages (`Lingua`) << 209 (`fasttext`),
we're still covering the majority of languages found on the web.

{{< figure
  src="images/f1score-supported.png" >}}

| F1 Score | fasttext | langid | lingua | stanza |
| :------- | -------: | -----: | -----: | -----: |
| mean     |    0.835 |  0.737 |  0.859 |  0.843 |
| std      |    0.005 |  0.004 |  0.005 |  0.006 |

#### Web Top Languages

Constraining the experiment further, I examine performance on the 20 most frequent languages found on the web ([according to wikipedia](https://en.wikipedia.org/wiki/Languages_used_on_the_Internet))

{{< figure
  src="images/f1score-web.png">}}

{{< figure
  src="images/f1score-web-heatmap.png">}}

{{< callout type="warning" >}}
Chinese ('zh') appears to pull down fasttext's performance considerably.
This may be erroneous, since fasttext actually classifies multiple subtypes of Chinese ('zh-Hans', 'zh-Hant') with script designations, while the datasets specify ['zh_cn', 'zh_tw', 'zh'] which regionalize the language.
{{< /callout >}}

## References

- [Comparison of language identification models | model.predict](https://modelpredict.com/language-identification-survey)
- [Language Identification for very short texts: a review | by Jade Moillic | Besedo Engineering Blog | Medium](https://medium.com/besedo-engineering/language-identification-for-very-short-texts-a-review-c9f2756773ad)

## Appendix

### Rejected options

- [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection) - slow (10x slower than others), fewer languages (20 vs 70+)
- [adbar/simplemma](https://github.com/adbar/simplemma) - primarily focuses on lemmatization; language detection is side effect
- [ssut/py-googletrans](https://github.com/ssut/py-googletrans) - uses API instead of local
- [Spacy FastLang](https://spacy.io/universe/project/spacy_fastlang) - uses fasttext library
- [mbanon/fastspell](https://github.com/mbanon/fastspell) - uses fasttext library
- [Mimino666/langdetect](https://github.com/Mimino666/langdetect) - langdetect is old
- [davebulaval/spacy-language-detection](https://github.com/davebulaval/spacy-language-detection) --> just use langdetect directly
- [Abhijit-2592/spacy-langdetect](https://github.com/Abhijit-2592/spacy-langdetect) --> just use langdetect directly
