# Language Detection

Current "best" language detection models [google/cld3](https://github.com/google/cld3) and [facebookresearch/fastText](https://github.com/facebookresearch/fastText/tree/main)
were archived in June and March of 2024, respectively.

What are the best (most accurate, fastest) alternatives that are still being actively managed/developed?

## Experiment

> The experiment must be run from the local experiment folder, not the repository root!
> This is because the environment is unique from other experiments and does not use the same dependencies.
>
> ```sh
> cd ./experiments/language-detection
> uv sync
> ```

### Data

Use `modelpredict` datasets

- [Subtitles - download movie and TV Series subtitles](https://www.opensubtitles.org/en/search/subs) via [OpenSubtitles corpus](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles)
- [Tatoeba: Collection of sentences and translations](https://tatoeba.org/en/) via [Tatoeba corpus](https://opus.nlpl.eu/Tatoeba/corpus/version/Tatoeba)

Download with:

```sh
python ./download_datasets.py
```

## References

- [Comparison of language identification models | model.predict](https://modelpredict.com/language-identification-survey)
- [Language Identification for very short texts: a review | by Jade Moillic | Besedo Engineering Blog | Medium](https://medium.com/besedo-engineering/language-identification-for-very-short-texts-a-review-c9f2756773ad)

## Options

- [saffsd/langid.py: Stand-alone language identification system](https://github.com/saffsd/langid.py)
- [pemistahl/lingua-rs: The most accurate natural language detection library for Rust, suitable for short text and mixed-language text](https://github.com/pemistahl/lingua-rs)
- Most recent FastText model
  [facebook/nllb-200-distilled-600M · Hugging Face](https://huggingface.co/facebook/nllb-200-distilled-600M) via [facebookresearch/fairseq at nllb](https://github.com/facebookresearch/fairseq/tree/nllb)
- [Language Identification - Stanza](https://stanfordnlp.github.io/stanza/langid.html)

## Rejected options

- [papluca/xlm-roberta-base-language-detection · Hugging Face](https://huggingface.co/papluca/xlm-roberta-base-language-detection) - slow (10x slower than others), fewer languages (20 vs 70+)
- [adbar/simplemma: Simple multilingual lemmatizer for Python, especially useful for speed and efficiency](https://github.com/adbar/simplemma) - primarily focuses on lemmatization; language detection is side effect
- [ssut/py-googletrans: (unofficial) Googletrans: Free and Unlimited Google translate API for Python. Translates totally free of charge.](https://github.com/ssut/py-googletrans) - uses API instead of local
- [Spacy FastLang · spaCy Universe](https://spacy.io/universe/project/spacy_fastlang) - uses fasttext
- [mbanon/fastspell: Targeted language identifier, based on FastText and Hunspell.](https://github.com/mbanon/fastspell) - uses fasttext
- [Mimino666/langdetect: Port of Google's language-detection library to Python.](https://github.com/Mimino666/langdetect) - langdetect is old
- [davebulaval/spacy-language-detection: Fully customizable language detection for spaCy pipeline](https://github.com/davebulaval/spacy-language-detection) --> just use langdetect directly
- [Abhijit-2592/spacy-langdetect: A fully customisable language detection pipeline for spaCy](https://github.com/Abhijit-2592/spacy-langdetect) --> just use langdetect directly
