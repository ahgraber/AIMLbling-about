# %%
from argparse import ArgumentError
from collections import defaultdict
import gc
import json
import logging
from pathlib import Path
import sys
import time

from IPython.display import display
from langcodes import Language, find as lcfind, standardize_tag
from tqdm.auto import tqdm, trange

import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import seaborn as sns

from aiml.utils import basic_log_config, get_repo_path

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "language-identification"

# %%
sys.path.insert(0, str(LOCAL_DIR))
from models import FastText, LangID, Lingua, StanzaNLP  # NOQA: E402

# %%
DATA_DIR = LOCAL_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Facebook's fasttext (https://github.com/facebookresearch/fastText) and Google's cl3d (https://github.com/google/cld3) were both deprecated mid-2024.

# Objective: identify model or implementation that is not deprecated.  This could be using a fasttext or cld3 model through huggingface rather than their own packages, or a new approach.
#
# Models under consideration
#
# - [saffsd/langid.py: Stand-alone language identification system](https://github.com/saffsd/langid.py)
# - [pemistahl/lingua-rs: The most accurate natural language detection library for Rust, suitable for short text and mixed-language text](https://github.com/pemistahl/lingua-rs)
# - [facebook/nllb-200-distilled-600M · Hugging Face](https://huggingface.co/facebook/nllb-200-distilled-600M) via [facebookresearch/fairseq at nllb](https://github.com/facebookresearch/fairseq/tree/nllb)
# - [Language Identification - Stanza](https://stanfordnlp.github.io/stanza/langid.html)

# %%
# Load Dataset(s)
_dfs = []
for file in (DATA_DIR).glob("*.parquet"):
    _df = pd.read_parquet(file)
    _df["corpus"] = file.stem
    _dfs.append(_df)

benchmark_df = pd.concat(_dfs, ignore_index=True)
del _dfs

# %%
fig, ax = plt.subplots()
benchmark_df["label"].value_counts().plot(kind="bar", ax=ax, figsize=(10, 6))
fig.suptitle("Sample Count x Language")
fig.tight_layout()
plt.show()

# with pd.option_context("display.max_columns", None, "display.max_rows", None):
#     display(benchmark_df["label"].value_counts())
#     display(benchmark_df["label"].value_counts(normalize=True).round(4))

# %%
benchmark_df["label"].value_counts()[-1:]
# 'lb' only has 961 samples

# %%
# TODO: getting "sentence length" is nontrivial, especially for chinese, korean, japanese ...
# could use spacy to tokenize or KoNLPy/jieba
# benchmark_df['length'] = benchmark_df['text'].apply(lambda x: len(x.split()))

# %%
# Run models on data
for factory in [FastText, LangID, Lingua, StanzaNLP]:
    gc.collect()
    time.sleep(3)
    model = factory()
    print(f"{model.name}: {model.memory:.2f}MB")
    gc.collect()

# FastText: 1137.64MB
# langid: 104.58MB
# lingua: 666.38MB
# stanza: 487.12MB

# %%
fasttext = FastText()
langid = LangID()
lingua = Lingua()
stanza = StanzaNLP()

models = [
    fasttext,
    langid,
    lingua,
    stanza,
]


# %%
print("Supported Language Count: " + str({model.name: len(model.supported_languages) for model in models}))
all_supported_langs = set.intersection(*[set(model.supported_languages) for model in models])

# %%
runtimes = defaultdict(list)
for i in trange(20):
    sample = benchmark_df.groupby("label").sample(n=100, replace=False)
    sample = sample.sample(frac=1, replace=False)  # shuffle sample

    sample["text"] = sample["text"].str.replace("\n", "  ")  # some models don't like newlines in text strings
    sample["sample_id"] = i

    for model in tqdm(models):
        print(f"Running benchmark for {model}...")
        start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        preds = model.predict_batch(sample["text"].tolist())
        elapsed_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start_time

        sample[model.name] = preds  # [pred[0] for pred in preds]  # add column/model
        runtimes[model.name].append(elapsed_time)

    sample.to_csv(DATA_DIR / f"results_run{i}.csv", index=False)

print("Done!")

# %% [markdown]
# ## Evaluation

# %%
runtimes_sec = {}
for key in runtimes:
    runtimes_sec[key] = [x / 1_000_000_000 for x in runtimes[key]]

display(pd.DataFrame(runtimes_sec).mean())
pd.DataFrame(runtimes_sec).plot(title="Time to Process 100 Samples", ylabel="runtime (sec)")

# %%
results_df = pd.concat([pd.read_csv(csv) for csv in DATA_DIR.glob("results*.csv")], ignore_index=True)

# convert norm_label
results_df["norm_label"] = results_df["label"].apply(lambda x: Language.get(x).language)

# fasttext (and others?) might use 'arb' instead of 'ar'
results_df = results_df.replace("arb", "ar")


# %%
def ci_mean(series: pd.Series, confidence: float = 0.95, decimals: int | None = 4) -> tuple[float, float]:
    """Get confidence interval of a mean."""
    if len(series) < 30:
        ci = stats.t.interval(confidence, len(series) - 1, series.mean(), series.sem())
    else:
        ci = stats.norm.interval(confidence, series.mean(), series.sem())

    if round:
        ci = (ci[0].round(decimals), ci[1].round(decimals))

    return ci


# %%
# overall metrics
accuracy = defaultdict(list)
f1 = defaultdict(list)

for model in models:
    for _, df in results_df.groupby("sample_id"):
        y_true = df["norm_label"]
        accuracy[model.name].append(
            accuracy_score(
                y_true=y_true,
                y_pred=df[model.name],
                normalize=True,
            )
        )
        f1[model.name].append(
            f1_score(
                y_true=y_true,
                y_pred=df[model.name],
                average="micro",
            )
        )


accuracy_df = pd.DataFrame(accuracy)
f1_df = pd.DataFrame(f1)

accuracy_df.plot(kind="box", title="Accuracy", ylim=(0, 1))
display(accuracy_df.aggregate(["mean", "std", ci_mean]))

f1_df.plot(kind="box", title="F1 Score", ylim=(0, 1))
display(f1_df.aggregate(["mean", "std", ci_mean]))


# %% [markdown]
# `FastText` outperforms all other models, but this is exacerbated by the fact that it also supports the most languages.

# %%
# metrics on subset of supported languages
accuracy = defaultdict(list)
f1 = defaultdict(list)

for model in models:
    supported_df = results_df.copy()
    supported_df = supported_df[supported_df["norm_label"].isin(model.supported_languages)]
    print(
        f"{model.name}: Dropping unsupported languages {set(results_df['norm_label']).difference(model.supported_languages)} from evaluation"
    )
    assert len(results_df) >= len(supported_df)  # NOQA: S101

    for _, df in supported_df.groupby("sample_id"):
        y_true = df["norm_label"]
        accuracy[model.name].append(
            accuracy_score(
                y_true=y_true,
                y_pred=df[model.name],
                normalize=True,
            )
        )
        f1[model.name].append(
            f1_score(
                y_true=y_true,
                y_pred=df[model.name],
                average="micro",
            )
        )
del supported_df

accuracy_df = pd.DataFrame(accuracy)
f1_df = pd.DataFrame(f1)

accuracy_df.plot(kind="box", title="Accuracy - Supported Languages", ylim=(0, 1))
display(accuracy_df.aggregate(["mean", "std", ci_mean]))

f1_df.plot(kind="box", title="F1 Score - Supported Languages", ylim=(0, 1))
display(f1_df.aggregate(["mean", "std", ci_mean]))

# %% [markdown]
# ## Results
#
# `Lingua` outperforms `FastText` once language support is taken into consideration; while 97 languages (`Lingua`) << 209 (`FastText`), we're still covering the majority of languages a LLM will likely encounter.
# Additionally, `Lingua` is just as fast as `FastText`

# %%
# metrics on most popular languages
WEB_TOP_LANGUAGES = [
    "English",
    "Spanish",
    "Russian",
    "German",
    "French",
    "Japanese",
    "Portuguese",
    "Turkish",
    "Italian",
    "Persian",
    "Dutch",
    "Polish",
    "Chinese",
    "Vietnamese",
    "Indonesian",
    "Czech",
    "Korean",
    "Ukrainian",
    "Arabic",
    "Greek",
]
WEB_TOP_LANG_CODES = [lcfind(lang).language for lang in WEB_TOP_LANGUAGES]

accuracy = defaultdict(list)
f1 = defaultdict(list)

for model in models:
    if not set(WEB_TOP_LANG_CODES).issubset(fasttext.supported_languages):
        print(
            f"Warning: {model.name} does not support web top languages {set(WEB_TOP_LANG_CODES) - fasttext.supported_languages}"
        )
    critical_df = results_df.copy()
    critical_df = critical_df[critical_df["norm_label"].isin(WEB_TOP_LANG_CODES)]

    assert len(results_df) >= len(critical_df)  # NOQA: S101

    for _, df in critical_df.groupby("sample_id"):
        y_true = df["norm_label"]
        accuracy[model.name].append(
            accuracy_score(
                y_true=y_true,
                y_pred=df[model.name],
                normalize=True,
            )
        )
        f1[model.name].append(
            f1_score(
                y_true=y_true,
                y_pred=df[model.name],
                average="micro",
            )
        )

del critical_df

accuracy_df = pd.DataFrame(accuracy)
f1_df = pd.DataFrame(f1)

accuracy_df.plot(kind="box", title="Accuracy - Web Top Languages", ylim=(0, 1))
display(accuracy_df.aggregate(["mean", "std", ci_mean]))

f1_df.plot(kind="box", title="F1 Score - Web Top Languages", ylim=(0, 1))
display(f1_df.aggregate(["mean", "std", ci_mean]))

# %% [markdown]
# FastText actually underperforms on these web critical languages

# %%
# what is the per-language performance
critical_df = results_df.copy()
critical_df = critical_df[critical_df["norm_label"].isin(WEB_TOP_LANG_CODES)]

f1_dfs = []
for model in models:
    accuracy = defaultdict(list)
    f1 = defaultdict(list)
    for label, label_df in critical_df.groupby("norm_label"):
        for _, df in label_df.groupby("sample_id"):
            y_true = df["norm_label"]
            accuracy[label].append(
                accuracy_score(
                    y_true=y_true,
                    y_pred=df[model.name],
                    normalize=True,
                )
            )
            f1[label].append(
                f1_score(
                    y_true=y_true,
                    y_pred=df[model.name],
                    average="micro",
                )
            )

    # accuracy_df = pd.DataFrame(accuracy)
    f1_df = pd.DataFrame(f1)
    # f1_df['model'] = model
    f1_dfs.append(f1_df.assign(model=model.name))
    # f1_df.drop(columns='model')
    # accuracy_df.plot(kind="box", title=f"{model.name} Accuracy - Web Top Languages", ylim=(0,1))
    # display(accuracy_df.aggregate(["mean", "std", ci_mean]))

    f1_df.plot(kind="box", title=f"{model.name} F1 Score - Web Top Languages", ylim=(0, 1))
    display(f1_df.aggregate(["mean", "std", ci_mean]))


# %%
f1_scores = (
    pd.concat(f1_dfs)
    .melt(id_vars="model", var_name="language", value_name="f1")
    .groupby(["language", "model"])
    .aggregate("mean")  # , "std"])
    .unstack(-1)
    .droplevel(level=0, axis="columns")
    .round(3)
    .reindex(WEB_TOP_LANG_CODES)
    .assign(language_name=WEB_TOP_LANGUAGES)
    .reset_index(drop=True)
    .set_index("language_name")
    .rename_axis("language")
)
display(f1_scores)

# %%
fig, ax = plt.subplots(figsize=(7, 10))
sns.heatmap(
    f1_scores,
    annot=True,
    cmap=cmc.managua,
    vmin=0,
    vmax=1,
    cbar_kws={"shrink": 0.5},
)

# %% [markdown]
# Chinese ('zh') pulls down FastText's performance considerably.
# This may be erratta, since FastText actually classifies multiple subtypes of Chinese ('zh-Hans', 'zh-Hant') with script designations,
# while the datasets specify ['zh_cn', 'zh_tw', 'zh'] which regionalize the language.

# %% [markdown]
# ## Conclusion
#
# `Lingua` is the best option to replace `FastText` (or Googles)
# Its classification performance meets `FastText` on supported languages, and critically exceeds `FastText` on web top languages.
# `Lingua` is as fast as `FastText` and has equivalent compute/memory requirements.

# %%
