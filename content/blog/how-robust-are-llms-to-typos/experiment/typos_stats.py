# %%
from collections import defaultdict
import difflib
import json
from pathlib import Path
import re
import string
from typing import Sequence
import yaml

import Levenshtein

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *
import seaborn as sns

# %% [markdown]
# ## Typo corpus
#
# Construct `typos` from corpus such that the final object is a dictionary with correct keys and with values containing set of typos
#
# - aspell, birbeck, holbrook, wikipedia
# - microsoft
# - typokit
#


# %%
def parse_dat_misspellings(dat: Path):
    """Extract misspellings data from files, where each correct word is preceded by a dollar sign and followed by its misspellings, each on one line, without duplicates."""
    d = defaultdict(set)
    key = ""
    with dat.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("$"):
                key = line.replace("$", "").strip()
            else:
                line = line.split()
                d[key].add(line[0])

    return d


# %%
typos = defaultdict(set)
for dat in Path("./data/").glob("*.dat"):
    parsed = parse_dat_misspellings(dat)
    for k, v in parsed.items():
        typos[k].update(v)


with Path("./data/microsoft.txt").open("r", encoding="utf-8") as file:
    for line in file:
        k, v = line.split()
        typos[v].update({k})


# typokit
for file in Path("./data/typokit/").glob("*.yaml"):
    y = yaml.safe_load(file.open("r", encoding="utf-8"))
    for k, v in y.items():
        if isinstance(v, str):
            typos[k].update({v})
        elif isinstance(v, Sequence):
            typos[k].update(v)
        else:
            raise TypeError(f"Expected string or Sequence, got {v}")

# %%
# remove 'unknown' category from typos
_ = typos.pop("?")
# %%
print(f"The final collection contains {sum(len(v) for v in typos.values())} misspellings of {len(typos)} words")


# %% [markdown]
# ## Correction identificaiton
#
# Identify the string edit(s) (deletion, insertion, substitution) required to correct the typo.
#
# Ref: https://rapidfuzz.github.io/Levenshtein/levenshtein.html#editops

# %%
edit_stats = []
for correction, typo_set in typos.items():
    for typo in typo_set:
        edits = Levenshtein.editops(typo, correction)
        for op, typo_idx, correct_idx in edits:
            edit_dict = {"op": op}
            if op == "delete":
                edit_dict["error"] = typo[typo_idx]
                edit_dict["fix"] = "<del>"
            elif op == "insert":
                edit_dict["error"] = "<ins>"
                edit_dict["fix"] = correction[correct_idx]
            else:  # replace
                edit_dict["error"] = typo[typo_idx]
                edit_dict["fix"] = correction[correct_idx]

            edit_dict["abs_pos"] = typo_idx
            edit_dict["pct_pos"] = typo_idx / len(typo)

            edit_stats.append(edit_dict)


# %%
df = pd.DataFrame.from_records(edit_stats)

# limit to ascii chars
df["error_ascii"] = df["error"].apply(
    lambda x: x in string.printable if (isinstance(x, str) and x != "<ins>") else True
)
df["fix_ascii"] = df["fix"].apply(lambda x: x in string.printable if (isinstance(x, str) and x != "<del>") else True)

df = df[df["error_ascii"] & df["fix_ascii"]].drop(columns=["error_ascii", "fix_ascii"])

# %%
print(f"This collection provides {len(df)} edits for statistical summarization.")

# %%
# Frequency of occurrence x Op
# fmt: off
plot_df = (
    pd.concat(
        [
            df.loc[df["op"].isin(["delete", "replace"]), ["op", "error"]].rename(columns={"error": "letter"}),
            df.loc[df["op"] == "insert", ["op", "fix"]].rename(columns={"fix": "letter"}),
        ]
    )
    .groupby("op")
    .value_counts(["letter"], normalize=True, dropna=False)
    .reset_index()
    .rename(columns={0: "percentage"})
    .reset_index(drop=True)
)

g = (
    ggplot(
        plot_df,
        aes(
            x="letter",
            y="percentage",
        )
    )
    + geom_bar(stat="identity")
    + geom_text(
        aes(label="percentage", group=1),
        ha="left",
        size=8,
        nudge_y=0.005,
        format_string="{:.1%}",

    )
    + scale_x_discrete(limits=sorted(plot_df['letter'].unique(), reverse=True))
    + scale_y_continuous(
        labels=percent_format(),
        limits=(0, 0.2),
    )
    + facet_wrap("op")
    + coord_flip()
    + theme_xkcd()
    + theme(figure_size=(9, 16))
    + labs(title="Error Likelihood by Edit Operation")
)
# fmt: on

g.save(Path("./editops.png"))
g.draw()

# %%
# Edit location
# fmt: off
g = (
    ggplot(df, aes(x="pct_pos"))
    + geom_density()
    + scale_x_continuous(labels=percent_format())
    + facet_wrap("op", nrow=3,)
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Error location by Edit Operation")

)
# fmt: on

g.save(Path("./locations.png"))
g.draw()

# %%
# Correction matrix
# 'op' is not needed (error -> None is delete; None -> fix is insert)
alpha_df = df.loc[
    df["error"].isin(list(string.ascii_letters) + ["<ins>"]) & df["fix"].isin(list(string.ascii_letters) + ["<del>"]),
    ["error", "fix"],
].copy()
alpha_df = (
    alpha_df.groupby("error")
    .value_counts(["fix"], normalize=True)
    .reset_index()
    .pivot(
        index="error",
        columns="fix",
        values=0,
    )
    .sort_index(ascending=True)
    .sort_index(axis=1, ascending=True)
    .fillna(0)
)
alpha_df

# %%
fig, ax = plt.subplots(figsize=(12, 12))
ax = sns.heatmap(
    alpha_df,
    cmap=sns.color_palette("YlOrBr", as_cmap=True),
)
ax.set_title("Correction Matrix: Letters")

plt.savefig(Path("./correction-letters.png"))
plt.show()

# %%
numeric_df = df.loc[
    df["error"].isin(list(string.digits + string.punctuation) + ["<ins>"])
    & df["fix"].isin(list(string.digits + string.punctuation) + ["<del>"]),
    ["error", "fix"],
].copy()
numeric_df = (
    numeric_df.groupby("error")
    .value_counts(["fix"], normalize=True)
    .reset_index()
    .pivot(
        index="error",
        columns="fix",
        values=0,
    )
    .sort_index(ascending=True)
    .sort_index(axis=1, ascending=True)
    .fillna(0)
)
numeric_df

# %%
fig, ax = plt.subplots(figsize=(12, 12))
ax = sns.heatmap(
    numeric_df,
    cmap=sns.color_palette("YlOrBr", as_cmap=True),
)
ax.set_title("Correction Matrix: Numbers + Punctuation")

plt.savefig(Path("./correction-numpunkt.png"))
plt.show()
# %%
# TODO: NOTE: these stats are misleading!
# Microsoft has multiple records per mispelling
# leading to more accurate likelihoods for specific errors
# while the other datasets just contain 'standard errors' without frequency information
