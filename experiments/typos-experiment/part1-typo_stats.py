# %%
from collections import defaultdict
import difflib
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import string
import subprocess
import sys
from typing import Sequence
import yaml

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *
import seaborn as sns

from aiml.utils import basic_log_config, get_repo_path, this_file

from src.damerau_levenshtein import DamerauLevenshtein

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
repo = get_repo_path(this_file())

# %% [markdown]
# ## Typo corpus
#
# Construct `typos` from corpus such that the final object is a dictionary with correct keys and with values containing set of typos
#
# - aspell, birbeck, holbrook, wikipedia
# - microsoft
# - typokit


# %%
def dat_parse_misspellings(dat: Path):
    """Extract misspellings data from files, where each correct word is preceded by a dollar sign and followed by its misspellings, each on one line, without duplicates."""
    d = {}
    # typo = ""
    with dat.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("$"):
                typo = line.replace("$", "").strip()
            else:
                line = line.split()
                if len(line) > 1 and line[1].isdigit():  # noqa: SIM108
                    correction = [line[0]] * int(line[1])
                else:
                    correction = line

                d[typo] = correction

    return d


def gh_get_edit_chunk(string: str, op: str, op_start: int, op_end: int):
    """Extract the minimal number of words required to represent an edit operation from the github dataset."""
    spaces = [0, *[i for i, _char in enumerate(string) if _char == " "], len(string)]

    if op != "equal" and " " in string[op_start:op_end]:
        op_start = max(0, op_start - 1)
        op_end = min(op_end + 1, len(string))

    start = max(s for s in spaces if s <= op_start)
    end = min(s for s in spaces if s >= op_end)
    return string[start:end].strip()


def gh_parse_diff(src: str, tgt: str):
    """Edits could be multiple lines long; identify edits to individual words."""
    matcher = difflib.SequenceMatcher(
        None,
        a=src,
        b=tgt,
    )
    editops = matcher.get_opcodes()

    typos = defaultdict(list)
    for op, src_start, src_end, tgt_start, tgt_end in editops:
        # editops are the bulk changes required to go from source to target.
        # if the op is equal, we don't care about it for typos
        if op != "equal":
            # keep only complete words wrapping an individual edit
            typo = gh_get_edit_chunk(src, op, src_start, src_end)
            correction = gh_get_edit_chunk(tgt, op, tgt_start, tgt_end)

            # exclude full-word insertions or deletions
            if len(typo) > 0 and len(correction) > 0:
                typos[typo].extend([correction])

    return typos


def collect_typo(typo: str, correction: str | Sequence):
    """Add (typo, correction) pair to global list."""
    global typos
    if isinstance(correction, str):
        typos[typo].extend([correction])
    elif isinstance(correction, Sequence):
        typos[typo].extend(correction)
    else:
        raise TypeError(f"Expected string or Sequence, got {correction}")


# %%
typos = defaultdict(list)

# typo corpus (aspell, birbeck, holbrook, wikipedia)
for dat in Path("./data/corpus").glob("*.dat"):
    parsed = dat_parse_misspellings(dat)
    for typo, correction in parsed.items():
        collect_typo(typo=typo, correction=correction)

# microsoft
with Path("./data/corpus/microsoft.txt").open("r", encoding="utf-8") as file:
    for line in file:
        typo, correction = line.split()
        collect_typo(typo=typo, correction=correction)


# typokit
for file in Path("./data/corpus/typokit/").glob("*.yaml"):
    y = yaml.safe_load(file.open("r", encoding="utf-8"))
    for typo, correction in y.items():
        collect_typo(typo=typo, correction=correction)

# github
with Path("./data/corpus/github.jsonl").open("r", encoding="utf-8") as file:
    for line in file:
        line = json.loads(line)
        # each line is a git diff, which can contain multiple edits
        for edit in line.get("edits"):
            if not edit.get("is_typo"):
                continue
            if not (edit["src"].get("lang") == edit["tgt"].get("lang") == "eng"):
                continue

            # each edit could contain multiple corrections
            parsed = gh_parse_diff(
                src=edit["src"]["text"].strip(),
                tgt=edit["tgt"]["text"].strip(),
            )
            for typo, correction in parsed.items():
                collect_typo(typo=typo, correction=correction)

ct = pd.read_csv(Path("./data/corpus/commit-typos.csv"))
for row in ct[["wrong", "correct"]].to_dict(orient="records"):
    typo = row["wrong"]
    correction = row["correct"]
    collect_typo(typo=typo, correction=correction)


# %%
# remove 'unknown' category from typos
_ = typos.pop("?")


# %% [markdown]
# ## Correction identification
#
# Identify the string edit(s) (delete, insert, substitute, transpose) required to correct the typo.
#
# Ref: https://github.com/nickeldan/dam_lev


# %%
edit_stats = []
for correction, typo_set in typos.items():
    for typo in typo_set:
        dl = DamerauLevenshtein(typo, correction)
        edits = [edit.todict().values() for edit in dl.edits]
        for op, err_idx, fix_idx in edits:
            if op == "transpose":
                edit_stats.append(
                    {
                        "op": op,
                        "error": typo[err_idx],
                        "fix": typo[err_idx + 1],
                        "abs_pos": err_idx,
                        "pct_pos": err_idx / len(typo),
                    }
                )
                edit_stats.append(
                    {
                        "op": op,
                        "error": typo[err_idx + 1],
                        "fix": typo[err_idx],
                        "abs_pos": err_idx + 1,
                        "pct_pos": (err_idx + 1) / len(typo),
                    }
                )
            else:
                edit_dict = {"op": op}
                if op == "delete":
                    edit_dict["error"] = typo[err_idx]
                    edit_dict["fix"] = "<del>"
                elif op == "insert":
                    edit_dict["error"] = "<ins>"
                    edit_dict["fix"] = correction[fix_idx]
                else:  # replace
                    edit_dict["error"] = typo[err_idx]
                    edit_dict["fix"] = correction[fix_idx]

                edit_dict["abs_pos"] = err_idx
                edit_dict["pct_pos"] = err_idx / len(typo)

                edit_stats.append(edit_dict)


# %%
df = pd.DataFrame.from_records(edit_stats)

# limit to ascii chars
df["error_ascii"] = df["error"].apply(
    lambda x: x in string.printable if (isinstance(x, str) and x != "<ins>") else True
)
df["fix_ascii"] = df["fix"].apply(lambda x: x in string.printable if (isinstance(x, str) and x != "<del>") else True)

df = df[df["error_ascii"] & df["fix_ascii"]].drop(columns=["error_ascii", "fix_ascii"])

print(f"This collection provides {len(df)} edits for statistical summarization.")

# %%
# save
df.to_csv(Path("./data/typo-stats.csv"), index=False)


# %%
# Edit Op Proportions
g = (
    ggplot(
        df.value_counts("op", normalize=True).reset_index().rename(columns={"proportion": "percentage"}),
        aes(
            x="op",
            y="percentage",
        ),
    )
    + geom_bar(stat="identity")
    + geom_text(
        aes(label="percentage", group=1),
        ha="center",
        size=8,
        nudge_y=0.01,
        format_string="{:.1%}",
    )
    + scale_y_continuous(
        labels=percent_format(),
    )
    + coord_cartesian(
        # xlim=(0, 1),
        ylim=(0, 1),
    )
    + theme_xkcd()
    + theme(figure_size=(6, 4))
    + labs(title="Likelihood of Edit Operation")
)
# fmt: on

g.save(Path("./editprob.png"))
g.draw()

# %%
# Frequency of occurrence x Op
# fmt: off
plot_df = (
    pd.concat(
        [
            df.loc[df["op"].isin(["delete", "substitute", "transpose"]), ["op", "error"]].rename(columns={"error": "letter"}),
            df.loc[df["op"].isin(["insert"]), ["op", "fix"]].rename(columns={"fix": "letter"}),
        ]
    )
    .groupby("op")
    .value_counts(["letter"], normalize=True, dropna=False)
    .reset_index()
    .rename(columns={"proportion": "percentage"})
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
    + scale_x_discrete(limits=sorted(plot_df['letter'].unique(), reverse=True,))
    + scale_y_continuous(
        labels=percent_format(),
    )
    + coord_cartesian(
        # xlim=(0, 1),
        ylim=(0, 0.2),
    )
    + facet_wrap("op", ncol=4)
    + coord_flip()
    + theme_xkcd()
    + theme(figure_size=(12, 16))
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
    + facet_wrap("op", nrow=4)
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Error Location by Edit Operation", x="relative position")
)
# fmt: on

g.save(Path("./locations.png"))
g.draw()

# %%
# Correction matrix
# 'op' is not needed (error -> None is delete; None -> fix is insert)
to_fix = (
    df[["error", "fix"]]
    .groupby("error")
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
if not np.allclose(to_fix.sum(axis=1), 1):
    raise ValueError("Error: rows do not sum to 100%")

to_fix.head()

# %%
alpha = list(string.ascii_letters) + ["<ins>", "<del>"]
alpha_df = (
    to_fix.copy()
    .loc[
        [row for row in to_fix.index if row in alpha],
        [col for col in to_fix.columns if col in alpha],
    ]
    .reset_index()
    .melt(id_vars="error", var_name="fix", value_name="prob")
)

# fmt: off
g = (
    ggplot(alpha_df, aes(x="fix", y="error", fill="prob"))
    + geom_tile()
    + scale_fill_cmap("YlOrBr")
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Correction Matrix: Letters")
)
# fmt: on

g.save(Path("./correction-letters.png"))
g.draw()


# %%
numeric = list(string.digits + string.punctuation) + ["<ins>", "<del>"]
numeric_df = (
    to_fix.copy()
    .loc[
        [row for row in to_fix.index if row in numeric],
        [col for col in to_fix.columns if col in numeric],
    ]
    .reset_index()
    .melt(id_vars="error", var_name="fix", value_name="prob")
)

# fmt: off
g = (
    ggplot(numeric_df, aes(x="fix", y="error", fill="prob"))
    + geom_tile()
    + scale_fill_cmap("YlOrBr")
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Correction Matrix: Numbers + Punctuation")
)
# fmt: on

g.save(Path("./correction-numpunct.png"))
g.draw()


# %%
# inverse correction
to_err = (
    df[["error", "fix"]]
    .groupby("fix")
    .value_counts(["error"], normalize=True)
    .reset_index()
    .pivot(
        index="fix",
        columns="error",
        values=0,
    )
    .sort_index(ascending=True)
    .sort_index(axis=1, ascending=True)
    .fillna(0)
)
if not np.allclose(to_err.sum(axis=1), 1):
    raise ValueError("Error: rows do not sum to 100%")

to_err.head()

# %%
alpha = list(string.ascii_letters) + ["<ins>", "<del>"]
alpha_df = (
    to_err.copy()
    .loc[
        [row for row in to_err.index if row in alpha],
        [col for col in to_err.columns if col in alpha],
    ]
    .reset_index()
    .melt(id_vars="fix", var_name="error", value_name="prob")
)

# fmt: off
g = (
    ggplot(alpha_df, aes(x="error", y="fix", fill="prob"))
    + geom_tile()
    + scale_fill_cmap("GnBu")
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Inverse Correction Matrix: Letters")
)
# fmt: on

g.save(Path("./inv_correction-letters.png"))
g.draw()


# %%
numeric = list(string.digits + string.punctuation) + ["<ins>", "<del>"]
numeric_df = (
    to_err.copy()
    .loc[
        [row for row in to_err.index if row in numeric],
        [col for col in to_err.columns if col in numeric],
    ]
    .reset_index()
    .melt(id_vars="fix", var_name="error", value_name="prob")
)

# fmt: off
g = (
    ggplot(numeric_df, aes(x="error", y="fix", fill="prob"))
    + geom_tile()
    + scale_fill_cmap("GnBu")
    + theme_xkcd()
    + theme(figure_size=(9, 9))
    + labs(title="Inverse Correction Matrix: Numbers + Punctuation")
)
# fmt: on

g.save(Path("./inv_correction-numpunct.png"))
g.draw()

# %%
