# %%
from collections import Counter, defaultdict
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

# %%
records = []
with Path("./data/github.jsonl").open("r", encoding="utf-8") as file:
    for line in file:
        line = json.loads(line)
        for edit in line.get("edits"):
            if edit.get("is_typo") and (edit["src"].get("lang") == edit["tgt"].get("lang") == "eng"):
                records.append(
                    {
                        "src": edit["src"]["text"],
                        "tgt": edit["tgt"]["text"],
                    }
                )


# %%
def get_edit_chunk(string: str, op: str, op_start: int, op_end: int):
    """Extract the minimal number of words required to represent an edit operation."""
    spaces = [0, *[i for i, _char in enumerate(string) if _char == " "], len(string)]

    if op != "equal" and " " in string[op_start:op_end]:
        op_start = max(0, op_start - 1)
        op_end = min(op_end + 1, len(string))

    start = max(s for s in spaces if s <= op_start)
    end = min(s for s in spaces if s >= op_end)
    return string[start:end].strip()


# %%
# records = [
#     # {"src": "thequick brown fox", "tgt": "the quick brown fox"},
#     # {"src": "theQuick brown fox", "tgt": "the quick brown fox"},
#     # {"src": "Thequick brown fox", "tgt": "the quick brown fox"},
#     # {"src": "the quick brown fox", "tgt": "thequick brown fox"},
#     # {"src": "the quick brown fox", "tgt": "theQuick brown fox"},
#     # {"src": "the quick brown fox", "tgt": "Thequick brown fox"},
#     # {"src": "the quickbrown fox", "tgt": "the quick brown fox"},
#     # {"src": "the quickBrown fox", "tgt": "the quick brown fox"},
#     # {"src": "the quickbrown fox", "tgt": "the quick Brown fox"},
#     # {"src": "the quick brown fox", "tgt": "the quickbrown fox"},
#     # {"src": "the quick Brown fox", "tgt": "the quickbrown fox"},
#     # {"src": "the quick brown fox", "tgt": "the quickBrown fox"},
#     {"src": "the quick brownfox", "tgt": "the quick brown fox"},
#     {"src": "the quick brownFox", "tgt": "the quick brown fox"},
#     {"src": "the quick Brownfox", "tgt": "the quick brown fox"},
#     {"src": "the quick brown fox", "tgt": "the quick brownfox"},
#     {"src": "the quick brown fox", "tgt": "the quick brownFox"},
#     {"src": "the quick brown fox", "tgt": "the quick Brownfox"},
# ]

# %%
# clean up
typos = []
for record in records:
    # ignore diffs from leading/trailing whitespace
    src = record["src"].strip()
    tgt = record["tgt"].strip()

    # extract edit operations from diff
    matcher = difflib.SequenceMatcher(None, a=src, b=tgt)
    editops = matcher.get_opcodes()

    chunks = []
    # keep only complete words wrapping an edit
    for op, src_start, src_end, tgt_start, tgt_end in editops:
        if op != "equal":
            typo = get_edit_chunk(src, op, src_start, src_end)
            correction = get_edit_chunk(tgt, op, tgt_start, tgt_end)

            # exclude full-word insertions or deletions
            if len(typo) > 0 and len(correction) > 0:
                chunks.append((typo, correction))

    typos.extend(chunks)

# %% [markdown]
# ## Correction identificaiton
#
# Identify the string edit(s) (deletion, insertion, substitution) required to correct the typo.
#
# Ref: https://rapidfuzz.github.io/Levenshtein/levenshtein.html#editops

# %%
edit_stats = []
for typo, correction in typos:
    edits = Levenshtein.editops(typo, correction)
    for op, typo_idx, correct_idx in edits:
        edit_dict = {"op": op}

        if op == "delete":
            edit_dict["error"] = typo[typo_idx]
            edit_dict["fix"] = ""
        elif op == "insert":
            edit_dict["error"] = ""
            edit_dict["fix"] = correction[correct_idx]
        else:  # replace
            edit_dict["error"] = typo[typo_idx]
            edit_dict["fix"] = correction[correct_idx]

        edit_dict["abs_pos"] = typo_idx
        edit_dict["pct_pos"] = typo_idx / len(typo)

        edit_stats.append(edit_dict)

# %%
print(f"This collection provides {len(edit_stats)} edits for statistical summarization.")


# %%
df = pd.DataFrame.from_records(edit_stats)

# limit to ascii chars
df["error_ascii"] = df["error"].apply(lambda x: x in string.printable if isinstance(x, str) else True)
df["fix_ascii"] = df["fix"].apply(lambda x: x in string.printable if isinstance(x, str) else True)

df = df[df["error_ascii"] & df["fix_ascii"]].drop(columns=["error_ascii", "fix_ascii"])

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
    .sort_values("letter")
    # .query('percentage > 0.01')
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
        format_string="{:.1%}",

    )
    + scale_y_continuous(
        labels=percent_format(),
        limits=(0, 0.2),
    )
    + facet_wrap("op")
    + coord_flip()
    + theme_xkcd()
    + theme(figure_size=(8, 16))
    + labs(title="Error Likelihood by Edit Operation")
)
# fmt: on

# g.save(Path("./editops.png"))
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

# g.save(Path("./locations.png"))
g.draw()

# %%
