# %%
from collections import defaultdict
import difflib
import json
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

from damerau_levenshtein import DamerauLevenshtein

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *
import seaborn as sns

# %%
repo = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"],  # NOQA: S603, S607
    cwd=Path(__file__).parent,
    encoding="utf-8",
).strip()
repo = Path(repo).resolve()

# %%
df = pd.read_csv(Path("./data/stats.csv"))


# %%
def ceil(x, precision=0):
    """Ceiling with precision."""
    return np.true_divide(np.ceil(x * 10**precision), 10**precision)


# %%
df = (
    df.drop(columns=["abs_pos"])
    .assign(pct_pos=df["pct_pos"].apply(ceil, precision=2))
    .rename(columns={"pct_pos": "position"})
)
df.head()

# %%
# %% [markdown]
# Normalising typo positions for all `<typo string, correct string>` pairs
# allows us to compute a probability of making a typing mistake by each of 100 percentiles
# (e.g. probability at 66th percentile corresponds to a probability of making a typo in first 2/3 of the string).
# Based on an input string length we convert probabilities for 100 percentiles
# to a probability mass function over all character positions in a string.
# With typo probabilities assigned to each individual character in an input string,
# it is trivial to iterate over such string and generate typos,
# following the patterns exhibited by humans.


# %%
# get cumulative probabilities (e.g. probability at 66th percentile corresponds to a probability of making a typo in first 2/3 of the string)
# index: location, value: probability
pos2prob = (
    df["position"]  # nowrap
    .value_counts(normalize=True)
    .sort_index()
    .cumsum()
    .rename("probability")
)
pos2prob.head()

# %%
pos2op = (
    df[["position", "op"]]  # nowrap
    .groupby("position")
    .value_counts(normalize=True)
    .sort_index()
    .fillna(0)
)
pos2op.head()

# %%
# op2err = (
#     df[["op", "fix", "error"]]  # nowrap
#     .groupby(["op", "fix"])
#     .value_counts(["error"], normalize=True)
#     .sort_index(ascending=True)
#     .fillna(0.000001)  # arbitrarily small value
# )
# op2err.head()
# op2err = {}
# op2err['delete'] = df[df['op']=='delete'][["error"]].value_counts(['error'], normalize=True).sort_values()

# invert directionality to find probabilities to induce typos
op2err = (
    df[["op", "error", "fix"]]  # nowrap
    .groupby(["op", "error"])
    .value_counts(["fix"], normalize=True)
    .sort_index(ascending=True)
    .fillna(0.000001)  # arbitrarily small value
)
op2err.head()
# 'fix' col becomes what is needed to fix


# %%
def generate_typo(word: str):
    """Generate a realistic typo."""
    global pos2prob, pos2op, op2err  # pandas Series

    loc = pos2prob[pos2prob < random.random()].index.max()
    loc = 0 if np.isnan(loc) else loc
    # identify character based on position
    idx = int(loc * len(word))
    char = word[idx]

    # identify 'fix' operation based on position
    op = random.choices(pos2op[loc].index, weights=pos2op[loc].values, k=1)[0]

    # invert fix op and induce probabilistic error
    if op == "delete":
        # inverse of 'delete' is 'insert'
        # get error likelihoods for inserting `char`
        _ins = random.choices(
            op2err["insert"].index.get_level_values("fix"),
            weights=op2err["insert"].values,
            k=1,
        )[0]  # [row][multiindex]
        typo = word[: idx + 1] + _ins + word[idx + 1 :]
    elif op == "insert":
        # inverse of 'insert' is 'delete'
        typo = word[:idx] + word[idx + 1 :]
    elif op == "substitute":
        # get error likelihoods for substituting `char`
        _choices = op2err["substitute"][op2err["substitute"].index.get_level_values("error") == char]
        _sub = random.choices(
            _choices.index,
            weights=_choices.values,
            k=1,
        )[0][1]  # [row][multiindex]
        typo = word[:idx] + _sub + word[idx + 1 :]
    elif op == "transpose":
        # fix is always to swap with subsequent letter
        typo = list(word)
        try:
            typo[idx], typo[idx + 1] = typo[idx + 1], typo[idx]
        except IndexError:
            # this means transposes might be underrepresented 🤷🏻‍♂️
            pass
        finally:
            typo = "".join(typo)

    return typo


# %%
tests = [
    """It is a period of civil war.
Rebel spaceships, striking
from a hidden base, have won
their first victory against
the evil Galactic Empire.

During the battle, Rebel
spies managed to steal secret
plans to the Empire's
ultimate weapon, the DEATH
STAR, an armored space
station with enough power to
destroy an entire planet.

Pursued by the Empire's
sinister agents, Princess
Leia races home aboard her
starship, custodian of the
stolen plans that can save
her people and restore
freedom to the galaxy....""",
    """def fizzbuzz(n):
    for x in range(n + 1):
        if x % 3 == 0 and x % 5 == 0:
            print("fizz buzz")
        elif x % 3 == 0:
            print("fizz")
        elif x % 5 == 0:
            print("buzz")
        else:
            print(x)
""",
]

# set rate to induce typos at a given rate
# 0.5 means approx half of the words will have typos
# but multiple could occur per word
RATE = 0.3
for test in tests:
    # word: typo
    words = re.split(r"(\S+)", test)
    typos = words.copy()

    # sample by index
    targets = random.choices(range(len(words)), k=int(RATE * len(words)))

    for idx in targets:
        if not words[idx].isspace() and words[idx]:
            typos[idx] = generate_typo(words[idx])
    print("".join(typos))
    print("\n")


# %%
