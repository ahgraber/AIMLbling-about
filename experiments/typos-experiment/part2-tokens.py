# %%
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from transformers import AutoTokenizer

import pandas as pd
from scipy.stats import sem

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "typo-experiment"

# %%
load_dotenv()
token = os.getenv("HF_TOKEN")

# %%
models = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
print(typos_df.shape)
print(typos_df.sample(10))

# %%
for name, model_id in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"{name}: {len(tokenizer.get_vocab())} token vocab")

    tokens = tokenizer(typos_df["question"].tolist(), return_attention_mask=False)["input_ids"]
    counts = [len(x) for x in tokens]
    typos_df[f"{name}_tokens"] = counts

# %%
baseline = typos_df[typos_df["rate"] == 0.0]
n_repeats = int(len(typos_df) / len(baseline))

for name in models:
    typos_df[f"{name}_delta"] = typos_df[f"{name}_tokens"] - (baseline[f"{name}_tokens"].tolist() * n_repeats)
    typos_df[f"{name}_deltapct"] = typos_df[f"{name}_delta"] / typos_df[f"{name}_tokens"]


# %%
plot_df = (
    typos_df[["rate", "llama2_delta", "llama3_delta"]]
    .rename(
        columns={
            "llama2_delta": "llama2",
            "llama3_delta": "llama3",
        }
    )
    .melt(id_vars="rate", value_vars=["llama2", "llama3"], var_name="tokenizer", value_name="count")
    .groupby(["rate", "tokenizer"])["count"]
    .agg(["mean", sem])
    .reset_index()
)

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean",
            color="tokenizer",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    # + coord_cartesian(xlim=(0, 1), ylim=(0, 30))
    + theme_xkcd()
    + theme(figure_size=(6, 6), legend_position=(0.26, 0.85))
    + labs(
        title="Token Increase by Typo Rate",
        x="Typo Occurrence Rate",
        y="Average Token Increase",
    )
)
# fmt: on

g.save(Path("./count-differences.png"))
g.draw()

# %%
plot_df = (
    typos_df[["rate", "llama2_deltapct", "llama3_deltapct"]]
    .rename(
        columns={
            "llama2_deltapct": "llama2",
            "llama3_deltapct": "llama3",
        }
    )
    .melt(id_vars="rate", value_vars=["llama2", "llama3"], var_name="tokenizer", value_name="count")
    .groupby(["rate", "tokenizer"])["count"]
    .agg(["mean", sem])
    .reset_index()
)
g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean",
            color="tokenizer",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + scale_y_continuous(
        labels=percent_format(),
    )
    + coord_cartesian(xlim=(0, 1), ylim=(0, 0.35))
    + theme_xkcd()
    + theme(figure_size=(6, 6), legend_position=(0.26, 0.85))
    + labs(
        title="Proportional Token Increase by Typo Rate",
        x="Typo Occurrence Rate",
        y="Average % Token Increase",
    )
)
# fmt: on

g.save(Path("./pct-differences.png"))
g.draw()

# %%
