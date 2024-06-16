# %%
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from scipy.stats import sem

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# specify debug logs from ds_utils to see behind-the-scenes updates
logging.getLogger("typo_gen").setLevel(logging.DEBUG)

# %%
load_dotenv()
token = os.getenv("HF_TOKEN")

# %%
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
typos_df = pd.read_csv(Path("./data/experiment.csv"))
print(typos_df.shape)
print(typos_df.sample(10))

# %%
for model, name in model_id.items():
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(f"{model}: {len(tokenizer.get_vocab())} token vocab")

    tokens = tokenizer(typos_df["questions"].tolist(), return_attention_mask=False)["input_ids"]
    counts = [len(x) for x in tokens]
    typos_df[f"{model}_tokens"] = counts

# %%
# baseline = typos_df.groupby(['rate','iter'], as_index=False).first()x
baseline = typos_df[typos_df["rate"] == 0.0]
n_repeats = int(len(typos_df) / len(baseline))

for model in model_id:
    typos_df[f"{model}_delta"] = typos_df[f"{model}_tokens"] - (baseline[f"{model}_tokens"].tolist() * n_repeats)
    typos_df[f"{model}_deltapct"] = typos_df[f"{model}_delta"] / typos_df[f"{model}_tokens"]


# %%
# temp/experimental plot
(
    typos_df[["rate", "llama2_deltapct", "llama3_deltapct"]]
    .groupby("rate")
    .mean()
    .reset_index()
    .plot(x="rate", y=["llama2_deltapct", "llama3_deltapct"])
)

# %%
(
    typos_df[["rate", "llama2_delta", "llama3_delta"]]
    .groupby("rate")
    .mean()
    .reset_index()
    .plot(x="rate", y=["llama2_delta", "llama3_delta"])
)
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
    # + geom_errorbar(
    #     plot_df,
    #     aes(
    #         x="rate",
    #         ymin="mean - sem",
    #         ymax="mean + sem",
    #         color="tokenizer",
    #     ),
    #     width=0.05,
    # )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
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
    # + geom_errorbar(
    #     plot_df,
    #     aes(
    #         x="rate",
    #         ymin="mean - sem",
    #         ymax="mean + sem",
    #         color="tokenizer",
    #     ),
    #     width=0.05,
    # )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + scale_y_continuous(
        labels=percent_format(),
        limits=(0, 0.35),
    )
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
