# %%
from collections import Counter
import gc
import logging
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from tqdm.auto import tqdm

import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F  # NOQA: N812  # NOQA: N812
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *

# %%
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"

logging.basicConfig(format=LOG_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
def check_torch_device():
    """Check which device pytorch will use."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Found pytorch device '{device.type}'")
    return device


device = check_torch_device()

# %%
load_dotenv()
token = os.getenv("HF_TOKEN")

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
print(typos_df.shape)
print(typos_df.sample(10))

# %%
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))


# %% [markdown]
# ## Can LLM's "heal" the typos?

# %%
models = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
questions = typos_df["question"].tolist()
healed = {"llama2": [], "llama3": []}
for name, model_id in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    llm = llm.to(device)
    llm.eval()

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for question in tqdm(questions):
        messages = [
            {
                "role": "system",
                "content": "Instructions:\n\nCorrect the typos in the following text.\nRespond ONLY with the corrected text.",
            },
            {
                "role": "user",
                "content": f"Text:\n\n{question}\n\nCorrection:\n\n",
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = llm.generate(
            input_ids,
            max_new_tokens=input_ids.shape[1],
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # do_sample=False makes sampling parms unnecessary
            temperature=None,  # 0.0,
            top_p=None,  # 0.9,
        )
        response = outputs[0][input_ids.shape[-1] :]

        # print(tokenizer.decode(response, skip_special_tokens=True))
        healed[name].append(tokenizer.decode(response, skip_special_tokens=True))

    del tokenizer, llm
    torch.cuda.empty_cache()
    gc.collect()

# ~ 11 hr / loop

# %%
healed_df = pd.DataFrame.from_records(healed)
healed_df.to_csv(Path("./data/typos-healed.csv"), header=True, index=False)
healed_df.sample(10)

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
healed_df = pd.read_csv(Path("./data/typos-healed.csv"))

baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

# %%
# llama2 often injects chat prior to the fix
pattern1 = re.compile(r".*text:(.*)")  # "Here is the corrected text:\n\n"
pattern2 = re.compile(r".*text is:(.*)")  # "The corrected texti is:\n\n"
pattern3 = re.compile(r".*[Cc]orrection:(.*)")  # "The corrected texti is:\n\n"

for col in ["llama2", "llama3"]:
    healed_df[col] = (
        healed_df[col]
        .str.replace(pattern1, r"\1", regex=True)
        .str.replace(pattern2, r"\1", regex=True)
        .str.replace(pattern3, r"\1", regex=True)
        .str.strip()
        .tolist()
    )

healed_df["baseline"] = baseline_q.tolist() * n_repeats

healed_df.sample(10)

# %%
# from collections import Counter

# baseline = [Counter(q.split()) for q in healed_df["baseline"]]
# llama2 = [Counter(q.split()) for q in healed_df["llama2"]]
# llama3 = [Counter(q.split()) for q in healed_df["llama3"]]


# %% [markdown]
# ### LLM healing accuracy
#
# uses multiset Jaccard (https://arxiv.org/abs/2110.09619)


# %%
def jaccard_multiset(actual: str, pred: str):
    """Multiset generalization of Jaccard index."""
    a = Counter(actual.split())
    p = Counter(pred.split())

    jaccard = (a & p).total() / (a | p).total()

    return jaccard


for col in ["llama2", "llama3"]:
    healed_df[f"{col}_acc"] = healed_df[["baseline", col]].apply(
        lambda row: jaccard_multiset(row["baseline"], row[col]), axis=1.0
    )


# %%
plot_df = (
    typos_df.join(healed_df)[["rate", "llama2_acc", "llama3_acc"]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(id_vars="rate", value_vars=["llama2_acc", "llama3_acc"], var_name="model", value_name="mean_jaccard")
    .replace(
        {
            "llama2_acc": "Llama 2 7B Chat",
            "llama3_acc": "Llama 3 8B Instruct",
        }
    )
)

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean_jaccard",
            color="model",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + coord_cartesian(xlim=(0, 1), ylim=(0, 1))
    + guides(color=guide_legend(title="Model", ncol=1, reverse=True))
    + labs(
        title="Typo Healing Performance by Typo Rate",
        x="Typo Occurrence Rate",
        y="Jaccard Equivalence to Ground Truth Baseline",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.8, 0.25),
    )
)
# fmt: on

g.save(Path("./healing-accuracy.png"))
g.draw()

# %%
# ### LLM healing similarity

# %%
model_id = SentenceTransformer("all-mpnet-base-v2")
model_id.to(device)

# %%
for col in ["llama2", "llama3"]:
    emb = model_id.encode(
        healed_df[col].tolist(),
        convert_to_tensor=True,
    )
    # ~1 min on 3090

    torch.save(emb, Path(f"./data/embeddings/{col}_healed_emb.pt"))

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

baseline = torch.load(Path("./data/embeddings/st_mpnet_baseline_emb.pt"), map_location=device)
llama2 = torch.load(Path("./data/embeddings/llama2_healed_emb.pt"), map_location=device)
llama3 = torch.load(Path("./data/embeddings/llama3_healed_emb.pt"), map_location=device)

typos_df["llama2_healed_sim"] = F.cosine_similarity(baseline.repeat(n_repeats, 1), llama2).tolist()
typos_df["llama3_healed_sim"] = F.cosine_similarity(baseline.repeat(n_repeats, 1), llama3).tolist()

# %%
plot_df = (
    typos_df.join(healed_df)[["rate", "llama2_healed_sim", "llama3_healed_sim"]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(
        id_vars="rate",
        value_vars=["llama2_healed_sim", "llama3_healed_sim"],
        var_name="model",
        value_name="mean_similarity",
    )
    .replace(
        {
            "llama2_healed_sim": "Llama 2 7B Chat",
            "llama3_healed_sim": "Llama 3 8B Instruct",
        }
    )
)

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean_similarity",
            color="model",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + coord_cartesian(xlim=(0, 1), ylim=(-1, 1))
    + guides(color=guide_legend(title="Model", ncol=1, reverse=True))
    + labs(
        title="Typo Healing Similarity by Typo Rate",
        x="Typo Occurrence Rate",
        y="Cosine Similarity to Ground Truth Baseline",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.33, 0.25),
    )
)
# fmt: on

g.save(Path("./healing-similarity.png"))
g.draw()

# %%
