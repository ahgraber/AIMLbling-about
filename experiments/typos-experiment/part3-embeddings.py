# %%
import gc
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from IPython.display import display
from tqdm.auto import tqdm

import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F  # NOQA: N812
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *

from aiml.utils import basic_log_config, get_repo_path, this_file, torch_device

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
repo = get_repo_path(this_file())


# %%
load_dotenv()
token = os.getenv("HF_TOKEN")

device = torch_device()

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

savedir = Path("./data/embeddings")
savedir.mkdir(parents=True, exist_ok=True)

# %%
print(typos_df.shape)
print(typos_df.sample(10))

# %% [markdown]
# ## Sentence Transformers

# %%
model_id = SentenceTransformer("all-mpnet-base-v2")
model_id.to(device)

baseline_emb = model_id.encode(
    baseline_q.tolist(),
    convert_to_tensor=True,
)
torch.save(baseline_emb, savedir / "st_mpnet_baseline_emb.pt")

st_emb = model_id.encode(
    typos_df["question"].tolist(),
    convert_to_tensor=True,
)

torch.save(st_emb, savedir / "st_mpnet_all_emb.pt")

del baseline_emb, st_emb
torch.cuda.empty_cache()
gc.collect()


# %% [markdown]
# ## Hack llama models for sentence embeddings
# ### Sentence Embedding
#
# use `watch -n0.5 nvidia-smi` to monitor GPU utilization
# where `0.5` is the time interval


# %%
# ref:
# https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
# https://stackoverflow.com/a/77441536
class SentEmbed:
    """Use various pooling strategies to calculate sentence embeddings."""

    def __init__(
        self,
        model_id: str | os.PathLike,
        device: torch.device | None = None,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self.llm = self.llm.to(self.device)
        self.llm.eval()

        torch.cuda.empty_cache()
        gc.collect()

    def _tokenize(self, text: str | list[str]) -> BatchEncoding:
        """Tokenize input."""
        inputs = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def _pool(
        self,
        inputs: BatchEncoding,  # dict[str, torch.Tensor],
        pooling_strategy: int | str | None = None,
    ):
        with torch.no_grad():
            last_hidden_state = self.llm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]

            if pooling_strategy == "max":
                outputs, _ = torch.max(last_hidden_state * inputs.attention_mask[:, :, None], dim=1)
            elif pooling_strategy == "mean":
                outputs = torch.div(
                    torch.sum(last_hidden_state * inputs.attention_mask[:, :, None], dim=1),
                    torch.sum(inputs.attention_mask),
                )
            elif pooling_strategy == "wtmean":
                # linear weights for weighted mean
                positional_weights = inputs.attention_mask * torch.arange(
                    start=1,
                    end=last_hidden_state.shape[1] + 1,
                    device=device,
                ).unsqueeze(0)

                outputs = torch.div(
                    torch.sum(last_hidden_state * positional_weights.unsqueeze(-1), dim=1),
                    torch.sum(inputs.attention_mask),
                )
            else:
                raise NotImplementedError("please specify pooling_strategy from [`max`, `mean`, `wtmean`]")

        return outputs

    def __call__(
        self,
        sentences: list[str],
        pooling_strategy: int | str | None = "wtmean",
        normalize: bool = True,
        to_numpy: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Calculate sentence embeddings."""
        if not isinstance(sentences, list):
            raise TypeError(f"`sentences` must be list. Was {type(sentences)}")

        sentence_embeddings = []
        for sentence in tqdm(sentences, leave=True):
            torch.cuda.empty_cache()
            gc.collect()

            inputs = self._tokenize(text=sentence)
            outputs = self._pool(inputs=inputs, pooling_strategy=pooling_strategy)

            if normalize:
                sentence_embeddings.append(F.normalize(outputs, p=2, dim=1))
            else:
                sentence_embeddings.append(outputs)

        sentence_embeddings = torch.concatenate(sentence_embeddings)
        if to_numpy:
            return sentence_embeddings.cpu().numpy()
        else:
            return sentence_embeddings


# %%
models = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
for name, model_id in models.items():
    se = SentEmbed(model_id)

    baseline_emb = se(
        sentences=baseline_q.tolist(),
        pooling_strategy="wtmean",
        normalize=True,
        to_numpy=False,
    )
    torch.save(baseline_emb, savedir / f"{name}_baseline_emb.pt")
    del baseline_emb
    torch.cuda.empty_cache()
    gc.collect()

    sentence_embeddings = se(
        sentences=typos_df["question"].tolist(),
        pooling_strategy="wtmean",
        normalize=True,
        to_numpy=False,
    )
    torch.save(sentence_embeddings, savedir / f"{name}_all_emb.pt")

    del se, sentence_embeddings
    torch.cuda.empty_cache()
    gc.collect()

    # ~30 min / loop on 3090

# %% [markdown]
# ## Similarity Analysis

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

savedir = Path("./data/embeddings")

for name in ["st_mpnet", "llama2", "llama3"]:
    torch.cuda.empty_cache()
    gc.collect()

    base_emb = torch.load(savedir / f"{name}_baseline_emb.pt", map_location=device)
    all_emb = torch.load(savedir / f"{name}_all_emb.pt", map_location=device)

    typos_df[f"{name}_similarity"] = F.cosine_similarity(
        base_emb.repeat(n_repeats, 1),
        all_emb,
    ).tolist()

    del base_emb, all_emb


# st_baseline = torch.load(Path("./data/st_mpnet_baseline_emb.pt"), map_location=device)
# st_all = torch.load(Path("./data/st_mpnet_all_emb.pt"), map_location=device)
# typos_df["st_similarity"] = F.cosine_similarity(st_baseline.repeat(n_repeats, 1), st_all).cpu().numpy()

# llama2_baseline = torch.load(Path("./data/llama2_baseline_emb.pt"), map_location=device)
# llama2_all = torch.load(Path("./data/llama2_all_emb.pt"), map_location=device)
# typos_df["llama2_similarity"] = F.cosine_similarity(llama2_baseline.repeat(n_repeats, 1), llama2_all).cpu().numpy()

# llama3_baseline = torch.load(Path("./data/llama3_baseline_emb.pt"), map_location=device)
# llama3_all = torch.load(Path("./data/llama3_all_emb.pt"), map_location=device)
# typos_df["llama3_similarity"] = F.cosine_similarity(llama3_baseline.repeat(n_repeats, 1), llama3_all).cpu().numpy()

# del st_baseline, st_all, llama2_baseline, llama2_all, llama3_baseline, llama3_all
# torch.cuda.empty_cache()
# gc.collect()

# %%
sim_cols = [
    "st_mpnet_similarity",
    "llama2_similarity",
    "llama3_similarity",
]
plot_df = (
    typos_df[["rate", *sim_cols]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(id_vars="rate", value_vars=sim_cols, var_name="model", value_name="mean_similarity")
    .replace(
        {
            "st_mpnet_similarity": "Sentence Transformer",
            "llama2_similarity": "Llama 2 7B Chat",
            "llama3_similarity": "Llama 3 8B Instruct",
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
    + scale_y_continuous(
        # breaks=[x / 10 for x in range(0, 11)],
    )
    + coord_cartesian(xlim=(0, 1), ylim=(-1, 1))
    + guides(color=guide_legend(title="Model", ncol=3, reverse=True))
    + labs(
        title="Cosine Similarity by Typo Rate",
        x="Typo Occurrence Rate",
        y="Cosine Similarity",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.55, 0.2),
    )
)
# fmt: on

g.save(Path("./sentence-similarity.png"))
g.draw()

# %%
