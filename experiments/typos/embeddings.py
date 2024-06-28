# %%
import gc
import itertools
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from IPython.display import display
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F  # NOQA: N812
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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
typos_df = pd.read_csv(Path("./data/experiment.csv"))
print(typos_df.shape)
print(typos_df.sample(10))

# %%
baseline_q = typos_df[typos_df["rate"] == 0]["questions"]
n_repeats = int(len(typos_df) / len(baseline_q))

# %% [markdown]
# ## Sentence Transformers

# %%
model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)

# %%
baseline_emb = model.encode(
    baseline_q.tolist(),
    convert_to_tensor=True,
)
torch.save(baseline_emb, Path("./data/st_mpnet_baseline.pt"))

# %%
st_emb = model.encode(
    typos_df["questions"].tolist(),
    convert_to_tensor=True,
)
# ~15 min on 3090

torch.save(st_emb, Path("./data/st_mpnet_all.pt"))

# %%
typos_df["st_similarity"] = F.cosine_similarity(baseline_emb.repeat(n_repeats, 1), st_emb).cpu().numpy()
typos_df.head()

# %%
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
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: torch.device | None = None,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

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
            last_hidden_state = self.model(
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

        for sentence in tqdm(sentences, leave=False):
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
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
for name, model in model_id.items():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model)

    se = SentEmbed(tokenizer=tokenizer, model=llm, device=device)

    baseline_emb = se(
        sentences=baseline_q.tolist(),
        pooling_strategy="wtmean",
        normalize=True,
        to_numpy=False,
    )
    torch.save(baseline_emb, Path(f"./data/{name}_baseline.pt"))
    torch.cuda.empty_cache()
    gc.collect()
    # ~5 min on 3090

    sentence_embeddings = se(
        sentences=typos_df["questions"].tolist(),
        pooling_strategy="wtmean",
        normalize=True,
        to_numpy=False,
    )
    # ~3-5hr on 3090

    torch.save(sentence_embeddings, Path(f"./data/{name}_all_emb.pt"))

    typos_df[f"{name}_similarity"] = (
        F.cosine_similarity(baseline_emb.repeat(n_repeats, 1), sentence_embeddings).cpu().numpy()
    )

    display(typos_df.sample(10))

    del tokenizer, llm, se, baseline_emb, sentence_embeddings
    torch.cuda.empty_cache()
    gc.collect()

# %%
typos_df = pd.read_csv(Path("./data/experiment.csv"))

st_baseline = torch.load(Path("./data/st_mpnet_baseline.pt"), map_location=device)
st_all = torch.load(Path("./data/st_mpnet_all.pt"), map_location=device)
typos_df["st_similarity"] = F.cosine_similarity(st_baseline.repeat(n_repeats, 1), st_all).cpu().numpy()

llama2_baseline = torch.load(Path("./data/llama2_baseline.pt"), map_location=device)
llama2_all = torch.load(Path("./data/llama2_all_emb.pt"), map_location=device)
typos_df["llama2_similarity"] = F.cosine_similarity(llama2_baseline.repeat(n_repeats, 1), llama2_all).cpu().numpy()

llama3_baseline = torch.load(Path("./data/llama3_baseline.pt"), map_location=device)
llama3_all = torch.load(Path("./data/llama3_all_emb.pt"), map_location=device)
typos_df["llama3_similarity"] = F.cosine_similarity(llama3_baseline.repeat(n_repeats, 1), llama3_all).cpu().numpy()

del st_baseline, st_all, llama2_baseline, llama2_all, llama3_baseline, llama3_all
torch.cuda.empty_cache()
gc.collect()

# %%
sim_cols = [
    "st_similarity",
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
            "st_similarity": "Sentence Transformer",
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
    + guides(color=guide_legend(title="Model", ncol=1, reverse=True))
    + labs(
        title="Cosine Similarity by Typo Rate",
        x="Typo Occurrence Rate",
        y="Cosine Similarity",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.8, 0.8),
    )
)
# fmt: on

g.save(Path("./sentence-sim.png"))
g.draw()
