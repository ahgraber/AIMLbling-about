# %%
import itertools
import logging
import os
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv
import math
import numpy as np
import pandas as pd
from scipy.stats import sem
import gc
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F  # NOQA: N812
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedModel
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
load_dotenv()
token = os.getenv("HF_TOKEN")

# %%
typos_df = pd.read_csv(Path("./data/experiment.csv"))
print(typos_df.shape)
print(typos_df.sample(10))


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

#%%
st_emb = model.encode(
    typos_df["questions"].tolist(),
    convert_to_tensor=True,
)
# ~15 min on 3090

# %%
typos_df["st_similarity"] = F.cosine_similarity(baseline_emb.repeat(n_repeats, 1), st_emb).cpu().numpy()
typos_df.head()

# %%
plot_df = typos_df[["rate", "st_similarity"]].groupby(["rate"])["st_similarity"].agg(["mean", sem]).reset_index()

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean",
            # color="st_similarity",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + theme_xkcd()
    + theme(figure_size=(6, 6), legend_position=(0.26, 0.85))
    + labs(
        title="Cosine Similarity by Typo Rate",
        x="Typo Occurrence Rate",
        y="Cosine Similarity",
    )
)
# fmt: on

g.save(Path("./sentence-transformer-sim.png"))
g.draw()


# %% [markdown]
# ## Hack llama models for sentence embeddings
# ### Sentence Embedding
#
# use `watch -n0.5 nvidia-smi` to monitor GPU utilization
# where `0.5` is the time interval

# %%
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}
name = model_id['llama3']

# %%
# for model, name in model_id.items():
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)


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


#%%
se = SentEmbed(tokenizer=tokenizer, model=model, device=device)

#%%
print(torch.cuda.memory_summary(device=None, abbreviated=False))

#%%
baseline_emb = se(
    sentences=baseline_q.tolist(),
    pooling_strategy="wtmean",
    normalize=True,
    to_numpy=False,
)
torch.save(baseline_emb, Path("./data/llama3_baseline.pt"))
torch.cuda.empty_cache()
# ~5 min on 3090

#%%
for name, group in tqdm(typos_df.groupby(['rate','ver'])):
    rate, ver = name
    sentence_embeddings = se(
        sentences=group['questions'].tolist(),
        pooling_strategy="wtmean",
        normalize=True,
        to_numpy=False,
    )
    torch.save(sentence_embeddings, Path(f"./data/llama3_sentence_emb_r{rate}v{ver}.pt"))
    torch.cuda.empty_cache()

# ~8hr on 3090
# runtimes increasing as typo incidence increases?

#%%
typos_df["llama3_similarity"] = F.cosine_similarity(baseline_emb.repeat(n_repeats, 1), sentence_embeddings).cpu().numpy()
typos_df.head()

# %%
# Mean Pooling - Take attention mask into account for correct averaging
# [artificial intelligence - Sentence embeddings from LLAMA 2 Huggingface opensource - Stack Overflow](https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource)
def wt_mean_pooling(inputs):
    """Weighted mean pooling for sentence embeddings."""
    with torch.no_grad():
        last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]

    # linear weights for weighted mean
    positional_weights = inputs.attention_mask * torch.arange(
        start=1,
        end=last_hidden_state.shape[1] + 1,
    ).unsqueeze(0)

    wt_token_embeddings = torch.sum(last_hidden_state * positional_weights.unsqueeze(-1), dim=1)
    n_nonpad_tokens = torch.sum(positional_weights, dim=-1).unsqueeze(-1)
    sentence_embeddings = wt_token_embeddings / n_nonpad_tokens
    sentence_embeddings = F.normalize(wt_token_embeddings / n_nonpad_tokens, p=2, dim=1)
    return sentence_embeddings


# %%
# Tokenize sentences
encoded_input = tokenizer(
    baseline_q.tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
).to(device)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input, batch_size=32, device=device)

# Perform sentence-level pooling
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()


# %%
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}
# %%
for model, name in model_id.items():
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(f"{model}: {len(tokenizer.get_vocab())} token vocab")

    tokens = tokenizer(typos_df["questions"].tolist(), return_attention_mask=False)["input_ids"]
    counts = [len(x) for x in tokens]
    typos_df[f"{model}_tokens"] = counts

# %%
# baseline = typos_df.groupby(['rate','ver'], as_index=False).first()x
baseline = typos_df[typos_df["rate"] == 0.0]
n_repeats = int(len(typos_df) / len(baseline))


# %%
plot_df = typos_df[["rate", "similarity"]].groupby(["rate"])["similarity"].agg(["mean", sem]).reset_index()

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean",
            color="similarity",
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
    # + scale_x_continuous(
    #     breaks=[x / 10 for x in range(0, 11)],
    #     labels=percent_format(),
    # )
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
