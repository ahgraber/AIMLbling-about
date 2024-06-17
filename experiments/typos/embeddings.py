# %%
import itertools
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from scipy.stats import sem

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

# %%
worstcase = model.encode(
    typos_df.iloc[-100:]["questions"].tolist(),
    convert_to_tensor=True,
)

# %%
F.cosine_similarity(
    baseline_emb,
    worstcase,
).cpu().numpy()


# %%
def baseline_cos_sim(
    sentences: pd.Series | list[str], *, baseline=torch.Tensor, model: SentenceTransformer
) -> np.ndarray:
    """Cosine similarity vs baseline data."""
    if isinstance(sentences, pd.Series):
        sentences = sentences.tolist()

    embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
    )

    similarity = F.cosine_similarity(baseline, embeddings).cpu().numpy()
    return similarity


# %%
typos_df["st_similarity"] = typos_df.groupby(["rate", "iter"]).transform(
    baseline_cos_sim, baseline=baseline_emb, model=model
)
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


# %%
model_id = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
# for model, name in model_id.items():
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_id["llama3"])
model = AutoModelForCausalLM.from_pretrained(
    model_id["llama3_gguf"],
    # device_map="auto",
)
model.to(device)
# %%
tokenizer.pad_token = tokenizer.eos_token

model.eval()
model = model.to(device)

# %%
inputs = tokenizer(
    baseline_q.tolist()[:10],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# %%
with torch.no_grad():
    last_hidden_state = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-1]

# %%
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

# %%
sentence_embeddings = F.normalize(outputs, p=2, dim=1)
sentence_embeddings.cpu().numpy()

# %% [markdown]
# ### Sentence Embedding


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
        batch_size: int = 10,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

        self._batch_size = batch_size

    def _batch(self, iterable, batch_size: int):
        """Batch data from the iterable into tuples of length n. The last batch may be shorter than n."""
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, batch_size)):
            yield batch

    def _tokenize(self, batch: list[str]) -> BatchEncoding:
        """Tokenize input."""
        inputs = self.tokenizer(
            batch,
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
        batch_size: int | None = None,
        pooling_strategy: int | str | None = "wtmean",
        normalize: bool = True,
        to_numpy: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """Calculate sentence embeddings."""
        if not isinstance(sentences, list):
            raise TypeError(f"`sentences` must be list. Was {type(sentences)}")

        sentence_embeddings = []
        for batch in self._batch(sentences, batch_size=batch_size):
            inputs = self._tokenize(batch)
            outputs = self._pool(inputs=inputs, pooling_strategy=pooling_strategy)

            if normalize:
                sentence_embeddings.append(F.normalize(outputs, p=2, dim=1))
            else:
                sentence_embeddings.append(outputs)

        if to_numpy:
            return np.concatenate([s.cpu().numpy() for s in sentence_embeddings])
        else:
            return torch.concatenate(sentence_embeddings)


# %%
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
# for model, name in model_id.items():
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(name, device=device)
model = AutoModelForCausalLM.from_pretrained(name)
model.to(device)


tokens = tokenizer(typos_df["questions"].tolist(), return_attention_mask=False)["input_ids"]
counts = [len(x) for x in tokens]
typos_df[f"{model}_tokens"] = counts


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
# baseline = typos_df.groupby(['rate','iter'], as_index=False).first()x
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
