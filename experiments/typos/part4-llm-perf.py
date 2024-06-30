# %%
import gc
import itertools
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import evaluate
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
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
print(typos_df.shape)
print(typos_df.sample(10))

# %%
baseline_q = typos_df[typos_df["rate"] == 0]["questions"]
n_repeats = int(len(typos_df) / len(baseline_q))


# %%
class Perplexity:
    """Calculate model perplexity."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        llm: PreTrainedModel,
        device: torch.device | None = None,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = llm
        self.llm.eval()
        self.llm = self.llm.to(self.device)

        torch.cuda.empty_cache()
        gc.collect()

    def _tokenize(self, text: str | list[str]) -> BatchEncoding:
        """Tokenize input."""
        inputs = self.tokenizer(
            text,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def __call__(self, sentences: list[str]) -> torch.Tensor | np.ndarray:
        """Calculate perplexity."""
        if not isinstance(sentences, list):
            raise TypeError(f"`sentences` must be list. Was {type(sentences)}")

        perplexities = []
        for sentence in tqdm(sentences, leave=False):
            torch.cuda.empty_cache()
            gc.collect()

            inputs = self._tokenize(text=sentence)
            # labels = inputs["input_ids"]

            with torch.no_grad():
                logits = self.llm(**inputs).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_attention_mask = inputs["attention_mask"][..., 1:].contiguous()

            cross_entropy = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )
            del shift_logits, shift_labels

            pplx = torch.exp(
                torch.div(
                    (cross_entropy * shift_attention_mask).sum(1),
                    shift_attention_mask.sum(1),
                )
            )
            del cross_entropy, shift_attention_mask

            perplexities.append(pplx)

        return torch.concatenate(perplexities).tolist()


# %%
model_id = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %%
for name, model in model_id.items():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)

    plx = Perplexity(
        tokenizer=tokenizer,
        llm=llm,
        device=device,
    )

    perplexities = plx(sentences=typos_df["questions"].tolist())
    # ~3-5hr on 3090

    typos_df[f"{name}_perplexity"] = perplexities
    display(typos_df.sample(10))

    del tokenizer, llm, plx, perplexities
    torch.cuda.empty_cache()
    gc.collect()

# %%
plot_df = (
    typos_df[["rate", "llama2_perplexity", "llama3_perplexity"]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(
        id_vars="rate",
        value_vars=["llama2_perplexity", "llama3_perplexity"],
        var_name="model",
        value_name="mean_perplexity",
    )
    .replace(
        {
            "llama2_perplexity": "Llama 2 7B Chat",
            "llama3_perplexity": "Llama 3 8B Instruct",
        }
    )
)

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean_perplexity",
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
        title="Model Perplexity by Typo Rate",
        x="Typo Occurrence Rate",
        y="Model Perplexity",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.8, 0.25),
    )
)
# fmt: on

g.save(Path("./perplexity.png"))
g.draw()

# %%
