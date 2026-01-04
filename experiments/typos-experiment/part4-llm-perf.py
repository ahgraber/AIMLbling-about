# %%
import gc
import logging
import os
from pathlib import Path
import re
import string
import textwrap

from dotenv import load_dotenv
from IPython.display import display
from jinja2 import Template
from setproctitle import setproctitle
from tqdm.auto import tqdm

import torch
from datasets import load_dataset
import torch.nn.functional as F  # NOQA: N812  # NOQA: N812
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    StoppingCriteriaList,
    StopStringCriteria,
)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mizani.formatters import percent_format
from plotnine import *

from aiml.utils import basic_log_config, get_repo_path, this_file, torch_device

# %%
# define python process name
setproctitle(Path(__file__).stem)

basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "typo-experiment"

# %%
load_dotenv()
token = os.getenv("HF_TOKEN")

device = torch_device()

# %%
models = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# %% [markdown]
# ## Perplexity
#
# Hypothetical representation of language modeling performance


# %%
class Perplexity:
    """Calculate model perplexity.

    ref: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py
    """

    def __init__(
        self,
        model_id: str | os.PathLike,
        device: torch.device | None = None,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.llm = self.llm.to(self.device)
        self.llm.eval()

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
        for sentence in tqdm(sentences, leave=True):
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
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

# %%
plx_df = typos_df[["question"]]
for name, model_id in models.items():
    plx = Perplexity(model_id=model_id)

    perplexities = plx(sentences=typos_df["question"].tolist())

    plx_df[f"{name}_perplexity"] = perplexities
    display(typos_df.sample(10))

    del plx, perplexities
    torch.cuda.empty_cache()
    gc.collect()
    # ~30 min / loop on 3090

plx_df.to_csv(Path("./data/typos-perplexity.csv"), header=True, index=False)

# %%
typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
plx_df = pd.read_csv(Path("./data/typos-perplexity.csv"))

plx_cols = ["llama2_perplexity", "llama3_perplexity"]

plot_df = (
    typos_df.join(plx_df[plx_cols])[["rate", *plx_cols]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(
        id_vars="rate",
        value_vars=plx_cols,
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
        limits=(0, 1),
    )
    # + coord_cartesian(xlim=(0, 1), ylim=(0, 1000))
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


# %% [markdown]
# ## MMLU performance
#
# Actual performance


# %%
class TinyMMLU:
    """Answer multiple-choice questions."""

    def __init__(
        self,
        model_id: str | os.PathLike,
        device: torch.device | None = None,
    ):
        self.device = device if device else check_torch_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.llm = self.llm.to(self.device)
        self.llm.eval()

        torch.cuda.empty_cache()
        gc.collect()

    def _tokenize(self, text: str | list[str]) -> BatchEncoding:
        """Tokenize input."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def _chat_tokenize(self, text: str | list[str]) -> BatchEncoding:
        """Tokenize input with chat template."""
        inputs = self.tokenizer.apply_chat_template(
            text,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def _generate(self, inputs: BatchEncoding, max_new_tokens: int = 10):
        stop_at_newlines = StopStringCriteria(tokenizer=self.tokenizer, stop_strings=["\n", "\n\n"])

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stop_at_newlines]),
            do_sample=False,
            # do_sample=False makes sampling params unnecessary
            temperature=None,  # 0.0,
            top_p=None,  # 0.9,
        )
        return outputs

    def _decode(self, outputs: torch.Tensor, n_input_tokens: int):
        response = outputs[0][n_input_tokens:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)
        return answer

    def parse_answer(self, response: str):
        """Extract multiple-choice letter selection from string response."""
        # select only A, B, C,or D;
        # if the response continues from the letter, then the subsequent characters must be nonword, a space, or line-end
        choices = re.compile(r"([ABCD])(?=\W|\s|$)")

        try:
            return choices.findall(response)[0]
        except IndexError:
            logger.info("No uppercase letters in A-E found, returning None")
            return None

    def zero_shot(self, question: str, choices: list[str], parse: bool = True):
        """Zero-Shot MMLU pipeline."""
        zeroshot_tmpl = Template(
            textwrap.dedent(
                """
                {{text}}
                {% for i, option in zip(letters, choices) %}{{i}}. {{option}}\n{% endfor %}
                Answer:
                """
            ).strip()
        )
        messages = [
            {
                "role": "system",
                "content": "What is the correct answer?\nDo not explain yourself.\nRespond ONLY with the letter representing the correct answer option.",
            },
            {
                "role": "user",
                "content": zeroshot_tmpl.render(
                    text=question,
                    letters=string.ascii_uppercase,
                    choices=choices,
                    zip=zip,
                ),
            },
        ]
        inputs = self._chat_tokenize(text=messages)
        outputs = self._generate(inputs=inputs)
        response = self._decode(outputs=outputs, n_input_tokens=inputs["input_ids"].shape[-1])

        if not parse:
            return response
        else:
            answer = self.parse_answer(response=response)
            return answer

    def five_shot(self, question: str, parse=True):
        """Five-Shot MMLU pipeline."""
        inputs = self._tokenize(text=question)
        outputs = self._generate(inputs=inputs)
        response = self._decode(outputs=outputs, n_input_tokens=inputs["input_ids"].shape[-1])

        if not parse:
            return response
        else:
            answer = self.parse_answer(response=response)
            return answer


# %%
tiny_data = load_dataset("tinyBenchmarks/tinyMMLU", "all")["test"]
mmlu_df = tiny_data.to_pandas()

typos_df = pd.read_csv(Path("./data/typos-variants.csv"))
baseline_q = typos_df[typos_df["rate"] == 0]["question"]
n_repeats = int(len(typos_df) / len(baseline_q))

# %%
# 'question' was modified by typo
# 'choices' contain the possible multiple choice answers
# 'answer' is the correct answer
# 'input_formatted' contains 5-shot ICL
typos_df["choices"] = mmlu_df["choices"].tolist() * n_repeats

# replace 'five-shot formatted' question in MMLU with typo'd version
# typos_df["input_formatted"] = mmlu_df["input_formatted"].tolist() * n_repeats
typos_df["fiveshot_formatted"] = [
    fewshot.replace(baseline, typo)
    for fewshot, baseline, typo in zip(
        mmlu_df["input_formatted"].tolist() * n_repeats,
        baseline_q.tolist() * n_repeats,
        typos_df["question"],
    )
]

# convert typos df to letters
typos_df["answer"] = [string.ascii_uppercase[x] for x in mmlu_df["answer"]] * n_repeats

# %%
for name, model_id in models.items():
    tmmlu = TinyMMLU(
        model_id=model_id,
        device=device,
    )

    zeroshot = []
    fiveshot = []
    for row in tqdm(typos_df.to_dict(orient="records"), leave=True):
        zeroshot.append(
            tmmlu.zero_shot(
                question=row["question"],
                choices=row["choices"],
                parse=False,  # parse separately
            )
        )
        fiveshot.append(
            tmmlu.five_shot(
                question=row["fiveshot_formatted"],
                parse=False,  # parse separately
            )
        )

    # print(zeroshot)
    # print(fiveshot)
    df = pd.DataFrame(
        {
            f"{name}_zeroshot": zeroshot,
            f"{name}_fiveshot": fiveshot,
        }
    )
    df[f"{name}_zeroshot_parsed"] = df[f"{name}_zeroshot"].apply(tmmlu.parse_answer)
    df[f"{name}_fiveshot_parsed"] = df[f"{name}_fiveshot"].apply(tmmlu.parse_answer)

    df.to_csv(Path(f"./data/{name}_mmlu_responses.csv"), header=True, index=False)

    print(df.sample(10))

    del tmmlu, zeroshot, fiveshot, df
    torch.cuda.empty_cache()
    gc.collect()
    # ~90 min / loop


# %%
# parse and score
def parse_answer(response: str):
    """Extract multiple-choice letter selection from string response."""
    # select only A, B, C,or D;
    # if the response continues from the letter, then the subsequent characters must be nonword, a space, or line-end
    choices = re.compile(r"([ABCD])(?=\W|\s|$)")

    try:
        return choices.findall(response)[0]
    except IndexError:
        logger.info(f"No uppercase letters in A-E found in {response}, returning None")

        return None


# %%
llama2_mmlu = pd.read_csv(Path("./data/llama2_mmlu_responses.csv"))
llama3_mmlu = pd.read_csv(Path("./data/llama3_mmlu_responses.csv"))

response_df = pd.concat([llama2_mmlu, llama3_mmlu], axis="columns")
response_df.head()

for col in response_df.filter(like="parsed").columns:
    typos_df[col.replace("parsed", "correct")] = response_df[col] == typos_df["answer"]

# %%
cols = typos_df.filter(like="correct").columns
plot_df = (
    typos_df[["rate", *cols]]
    .groupby(["rate"])
    .mean()
    .reset_index()
    .melt(
        id_vars="rate",
        value_vars=cols,
        var_name="model",
        value_name="mean_accuracy",
    )
    .replace(
        {
            "llama2_zeroshot_correct": "Llama 2 7B Chat - 0-shot",
            "llama2_fiveshot_correct": "Llama 2 7B Chat - 5-shot",
            "llama3_zeroshot_correct": "Llama 3 8B Instruct - 0-shot",
            "llama3_fiveshot_correct": "Llama 3 8B Instruct - 5-shot",
        }
    )
)

g = (
    ggplot()
    + geom_line(
        plot_df,
        aes(
            x="rate",
            y="mean_accuracy",
            color="model",
        ),
        size=1,
    )
    + scale_x_continuous(
        breaks=[x / 10 for x in range(0, 11)],
        labels=percent_format(),
    )
    + scale_y_continuous(
        breaks=[x / 10 for x in range(0, 8)],
        labels=percent_format(),
    )
    + coord_cartesian(xlim=(0, 1), ylim=(0, 0.7))
    + guides(color=guide_legend(title="Model", ncol=2, reverse=True))
    + labs(
        title="tinyBenchmarks MMLU Accuracy by Typo Rate",
        x="Typo Occurrence Rate",
        y="Model Accuracy",
    )
    + theme_xkcd()
    + theme(
        figure_size=(6, 6),
        legend_position=(0.55, 0.2),
    )
)
# fmt: on

g.save(Path("./mmlu-accuracy.png"))
g.draw()

# %%
