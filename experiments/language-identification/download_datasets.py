# %%
import argparse
import gzip
from io import BytesIO
import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

import requests

import pandas as pd

from aiml.utils import basic_log_config, get_repo_path

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_DIR = get_repo_path(__file__)

# %%
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Languages commented out are not supported by some benchmarks
unsupported_languages = [
    "br",
    "bs",
    "eo",
    "eu",
    "gl",
    "hy",
    "is",
    "ka",
    "kk",
    "ms",
    "pt_br",
    "si",
    "sr",
    "ze_en",
    "ze_zh",
    "zh_tw",
]


# %%
def get_opensubtitles_languages() -> list[str]:
    """Get available languages for opensubtitles corpus."""
    url = "https://opus.nlpl.eu/opusapi/?languages=True&corpus=OpenSubtitles"
    response = requests.get(url, allow_redirects=True, timeout=120)

    try:
        response.raise_for_status()
    except Exception:
        logger.warning(f"Failed to download the file for {url}. Status code: {response.status_code}")
        pass

    content = json.loads(response.content)
    return sorted(content["languages"])


def download_opensubtitles_sample(lang: str, sample_size: int = 100_000):
    """Download language sample from opensubtitles corpus.

    Failures expected for ['az','be','ic','lb','nb', 'oc', 'scc', 'scr', 'sw', 'zh', 'zh_en', 'zh_zh',]
    """
    if lang in ["az", "be", "ic", "lb", "nb", "oc", "scc", "scr", "sw", "zh", "zh_en", "zh_zh"]:
        pass

    url = f"https://opus.nlpl.eu/legacy/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.{lang}.gz"
    response = requests.get(url, stream=True, allow_redirects=True, timeout=120)

    try:
        response.raise_for_status()
    except Exception:
        logger.warning(f"Failed to download the file for {url}. Status code: {response.status_code}")
        pass

    if response.status_code == 200:
        (DATA_DIR / "opensubtitles").mkdir(exist_ok=True)

        # Create a buffer to hold the decompressed data
        with (
            gzip.GzipFile(fileobj=BytesIO(response.content)) as f_in,
            (DATA_DIR / "opensubtitles" / f"{lang}.txt").open("w") as f_out,
        ):
            # Read the first 100,000 lines and write them to a file
            for i, line in enumerate(f_in):
                if i >= sample_size:
                    break
                f_out.write(line.decode("utf-8"))


def opensubtitles_to_parquet(sample_size: int = 100_000):
    """Read opensubtitles text files into local parquet."""
    opensubtitles = {}
    for file in (DATA_DIR / "opensubtitles").glob("*.txt"):
        lang = file.stem
        with file.open("r") as f:
            sample = f.readlines()

        # Extend the list with None values using slicing
        sample[len(sample) :] = [None] * (sample_size - len(sample))
        if not all(len(v) == sample_size for _, v in opensubtitles.items()):
            raise ValueError("Error: opensubtitles has ragged arrays")

        opensubtitles[lang] = sample

    df = pd.DataFrame.from_dict(opensubtitles).melt(var_name="label", value_name="text").dropna()
    df.to_parquet(DATA_DIR / "opensubtitles.parquet")


# %%
def get_tatoeba_languages() -> list[str]:
    """Get available languages for tatoeba corpus."""
    url = "https://opus.nlpl.eu/opusapi/?languages=True&corpus=Tatoeba"
    response = requests.get(url, allow_redirects=True, timeout=120)

    try:
        response.raise_for_status()
    except Exception:
        logger.exception(f"Failed to download the file for {url}. Status code: {response.status_code}")
        raise

    content = json.loads(response.content)
    return sorted(content["languages"])


def download_tatoeba_sample(lang: str, sample_size: int = 100_000):
    """Download language sample from tatoeba corpus.

    Failures expected for []
    """
    url = f"https://opus.nlpl.eu/legacy/download.php?f=Tatoeba/v2023-04-12/mono/{lang}.txt.gz"
    response = requests.get(url, stream=True, allow_redirects=True, timeout=120)

    try:
        response.raise_for_status()
    except Exception:
        logger.warning(f"Failed to download the file for {url}. Status code: {response.status_code}")
        pass

    if response.status_code == 200:
        (DATA_DIR / "tatoeba").mkdir(exist_ok=True)
        with (
            gzip.GzipFile(fileobj=BytesIO(response.content)) as f_in,
            (DATA_DIR / "tatoeba" / f"{lang}.txt").open("w") as f_out,
        ):
            # Read the first 100,000 lines and write them to a file
            for i, line in enumerate(f_in):
                if i >= sample_size:
                    break
                f_out.write(line.decode("utf-8"))


def tatoeba_to_parquet(sample_size: int = 100_000):
    """Read tatoeba text files into local parquet."""
    tatoeba = {}
    for file in (DATA_DIR / "tatoeba").glob("*.txt"):
        lang = file.stem
        with file.open("r") as f:
            sample = f.readlines()

        # Extend the list with None values using slicing
        sample[len(sample) :] = [None] * (sample_size - len(sample))
        if not all(len(v) == sample_size for _, v in tatoeba.items()):
            raise ValueError("Error: opensubtitles has ragged arrays")

        tatoeba[lang] = sample

    df = pd.DataFrame.from_dict(tatoeba).melt(var_name="label", value_name="text").dropna()
    df.to_parquet(DATA_DIR / "tatoeba.parquet")


# %%
def papluca_to_parquet():
    """Read papluca/language-identification into local parquet."""
    df = pd.read_csv("hf://datasets/papluca/language-identification/test.csv")
    df = df.rename(columns={"labels": "label"})
    df.to_parquet(DATA_DIR / "papluca.parquet")


def global_mmlu_to_parquet():
    """Read global_mmlu into local parquet."""
    gmmlu_lang_codes = ['am', 'ar', 'bn', 'cs', 'de', 'el', 'en', 'es', 'fa', 'fil',
                        'fr', 'ha', 'he', 'hi', 'id', 'ig', 'it', 'ja', 'ko', 'ky',
                        'lt', 'mg', 'ms', 'ne', 'nl', 'ny', 'pl', 'pt', 'ro', 'ru',
                        'si', 'sn', 'so', 'sr', 'sv', 'sw', 'te', 'tr', 'uk', 'vi',
                        'yo', 'zh']  # fmt: skip
    _dfs = []
    for code in tqdm(gmmlu_lang_codes):
        _df = pd.read_parquet(f"hf://datasets/CohereForAI/Global-MMLU/{code}/test-00000-of-00001.parquet")
        _df["label"] = code
        _df = _df[["label", "question"]].rename(columns={"question": "text"})
        _dfs.append(_df)
    df = pd.concat(_dfs, ignore_index=True)
    df.to_parquet(DATA_DIR / "global_mmlu.parquet")


# %%
if __name__ == "__main__":
    os_langs = get_opensubtitles_languages()
    t_langs = get_tatoeba_languages()
    logger.info("Downloading opensubtitles and tatoeba corpuses...")
    for lang in tqdm(os_langs):
        download_opensubtitles_sample(lang)
        if lang in t_langs:  # don't need all 300 tatoeba langs
            download_tatoeba_sample(lang)

    logger.info("Writing to parquet files...")
    opensubtitles_to_parquet()
    tatoeba_to_parquet()
    papluca_to_parquet()
    global_mmlu_to_parquet()

# %%
