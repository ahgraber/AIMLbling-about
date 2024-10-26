import itertools
import logging
import re
import typing as t


def check_torch_device():
    """Check which device pytorch will use."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")

    logging.info(f"Found pytorch device '{device.type}'")
    return device


def batched(iterable: t.Iterable, n: int):
    """Batch data from the iterable into tuples of length n. The last batch may be shorter than n."""
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def hugo_title_to_h1(text: str):
    """Extract title hugo front matter section and convert to markdown H1."""
    frontmatter = re.match(r"(?P<frontmatter>---[\s\S]*?---)", text)

    try:
        frontmatter = frontmatter["frontmatter"]
    except TypeError:
        # no frontmatter; return without changes
        return text

    title = re.match(r"[\s\S]*title: (?P<title>.*)\n", frontmatter)
    try:
        title = f"# {title['title']}\n\n"  # ensure title has trailing newlines
    except TypeError:
        logging.info("Could not parse title from frontmatter")
        logging.debug({frontmatter["frontmatter"]})
        title = ""
    else:
        text = text.replace(frontmatter, title)
        text = re.sub(r"\n{3,}", "\n\n", text)  # clean up overzealous newlines
        return text
