import itertools
import logging
from pathlib import Path
import typing as t

from rouge_score import rouge_scorer

from ragas import EvaluationDataset, evaluate
from ragas.metrics.base import (
    MetricWithEmbeddings,
    MetricWithLLM,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopKRougeScorer:
    """Calculate similarity between lists of strings."""

    def __init__(
        self,
        rouge_type: str,
        use_stemmer: bool = False,
        split_summaries: bool = False,
        metric: str = "fmeasure",
        weight: bool = True,
        k: int = 5,
    ):
        self.rouge_type = rouge_type
        if metric in ["precision", "recall", "fmeasure"]:
            self.metric = metric
        else:
            raise ValueError("metric not in ['precision','recall','fmeasure']")

        self.scorer = rouge_scorer.RougeScorer(
            [rouge_type],  # use rougeL, which ignores/concats newlines
            use_stemmer=use_stemmer,
            split_summaries=split_summaries,
        )

        self.weights = self._calculate_rank_weights(k) if weight else None

    def _calculate_rank_weights(self, k: int) -> np.ndarray:
        """Create a weight matrix of shape (m, n) with weights diminishing from the identity."""
        # Create arrays of row and column indices
        indices = np.arange(k)

        # Calculate the distance matrix
        distance_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])

        # Create the weight matrix using the distance
        weight_matrix = 1 / (1 + distance_matrix)
        return weight_matrix

    def score(self, x: str, y: str) -> float:
        """Evaluate Rouge score for string pairs and extract the specified metric."""
        return self.scorer.score(x, y)[self.rouge_type]._asdict()[self.metric]

    def score_lists(self, a: t.List[str], b: t.List[str]) -> float:
        """Compare lists of retrieved text using ROUGE."""
        pairs = itertools.product(a, b)
        scores = np.array([self.score(x, y) for x, y in pairs]).reshape(len(a), len(b))

        if self.weights is None:
            return scores.mean()
        else:
            return float(np.average(scores, weights=self.weights))


def validate_metrics(metrics: t.List):
    """Ensure metrics have been initialized with required model access."""
    for metric in metrics:
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            logger.warning(f"{metric.name} does not have an LLM assigned")
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            logger.warning(f"{metric.name} does not have an embedding model assigned")


def run_ragas_evals(source_df: pd.DataFrame, metrics: t.List, outfile: Path | str, batch_size: int = 20):
    """Run evaluation of set of metrics for source data."""
    # check if output file exists
    outfile = Path(outfile) if isinstance(outfile, str) else outfile
    if outfile.exists():
        logger.info(f"Prior '{outfile}' exists, will not rerun.")
    else:
        # ensure source data has features required for metrics
        required_cols = set(
            itertools.chain.from_iterable(metric.required_columns["SINGLE_TURN"] for metric in metrics)
        )
        missing = required_cols - set(source_df.columns)
        assert not missing, f"Error: source_df missing column(s) {missing}"  # NOQA: S101

        # run evals
        eval_dataset = EvaluationDataset.from_list(source_df.to_dict(orient="records"))
        eval_results = evaluate(dataset=eval_dataset, metrics=metrics, batch_size=batch_size)
        logger.info(f"Summary metrics:\n{eval_results}")

        # save work
        eval_df = pd.concat([source_df, pd.DataFrame.from_records(eval_results.scores)], axis="columns")
        logger.info("Saving eval_results...")
        eval_df.to_json(outfile, orient="records", lines=True)

        logger.info("Evaluation run complete.")
