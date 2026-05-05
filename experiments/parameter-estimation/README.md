# parameter-estimation

Companion code for [_Benchmarks predict model size_](../../hugo/content/blog/benchmarks-predict-model-size/index.md):
fit a small log-linear regression that maps Artificial Analysis benchmark scores
back to (estimated) total parameter counts, then predict params for closed-weight
models such as Claude, GPT, Gemini, Muse Spark, etc.

## Layout

| Path                   | Purpose                                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `rsc_fetch.py`         | Capture Next.js RSC payloads from Artificial Analysis pages with Playwright.                                                |
| `rsc_parse.py`         | Pull `(model, value)` pairs out of the RSC payloads into per-metric CSVs.                                                   |
| `param_estimation.py`  | Build features, fit/eval the GLMs, plot diagnostics, persist artifacts.                                                     |
| `data/params.csv`      | Hand-curated `totalParams`, `activeParams`, expert-count metadata.                                                          |
| `data/<benchmark>.csv` | Auto-generated per-metric tables (`gdpval`, `intelligenceIndex`, `mmlu_pro`, `omniscience_accuracy`, `tau2`, `price_1m_*`). |
| `data/prices.csv`      | Backfill prices from Simon Willison's `llm-prices` repo.                                                                    |
| `data/rsc_*.txt`       | Raw RSC payloads — keep these so parsing is reproducible.                                                                   |
| `data/model_keys.json` | Snapshot of the AA model-object schema (handy when fields drift).                                                           |
| `models/*.pkl`         | Persisted regression models for reuse (see _Persisted models_ below).                                                       |
| `data_17JAN2026.zip`   | Frozen snapshot of the Jan 2026 inputs that backed the original blog post.                                                  |

## Setup

```bash
# from the repo root — uv workspace handles the package
just experiments sync parameter-estimation

# Playwright Chromium is fetched on-demand by `rsc_fetch.py` via
# `aiml.utils.ensure_playwright_chromium`; if you'd rather grab it eagerly:
uv run --package parameter-estimation playwright install chromium
```

## Refreshing data from Artificial Analysis

AA's site is a Next.js app.
Each page emits one or more RSC (React Server Component) payloads carrying the structured data the page renders.
We capture those payloads with Playwright (`rsc_fetch.py`), then pluck out the model values we care about (`rsc_parse.py`).

As of 2026-05 the App Router on `gdpval-aa` is the one-stop-shop: its `data[]` array carries every per-model metric (gdpval, intelligence_index, mmlu_pro, tau2, omniscience, prices) for ~360 models in a single payload.
Other evaluation pages (`/evaluations/omniscience`, `/evaluations/tau2-bench`) only ship a top-N filtered subset.
The metrics use snake_case keys here even though the rendered table sometimes shows camelCase elsewhere.

| URL                                                   | Output filename       | What's in it                                                                                                                        |
| ----------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `https://artificialanalysis.ai/evaluations/gdpval-aa` | `rsc_gdpval.txt`      | Full table: `gdpval`, `intelligence_index`, `mmlu_pro`, `tau2`, `omniscience`, `price_1m_input_tokens`, `price_1m_output_tokens`, … |
| `https://artificialanalysis.ai/leaderboards/models`   | `rsc_leaderboard.txt` | Same metrics under camelCase (`intelligenceIndex`); useful as a fallback if the gdpval page is down                                 |

> **MMLU-Pro:** AA dropped MMLU-Pro from Intelligence Index v4, so the field
> is null for any model added after Jan 2026. We still keep `data/mmlu_pro.csv`
> and the trained regressor — new models simply won't appear in it.
>
> **Omniscience:** AA replaced the old `omniscience_breakdown.total.accuracy`
> (0–1) with a composite `omniscience` score that can range from roughly −60
> to +30. The CSV file keeps the legacy filename, but the values are no longer
> a probability. If you mix old and new models in the regression, document the
> scoring change.

### 1. Fetch the payload

```bash
uv run --package parameter-estimation python experiments/parameter-estimation/rsc_fetch.py \
    --url https://artificialanalysis.ai/evaluations/gdpval-aa \
    --output experiments/parameter-estimation/data/rsc_gdpval.txt
```

The captured file should be **~9–10 MB** of HTML containing 100+ inline `self.__next_f.push([N, "..."])` chunks.
If you get a tiny file (\<1 MB) the filter in `rsc_fetch.py` is missing the document response — re-pull `main`.

### 2. Inspect the model schema (optional)

`--dump-keys` writes a merged key/type tree across `models[]`, `data[]`, and
`defaultData[]` so you can confirm where each metric lives before extracting
it.

```bash
uv run --package parameter-estimation python experiments/parameter-estimation/rsc_parse.py \
    --rsc experiments/parameter-estimation/data/rsc_gdpval.txt \
    --dump-keys
```

### 3. Extract per-metric CSVs

```bash
uv run --package parameter-estimation python experiments/parameter-estimation/rsc_parse.py \
    --rsc experiments/parameter-estimation/data/rsc_gdpval.txt \
    --metric gdpval=gdpval \
    --metric intelligenceIndex=intelligence_index \
    --metric mmlu_pro=mmlu_pro \
    --metric tau2=tau2 \
    --metric omniscience_accuracy=omniscience \
    --metric price_1m_input_tokens=price_1m_input_tokens \
    --metric price_1m_output_tokens=price_1m_output_tokens
```

Each `--metric label=path` writes `data/<label>.csv`.
The label is what the analysis script expects; the path is the snake_case field on the AA record.
Pass multiple `--metric` flags to write all columns from one payload — much faster than fetching/parsing one URL per metric.

### 4. Pricing CSVs

`price_1m_input_tokens.csv` and `price_1m_output_tokens.csv` are extracted
from the same payload as the benchmarks (see step 3). `prices.csv` is a
manual backfill from Simon Willison's [`llm-prices`](https://github.com/simonw/llm-prices)
data — kept around so models the AA endpoint omits still get a price.

### 5. Update `params.csv`

`data/params.csv` is the only file that is hand-maintained.
Add one row per base model name (the analysis aggregates parenthetical reasoning-effort suffixes, so write `Kimi K2.6` rather than `Kimi K2.6 (Reasoning)`).
For open-weight models grab `n_routed_experts` / `num_experts_per_tok` from each model's `config.json` on Hugging Face.
For closed-weight models leave the parameter columns empty — they become prediction targets.

## Running the regression

```bash
MPLBACKEND=Agg \
uv run --package parameter-estimation python experiments/parameter-estimation/param_estimation.py
```

The script:

1. Loads `params.csv`, the per-metric CSVs, and the price tables.
2. Computes `expert_sparsity` and `token_active_ratio` features.
3. Cross-validates a positive-coefficient log-linear regressor for every
   `(benchmark, [+price | +active])` spec.
4. Fits the top-3 specs on the full data and predicts `totalParams` for every
   model in `LLMS`.
5. Sweeps assumed `token_active_ratio` values for the closed-weight predictions
   to bound the estimate under MoE / dense assumptions.
6. Persists the `mmlu_pro` and `intelligenceIndex` fits to `models/`.

## Persisted models

`models/mmlu_pro.pkl` and `models/intelligenceIndex.pkl` are
[cloudpickle](https://github.com/cloudpipe/cloudpickle) artifacts containing:

```python
{
    "model": TransformedTargetRegressor,  # MinMaxScaler → LinearRegression(positive=True), log10 target
    "feature_cols": [...],
    "target_col": "totalParams",
    "n_train": int,
    "r2_mean": float,
    "mae_mean": float,
    "rmse_mean": float,
}
```

cloudpickle is required because the regressor's `inverse_func` is a function
defined in `param_estimation.py`; cloudpickle serialises it by value so the
artifact loads cleanly in a fresh interpreter without re-importing the script.

```python
import cloudpickle, pandas as pd

with open("experiments/parameter-estimation/models/intelligenceIndex.pkl", "rb") as fh:
    artifact = cloudpickle.load(fh)

X = pd.DataFrame({"intelligenceIndex": [42.5, 50.6]})
estimated_total_params = artifact["model"].predict(X)
```

The `mmlu_pro` model exists for posterity (AA stopped publishing MMLU-Pro after
Intelligence Index v4); it can still be used against historical scores.

## Caveats

- The regressor only sees ~25 open-weight training points.
  Take individual predictions with a grain of salt; trust the order-of-magnitude.
- `unify_model_names(strategy="optimistic")` keeps the column-wise max across reasoning-effort variants of the same base model.
  Switch to `"average"` if you want a more conservative estimate.
- AA periodically renames slugs (`gdpval` → `gdpval-aa`, `tau2` →
  `tau2-bench-telecom`); re-check the URL table above before each refresh.
