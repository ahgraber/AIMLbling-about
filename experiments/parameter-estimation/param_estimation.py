# %%
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import plotnine as ggplot

from aiml.utils import basic_log_config, get_repo_path, this_file

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
REPO_DIR = get_repo_path(Path.cwd())
LOCAL_DIR = REPO_DIR / "experiments" / "parameter-estimation"
DATA_DIR = LOCAL_DIR / "data"

# %%
# convert params from string to numeric
M = 1000**2
B = 1000**3
T = 1000**4


def parse_param_string(param_str):
    """Convert parameter string with suffixes to numeric value."""
    if isinstance(param_str, (int, float)):
        return param_str
    param_str = param_str.strip().upper()
    if param_str.endswith("T"):
        return float(param_str[:-1]) * T
    elif param_str.endswith("B"):
        return float(param_str[:-1]) * B
    elif param_str.endswith("M"):
        return float(param_str[:-1]) * M
    else:
        try:
            return float(param_str)
        except ValueError as e:
            raise ValueError(f"Could not parse parameter string: {param_str}") from e


def format_param_value(param_value: float | int | None, decimals: int = 2) -> str | None:
    """Format a numeric parameter count with M/B/T suffixes.

    Args:
        param_value: Parameter count as a number.
        decimals: Number of decimal places for M/B/T formatting.

    Returns:
        Human-readable string with M/B/T suffix, or None if input is missing.
    """
    if param_value is None or pd.isna(param_value):
        return None

    value = float(param_value)
    if value >= T:
        scaled = value / T
        if decimals == 0:
            return f"{int(np.floor(scaled))}T"
        return f"{scaled:.{decimals}f}T"
    if value >= B:
        scaled = value / B
        if decimals == 0:
            return f"{int(np.floor(scaled))}B"
        return f"{scaled:.{decimals}f}B"
    if value >= M:
        scaled = value / M
        if decimals == 0:
            return f"{int(np.floor(scaled))}M"
        return f"{scaled:.{decimals}f}M"
    return f"{value:.0f}"


def unify_model_names(data: pd.DataFrame, strategy: Literal["optimistic", "average"] = "optimistic") -> pd.DataFrame:
    """Unify model names by stripping parenthetical suffixes and aggregating values.

    Args:
        data: Input data with model names as the index.
        strategy: "optimistic" to take column-wise maxima, "average" to take means.

    Returns:
        Aggregated data indexed by base model name.
    """
    base_names = data.index.str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
    grouped = data.groupby(base_names, sort=False)
    if strategy == "optimistic":
        return grouped.max()
    if strategy == "average":
        return grouped.mean()
    raise ValueError(f"Unsupported strategy: {strategy}")


# %%
BENCHMARK_COLS = [
    "gdpval",
    "intelligenceIndex",
    "mmlu_pro",
    "omniscience_accuracy",
    "tau2",
]
PRICE_COLS = [
    "price_1m_3_to_1",
    "price_1m_input_tokens",
    "price_1m_output_tokens",
]
MODEL_UNIFICATION_STRATEGY = "optimistic"
# %%
df = pd.read_csv(DATA_DIR / "params.csv").drop(columns=["references", "notes"])
df["totalParams"] = df["totalParams"].apply(parse_param_string)
df["activeParams"] = df["activeParams"].apply(parse_param_string)
df = df.set_index("model")
df = unify_model_names(df, strategy=MODEL_UNIFICATION_STRATEGY)

# %%
# We define sparsity as the ratio of inactive experts to the total number of experts, which controls
# the ratio of the total number of parameters to FLOPs per example in MoEs
# ref: [[2501.12370] Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370)
df["expert_sparsity"] = 1.0 - df["activeExperts"] / df["totalExperts"].replace({0: np.nan})

# parameter-level (per-token, per forward pass)
df["token_active_ratio"] = df["activeParams"] / df["totalParams"].replace({0: np.nan})  # density

df = df.drop(columns=["activeParams", "totalExperts", "activeExperts"])

# %%
_feature_csvs = sorted(
    csv_path
    for csv_path in DATA_DIR.glob("*.csv")
    if (csv_path.name != "params.csv") and ("price" not in csv_path.name)
)
_feature_dfs = [pd.read_csv(csv).set_index("model") for csv in _feature_csvs]
_feature_dfs = [unify_model_names(fdf, strategy=MODEL_UNIFICATION_STRATEGY) for fdf in _feature_dfs]

feature_df = pd.concat(_feature_dfs, axis="columns")
del _feature_csvs, _feature_dfs

# %%
# start with price_1m from artificial analysis
_price_csvs = sorted(csv_path for csv_path in DATA_DIR.glob("*.csv") if "price_1m" in csv_path.name)
_price_dfs = [pd.read_csv(csv).set_index("model") for csv in _price_csvs]
_price_dfs = [unify_model_names(fdf, strategy=MODEL_UNIFICATION_STRATEGY) for fdf in _price_dfs]

price_df = pd.concat(_price_dfs, axis="columns")
del _price_csvs

# %%
# "free" doesn't exist, treat as NaN
price_df = price_df.replace({0: np.nan})

# use Simon Willison's pricing info to fill missing prices where available
fill_df = pd.read_csv(DATA_DIR / "prices.csv").set_index("model")
for col in price_df.columns:
    if col in fill_df.columns:
        candidate = fill_df[col].reindex(price_df.index)
        use_mask = candidate.notna() & (candidate != 0)
        price_df.loc[use_mask, col] = candidate.loc[use_mask]

# add 3:1 input:output blended price
price_df["price_1m_3_to_1"] = ((price_df["price_1m_input_tokens"] * 3) + price_df["price_1m_output_tokens"]) / 4

# %%
df = pd.concat([df, feature_df, price_df], axis="columns")
print(df.head(10).to_markdown())


# %%
def _build_lm() -> TransformedTargetRegressor:
    lm = LinearRegression(positive=True)
    pipeline = Pipeline(
        [
            ("scale", MinMaxScaler()),
            ("glm", lm),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log10,
        inverse_func=lambda vals: np.power(10.0, vals),
        check_inverse=False,
    )


def _evaluate_glm(
    features: pd.DataFrame,
    target: pd.Series,
    cv_splits: int = 5,
    test_size: int = 5,
) -> dict[str, float]:
    if len(features) <= test_size:
        return {
            "r2_mean": np.nan,
            "r2_std": np.nan,
            "mae_mean": np.nan,
            "mae_std": np.nan,
            "rmse_mean": np.nan,
            "rmse_std": np.nan,
        }

    cv = ShuffleSplit(n_splits=cv_splits, test_size=test_size, random_state=7)
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    for train_idx, test_idx in cv.split(features):
        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        model = _build_lm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        r2_scores.append(r2_score(np.log10(y_test), np.log10(y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores, ddof=1)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores, ddof=1)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores, ddof=1)),
    }


# %%
def _format_param_ticks(vals) -> list[str]:
    return [format_param_value(val, decimals=0) or "" for val in vals]


def _format_log10_billions(vals) -> list[str]:
    formatted = []
    for val in vals:
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            formatted.append("")
            continue
        formatted.append(format_param_value((10**numeric) * B, decimals=0) or "")
    return formatted


def _log_ticks(min_val: float, max_val: float) -> tuple[list[float], list[float]]:
    min_val = max(min_val, 1.0)
    max_val = max(max_val, min_val * 1.01)
    min_pow = int(np.floor(np.log10(min_val)))
    max_pow = int(np.ceil(np.log10(max_val)))
    major = [10**power for power in range(min_pow, max_pow + 1)]
    minor = [mult * (10**power) for power in range(min_pow, max_pow + 1) for mult in range(2, 10)]
    return major, minor


def _format_mixed_axis(vals) -> list[str]:
    formatted = []
    for val in vals:
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            formatted.append("")
            continue
        if not np.isfinite(numeric):
            formatted.append("")
            continue
        if abs(numeric) >= M:
            formatted.append(format_param_value(numeric, decimals=0) or "")
        else:
            formatted.append(f"{numeric:.2f}")
    return formatted


def _find_line_equation(
    model: TransformedTargetRegressor,
    features: pd.DataFrame,
    benchmark: str,
    points: int = 200,
) -> tuple[float, float, pd.DataFrame]:
    """Build regression line inputs and equation parameters for plotting.

    Args:
        model: Fitted transformed-target regression model.
        features: Feature matrix used for predictions (non-null rows).
        benchmark: Benchmark column name plotted on the x-axis.
        points: Number of points to sample for the line.

    Returns:
        Tuple containing slope, intercept, line dataframe.
    """
    x_values = features[benchmark]
    x_min = float(x_values.min())
    x_max = float(x_values.max())

    line_x = np.linspace(x_min, x_max, points)
    feature_means = features.mean(axis=0)
    line_features = pd.DataFrame(
        np.tile(feature_means, (len(line_x), 1)),
        columns=features.columns,
        index=np.arange(len(line_x)),
    )
    line_features[benchmark] = line_x
    line_y = model.predict(line_features)
    # slope, intercept = np.polyfit(line_x, np.log10(line_y), 1)
    slope, intercept = np.polyfit(line_x, line_y, 1)

    line_df = pd.DataFrame(
        {
            "benchmark_score": line_x,
            "line_totalParams": line_y,
        }
    )

    return slope, intercept, line_df


def plot_model_fit(
    model: TransformedTargetRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    benchmark: str,
    r2: float | None = None,
    title: str | None = None,
) -> ggplot.ggplot:
    """Plot model fit with actual/predicted points, regression line, and labels.

    Args:
        model: Fitted transformed-target regression model.
        source_df: Source dataframe with benchmark and parameter columns.
        features: Feature matrix used for predictions (non-null rows).
        target: Target series aligned to the training subset.
        benchmark: Benchmark column name plotted on the x-axis.
        r2: Optional R² value to display on the plot.
        title: Plot title.

    Returns:
        Plotnine plot object.
    """
    plot_df = features[features[benchmark].notna()].copy()

    # predicted and actual
    plot_df["predicted_totalParams"] = model.predict(plot_df[[benchmark]])
    plot_df = plot_df.join(target, how="left")
    plot_df["abs_error"] = (plot_df["predicted_totalParams"] - plot_df["totalParams"]).abs()
    plot_df["error_log10_b"] = np.where(
        np.isfinite(plot_df["abs_error"]) & (plot_df["abs_error"] > 0), np.log10(plot_df["abs_error"] / B), np.nan
    )

    # housekeeping
    plot_df = plot_df.reset_index().rename(columns={"index": "model"})
    plot_df = plot_df.rename(columns={benchmark: "benchmark_score"})
    plot_df["predicted_totalParams_str"] = plot_df["predicted_totalParams"].apply(format_param_value)
    plot_df["actual_marker"] = "Actual"
    plot_df["pred_marker"] = "Predicted"

    slope, intercept, line_df = _find_line_equation(
        model=model,
        features=features,
        benchmark=benchmark,
    )
    equation = f"totalParams = {intercept:.2e} + {slope:.2e}*{benchmark}"
    if r2:
        equation += f"\nR^2 = {r2:.2f}"

    y_values = plot_df["totalParams"]
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    x_min = float(plot_df["benchmark_score"].min())
    x_max = float(plot_df["benchmark_score"].max())
    y_breaks, y_minor = _log_ticks(y_min, y_max)

    label_models = set(features.index) - set(target.index)
    label_df = plot_df[plot_df["model"].isin(label_models)].copy()
    label_df["label"] = label_df["model"] + "\n" + label_df["predicted_totalParams_str"].fillna("")
    label_adjust = {
        "expand_text": (1.05, 1.15),
        "expand_points": (1.05, 1.15),
        "arrowprops": {
            "arrowstyle": "-",
            "color": "#888888",
            "lw": 0.6,
            "alpha": 0.6,
            "shrinkA": 6,
            "shrinkB": 6,
        },
    }

    plot = (
        ggplot.ggplot(plot_df, ggplot.aes(x="benchmark_score", y="totalParams"))
        + ggplot.geom_line(
            data=line_df,
            mapping=ggplot.aes(x="benchmark_score", y="line_totalParams"),
            color="#3f7f8c",
            size=0.9,
            alpha=0.8,
            inherit_aes=False,
        )
        + ggplot.geom_segment(
            data=plot_df[plot_df["totalParams"].notna()],
            mapping=ggplot.aes(
                x="benchmark_score",
                y="totalParams",
                xend="benchmark_score",
                yend="predicted_totalParams",
            ),
            color="#8a8a8a",
            size=0.4,
            alpha=0.6,
            inherit_aes=False,
        )
        + ggplot.geom_point(
            data=plot_df[plot_df["totalParams"].notna()],
            mapping=ggplot.aes(
                x="benchmark_score",
                y="totalParams",
                shape="actual_marker",
            ),
            color="#5a5a5a",
            alpha=0.6,
            size=2.4,
            inherit_aes=False,
        )
        + ggplot.geom_point(
            data=plot_df[plot_df["totalParams"].notna()],
            mapping=ggplot.aes(
                x="benchmark_score",
                y="predicted_totalParams",
                color="error_log10_b",
                shape="pred_marker",
            ),
            alpha=0.75,
            size=2.2,
            inherit_aes=False,
        )
        + ggplot.geom_point(
            data=plot_df[plot_df["totalParams"].isna()],
            mapping=ggplot.aes(
                x="benchmark_score",
                y="predicted_totalParams",
                shape="pred_marker",
            ),
            color="#235099",
            alpha=0.75,
            size=3,
            inherit_aes=False,
        )
        + ggplot.geom_text(
            data=label_df,
            mapping=ggplot.aes(
                x="benchmark_score",
                y="predicted_totalParams",
                label="label",
            ),
            adjust_text=label_adjust,
            size=7,
            color="#2b2b2b",
            inherit_aes=False,
        )
        + ggplot.scale_color_cmap(
            name="Abs. Error (log10)",
            cmap_name="Spectral_r",
            labels=_format_log10_billions,
            na_value="#bdbdbd",
        )
        + ggplot.scale_shape_manual(
            values={"Actual": "o", "Predicted": "^"},
        )
        + ggplot.scale_y_log10(
            breaks=y_breaks,
            minor_breaks=y_minor,
            labels=_format_param_ticks,
        )
        + ggplot.annotate(
            "text",
            x=x_min + (x_max - x_min) * 0.01,
            y=y_max / 0.2,
            label=equation,
            ha="left",
            va="top",
            size=7,
            color="#2b2b2b",
        )
        + ggplot.labs(
            title=title,
            x=f"{benchmark} score",
            y="Total parameters (log scale)",
            color="",
            shape="",
        )
        + ggplot.theme_minimal()
        + ggplot.theme(
            legend_position="right",
            legend_title=ggplot.element_text(size=9),
            legend_text=ggplot.element_text(size=8),
            axis_line=ggplot.element_line(color="#444444", size=0.6),
            axis_text_x=ggplot.element_text(size=8),
            axis_text_y=ggplot.element_text(size=8),
            axis_ticks_minor=ggplot.element_line(color="#666666", size=0.4),
            axis_ticks_length_minor=3,
        )
    )
    return plot


# %%
target_col = "totalParams"
price_cols = [col for col in PRICE_COLS if col in df.columns]

# %%
# define all model specifications for experiments
model_specs = {}
for benchmark in BENCHMARK_COLS:
    model_specs[benchmark] = {
        "benchmark": benchmark,
        "feature_cols": [benchmark],
    }
    model_specs[f"{benchmark}+price"] = {
        "benchmark": benchmark,
        "feature_cols": [benchmark] + price_cols,
    }
    model_specs[f"{benchmark}+active"] = {
        "benchmark": benchmark,
        "feature_cols": [benchmark] + ["token_active_ratio"],
    }

# %%
metrics_rows = []
for name, spec in model_specs.items():
    feature_cols = spec["feature_cols"]
    subset = df[feature_cols + [target_col]].dropna()
    features = subset[feature_cols]
    target = subset[target_col]

    metrics = _evaluate_glm(features, target)
    metrics_rows.append(
        {
            "model_name": name,
            # "benchmark": spec["benchmark"],
            "r2_mean": metrics["r2_mean"],
            # "r2_std": metrics["r2_std"],
            "mae_mean": metrics["mae_mean"],
            # "mae_std": metrics["mae_std"],
            "rmse_mean": metrics["rmse_mean"],
            # "rmse_std": metrics["rmse_std"],
        }
    )

metrics_df = pd.DataFrame(metrics_rows).dropna(subset=["r2_mean"]).sort_values("r2_mean", ascending=False)
metrics_tall = metrics_df.melt(
    id_vars=["model_name"],
    value_vars=["r2_mean", "mae_mean", "rmse_mean"],
    var_name="metric",
    value_name="value",
)

metrics_tall["metric"] = pd.Categorical(
    metrics_tall["metric"],
    categories=["r2_mean", "mae_mean", "rmse_mean"],
    ordered=True,
)
metrics_tall["value_label"] = metrics_tall.apply(
    lambda row: f"{row['value']:.2f}"
    if row["metric"] == "r2_mean"
    else (format_param_value(row["value"], decimals=2) or ""),
    axis=1,
)
label_offset_fraction = 0.05
metric_ranges = metrics_tall.groupby("metric")["value"].transform(lambda values: values.max() - values.min())
fallback_offsets = metrics_tall["value"].abs().where(metrics_tall["value"].abs() > 0, 1.0)
metrics_tall["y_text"] = metrics_tall["value"] + np.where(
    metric_ranges > 0,
    metric_ranges * label_offset_fraction,
    fallback_offsets * label_offset_fraction,
)

# %%
metrics_plot = (
    ggplot.ggplot(metrics_tall, ggplot.aes(x="model_name", y="value"))
    + ggplot.geom_point(size=2.5, color="#3b3b3b")
    + ggplot.geom_text(
        ggplot.aes(label="value_label", y="y_text"),
        size=7,
        va="bottom",
    )
    + ggplot.facet_wrap("metric", scales="free_y", ncol=1)
    + ggplot.scale_y_continuous(labels=_format_mixed_axis)
    + ggplot.labs(
        title="Regression fit quality across metrics (mean ± std)",
        x="Model specification",
        y="Score",
    )
    + ggplot.theme_minimal()
    + ggplot.theme(
        axis_text_x=ggplot.element_text(size=7, rotation=90),
        strip_text=ggplot.element_text(size=9, weight="bold"),
        axis_line=ggplot.element_line(color="#444444", size=0.6),
        figure_size=(7, 9),
    )
)
metrics_plot.show()

# %%
metrics_df = metrics_df.set_index("model_name")
print(metrics_df.to_markdown())

# %%
metrics_df["rank"] = (
    metrics_df["r2_mean"].rank(ascending=False)
    + metrics_df["mae_mean"].rank(ascending=True)
    + metrics_df["rmse_mean"].rank(ascending=True)
) / 3
metrics_df = metrics_df.sort_values("rank", ascending=True)
print(metrics_df[["rank"]].to_markdown())

# %%
# top3 = metrics_df.sort_values("rank", ascending=True).index.to_list()[:3]
top3 = [
    "omniscience_accuracy",
    "mmlu_pro",
    "intelligenceIndex",
]
fitted_models = {}

for spec_name in top3:
    spec = model_specs[spec_name]
    feature_cols = spec["feature_cols"]
    subset = df[feature_cols + [target_col]].dropna()
    features = subset[feature_cols]
    target = subset[target_col]

    model = _build_lm()
    model.fit(features, target)
    fitted_models[spec_name] = model

# %%
LLMS = [
    "Claude 3.7 Sonnet",
    "Claude 4.5 Haiku",
    "Claude 4.5 Sonnet",
    "Claude Opus 4.5",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Pro",
    "Gemini 3 Flash",
    "Gemini 3 Pro Preview",
    "GPT-4.1",
    "GPT-4o",
    "GPT-5",
    "GPT-5 mini",
    "GPT-5 nano",
    "GPT-5.1",
    "GPT-5.2",
]

prediction_rows = []
for spec_name, model in fitted_models.items():
    spec = model_specs[spec_name]
    llm_features = df.loc[df.index.intersection(LLMS), spec["feature_cols"]].dropna()
    if llm_features.empty:
        continue
    preds = model.predict(llm_features)
    for model_name, pred in zip(llm_features.index, preds):
        prediction_rows.append(
            {
                "llm": model_name,
                "model_spec": spec_name,
                "predicted_totalParams": pred,
                "predicted_totalParams_str": format_param_value(pred),
            }
        )

predictions_df = pd.DataFrame(prediction_rows).sort_values(["model_spec", "llm"])
print(predictions_df[["llm", "model_spec", "predicted_totalParams_str"]].to_markdown(index=False))

# %%
for spec_name, model in fitted_models.items():
    spec = model_specs[spec_name]
    feature_cols: list[str] = spec["feature_cols"]
    subset: pd.DataFrame = df[feature_cols + [target_col]].dropna()
    features: pd.DataFrame = pd.concat(
        [
            subset[feature_cols],
            df.loc[df.index.intersection(LLMS), feature_cols],
        ],
        axis="index",
    )
    target: pd.Series = subset[target_col]

    plot = plot_model_fit(
        model=model,
        features=features,
        target=target,
        benchmark=spec["benchmark"],
        r2=float(metrics_df.loc[spec_name, "r2_mean"].round(3)),
        title=f"Total Parameters x {spec_name}",
    )
    plot.show()

# %%
ASSUMED_ACTIVE_RATIOS = [0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01]
sensitivity_rows = []

top3_active = [
    "omniscience_accuracy+active",
    "mmlu_pro+active",
    "intelligenceIndex+active",
]
fitted_models = {}

for spec_name in top3_active:
    spec = model_specs[spec_name]
    feature_cols = spec["feature_cols"]
    subset = df[feature_cols + [target_col]].dropna()
    features = subset[feature_cols]
    target = subset[target_col]

    model = _build_lm()
    model.fit(features, target)
    fitted_models[spec_name] = model

for spec_name, model in fitted_models.items():
    spec = model_specs[spec_name]
    benchmark = spec["benchmark"]
    llm_benchmark = df.loc[df.index.intersection(LLMS), [benchmark]].dropna()
    if llm_benchmark.empty:
        continue

    ratios = np.array(ASSUMED_ACTIVE_RATIOS, dtype=float)
    llm_index = llm_benchmark.index.to_numpy()
    bench_vals = llm_benchmark[benchmark].to_numpy()
    repeat_count = len(ratios)
    features = pd.DataFrame(
        {
            benchmark: np.repeat(bench_vals, repeat_count),
            "token_active_ratio": np.tile(ratios, len(bench_vals)),
        },
        index=np.repeat(llm_index, repeat_count),
    )
    preds = model.predict(features)
    sensitivity_rows.append(
        pd.DataFrame(
            {
                "llm": features.index,
                "benchmark": benchmark,
                "token_active_ratio": features["token_active_ratio"],
                "predicted_totalParams": preds,
            }
        )
    )

# %%
sensitivity_df = pd.concat(sensitivity_rows, ignore_index=True).sort_values(["benchmark", "llm", "token_active_ratio"])

plot = (
    ggplot.ggplot(
        sensitivity_df,
        ggplot.aes(
            x="token_active_ratio",
            y="predicted_totalParams",
            color="llm",
            group="llm",
        ),
    )
    + ggplot.geom_line(size=0.7)
    + ggplot.geom_point(size=1.3)
    + ggplot.scale_y_log10(labels=_format_param_ticks)
    + ggplot.facet_wrap("~benchmark", scales="free_y")
    + ggplot.labs(
        title="Predicted total parameters vs token active ratio",
        x="Token active ratio",
        y="Total parameters (log scale)",
        color="",
    )
    + ggplot.theme_minimal()
    + ggplot.theme(
        legend_position="right",
        legend_title=ggplot.element_text(size=9),
        legend_text=ggplot.element_text(size=8),
        axis_line=ggplot.element_line(color="#444444", size=0.6),
        axis_text_x=ggplot.element_text(size=8),
        axis_text_y=ggplot.element_text(size=8),
    )
)
plot.show()

# %%
