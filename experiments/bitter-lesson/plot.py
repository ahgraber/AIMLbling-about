# ruff: NOQA: C408 # allow dict()

# %%
import logging
from pathlib import Path

import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

from aiml.utils import basic_log_config, get_repo_path

# %%
basic_log_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_DIR = get_repo_path(__file__)

# %%
# Create grid of points with adjusted range
_x = np.linspace(-1, 1, 100)  # specialization
_y = np.linspace(-1, 1, 100)  # timeline
X, Y = np.meshgrid(_x, _y)


def gaussian_2d(X, Y, amplitude, x0, y0, sigma_x, sigma_y):  # noqa: N803
    """Represent a 2D gaussian surface."""
    return amplitude * np.exp(-((X - x0) ** 2 / (2 * sigma_x**2) + (Y - y0) ** 2 / (2 * sigma_y**2)))


def exponential_opacity(Z, decay_rate=5):  # noqa: N803
    """Create exponentially decaying opacity values relative to max height."""
    normalized = Z / Z.max()  # Scale relative to this model's maximum
    return np.exp(-decay_rate * (1 - normalized))  # Exponential decay


# %%
# Adjust model parameters for new scale
# fmt: off
#                   perf, time, spec, xwidth, ywidth
models = {  #        (z),  (x),  (y), xwidth, ywidth
    "GPT-3.5":      [0.5, -0.7, 0.0, 1.0, 1.0],
    "BloombergGPT": [0.7, -0.5, 0.0, 0.1, 0.1],  # Narrow spike
    "GPT-4":        [1.0,  0.0, 0.0, 1.2, 1.2],
    "o1-IOI":       [1.2,  0.5, 0.0, 0.3, 0.3],
    "o3":           [1.5,  0.7, 0.0, 1.5, 1.5],
}
# fmt: on

colors = {
    "BloombergGPT": "#228B22",  # Forest Green
    "GPT-3.5": "#20B2AA",  # Light Teal
    "GPT-4": "#008B8B",  # Dark Teal
    "o1-IOI": "#6A5ACD",  # Light Indigo
    "o3": "#4B0082",  # Dark Indigo
}

# Create figure
fig = go.Figure()

# Add reference planes
# XY plane at z=0
fig.add_trace(
    go.Surface(
        x=np.array([[-1, 1], [-1, 1]]),
        y=np.array([[-1, -1], [1, 1]]),
        z=np.zeros((2, 2)),
        showscale=False,
        opacity=0.2,
        colorscale=[[0, "#dddddd"], [1, "#dddddd"]],
        name="z=0 plane",
    )
)

# XZ plane at y=0
fig.add_trace(
    go.Surface(
        x=np.array([[-1, 1], [-1, 1]]),
        y=np.zeros((2, 2)),
        z=np.array([[0, 0], [2, 2]]),
        showscale=False,
        opacity=0.2,
        colorscale=[[0, "#dddddd"], [1, "#dddddd"]],
        name="y=0 plane",
    )
)

# YZ plane at x=0
fig.add_trace(
    go.Surface(
        x=np.zeros((2, 2)),
        y=np.array([[-1, -1], [1, 1]]),
        z=np.array([[0, 2], [0, 2]]),
        showscale=False,
        opacity=0.2,
        colorscale=[[0, "#dddddd"], [1, "#dddddd"]],
        name="x=0 plane",
    )
)


# Add surface for each model with variable opacity
for model, params in models.items():
    color = colors[model]
    Z = gaussian_2d(X, Y, *params)

    # Create exponentially decaying opacity values specific to this model
    opacities = exponential_opacity(Z, decay_rate=7)

    # Add Gaussian surface
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            surfacecolor=opacities,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, color]],
            name=model,
        )
    )

    # Add an invisible scatter point for the legend
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],  # Invisible point
            mode="markers",
            marker=dict(size=10, color=color),
            name=model,  # Adds to legend
        )
    )

    # Add annotation at the model's peak
    fig.add_trace(
        go.Scatter3d(
            x=[params[1]],  # x0
            y=[params[2]],  # y0
            z=[params[0]],  # peak height (amplitude)
            mode="text",
            text=[f"<b>{model}</b>"],  # Bold text
            textposition="top center",
            showlegend=False,
        )
    )
# Update layout with adjusted ranges and aspect ratio
fig.update_layout(
    title_automargin=True,
    title=dict(
        text="AI Model Capability Distributions",
        pad=dict(l=20, r=20, b=10, t=25),
        x=0.5,
        xref="container",
        yref="container",
        xanchor="center",
        yanchor="top",
    ),
    scene=dict(
        xaxis=dict(title="Timeline", range=[-1.1, 1.1], ticktext=["2022", "2025"], tickvals=[-1, 1]),
        yaxis=dict(title="Specificity", range=[-1.1, 1.1], showticklabels=False),
        zaxis=dict(title="Performance", range=[-0.1, 2.21], ticktext=["Low", "High"], tickvals=[0, 2]),
        aspectmode="cube",  # Force cubic display
        camera=dict(eye=dict(x=0.1, y=-2.5, z=0.1)),
    ),
    margin=dict(l=20, r=20, b=20, t=20),
    width=600,
    height=600,
    showlegend=True,
    legend=dict(orientation="h", x=0.5, y=1, xanchor="center", yanchor="top"),
)

fig.show()

# %%
pio.write_json(fig, Path(__file__).parent / "ai_model_distributions.json")


# %%
