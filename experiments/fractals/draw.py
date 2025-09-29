# %% Imports & constants
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import matplotlib.pyplot as plt

Point = Tuple[float, float]
Path = Tuple[int, ...]

BACKGROUND_COLOR = "#2f2f2f"
TREE_BASE_COLOR = "#4d78ae"
TREE_BASE_ALPHA = 0.55
ACCENT_COLOR = "#f97316"
ACCENT_ALPHA = 0.95
NODE_COLOR = "#f4b300"
NODE_EDGE = "#1f2937"

RNG_SEED = 20240205
LOCAL_NODE_PATH: Path = (3, 1)


@dataclass
class Branch:
    path: Path
    level: int
    start: Point
    end: Point


@dataclass
class TreeGeometry:
    branches: Dict[Path, Branch]
    children: Dict[Path, List[Path]]
    max_depth: int


def is_prefix(candidate: Path, reference: Path) -> bool:
    if len(candidate) > len(reference):
        return False
    return reference[: len(candidate)] == candidate


def build_tree(
    depth: int,
    base_length: float,
    initial_branch_count: int,
    branch_overrides: Mapping[Path, int],
    rng_seed: int,
) -> TreeGeometry:
    rng = random.Random(rng_seed)
    branches: Dict[Path, Branch] = {}
    children: MutableMapping[Path, List[Path]] = {}

    def choose_branch_count(path: Path, remaining_depth: int) -> int:
        override = branch_overrides.get(path)
        if override is not None:
            return override
        if remaining_depth >= 4:
            base = 3
            if rng.random() < 0.4:
                base += 1
        else:
            base = 2
            if rng.random() < 0.4:
                base -= 1
        return max(2, min(base, 4))

    def grow(path: Path, start: Point, angle: float, remaining_depth: int, length: float) -> None:
        if remaining_depth == 0:
            return

        direction = (math.cos(angle), math.sin(angle))
        level = len(path)

        child_specs: List[Tuple[float, float, float]] = []  # (sprout_position, child_angle, child_length)
        if remaining_depth > 1:
            branch_count = choose_branch_count(path, remaining_depth)
            spread = math.radians(28 + 9 * rng.random())
            offsets = []
            for idx in range(branch_count):
                center = idx - (branch_count - 1) / 2
                offsets.append(center + (rng.random() - 0.5) * 0.85)
            offsets.sort()

            for offset in offsets:
                sprout_position = 0.55 + 0.38 * rng.random()
                angle_jitter = (rng.random() - 0.5) * math.radians(7)
                child_angle = angle + spread * offset + angle_jitter
                child_length = length * (0.58 + 0.16 * rng.random())
                child_specs.append((sprout_position, child_angle, child_length))

        effective_length = length
        if child_specs:
            max_position = max(spec[0] for spec in child_specs)
            effective_length = length * max_position

        end_point = (
            start[0] + direction[0] * effective_length,
            start[1] + direction[1] * effective_length,
        )

        branches[path] = Branch(path=path, level=level, start=start, end=end_point)
        children[path] = []

        if remaining_depth == 1:
            return

        for child_index, (sprout_position, child_angle, child_length) in enumerate(child_specs):
            child_path = path + (child_index,)
            sprout = (
                start[0] + direction[0] * length * sprout_position,
                start[1] + direction[1] * length * sprout_position,
            )
            children[path].append(child_path)
            grow(child_path, sprout, child_angle, remaining_depth - 1, child_length)

    origin = (0.0, 0.0)
    children[()] = []

    for idx in range(initial_branch_count):
        angle_variation = (rng.random() - 0.5) * math.radians(12)
        angle = math.tau * idx / initial_branch_count + math.pi / 2 + angle_variation
        length = base_length * (0.95 + 0.25 * rng.random())
        path = (idx,)
        children[()].append(path)
        grow(path, origin, angle, depth, length)

    return TreeGeometry(branches=branches, children=dict(children), max_depth=depth)


def collect_visible_paths(tree: TreeGeometry, restrictions: Mapping[Path, Sequence[int]]) -> Set[Path]:
    visible: Set[Path] = set()
    stack: List[Path] = [()]
    while stack:
        node_path = stack.pop()
        for idx, child_path in enumerate(tree.children.get(node_path, [])):
            allowed = restrictions.get(node_path)
            if allowed is not None and idx not in allowed:
                continue
            if child_path in visible:
                continue
            visible.add(child_path)
            stack.append(child_path)
    return visible


def configure_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)


def render_view(
    ax: plt.Axes,
    tree: TreeGeometry,
    visible_paths: Iterable[Path],
    emphasis_path: Path,
) -> None:
    visible_set = set(visible_paths)
    ordered_paths = sorted(visible_set, key=len)

    for path in ordered_paths:
        branch = tree.branches[path]
        if is_prefix(path, emphasis_path) or is_prefix(emphasis_path, path):
            color = ACCENT_COLOR
            alpha = ACCENT_ALPHA
        else:
            color = TREE_BASE_COLOR
            alpha = TREE_BASE_ALPHA
        width = 4.6 * (0.72 ** max(0, branch.level - 1))
        ax.plot(
            (branch.start[0], branch.end[0]),
            (branch.start[1], branch.end[1]),
            color=color,
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
        )

    if emphasis_path in tree.branches:
        node_point = tree.branches[emphasis_path].start
        ax.scatter(
            [node_point[0]],
            [node_point[1]],
            s=28,
            color=NODE_COLOR,
            edgecolors=NODE_EDGE,
            linewidths=0.6,
            zorder=5,
        )


# %% Bias pairs (functions)
def draw_bias_pairs(*, dpi: int = 300, output: str = "fractals_bias_pairs.png", show: bool = True) -> None:
    depth = 5
    initial_branch_count = 7
    branch_overrides = {LOCAL_NODE_PATH: 8}

    tree = build_tree(
        depth=depth,
        base_length=0.95,
        initial_branch_count=initial_branch_count,
        branch_overrides=branch_overrides,
        rng_seed=RNG_SEED,
    )

    scenarios = [
        (
            "Objective: 7 branches observed",
            {LOCAL_NODE_PATH: list(range(7))},
            "Individual: perceives 4",
            {LOCAL_NODE_PATH: [0, 1, 2, 3]},
        ),
        (
            "Individual: perceives 4",
            {LOCAL_NODE_PATH: [0, 1, 2, 3]},
            "Perceived average: 3",
            {LOCAL_NODE_PATH: [0, 1, 2]},
        ),
        (
            "Individual: perceives 7",
            {LOCAL_NODE_PATH: [0, 1, 2, 3, 4, 5, 6]},
            "Perceived average: 8",
            {LOCAL_NODE_PATH: list(range(8))},
        ),
    ]

    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=2,
        figsize=(8.0, 10.0),
        facecolor=BACKGROUND_COLOR,
    )

    row_labels = [
        "Dunning-Kruger effect",
        "Lake Wobegon effect",
        "Impostor syndrome",
    ]

    for row_index, (left_title, left_restrictions, right_title, right_restrictions) in enumerate(scenarios):
        for col_index, (title, restrictions) in enumerate(
            ((left_title, left_restrictions), (right_title, right_restrictions))
        ):
            ax = axes[row_index, col_index]
            configure_axes(ax)
            visible = collect_visible_paths(tree, restrictions)
            render_view(ax, tree, visible, LOCAL_NODE_PATH)
            ax.set_title(title, color="white", fontsize=10, pad=10)
        axes[row_index, 0].text(
            -0.12,
            0.5,
            row_labels[row_index],
            color="#cbd5f5",
            fontsize=10,
            ha="right",
            va="center",
            rotation=90,
            transform=axes[row_index, 0].transAxes,
        )

    plt.tight_layout(h_pad=2.5, w_pad=1.8)
    plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


# %% Knowledge tree (functions)

Point2D = Tuple[float, float]


def draw_branch_knowledge(
    ax: plt.Axes,
    start: Point2D,
    angle: float,
    depth: int,
    max_depth: int,
    length: float,
    highlight_path: Sequence[int] | None,
    highlight_nodes: List[Point2D],
    rng: random.Random,
) -> None:
    if depth == 0:
        return

    direction = (math.cos(angle), math.sin(angle))

    level = max_depth - depth
    taper_factor = 0.6**level
    base_width = 5 * taper_factor

    is_highlight = highlight_path is not None
    line_width = base_width * (1.35 if is_highlight else 0.9)
    color = ACCENT_COLOR if is_highlight else TREE_BASE_COLOR

    if depth == 1:
        end_point = (start[0] + direction[0] * length, start[1] + direction[1] * length)
        ax.plot(
            (start[0], end_point[0]),
            (start[1], end_point[1]),
            color=color,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=2 if not is_highlight else 4,
        )
        if is_highlight:
            highlight_nodes.append(end_point)
        return

    branch_count = 3 if depth > 2 else 2
    if depth > 3 and rng.random() < 0.4:
        branch_count += 1
    if depth <= 2 and rng.random() < 0.4:
        branch_count -= 1
    branch_count = max(2, min(branch_count, 4))

    spread = math.radians(30 + 8 * rng.random())
    offsets: List[float] = []
    for idx in range(branch_count):
        center = idx - (branch_count - 1) / 2
        offsets.append(center + (rng.random() - 0.5) * 0.9)
    offsets.sort()

    target_index: int | None = None
    if highlight_path:
        target_index = min(highlight_path[0], branch_count - 1)

    child_specs: List[Tuple[int, float, float, float]] = []
    max_sprout_position = 0.0
    highlight_sprout_position: float | None = None

    for child_index, offset in enumerate(offsets):
        sprout_position = 0.55 + 0.38 * rng.random()
        max_sprout_position = max(max_sprout_position, sprout_position)

        angle_jitter = (rng.random() - 0.5) * math.radians(6)
        child_angle = angle + spread * offset + angle_jitter
        child_length = length * (0.58 + 0.16 * rng.random())

        child_specs.append((child_index, sprout_position, child_angle, child_length))

        if target_index is not None and child_index == target_index:
            highlight_sprout_position = sprout_position

    effective_length = length * max_sprout_position if max_sprout_position > 0 else length
    end_point = (start[0] + direction[0] * effective_length, start[1] + direction[1] * effective_length)

    highlight_tip = end_point
    if highlight_path and highlight_sprout_position is not None:
        highlight_tip = (
            start[0] + direction[0] * length * highlight_sprout_position,
            start[1] + direction[1] * length * highlight_sprout_position,
        )

    ax.plot(
        (start[0], end_point[0]),
        (start[1], end_point[1]),
        color=color,
        linewidth=line_width,
        solid_capstyle="round",
        zorder=2 if not is_highlight else 4,
    )

    if is_highlight:
        highlight_nodes.append(highlight_tip)

    for child_index, sprout_position, child_angle, child_length in child_specs:
        sprout = (
            start[0] + direction[0] * length * sprout_position,
            start[1] + direction[1] * length * sprout_position,
        )

        child_highlight: Sequence[int] | None = None
        if target_index is not None and child_index == target_index:
            child_highlight = highlight_path[1:]

        draw_branch_knowledge(
            ax,
            sprout,
            child_angle,
            depth - 1,
            max_depth,
            child_length,
            child_highlight,
            highlight_nodes,
            rng,
        )


def draw_fractal_knowledge_tree(
    ax: plt.Axes,
    origin: Point2D,
    depth: int,
    base_length: float,
    initial_branch_count: int,
    highlight_route: Sequence[int],
    *,
    rng: random.Random | None = None,
) -> List[Point2D]:
    rng = rng or random.Random(RNG_SEED)
    highlight_nodes: List[Point2D] = [origin]

    if not highlight_route:
        raise ValueError("highlight_route must contain at least one index")

    root_target = min(highlight_route[0], initial_branch_count - 1)

    for idx in range(initial_branch_count):
        angle_variation = (rng.random() - 0.5) * math.radians(10)
        initial_angle = math.tau * idx / initial_branch_count + math.pi / 2 + angle_variation
        branch_length = base_length * (0.95 + 0.25 * rng.random())

        highlight_path: Sequence[int] | None = None
        if idx == root_target:
            highlight_path = highlight_route[1:]

        draw_branch_knowledge(
            ax,
            origin,
            initial_angle,
            depth,
            depth,
            branch_length,
            highlight_path,
            highlight_nodes,
            rng,
        )

    return highlight_nodes


def annotate_path_knowledge(ax: plt.Axes, nodes: Sequence[Point2D], *, label: bool = True) -> None:
    xs, ys = zip(*nodes)
    sizes = [80] + [55] * (len(nodes) - 2) + [90]

    ax.scatter(
        xs,
        ys,
        s=sizes,
        color=NODE_COLOR,
        edgecolors=NODE_EDGE,
        linewidths=0.6,
        zorder=5,
    )

    if label and nodes:
        limit = nodes[-1]
        ax.text(
            limit[0] + 0.08,
            limit[1] + 0.08,
            "Limit node\n(PhD research)",
            color=NODE_COLOR,
            fontsize=10,
            ha="left",
            va="top",
            zorder=6,
        )


def draw_knowledge_single(*, dpi: int = 300, output: str = "fractals_knowledge_tree.png", show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor=BACKGROUND_COLOR)
    configure_axes(ax)

    depth = 5
    initial_branch_count = 11
    highlight_route = (4, 2, 1, 0, 1)

    highlight_nodes = draw_fractal_knowledge_tree(
        ax,
        origin=(0.0, 0.0),
        depth=depth,
        base_length=0.95,
        initial_branch_count=initial_branch_count,
        highlight_route=highlight_route,
    )

    annotate_path_knowledge(ax, highlight_nodes, label=True)

    ax.set_title(
        "Research at the limits of human knowledge",
        color="white",
        fontsize=12,
        pad=16,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def _random_route(depth: int, root_max: int, *, rng: random.Random) -> Sequence[int]:
    return [rng.randrange(max(1, root_max))] + [rng.randrange(4) for _ in range(depth - 1)]


# %%
draw_knowledge_single()

# %%
draw_bias_pairs()

# %%
