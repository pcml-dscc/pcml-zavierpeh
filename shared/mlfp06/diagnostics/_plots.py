# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Shared Plotly theme and small chart helpers.

Matches the M5 Doctor's Bag visual style (clean ``plotly_white`` template,
``steelblue`` / ``firebrick`` / ``orange`` palette) so M5 and M6 dashboards
sit side-by-side in the capstone deliverable.
"""
from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

__all__ = [
    "TEMPLATE",
    "PRIMARY",
    "WARN",
    "ACCENT",
    "MUTED",
    "PALETTE",
    "empty_figure",
    "style",
    "color_for",
    "bar_with_threshold",
]

TEMPLATE = "plotly_white"
PRIMARY = "steelblue"
WARN = "firebrick"
ACCENT = "orange"
MUTED = "lightgray"

PALETTE: tuple[str, ...] = (
    "steelblue",
    "firebrick",
    "seagreen",
    "darkorange",
    "mediumpurple",
    "teal",
    "goldenrod",
    "slategray",
)


def empty_figure(title: str, note: str = "no data") -> go.Figure:
    """Return a blank Plotly figure with a centred annotation.

    Used when a lens is asked to plot but has no data — never ``None``
    and never raises; a visible "no data" cue beats a blank screen.
    """
    fig = go.Figure()
    fig.update_layout(
        title=f"{title} — {note}",
        template=TEMPLATE,
        annotations=[
            dict(
                text=note,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color=MUTED),
            )
        ],
    )
    return fig


def style(fig: go.Figure, title: str, *, x: str = "", y: str = "") -> go.Figure:
    """Apply the shared MLFP observatory style to a figure in-place."""
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        xaxis_title=x,
        yaxis_title=y,
        hovermode="x unified",
    )
    return fig


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def bar_with_threshold(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    threshold: float | None = None,
    title: str,
    y_label: str = "",
    above_color: str = WARN,
    below_color: str = PRIMARY,
) -> go.Figure:
    """Bar chart where bars above the threshold are flagged in ``above_color``."""
    if threshold is None:
        colors = [below_color] * len(values)
    else:
        colors = [above_color if v > threshold else below_color for v in values]
    fig = go.Figure(go.Bar(x=list(categories), y=list(values), marker_color=colors))
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(color=ACCENT, dash="dash"),
            annotation_text=f"threshold={threshold:g}",
        )
    style(fig, title, x="", y=y_label)
    return fig
