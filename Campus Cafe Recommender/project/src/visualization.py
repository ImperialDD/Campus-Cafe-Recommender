"""Plotly visualizations for the PCA taste map."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.pca_analysis import PCAResult


def plot_taste_map(
    pca: PCAResult,
    student_ids: list[str],
    new_user_coords: np.ndarray | None = None,
    neighbor_ids: list[str] | None = None,
) -> go.Figure:
    """2D PCA taste map with a clear four-quadrant coordinate plane.

    - Existing students as blue circle markers.
    - Nearest neighbors highlighted in amber.
    - New user as a distinct red star (if provided).
    - Prominent vertical (PC1=0) and horizontal (PC2=0) reference lines.
    - Quadrant labels (I–IV) in the corners.
    """
    fig = go.Figure()

    neighbor_set = set(neighbor_ids or [])
    xs = pca.student_coords[:, 0]
    ys = pca.student_coords[:, 1]

    # Axis range with padding so quadrant labels sit clear of the data.
    pad = 0.18
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_range = x_max - x_min
    y_range = y_max - y_min
    x0 = x_min - pad * x_range
    x1 = x_max + pad * x_range
    y0 = y_min - pad * y_range
    y1 = y_max + pad * y_range

    # --- Quadrant label annotations (corners) ---
    qx = x1 - 0.04 * x_range
    qx_left = x0 + 0.04 * x_range
    qy = y1 - 0.05 * y_range
    qy_bottom = y0 + 0.05 * y_range
    quad_labels = [
        ("Quadrant II", qx_left, qy),        # PC1<0, PC2>0
        ("Quadrant I", qx, qy),              # PC1>0, PC2>0
        ("Quadrant III", qx_left, qy_bottom),# PC1<0, PC2<0
        ("Quadrant IV", qx, qy_bottom),      # PC1>0, PC2<0
    ]
    for label, lx, ly in quad_labels:
        fig.add_annotation(
            x=lx, y=ly, text=label, showarrow=False,
            font=dict(size=12, color="#94a3b8"),
            xanchor="left" if lx > 0 else "left",
            yanchor="top" if ly > 0 else "bottom",
        )

    # --- Quadrant reference lines at PC1=0 and PC2=0 ---
    fig.add_hline(y=0, line=dict(color="#64748b", width=2, dash="dash"),
                  annotation_text="PC2 = 0", annotation_position="top left",
                  annotation_font=dict(size=11, color="#64748b"))
    fig.add_vline(x=0, line=dict(color="#64748b", width=2, dash="dash"),
                  annotation_text="PC1 = 0", annotation_position="top left",
                  annotation_font=dict(size=11, color="#64748b"))

    # --- Student markers ---
    rest_idx = [i for i, sid in enumerate(student_ids) if sid not in neighbor_set]
    nb_idx = [i for i, sid in enumerate(student_ids) if sid in neighbor_set]

    fig.add_trace(
        go.Scatter(
            x=xs[rest_idx],
            y=ys[rest_idx],
            mode="markers+text",
            text=[student_ids[i] for i in rest_idx],
            textposition="top center",
            name="Existing students",
            marker=dict(size=12, color="#2563eb", symbol="circle",
                        line=dict(width=1, color="white")),
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        )
    )

    if nb_idx:
        fig.add_trace(
            go.Scatter(
                x=xs[nb_idx],
                y=ys[nb_idx],
                mode="markers+text",
                text=[student_ids[i] for i in nb_idx],
                textposition="top center",
                name="Nearest neighbors",
                marker=dict(size=14, color="#f59e0b", symbol="circle",
                            line=dict(width=2, color="white")),
                hovertemplate="<b>%{text}</b> (neighbor)<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            )
        )

    if new_user_coords is not None:
        fig.add_trace(
            go.Scatter(
                x=[new_user_coords[0]],
                y=[new_user_coords[1]],
                mode="markers+text",
                text=["You"],
                textposition="top center",
                name="New user (You)",
                marker=dict(size=18, color="#ef4444", symbol="star",
                            line=dict(width=2, color="white")),
                hovertemplate="<b>You</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            )
        )

    pc1_label = f"PC1 ({pca.pc1_variance*100:.1f}% variance)"
    pc2_label = f"PC2 ({pca.pc2_variance*100:.1f}% variance)"

    fig.update_layout(
        title="2D PCA Taste Map — Four Quadrants",
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
        xaxis=dict(range=[x0, x1], zeroline=False),
        yaxis=dict(range=[y0, y1], zeroline=False),
    )
    return fig


def plot_explained_variance(pca: PCAResult) -> go.Figure:
    """Bar chart of explained variance per component."""
    fig = go.Figure(
        go.Bar(
            x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio))],
            y=pca.explained_variance_ratio * 100,
            marker_color="#2563eb",
            text=[f"{v*100:.1f}%" for v in pca.explained_variance_ratio],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Explained Variance by Principal Component",
        xaxis_title="Component",
        yaxis_title="Variance explained (%)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
