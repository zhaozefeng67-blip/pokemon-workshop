from typing import Optional

import plotly.graph_objects as go
import streamlit as st


def _is_dark_theme() -> bool:
    theme = getattr(st.context, "theme", None)
    if theme is None:
        return False
    theme_type = theme.get("type") if isinstance(theme, dict) else getattr(theme, "type", None)
    return theme_type == "dark"


def clean_plotly_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    dark = _is_dark_theme()
    fig.update_layout(
        template="plotly_dark" if dark else "simple_white",
        font=dict(family="Arial, sans-serif", size=13, color="#e8e8e8" if dark else "#2b2b2b"),
        title_font=dict(size=18, color="#fafafa" if dark else "#1f1f1f"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=62, b=38),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig
