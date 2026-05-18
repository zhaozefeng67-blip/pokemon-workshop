import html
import re
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import CARD_HEIGHT_HALF

def searchable_dataframe(df: pd.DataFrame, key_prefix: str, default_rows: int = 25) -> pd.DataFrame:
    query = st.text_input("Search table", key=f"{key_prefix}_search", placeholder="Search by name, type, or any visible value")
    visible = df.copy()
    if query:
        mask = visible.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        visible = visible.loc[mask]
    st.dataframe(visible.head(default_rows), use_container_width=True, hide_index=True)
    return visible


def format_filter_summary(label: str, selected: list[str]) -> str:
    if not selected:
        return f"{label} · none"
    if len(selected) <= 3:
        return f"{label} · {', '.join(selected)}"
    return f"{label} · {len(selected)} selected"


def multiselect_popover(
    label: str,
    options: list[str],
    key: str,
    default: Optional[list[str]] = None,
) -> list[str]:
    options = [str(option) for option in options]
    if key not in st.session_state:
        st.session_state[key] = list(default if default is not None else options)

    with st.popover(format_filter_summary(label, st.session_state.get(key, []))):
        action_all, action_clear = st.columns(2)
        if action_all.button("Select all", key=f"{key}_all", use_container_width=True):
            st.session_state[key] = options
            st.rerun()
        if action_clear.button("Clear", key=f"{key}_clear", use_container_width=True):
            st.session_state[key] = []
            st.rerun()
        st.multiselect(
            "Options",
            options,
            key=key,
            label_visibility="collapsed",
        )

    selected = st.session_state.get(key, [])
    return selected if selected else options


def chart_panel_title(fig: go.Figure, fallback: str = "Chart") -> str:
    layout_title = fig.layout.title
    text = ""
    if layout_title is not None:
        text = layout_title.text if hasattr(layout_title, "text") else str(layout_title)
    if not text:
        return fallback
    text = re.sub(r"<[^>]+>", "", text)
    return text.split("<br>")[0].strip() or fallback


def plot_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})


def render_chart_panel(
    fig: go.Figure,
    title: Optional[str] = None,
    *,
    expanded: bool = False,
    caption: str = "",
) -> None:
    with st.expander(title or chart_panel_title(fig), expanded=expanded):
        plot_chart(fig)
        if caption:
            st.caption(caption)


def render_chart_expander(title: str, *, expanded: bool = False):
    return st.expander(title, expanded=expanded)


def render_chart_card(
    fig: go.Figure,
    caption: str = "",
    *,
    card_height: int = CARD_HEIGHT_HALF,
    title: Optional[str] = None,
    expanded: bool = False,
) -> None:
    del card_height
    render_chart_panel(fig, title=title, expanded=expanded, caption=caption)


def render_chart(fig: go.Figure, explanation: str = "") -> None:
    render_chart_panel(fig, caption=explanation)


def render_page_top(title: str) -> None:
    st.markdown(
        f'<div class="page-top"><p class="page-title">{html.escape(title)}</p></div>',
        unsafe_allow_html=True,
    )


def render_page_body(render_fn: Callable[[], None]) -> None:
    with st.container():
        render_fn()
