import streamlit as st

from dashboard.config import (
    DATA_PATH,
    NAV_SECTIONS,
    SECTION_CAPTURE,
    SECTION_DATA,
    SECTION_DUAL,
    SECTION_ML,
    SECTION_PROFILE,
)
from dashboard.data.loading import load_dataset
from dashboard.ui.pages.capture import render_capture_section
from dashboard.ui.pages.dual import render_dual_section
from dashboard.ui.pages.ml import render_ml_section
from dashboard.ui.pages.preview import render_data_preview_section
from dashboard.ui.pages.profile import render_profile_section
from dashboard.ui.theme import configure_page, inject_styles
from dashboard.ui.widgets import render_page_body, render_page_top


def main() -> None:
    configure_page()
    inject_styles()

    df = load_dataset()

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose section",
        NAV_SECTIONS,
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(DATA_PATH.name)

    render_page_top(section)

    if section == SECTION_DATA:
        render_page_body(lambda: render_data_preview_section(df))
    elif section == SECTION_DUAL:
        render_page_body(lambda: render_dual_section(df))
    elif section == SECTION_PROFILE:
        render_page_body(lambda: render_profile_section(df))
    elif section == SECTION_CAPTURE:
        render_page_body(lambda: render_capture_section(df))
    else:
        render_page_body(lambda: render_ml_section(df))
