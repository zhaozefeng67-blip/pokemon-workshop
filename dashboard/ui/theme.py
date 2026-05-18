import streamlit as st


APP_CSS = """
    <style>
    /* Streamlit default top bar (Deploy ⋮) — not part of our layout */
    header[data-testid="stHeader"] {
        display: none;
        height: 0;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    .stApp {
        background: #f7f7f4;
        color: #252525;
    }
    [data-testid="stSidebar"] {
        background: #efeee9;
        border-right: 1px solid #dedbd1;
    }
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }
    section.main > div {
        padding-top: 0.5rem;
    }
    .page-top {
        margin: 0 0 0.85rem 0;
        padding: 0 0 0.7rem 0;
        border-bottom: 1px solid #dedbd1;
    }
    .page-top .page-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #202020;
        margin: 0;
        line-height: 1.25;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: #202020;
    }
    .chart-note {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }
    div[data-testid="stPlotlyChart"] {
        min-height: 420px;
    }
    </style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title="Pokémon Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
