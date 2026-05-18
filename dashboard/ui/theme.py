import streamlit as st


APP_CSS = """
    <style>
    /* Do not hide stHeader: the sidebar open/close control lives there when collapsed. */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999;
    }

    [data-testid="stSidebarHeader"] {
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: var(--secondary-background-color);
    }

    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--text-color);
    }

    section[data-testid="stMain"] {
        color: var(--text-color);
    }

    section[data-testid="stMain"] label,
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] .stMarkdown {
        color: var(--text-color);
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
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }

    .page-top .page-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 0;
        line-height: 1.25;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--text-color);
    }

    .chart-note {
        color: var(--text-color);
        opacity: 0.72;
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
