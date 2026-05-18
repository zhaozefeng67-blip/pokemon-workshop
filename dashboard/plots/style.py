from typing import Optional

import plotly.graph_objects as go

def clean_plotly_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial, sans-serif", size=13, color="#2b2b2b"),
        title_font=dict(size=18, color="#1f1f1f"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=62, b=38),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig
