"""
Plotly visualization helpers for the LTP model.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = px.colors.qualitative.Set2


def plot_gdp_trajectories(
    projections: pd.DataFrame,
    iso3_list: list[str],
    title: str = "Projected GDP of Largest Economies",
) -> go.Figure:
    """Plot GDP trajectories for selected countries (reproduces Figure 2)."""
    fig = go.Figure()

    for i, iso3 in enumerate(iso3_list):
        country = projections[projections["iso3"] == iso3].sort_values("year")
        if country.empty:
            continue

        name = country["country_name"].iloc[0]
        fig.add_trace(go.Scatter(
            x=country["year"],
            y=country["gdp_billions"] / 1e3,  # trillions
            mode="lines",
            name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="GDP (trillion USD, 2012 PPP)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_per_capita_comparison(
    projections: pd.DataFrame,
    iso3_list: list[str],
    year: int = 2050,
    title: str = "Per Capita GDP Comparison",
) -> go.Figure:
    """Bar chart of per capita GDP for selected countries."""
    year_data = projections[
        (projections["year"] == year) & projections["iso3"].isin(iso3_list)
    ].sort_values("gdp_per_capita", ascending=True)

    fig = go.Figure(go.Bar(
        x=year_data["gdp_per_capita"] / 1e3,  # thousands
        y=year_data["country_name"],
        orientation="h",
        marker_color=COLORS[0],
    ))

    fig.update_layout(
        title=f"{title} ({year})",
        xaxis_title="GDP per capita (thousand USD, 2012 PPP)",
        template="plotly_white",
    )
    return fig


def plot_kernel_regression(
    ss_gci: np.ndarray,
    ss_phi: np.ndarray,
    gci_grid: np.ndarray,
    phi_curve: np.ndarray,
    ss_labels: list[str] | None = None,
    title: str = "Kernel Estimate: GCI → Steady-State Relative Productivity",
) -> go.Figure:
    """Plot kernel regression curve with scatter of steady-state countries."""
    fig = go.Figure()

    # Scatter of steady-state countries
    fig.add_trace(go.Scatter(
        x=ss_gci,
        y=ss_phi,
        mode="markers",
        name="Steady-state countries",
        marker=dict(size=8, color=COLORS[1], opacity=0.7),
        text=ss_labels,
        hovertemplate="%{text}<br>GCI: %{x:.2f}<br>φ: %{y:.3f}<extra></extra>",
    ))

    # Kernel curve
    fig.add_trace(go.Scatter(
        x=gci_grid,
        y=phi_curve,
        mode="lines",
        name="Kernel estimate",
        line=dict(color=COLORS[0], width=3),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Global Competitiveness Index (GCI)",
        yaxis_title="Relative Labour Productivity (US = 1.0)",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_bandwidth_comparison(
    ss_gci: np.ndarray,
    ss_phi: np.ndarray,
    gci_grid: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str = "Effect of Bandwidth on Kernel Estimate",
) -> go.Figure:
    """Plot multiple kernel curves with different bandwidths."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ss_gci, y=ss_phi, mode="markers",
        name="Steady-state countries",
        marker=dict(size=6, color="gray", opacity=0.5),
    ))

    for i, (label, curve) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=gci_grid, y=curve, mode="lines",
            name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="GCI",
        yaxis_title="Relative Productivity (US = 1.0)",
        yaxis_type="log",
        template="plotly_white",
    )
    return fig


def plot_convergence_path(
    sim: dict[str, np.ndarray],
    country_name: str = "Country",
    title: str | None = None,
) -> go.Figure:
    """Plot simulated convergence path with component decomposition."""
    if title is None:
        title = f"Convergence Path: {country_name}"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Relative Productivity Over Time", "Growth Rate Decomposition"),
        vertical_spacing=0.15,
    )

    # Top: relative productivity path
    fig.add_trace(go.Scatter(
        x=sim["years"],
        y=sim["relative_productivity"] * 100,
        mode="lines",
        name="Relative productivity",
        line=dict(color=COLORS[0], width=2.5),
    ), row=1, col=1)

    # Steady-state target line
    target = sim["relative_productivity"][-1] * 100
    fig.add_hline(
        y=target, line_dash="dash", line_color="gray",
        annotation_text=f"Target φ ≈ {target:.0f}%",
        row=1, col=1,
    )

    # Bottom: growth decomposition
    years = sim["years"][:-1]
    fig.add_trace(go.Scatter(
        x=years, y=sim["momentum_component"] * 100,
        mode="lines", name="Momentum (γ × lag)",
        line=dict(color=COLORS[1]),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=years, y=sim["frontier_component"] * 100,
        mode="lines", name="Frontier ((1-γ) × US)",
        line=dict(color=COLORS[2]),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=years, y=sim["catchup_component"] * 100,
        mode="lines", name="Catch-up (β × gap)",
        line=dict(color=COLORS[3]),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=years, y=sim["annual_growth"] * 100,
        mode="lines", name="Total growth",
        line=dict(color=COLORS[0], width=2.5, dash="dash"),
    ), row=2, col=1)

    fig.update_yaxes(title_text="% of US level", row=1, col=1)
    fig.update_yaxes(title_text="Annual growth rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Years", row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
        showlegend=True,
    )
    return fig


def plot_regional_shares(
    shares_df: pd.DataFrame,
    title: str = "Share of Global GDP by Region",
) -> go.Figure:
    """Stacked area chart of regional GDP shares."""
    fig = go.Figure()

    regions = shares_df["region"].unique()
    for i, region in enumerate(regions):
        region_data = shares_df[shares_df["region"] == region].sort_values("year")
        fig.add_trace(go.Scatter(
            x=region_data["year"],
            y=region_data["share_pct"],
            mode="lines",
            name=region,
            stackgroup="one",
            line=dict(color=COLORS[i % len(COLORS)]),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Share of Global GDP (%)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_growth_decomposition_table(
    decomp_df: pd.DataFrame,
    country_name: str,
) -> go.Figure:
    """Display growth decomposition as a formatted table."""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Period", "GDP Growth (%)", "Productivity (%)", "Workforce (%)"],
            fill_color=COLORS[0],
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=[
                decomp_df["period"],
                decomp_df["gdp_growth"].round(1),
                decomp_df["productivity_growth"].round(1),
                decomp_df["workforce_growth"].round(1),
            ],
            fill_color="white",
            align="center",
            font=dict(size=12),
        ),
    )])

    fig.update_layout(
        title=f"Decade-Average Growth Rates: {country_name}",
        template="plotly_white",
        height=300,
    )
    return fig


def plot_gdp_decomposition(
    years: np.ndarray,
    gdp: np.ndarray,
    productivity: np.ndarray,
    workforce: np.ndarray,
    country_name: str = "Country",
) -> go.Figure:
    """Plot GDP level with productivity and workforce index overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Normalize to base year = 100
    base_prod = productivity[0]
    base_wf = workforce[0]
    base_gdp = gdp[0]

    fig.add_trace(go.Scatter(
        x=years, y=gdp / base_gdp * 100,
        name="GDP", line=dict(color=COLORS[0], width=2.5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=years, y=productivity / base_prod * 100,
        name="Labour Productivity", line=dict(color=COLORS[1], width=2, dash="dash"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=years, y=workforce / base_wf * 100,
        name="Working-Age Population", line=dict(color=COLORS[2], width=2, dash="dot"),
    ), secondary_y=False)

    fig.update_layout(
        title=f"GDP Decomposition: {country_name} (base year = 100)",
        xaxis_title="Year",
        yaxis_title="Index (base year = 100)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_scenario_comparison(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    iso3: str,
    metric: str = "gdp_per_capita",
    title: str | None = None,
) -> go.Figure:
    """Compare baseline and scenario projections for a country."""
    base = baseline[baseline["iso3"] == iso3].sort_values("year")
    scen = scenario[scenario["iso3"] == iso3].sort_values("year")
    name = base["country_name"].iloc[0] if not base.empty else iso3

    if title is None:
        title = f"Scenario Comparison: {name}"

    divisor = 1e3 if metric == "gdp_per_capita" else 1.0
    ylabel = "GDP per capita (thousand USD)" if metric == "gdp_per_capita" else metric

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["year"], y=base[metric] / divisor,
        name="Baseline", line=dict(color=COLORS[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=scen["year"], y=scen[metric] / divisor,
        name="Reform scenario", line=dict(color=COLORS[3], width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    return fig
