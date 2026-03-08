"""
Page 1: What Drives GDP?

Builds intuition from first principles — the GDP identity
and how demographics vs productivity drive economic growth.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ltp.data import load_gdp, load_population, compute_labour_productivity

st.set_page_config(page_title="What Drives GDP?", page_icon="📊", layout="wide")
st.title("1. What Drives GDP?")

# --- Narrative ---
st.markdown("""
### From Malthus to Modern Growth

Before the industrial revolution, a country's GDP was roughly proportional to its population.
Thomas Malthus (1798) argued that any economic surplus would be consumed by population growth,
keeping per capita income near subsistence.

> **Pre-industrial:** GDP ≈ Population × (subsistence level)

The industrial revolution changed everything. Capital accumulation, technology, and
specialisation allowed workers to become far more productive. GDP now depends on
**how many workers** there are AND **how productive** each worker is:
""")

st.info("""
**The GDP Identity:**

**GDP = Working-Age Population × Labour Productivity**

**GDP per capita = (Working-Age Pop / Total Pop) × Labour Productivity**
""")

st.markdown("""
This is the foundation of the Hubbard & Sharma model. To project future GDP,
we need to project two things separately:

1. **Demographics** (working-age population) — from UN population projections
2. **Labour productivity** — from the convergence model (Pages 2-4)

---
### Explore the Data
""")

# --- Load data ---
try:
    gdp_df = load_gdp()
    pop_df = load_population()
    prod_df = compute_labour_productivity(gdp_df, pop_df)

    countries = sorted(prod_df["country_name"].unique())

    # Country selector
    default_countries = ["United States", "China", "India", "Japan", "Germany", "Indonesia", "Brazil"]
    available_defaults = [c for c in default_countries if c in countries]

    selected = st.multiselect(
        "Select countries to compare:",
        countries,
        default=available_defaults[:4],
    )

    if selected:
        tab1, tab2, tab3 = st.tabs([
            "GDP Levels", "Productivity vs Workforce", "Growth Decomposition"
        ])

        with tab1:
            st.subheader("GDP Over Time")
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, country in enumerate(selected):
                c_data = prod_df[prod_df["country_name"] == country].sort_values("year")
                fig.add_trace(go.Scatter(
                    x=c_data["year"],
                    y=c_data["gdp_billions_2012ppp"],
                    mode="lines",
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=2.5),
                ))
            fig.update_layout(
                yaxis_title="GDP (billion USD, 2012 PPP)",
                template="plotly_white",
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Productivity vs Working-Age Population (indexed to earliest year = 100)")
            col1, col2 = st.columns(2)

            for i, country in enumerate(selected):
                c_data = prod_df[prod_df["country_name"] == country].sort_values("year")
                if len(c_data) < 2:
                    continue

                base_prod = c_data["labour_productivity"].iloc[0]
                base_wap = c_data["working_age_pop_thousands"].iloc[0]

                fig = make_subplots(specs=[[{"secondary_y": False}]])
                fig.add_trace(go.Scatter(
                    x=c_data["year"],
                    y=c_data["labour_productivity"] / base_prod * 100,
                    name="Labour Productivity",
                    line=dict(color=colors[0], width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=c_data["year"],
                    y=c_data["working_age_pop_thousands"] / base_wap * 100,
                    name="Working-Age Pop",
                    line=dict(color=colors[1], width=2, dash="dash"),
                ))
                fig.update_layout(
                    title=country,
                    yaxis_title="Index (base = 100)",
                    template="plotly_white",
                    height=350,
                    showlegend=True,
                    legend=dict(x=0.02, y=0.98),
                )

                target = col1 if i % 2 == 0 else col2
                target.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("GDP Growth Decomposition")
            st.markdown("""
            GDP growth ≈ Productivity growth + Workforce growth

            This decomposition shows how much of each country's growth came from
            more productive workers versus more workers.
            """)

            for country in selected:
                c_data = prod_df[prod_df["country_name"] == country].sort_values("year")
                if len(c_data) < 3:
                    continue

                prod_vals = c_data["labour_productivity"].values
                wap_vals = c_data["working_age_pop_thousands"].values
                years = c_data["year"].values

                prod_growth = np.diff(np.log(prod_vals)) * 100
                wap_growth = np.diff(np.log(wap_vals)) * 100
                gdp_growth = prod_growth + wap_growth

                # 5-year moving average for smoother visualization
                window = min(5, len(prod_growth))
                if window > 1:
                    kernel = np.ones(window) / window
                    prod_growth_smooth = np.convolve(prod_growth, kernel, mode="valid")
                    wap_growth_smooth = np.convolve(wap_growth, kernel, mode="valid")
                    gdp_growth_smooth = np.convolve(gdp_growth, kernel, mode="valid")
                    years_smooth = years[window:]
                else:
                    prod_growth_smooth = prod_growth
                    wap_growth_smooth = wap_growth
                    gdp_growth_smooth = gdp_growth
                    years_smooth = years[1:]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=years_smooth, y=prod_growth_smooth,
                    name="Productivity growth",
                    marker_color=colors[0],
                ))
                fig.add_trace(go.Bar(
                    x=years_smooth, y=wap_growth_smooth,
                    name="Workforce growth",
                    marker_color=colors[1],
                ))
                fig.add_trace(go.Scatter(
                    x=years_smooth, y=gdp_growth_smooth,
                    name="Total GDP growth",
                    line=dict(color="black", width=2),
                ))
                fig.update_layout(
                    title=f"{country} — GDP Growth Decomposition ({window}-yr moving avg)",
                    yaxis_title="Annual growth rate (%)",
                    barmode="stack",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ---
    ### Key Takeaway

    To project future GDP, we need:
    1. **Population projections** — available from the UN (relatively reliable)
    2. **Productivity projections** — this is the hard part!

    The next pages explain how the Hubbard & Sharma model projects labour productivity
    using *conditional convergence* — the idea that countries with good institutions
    can catch up to the productivity frontier.

    """)
    st.page_link("pages/2_Convergence_Intuition.py", label="→ Continue to Page 2: Convergence Intuition", icon="📖")

except FileNotFoundError:
    st.error(
        "Data files not found. Please ensure CSV files are in the `data/` directory. "
        "See data/README.md for details."
    )
