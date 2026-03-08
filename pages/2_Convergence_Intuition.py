"""
Page 2: Why Do Some Countries Grow Faster?

Builds intuition for conditional convergence — the core idea
that institutional quality determines a country's productivity potential.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ltp.data import load_gdp, load_population, load_steady_state_countries, load_gci_scores
from ltp.data import compute_labour_productivity

st.set_page_config(page_title="Convergence Intuition", page_icon="🔄", layout="wide")
st.title("2. Why Do Some Countries Grow Faster?")

st.markdown("""
### The Puzzle of Unequal Growth

Not all countries grow at the same rate. Some, like China and India, have sustained
rapid growth for decades. Others, like many in Sub-Saharan Africa, have seen productivity
stagnate at low levels. Why?

The **conditional convergence** framework offers an answer:

1. Each country has a **steady-state** productivity level determined by its institutions
2. Countries **below** their potential grow faster (catch-up growth)
3. Countries **at** their potential grow at the global technology rate (~1.5-2%/year)

This means growth is *conditional* on institutional quality — not all countries
converge to the same level.
""")

try:
    gdp_df = load_gdp()
    pop_df = load_population()
    ss_df = load_steady_state_countries()
    gci_df = load_gci_scores()
    prod_df = compute_labour_productivity(gdp_df, pop_df)

    colors = px.colors.qualitative.Set2

    tab1, tab2, tab3 = st.tabs([
        "Relative Productivity", "Steady State vs Transitioning", "GCI & Institutions"
    ])

    with tab1:
        st.subheader("Labour Productivity Relative to the United States")
        st.markdown("""
        This chart shows each country's labour productivity as a fraction of US levels.
        Notice how some countries maintain a **stable ratio** (steady state) while others
        are **catching up** or **falling behind**.
        """)

        countries = sorted(prod_df["country_name"].unique())
        highlight = st.multiselect(
            "Highlight countries:",
            countries,
            default=["China", "India", "Japan", "Germany", "Brazil", "Indonesia"],
        )

        fig = go.Figure()

        # Background: all countries in light gray
        for iso3 in prod_df["iso3"].unique():
            c = prod_df[prod_df["iso3"] == iso3].sort_values("year")
            if c["country_name"].iloc[0] in highlight:
                continue
            fig.add_trace(go.Scatter(
                x=c["year"], y=c["relative_productivity"],
                mode="lines", line=dict(color="lightgray", width=0.5),
                showlegend=False, hoverinfo="skip",
            ))

        # Highlighted countries
        for i, country in enumerate(highlight):
            c = prod_df[prod_df["country_name"] == country].sort_values("year")
            if c.empty:
                continue
            fig.add_trace(go.Scatter(
                x=c["year"], y=c["relative_productivity"],
                mode="lines", name=country,
                line=dict(color=colors[i % len(colors)], width=2.5),
            ))

        fig.update_layout(
            yaxis_title="Productivity relative to US (US = 1.0)",
            yaxis_type="log",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Steady State vs Transitioning Countries")
        st.markdown("""
        The model classifies countries into two groups:

        - **Steady-state** (55 countries): Their productivity relative to the US has been
          roughly constant over the past two decades. They're already at their institutional
          potential — for better or worse.

        - **Transitioning** (85 countries): They're catching up (or in rare cases, falling)
          toward a potential level determined by their institutional quality.

        A country in the **"middle income trap"** is one whose steady-state productivity
        is well below frontier levels — it's *not* catching up because it's already
        *at* its (low) potential.
        """)

        # Show which countries are in each group
        ss_countries = ss_df[ss_df["is_steady_state"] == True].sort_values(
            "relative_productivity", ascending=False
        )
        trans_countries = ss_df[ss_df["is_steady_state"] == False].sort_values(
            "relative_productivity", ascending=False
        ) if "is_steady_state" in ss_df.columns and (ss_df["is_steady_state"] == False).any() else pd.DataFrame()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Steady-State Countries** (55)")
            st.markdown("These countries' relative productivity is stable.")

            fig_ss = go.Figure(go.Bar(
                x=ss_countries["relative_productivity"],
                y=ss_countries["country_name"],
                orientation="h",
                marker_color=colors[0],
            ))
            fig_ss.update_layout(
                xaxis_title="Relative Productivity (US = 1.0)",
                template="plotly_white",
                height=max(400, len(ss_countries) * 18),
                margin=dict(l=120),
            )
            st.plotly_chart(fig_ss, use_container_width=True)

        with col2:
            st.markdown("**Transitioning Countries**")
            st.markdown("These countries are converging toward their potential.")

            # Show all non-steady-state countries from GCI data
            trans_from_gci = gci_df[~gci_df["iso3"].isin(
                ss_countries["iso3"].values
            )].sort_values("gci_score", ascending=False)

            if not trans_from_gci.empty:
                fig_tr = go.Figure(go.Bar(
                    x=trans_from_gci["gci_score"],
                    y=trans_from_gci["country_name"],
                    orientation="h",
                    marker_color=colors[1],
                ))
                fig_tr.update_layout(
                    xaxis_title="GCI Score (proxy for potential)",
                    template="plotly_white",
                    height=max(400, len(trans_from_gci) * 18),
                    margin=dict(l=120),
                )
                st.plotly_chart(fig_tr, use_container_width=True)

    with tab3:
        st.subheader("Institutional Quality and Productivity")
        st.markdown("""
        The **Global Competitiveness Index (GCI)** from the World Economic Forum
        serves as a proxy for institutional quality. It combines 114 indicators covering:

        - **Basic requirements** (institutions, infrastructure, macro environment, health/education)
        - **Efficiency enhancers** (higher education, goods/labour/financial market efficiency, tech readiness, market size)
        - **Innovation factors** (business sophistication, innovation)

        For steady-state countries, there's a clear positive relationship between
        GCI score and labour productivity relative to the US.
        """)

        # Merge GCI with steady-state data
        ss_with_gci = pd.merge(ss_countries, gci_df[["iso3", "gci_score"]], on="iso3", how="inner")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ss_with_gci["gci_score"],
            y=ss_with_gci["relative_productivity"],
            mode="markers+text",
            text=ss_with_gci["iso3"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=10, color=colors[0], opacity=0.7),
            name="Steady-state countries",
        ))

        fig.update_layout(
            title="GCI Score vs Relative Productivity (Steady-State Countries)",
            xaxis_title="Global Competitiveness Index (GCI)",
            yaxis_title="Relative Productivity (US = 1.0)",
            yaxis_type="log",
            template="plotly_white",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        This relationship is **not linear** — it follows an S-shaped curve. In the next page,
        we'll estimate this curve using **kernel regression** to predict the potential
        productivity of every transitioning country.

        ---
        ### Key Takeaway

        - Institutional quality (GCI) determines a country's **steady-state** productivity level
        - The **gap** between current and potential productivity determines catch-up growth speed
        - Countries with poor institutions can be "trapped" at low productivity even if they're
          in "steady state"

        """)
        st.page_link("pages/3_Kernel_Regression.py", label="→ Continue to Page 3: Kernel Regression", icon="📖")

except FileNotFoundError:
    st.error("Data files not found. Please ensure CSV files are in the `data/` directory.")
