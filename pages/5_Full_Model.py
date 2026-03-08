"""
Page 5: Full Model — Complete Replication

Runs the full Hubbard & Sharma model for all 140 countries
and provides scenario analysis tools.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ltp.model import HubbardSharmaModel
from ltp.data import load_gci_scores
from ltp.viz import (
    plot_gdp_trajectories,
    plot_per_capita_comparison,
    plot_regional_shares,
    plot_growth_decomposition_table,
    plot_scenario_comparison,
)

st.set_page_config(page_title="Full Model", page_icon="🌍", layout="wide")
st.title("5. Full Model — GDP Projections to 2050")

st.markdown("""
This page runs the complete Hubbard & Sharma model for all 140 countries,
reproducing the paper's key results and allowing scenario analysis.
""")

# --- Sidebar parameters ---
st.sidebar.header("Model Parameters")
beta = st.sidebar.slider("β (convergence speed)", 0.005, 0.10, 0.025, 0.005)
gamma = st.sidebar.slider("γ (momentum)", 0.0, 1.0, 0.5, 0.05)
us_growth = st.sidebar.slider("US productivity growth (%)", 0.5, 3.0, 1.5, 0.1) / 100
bandwidth = st.sidebar.slider("Kernel bandwidth h", 0.1, 1.0, 0.294, 0.01)

# --- Run model ---
@st.cache_resource
def run_baseline_model(beta, gamma, us_growth, bandwidth):
    model = HubbardSharmaModel(
        beta=beta, gamma=gamma, us_growth=us_growth, bandwidth=bandwidth,
    )
    model.fit()
    return model


def run_scenario_model(beta, gamma, us_growth, bandwidth, gci_overrides):
    """Run model with GCI overrides — not cached since it depends on slider state."""
    model = HubbardSharmaModel(
        beta=beta, gamma=gamma, us_growth=us_growth, bandwidth=bandwidth,
    )
    model.fit(gci_overrides=gci_overrides)
    return model


try:
    model = run_baseline_model(beta, gamma, us_growth, bandwidth)
    projections = model.projections

    if projections is None or projections.empty:
        st.error("Model produced no projections. Check data files.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Top Economies", "Growth Decomposition", "Regional Shares",
        "Per Capita", "Scenario Builder"
    ])

    colors = px.colors.qualitative.Set2

    with tab1:
        st.subheader("GDP of Largest Economies (Figure 2)")
        st.markdown("Projected GDP trajectories for the seven largest economies by 2050.")

        top7_2050 = model.get_top_economies(2050, 7)
        top7_codes = top7_2050["iso3"].tolist()

        # Also include historical data
        all_data = model.data["productivity"]

        # Let user customize which countries to show
        all_countries = sorted(projections["country_name"].unique())
        default_show = [
            projections[projections["iso3"] == c]["country_name"].iloc[0]
            for c in top7_codes
            if not projections[projections["iso3"] == c].empty
        ]

        selected = st.multiselect(
            "Countries to display:",
            all_countries,
            default=default_show[:7],
        )

        if selected:
            selected_iso3 = projections[
                projections["country_name"].isin(selected)
            ]["iso3"].unique().tolist()

            # Combine historical + projected
            hist = all_data[all_data["iso3"].isin(selected_iso3)][
                ["iso3", "country_name", "year", "gdp_billions_2012ppp"]
            ].rename(columns={"gdp_billions_2012ppp": "gdp_billions"})

            proj = projections[
                (projections["iso3"].isin(selected_iso3)) & (projections["is_projection"])
            ][["iso3", "country_name", "year", "gdp_billions"]]

            combined = pd.concat([hist, proj]).drop_duplicates(
                ["iso3", "year"], keep="last"
            ).sort_values(["iso3", "year"])

            fig = go.Figure()
            for i, iso3 in enumerate(selected_iso3):
                c = combined[combined["iso3"] == iso3]
                name = c["country_name"].iloc[0]

                # Historical portion
                c_hist = c[c["year"] <= 2020]
                c_proj = c[c["year"] >= 2020]

                fig.add_trace(go.Scatter(
                    x=c_hist["year"], y=c_hist["gdp_billions"] / 1e3,
                    mode="lines", name=f"{name}",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    legendgroup=name,
                ))
                if not c_proj.empty:
                    fig.add_trace(go.Scatter(
                        x=c_proj["year"], y=c_proj["gdp_billions"] / 1e3,
                        mode="lines", name=f"{name} (projected)",
                        line=dict(color=colors[i % len(colors)], width=2.5, dash="dash"),
                        legendgroup=name,
                        showlegend=False,
                    ))

            fig.add_vline(x=2020, line_dash="dot", line_color="gray",
                          annotation_text="Projection starts")
            fig.update_layout(
                title="GDP Trajectories (trillion USD, 2012 PPP)",
                xaxis_title="Year",
                yaxis_title="GDP (trillion USD, 2012 PPP)",
                template="plotly_white",
                height=550,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Growth Decomposition by Decade (Table 1)")
        st.markdown("Decade-average GDP growth decomposed into productivity growth and workforce growth.")

        decomp_country = st.selectbox(
            "Select country for decomposition:",
            all_countries,
            index=all_countries.index("China") if "China" in all_countries else 0,
            key="decomp_country",
        )

        iso3 = projections[projections["country_name"] == decomp_country]["iso3"].iloc[0]
        decomp = model.get_growth_decomposition(iso3)

        if not decomp.empty:
            fig = plot_growth_decomposition_table(decomp, decomp_country)
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart with proper positive/negative stacking and total line
            fig_bar = go.Figure()

            # Productivity growth bars (can be positive or negative)
            fig_bar.add_trace(go.Bar(
                x=decomp["period"], y=decomp["productivity_growth"],
                name="Productivity growth", marker_color=colors[0],
            ))
            # Workforce growth bars (negative in later decades for many countries)
            fig_bar.add_trace(go.Bar(
                x=decomp["period"], y=decomp["workforce_growth"],
                name="Workforce growth", marker_color=colors[1],
            ))
            # Total GDP growth as black line
            fig_bar.add_trace(go.Scatter(
                x=decomp["period"], y=decomp["gdp_growth"],
                name="Total GDP growth", mode="lines+markers",
                line=dict(color="black", width=2.5),
                marker=dict(size=7, color="black"),
            ))

            fig_bar.update_layout(
                title=f"Growth Decomposition: {decomp_country}",
                yaxis_title="Average annual growth (%)",
                barmode="relative",
                template="plotly_white",
                height=400,
            )
            fig_bar.add_hline(y=0, line_color="gray", line_width=0.5)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning(f"Insufficient data for {decomp_country}")

    with tab3:
        st.subheader("Regional GDP Shares (Figure 3)")

        # Define regions
        region_mapping = {
            "North America": ["USA", "CAN", "MEX"],
            "Europe": ["GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE",
                       "SWE", "NOR", "DNK", "FIN", "IRL", "PRT", "GRC", "POL", "CZE",
                       "ROU", "HUN", "SVK", "BGR", "HRV", "SVN", "LTU", "LVA", "EST"],
            "East Asia": ["CHN", "JPN", "KOR", "TWN", "HKG"],
            "South Asia": ["IND", "PAK", "BGD", "LKA"],
            "Southeast Asia": ["IDN", "THA", "MYS", "PHL", "VNM", "SGP", "MMR"],
            "Middle East & Africa": ["SAU", "ARE", "EGY", "NGA", "ZAF", "KEN", "ETH",
                                     "DZA", "MAR", "IRN", "IRQ", "ISR"],
            "Latin America": ["BRA", "ARG", "COL", "CHL", "PER", "VEN", "ECU"],
        }

        shares = model.get_regional_shares(region_mapping)

        if not shares.empty:
            fig = plot_regional_shares(shares)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Per Capita GDP Comparison (Figure 5)")

        year_select = st.slider("Year", 2020, 2050, 2050, 5, key="percap_year")

        compare_countries = st.multiselect(
            "Countries to compare:",
            all_countries,
            default=["United States", "China", "India", "Japan", "Germany",
                     "Indonesia", "Mexico", "Brazil"][:min(8, len(all_countries))],
            key="percap_countries",
        )

        if compare_countries:
            compare_iso3 = projections[
                projections["country_name"].isin(compare_countries)
            ]["iso3"].unique().tolist()

            fig = plot_per_capita_comparison(
                projections, compare_iso3, year_select,
                title="GDP per Capita Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Scenario Builder (Figure 6)")
        st.markdown("""
        What happens if a country improves its institutions? Modify a country's
        GCI score to simulate policy reform and see the impact on GDP projections.

        The paper's example scenarios:
        - **China** → South Korea's GCI level
        - **India** → China's GCI level
        - **Indonesia** → China's GCI level
        """)

        gci_df = load_gci_scores()

        col1, col2 = st.columns(2)
        with col1:
            scenario_country = st.selectbox(
                "Country to modify:",
                all_countries,
                index=all_countries.index("China") if "China" in all_countries else 0,
                key="scenario_country",
            )

        scenario_iso3 = projections[
            projections["country_name"] == scenario_country
        ]["iso3"].iloc[0]

        current_gci = gci_df[gci_df["iso3"] == scenario_iso3]["gci_score"]
        current_gci_val = current_gci.iloc[0] if not current_gci.empty else 4.0

        with col2:
            new_gci = st.slider(
                f"New GCI score for {scenario_country}",
                2.5, 6.0, float(current_gci_val), 0.05,
                key="scenario_gci",
            )

        st.caption(f"Current GCI: {current_gci_val:.2f} → Modified: {new_gci:.2f}")

        # Show comparable steady-state countries for context
        ss_df = model.data["steady_state"]
        ss_countries_only = ss_df[ss_df["is_steady_state"] == True].copy()
        ss_with_gci = pd.merge(
            ss_countries_only, gci_df[["iso3", "gci_score"]], on="iso3", how="inner"
        )
        if not ss_with_gci.empty:
            active_gci = new_gci
            ss_with_gci["gci_dist"] = (ss_with_gci["gci_score"] - active_gci).abs()
            nearest = ss_with_gci.nsmallest(3, "gci_dist")
            comparables = ", ".join(
                f"{r['country_name']} ({r['gci_score']:.2f})"
                for _, r in nearest.iterrows()
            )
            # Compute kernel-estimated φ for this GCI score
            from ltp.kernel import kernel_estimate
            phi_est = kernel_estimate(
                model.ss_gci, model.ss_phi, active_gci, model.optimal_bandwidth
            )
            st.info(
                f"**GCI {active_gci:.2f}** is comparable to: {comparables}. "
                f"Kernel-estimated steady-state productivity: **{phi_est:.2f}** (US = 1.0)"
            )

        # Pre-built scenarios
        preset = st.selectbox(
            "Or use a preset scenario:",
            ["Custom (use slider above)", "China → Korea's GCI",
             "India → China's GCI", "Indonesia → China's GCI"],
            key="preset_scenario",
        )

        gci_override = {}
        if preset == "China → Korea's GCI":
            kor_gci = gci_df[gci_df["iso3"] == "KOR"]["gci_score"]
            if not kor_gci.empty:
                gci_override = {"CHN": kor_gci.iloc[0]}
                scenario_iso3 = "CHN"
                scenario_country = "China"
        elif preset == "India → China's GCI":
            chn_gci = gci_df[gci_df["iso3"] == "CHN"]["gci_score"]
            if not chn_gci.empty:
                gci_override = {"IND": chn_gci.iloc[0]}
                scenario_iso3 = "IND"
                scenario_country = "India"
        elif preset == "Indonesia → China's GCI":
            chn_gci = gci_df[gci_df["iso3"] == "CHN"]["gci_score"]
            if not chn_gci.empty:
                gci_override = {"IDN": chn_gci.iloc[0]}
                scenario_iso3 = "IDN"
                scenario_country = "Indonesia"
        else:
            gci_override = {scenario_iso3: new_gci}

        if gci_override:
            scenario_model = run_scenario_model(beta, gamma, us_growth, bandwidth,
                                                gci_overrides=gci_override)
            scenario_proj = scenario_model.projections

            if scenario_proj is not None and not scenario_proj.empty:
                fig = plot_scenario_comparison(
                    projections, scenario_proj, scenario_iso3,
                    metric="gdp_per_capita",
                    title=f"Scenario: {scenario_country} — Per Capita GDP",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show GDP comparison too
                fig2 = plot_scenario_comparison(
                    projections, scenario_proj, scenario_iso3,
                    metric="gdp_billions",
                    title=f"Scenario: {scenario_country} — Total GDP",
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Compute uplift
                base_2050 = projections[
                    (projections["iso3"] == scenario_iso3) & (projections["year"] == 2050)
                ]["gdp_per_capita"]
                scen_2050 = scenario_proj[
                    (scenario_proj["iso3"] == scenario_iso3) & (scenario_proj["year"] == 2050)
                ]["gdp_per_capita"]

                if not base_2050.empty and not scen_2050.empty:
                    uplift = (scen_2050.iloc[0] / base_2050.iloc[0] - 1) * 100
                    st.metric(
                        f"Per capita GDP uplift by 2050",
                        f"{uplift:+.0f}%",
                    )

    st.markdown("""
    ---
    ### Model Limitations (from the paper)

    This is a deliberately simple model. Key limitations include:

    - **Fixed institutions**: GCI scores are frozen — no room for reform or deterioration
    - **Single sector**: No structural change, rural-urban migration, or sectoral shifts
    - **No interaction**: Countries don't trade with or affect each other
    - **No finance**: No capital flows, debt, or financial crises
    - **Demographics as given**: No feedback from economics to population
    - **PPP only**: No exchange rates or terms-of-trade effects
    - **Political stability assumed**: Countries are assumed to persist unchanged

    Despite these limitations, the model provides a **coherent, transparent baseline**
    for thinking about long-term economic trajectories — and a framework for
    asking "what if?" through scenario analysis.
    """)

except FileNotFoundError as e:
    st.error(f"Data files not found: {e}. Please ensure CSV files are in the `data/` directory.")
except Exception as e:
    st.error(f"Error running model: {e}")
    st.exception(e)
