"""
Page 3: Estimating Potential Productivity

Deep dive into the kernel regression that maps GCI → steady-state
relative productivity (Equation A.1).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ltp.data import load_steady_state_countries, load_gci_scores
from ltp.kernel import kernel_estimate, kernel_curve, find_optimal_bandwidth, is_monotonically_nondecreasing
from ltp.viz import plot_kernel_regression, plot_bandwidth_comparison

st.set_page_config(page_title="Kernel Regression", page_icon="📐", layout="wide")
st.title("3. Estimating Potential Productivity")

st.markdown("""
### The Problem

We know the steady-state productivity of 55 countries (from observing stable ratios over decades).
But what about the other 85 transitioning countries? We need to estimate **where they're heading**.

### The Solution: Kernel Regression

We use a **non-parametric** approach — kernel regression — to estimate the relationship
between GCI score and steady-state relative productivity. This avoids imposing a functional
form (like linear or quadratic) and lets the data speak.

### The Math (Equation A.1)

The Nadaraya-Watson kernel estimator:
""")

st.latex(r"""
\hat{\varphi}(GCI) = \frac{\sum_{i=1}^{n} K(\omega_i) \cdot \varphi_i}{\sum_{i=1}^{n} K(\omega_i)}
""")

st.latex(r"""
K(\omega_i) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{GCI_i - GCI}{h}\right)^2\right)
""")

st.markdown("""
In plain English: to estimate the steady-state productivity for a given GCI score,
we take a **weighted average** of all known steady-state countries' productivities,
where countries with **similar GCI scores** get more weight (via the Gaussian kernel).

The bandwidth **h** controls how much weight distant comparators get:
- **h too large** → everyone gets similar weight → estimate is just the average
- **h too small** → only nearest neighbors matter → estimate is noisy and non-monotone
- **h = 0.294** → smallest value ensuring monotonically non-decreasing (the paper's choice)

---
""")

try:
    ss_df = load_steady_state_countries()
    gci_df = load_gci_scores()

    ss_countries = ss_df[ss_df["is_steady_state"] == True]
    ss_with_gci = pd.merge(ss_countries, gci_df[["iso3", "gci_score"]], on="iso3", how="inner")

    ss_gci = ss_with_gci["gci_score"].values
    ss_phi = ss_with_gci["relative_productivity"].values
    ss_labels = ss_with_gci["iso3"].values.tolist()

    tab1, tab2, tab3 = st.tabs([
        "Interactive Bandwidth", "Optimal Estimate", "Predict Any Country"
    ])

    with tab1:
        st.subheader("How Bandwidth Affects the Estimate")
        st.markdown("Drag the slider to see how changing bandwidth **h** changes the curve.")

        h_value = st.slider(
            "Bandwidth (h)",
            min_value=0.05,
            max_value=2.0,
            value=0.294,
            step=0.01,
            help="Paper uses h=0.294. Try different values to see the effect.",
        )

        gci_grid, phi_curve = kernel_curve(ss_gci, ss_phi, bandwidth=h_value)
        is_mono = is_monotonically_nondecreasing(ss_gci, ss_phi, h_value)

        col1, col2, col3 = st.columns(3)
        col1.metric("Bandwidth h", f"{h_value:.3f}")
        col2.metric("Monotonic?", "Yes ✓" if is_mono else "No ✗")
        col3.metric("Paper's h", "0.294")

        fig = plot_kernel_regression(
            ss_gci, ss_phi, gci_grid, phi_curve, ss_labels,
            title=f"Kernel Estimate (h = {h_value:.3f})"
            + (" — Monotonically non-decreasing" if is_mono else " — NOT monotonic!"),
        )
        st.plotly_chart(fig, use_container_width=True)

        if not is_mono:
            st.warning(
                "With this bandwidth, the curve is NOT monotonically non-decreasing. "
                "This means a higher GCI could predict lower productivity — which is "
                "economically unreasonable."
            )

    with tab2:
        st.subheader("Compare Multiple Bandwidths")

        # Compute optimal bandwidth
        h_opt = find_optimal_bandwidth(ss_gci, ss_phi)
        st.markdown(f"**Computed optimal bandwidth:** h = {h_opt:.3f} "
                     f"(paper reports h = 0.294)")

        bandwidths = {
            f"h = 0.10 (too low)": 0.10,
            f"h = {h_opt:.3f} (optimal)": h_opt,
            f"h = 0.50 (moderate)": 0.50,
            f"h = 1.50 (too high)": 1.50,
        }

        gci_grid_compare = np.linspace(ss_gci.min() - 0.2, ss_gci.max() + 0.2, 200)
        curves = {}
        for label, h in bandwidths.items():
            curves[label] = kernel_estimate(ss_gci, ss_phi, gci_grid_compare, h)

        fig = plot_bandwidth_comparison(ss_gci, ss_phi, gci_grid_compare, curves)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Notice:
        - **h = 0.10**: Too noisy — the curve dips and rises erratically
        - **h = optimal**: Smooth, monotonically non-decreasing — the paper's choice
        - **h = 0.50**: Smoother but starts to flatten the important middle section
        - **h = 1.50**: Almost flat — barely distinguishes between GCI scores
        """)

    with tab3:
        st.subheader("Predict Steady-State Productivity for Any Country")
        st.markdown("""
        Select a transitioning country to see its predicted steady-state relative
        productivity based on its GCI score.
        """)

        # Get transitioning countries
        trans = gci_df[~gci_df["iso3"].isin(ss_countries["iso3"].values)].sort_values(
            "country_name"
        )

        selected_country = st.selectbox(
            "Select a transitioning country:",
            trans["country_name"].unique(),
        )

        if selected_country:
            country_row = trans[trans["country_name"] == selected_country].iloc[0]
            country_gci = country_row["gci_score"]

            # Use optimal bandwidth
            gci_grid_opt, phi_curve_opt = kernel_curve(ss_gci, ss_phi, bandwidth=h_opt)
            predicted_phi = kernel_estimate(ss_gci, ss_phi, country_gci, h_opt)

            col1, col2, col3 = st.columns(3)
            col1.metric("Country", selected_country)
            col2.metric("GCI Score", f"{country_gci:.2f}")
            col3.metric(
                "Predicted φ (% of US)",
                f"{predicted_phi * 100:.1f}%",
            )

            # Plot with the country highlighted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ss_gci, y=ss_phi, mode="markers",
                name="Steady-state countries",
                marker=dict(size=8, color=px.colors.qualitative.Set2[1], opacity=0.6),
                text=ss_labels,
            ))
            fig.add_trace(go.Scatter(
                x=gci_grid_opt, y=phi_curve_opt, mode="lines",
                name="Kernel estimate",
                line=dict(color=px.colors.qualitative.Set2[0], width=3),
            ))
            # Highlight selected country
            fig.add_trace(go.Scatter(
                x=[country_gci], y=[predicted_phi], mode="markers+text",
                name=selected_country,
                text=[f"{selected_country}<br>φ = {predicted_phi:.3f}"],
                textposition="top center",
                marker=dict(size=15, color="red", symbol="star"),
            ))
            # Add vertical and horizontal reference lines
            fig.add_vline(x=country_gci, line_dash="dot", line_color="red", opacity=0.5)
            fig.add_hline(y=predicted_phi, line_dash="dot", line_color="red", opacity=0.5)

            fig.update_layout(
                title=f"Predicted Steady-State Productivity: {selected_country}",
                xaxis_title="GCI Score",
                yaxis_title="Relative Productivity (US = 1.0)",
                yaxis_type="log",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **Interpretation:** Based on {selected_country}'s institutional quality
            (GCI = {country_gci:.2f}), the model predicts its long-run labour productivity
            will converge to **{predicted_phi * 100:.1f}%** of US levels.

            This doesn't mean it will get there immediately — the *speed* of convergence
            is the subject of the next page.
            """)

    st.markdown("""
    ---
    ### Key Takeaway

    The kernel regression provides a **non-parametric mapping** from institutional quality
    (GCI) to potential productivity. It uses only the 55 steady-state countries as
    calibration points and makes minimal assumptions about the functional form.

    The key insight is the **S-shaped** relationship:
    - Below GCI ~3.8: Low productivity, modest gains from institutional improvement
    - GCI 3.8-4.8: **Steep gains** — this is where institutional reform pays off most
    - Above GCI ~4.8: High productivity, diminishing returns to further improvement

    """)
    st.page_link("pages/4_Convergence_Dynamics.py", label="→ Continue to Page 4: Convergence Dynamics", icon="📖")

except FileNotFoundError:
    st.error("Data files not found. Please ensure CSV files are in the `data/` directory.")
