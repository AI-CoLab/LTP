"""
Page 4: How Fast Do Countries Catch Up?

Explains and demonstrates the convergence equation (Equation A.2) —
the mechanics of how countries close the gap to their potential.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ltp.convergence import (
    simulate_convergence_path,
    convergence_half_life,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_US_GROWTH,
)
from ltp.viz import plot_convergence_path

st.set_page_config(page_title="Convergence Dynamics", page_icon="⚡", layout="wide")
st.title("4. How Fast Do Countries Catch Up?")

st.markdown("""
### The Convergence Equation (A.2)

Now we know *where* each country is heading (from the kernel regression).
The question is: **how fast do they get there?**

The model decomposes a country's annual productivity growth into three components:
""")

st.latex(r"""
\ln\frac{\Lambda_{i,t}}{\Lambda_{i,t-1}} = \underbrace{\gamma \cdot \ln\frac{\Lambda_{i,t-1}}{\Lambda_{i,t-2}}}_{\text{Momentum}}
+ \underbrace{(1-\gamma) \cdot \ln\frac{\Lambda_{US,t}}{\Lambda_{US,t-1}}}_{\text{Frontier}}
+ \underbrace{\beta \cdot \left[\ln(\varphi_i) - \ln\frac{\Lambda_{i,t-1}}{\Lambda_{US,t-1}}\right]}_{\text{Catch-up}}
""")

st.markdown("""
| Component | Parameter | Meaning |
|-----------|-----------|---------|
| **Momentum** | γ = 0.5 | Half of last year's growth carries over (inertia) |
| **Frontier** | 1-γ = 0.5 | Half comes from global technology growth (US rate) |
| **Catch-up** | β = 0.025 | 2.5% of the gap to potential is closed each year |

The **catch-up term** is the key innovation: it's proportional to the *log distance*
between current and potential relative productivity. Countries far below their
potential grow faster; countries at their potential have zero catch-up growth.

---
### Interactive Simulation
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
beta = st.sidebar.slider("β (convergence speed)", 0.005, 0.10, DEFAULT_BETA, 0.005,
                          help="Paper default: 0.025")
gamma = st.sidebar.slider("γ (momentum weight)", 0.0, 1.0, DEFAULT_GAMMA, 0.05,
                           help="Paper default: 0.5")
us_growth = st.sidebar.slider("US growth rate (%)", 0.5, 3.0, DEFAULT_US_GROWTH * 100, 0.1,
                               help="Paper default: 1.5%") / 100
n_years = st.sidebar.slider("Projection horizon (years)", 10, 100, 50, 5)

# Country-like scenarios
st.sidebar.header("Country Scenario")
initial_rel = st.sidebar.slider(
    "Current relative productivity (% of US)",
    1, 100, 20, 1,
    help="e.g., China ≈ 20%, India ≈ 10%, Indonesia ≈ 12%",
) / 100

target_rel = st.sidebar.slider(
    "Target φ (% of US)",
    5, 150, 70, 5,
    help="Steady-state potential from kernel regression",
) / 100

# Half-life info
hl = convergence_half_life(beta)
st.sidebar.metric("Half-life", f"{hl:.0f} years")
st.sidebar.caption(f"At β={beta}, it takes ~{hl:.0f} years to close half the gap.")

# Run simulation
sim = simulate_convergence_path(
    initial_relative_prod=initial_rel,
    phi_target=target_rel,
    beta=beta,
    gamma=gamma,
    us_growth=us_growth,
    n_years=n_years,
)

tab1, tab2, tab3 = st.tabs([
    "Convergence Path", "Growth Decomposition", "Sensitivity Analysis"
])

with tab1:
    st.subheader("Relative Productivity Over Time")

    colors = px.colors.qualitative.Set2
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sim["years"], y=sim["relative_productivity"] * 100,
        mode="lines", name="Projected path",
        line=dict(color=colors[0], width=3),
    ))

    fig.add_hline(y=target_rel * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"Target φ = {target_rel*100:.0f}%")
    fig.add_hline(y=initial_rel * 100, line_dash="dot", line_color="lightgray",
                  annotation_text=f"Start = {initial_rel*100:.0f}%")

    # Mark half-life point
    if hl < n_years:
        half_gap_level = initial_rel + (target_rel - initial_rel) * 0.5
        fig.add_trace(go.Scatter(
            x=[hl], y=[half_gap_level * 100],
            mode="markers+text",
            text=[f"Half-life ≈ {hl:.0f} yr"],
            textposition="top right",
            marker=dict(size=12, color="red", symbol="diamond"),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis_title="Years from start",
        yaxis_title="Productivity (% of US level)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Start", f"{initial_rel*100:.0f}% of US")
    col2.metric("After 30 years", f"{sim['relative_productivity'][min(30, n_years)]*100:.1f}%")
    if n_years >= 50:
        col3.metric("After 50 years", f"{sim['relative_productivity'][50]*100:.1f}%")
    col4.metric("Target φ", f"{target_rel*100:.0f}%")

with tab2:
    st.subheader("Growth Rate Decomposition")
    st.markdown("""
    This chart breaks down annual growth into its three components.
    Notice how the **catch-up** component shrinks over time as the country
    approaches its steady state.
    """)

    fig = plot_convergence_path(sim, "Simulated Country")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Sensitivity to Convergence Speed (β)")
    st.markdown("""
    How much does the convergence speed matter? Compare different values of β:
    - **β = 0.02** (Au-Yeung et al. 2013 baseline)
    - **β = 0.025** (Hubbard & Sharma, slightly faster)
    - **β = 0.05** (hypothetical — much faster convergence)
    """)

    betas = [0.02, 0.025, 0.05]
    fig = go.Figure()

    for i, b in enumerate(betas):
        s = simulate_convergence_path(
            initial_rel, target_rel, beta=b, gamma=gamma,
            us_growth=us_growth, n_years=n_years,
        )
        hl_b = convergence_half_life(b)
        fig.add_trace(go.Scatter(
            x=s["years"], y=s["relative_productivity"] * 100,
            mode="lines",
            name=f"β = {b} (half-life = {hl_b:.0f}yr)",
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))

    fig.add_hline(y=target_rel * 100, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Productivity (% of US level)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    | β | Half-life | Gap closed after 30 years |
    |---|-----------|---------------------------|
    | 0.02 | ~35 years | ~45% |
    | 0.025 | ~28 years | ~53% |
    | 0.05 | ~14 years | ~78% |

    Even "optimistic" convergence speeds imply it takes **decades** for countries
    to substantially close the productivity gap.
    """)

st.markdown("""
---
### Key Takeaway

Convergence is **slow**. At the paper's β = 0.025, it takes ~28 years to close
half the gap between current and potential productivity. This is why:

- China's growth is projected to gradually slow from ~7% to ~2% over 30 years
- India's demographic dividend can partially compensate for the slow productivity catch-up
- Advanced countries with shrinking workforces face very slow GDP growth

The model's simplicity is also its limitation: it assumes **fixed institutions** and
**smooth convergence**. Reality involves policy reforms, crises, and structural breaks.

""")
prev_col, next_col = st.columns(2)
with prev_col:
    st.page_link("pages/3_Kernel_Regression.py", label="← Back to Page 3: Kernel Regression", icon="📖")
with next_col:
    st.page_link("pages/5_Full_Model.py", label="→ Continue to Page 5: Full Model", icon="📖")
