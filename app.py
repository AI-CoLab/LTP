"""
Hubbard & Sharma (2016) Long-Term GDP Projection Model
Interactive Streamlit Application

Navigate through the pages to build intuition step-by-step,
then explore the full model with scenario analysis.
"""

import streamlit as st

st.set_page_config(
    page_title="Long-Term GDP Projections",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Long-Term GDP Projections")
st.subheader("Based on Hubbard & Sharma (2016)")

st.markdown("""
This interactive application walks through the conditional convergence model
for projecting GDP of 140 world economies from 2020 to 2050, as described in:

> **Hubbard, P. & Sharma, D. (2016).** *Understanding and Applying Long-Term GDP Projections.*
> EABER Working Paper No. 119.

The model builds on the Australian Treasury methodology (Au-Yeung et al., 2013).

---

### How to use this app

Use the **sidebar** to navigate through five progressive pages:

1. **What Drives GDP?** — The GDP identity and its components
2. **Convergence Intuition** — Why some countries grow faster than others
3. **Kernel Regression** — Estimating each country's productivity potential
4. **Convergence Dynamics** — The speed and mechanics of catching up
5. **Full Model** — Complete replication with scenario analysis

Each page builds on the previous one, constructing the full model piece by piece.

---

### Model at a Glance

The model rests on a simple but powerful framework:

| Component | Description |
|-----------|-------------|
| **GDP = Workforce x Productivity** | The fundamental identity |
| **Demographics** | UN population projections determine workforce |
| **Institutional Quality** | WEF Global Competitiveness Index (GCI) |
| **Steady-State Productivity** | Kernel regression maps GCI → potential productivity |
| **Convergence** | Countries close the gap to their potential at ~2.5% per year |

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| β (beta) | 0.025 | Speed of convergence (2.5% of gap closed per year) |
| γ (gamma) | 0.5 | Weight on momentum (lagged growth) |
| μ (mu) | 1.5% | US frontier productivity growth rate |
| h | 0.294 | Kernel regression bandwidth |
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** Bundled CSVs matching the paper's 2015-era vintage. "
    "Toggle live API data on the Full Model page."
)
