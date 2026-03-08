"""
Convergence equation and GDP projection engine.

Implements Equation A.2 from Hubbard & Sharma (2016):

    ln(Λᵢ,t / Λᵢ,t-1) = γ·ln(Λᵢ,t-1 / Λᵢ,t-2)
                        + (1-γ)·ln(ΛUS,t / ΛUS,t-1)
                        + β·[ln(φᵢ) - ln(Λᵢ,t-1 / ΛUS,t-1)]

Where:
    γ = 0.5  (weight on lagged own growth / momentum)
    β = 0.025 (speed of convergence)
    φᵢ = steady-state relative productivity for country i
    ΛUS growth = 1.5% per year (frontier productivity growth)
"""

import numpy as np
import pandas as pd


DEFAULT_GAMMA = 0.5
DEFAULT_BETA = 0.025
DEFAULT_US_GROWTH = 0.015


def us_productivity_path(
    us_prod_base: float,
    us_growth_rate: float,
    n_years: int,
    gamma: float = DEFAULT_GAMMA,
    us_growth_history: float | None = None,
) -> np.ndarray:
    """Project US labour productivity path.

    Implements Eq 9 from Au-Yeung et al (2013):
        ln(w_US,t / w_US,t-1) = (1-γ)·μ + γ·ln(w_US,t-1 / w_US,t-2)

    In steady state this converges to μ (the long-run US growth rate).

    Args:
        us_prod_base: US productivity level at start (year 0)
        us_growth_rate: Long-run US productivity growth rate μ
        n_years: Number of years to project
        gamma: Momentum parameter
        us_growth_history: Growth rate in the year before projection starts.
            If None, assumes already at steady-state (= us_growth_rate).

    Returns:
        Array of US productivity levels, length n_years + 1 (includes base year)
    """
    us_prod = np.empty(n_years + 1)
    us_prod[0] = us_prod_base

    prev_growth = us_growth_history if us_growth_history is not None else us_growth_rate

    for t in range(1, n_years + 1):
        growth = (1 - gamma) * us_growth_rate + gamma * prev_growth
        us_prod[t] = us_prod[t - 1] * np.exp(growth)
        prev_growth = growth

    return us_prod


def project_country_productivity(
    prod_history: np.ndarray,
    us_prod: np.ndarray,
    phi_i: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    is_steady_state: bool = False,
) -> np.ndarray:
    """Project a single country's labour productivity path.

    Args:
        prod_history: Country's productivity for at least 2 historical years
            (the last 2 values are used as initial conditions)
        us_prod: US productivity path (full projection, from historical base)
        phi_i: Steady-state relative productivity (φᵢ) for this country
        beta: Speed of convergence
        gamma: Momentum parameter
        is_steady_state: If True, β=0 (country is already at steady state)

    Returns:
        Projected productivity array starting from end of history
    """
    n_proj = len(us_prod) - 1
    # We need at least the last 2 historical points
    prod_t_minus_2 = prod_history[-2]
    prod_t_minus_1 = prod_history[-1]
    us_t_minus_1 = us_prod[0]

    effective_beta = 0.0 if is_steady_state else beta

    result = np.empty(n_proj + 1)
    result[0] = prod_t_minus_1

    prev_growth = np.log(prod_t_minus_1 / prod_t_minus_2)

    for t in range(1, n_proj + 1):
        us_growth = np.log(us_prod[t] / us_prod[t - 1])
        relative_prod = result[t - 1] / us_prod[t - 1]

        # Eq A.2: convergence dynamics
        growth = (
            gamma * prev_growth
            + (1 - gamma) * us_growth
            + effective_beta * (np.log(phi_i) - np.log(relative_prod))
        )

        result[t] = result[t - 1] * np.exp(growth)
        prev_growth = growth

    return result


def project_gdp(
    productivity: np.ndarray,
    working_age_pop: np.ndarray,
) -> np.ndarray:
    """Calculate GDP from productivity and working-age population.

    GDP = labour_productivity × working_age_population
    """
    return productivity * working_age_pop


def project_gdp_per_capita(
    gdp: np.ndarray,
    total_pop: np.ndarray,
) -> np.ndarray:
    """Calculate GDP per capita."""
    return gdp / total_pop


def decompose_growth(
    productivity: np.ndarray,
    working_age_pop: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose GDP growth into productivity growth + workforce growth.

    Returns dict with keys: 'gdp_growth', 'productivity_growth', 'workforce_growth'
    All as annual log growth rates.
    """
    prod_growth = np.diff(np.log(productivity))
    pop_growth = np.diff(np.log(working_age_pop))
    gdp_growth = prod_growth + pop_growth
    return {
        "gdp_growth": gdp_growth,
        "productivity_growth": prod_growth,
        "workforce_growth": pop_growth,
    }


def decade_average_growth(
    years: np.ndarray,
    growth_rates: np.ndarray,
    decades: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Compute decade-average growth rates.

    Args:
        years: Array of years corresponding to growth_rates
        growth_rates: Dict or array of annual growth rates
        decades: List of (start_year, end_year) tuples.
            Default: 1991-2000, 2001-2010, ..., 2041-2050

    Returns:
        DataFrame with decade averages
    """
    if decades is None:
        decades = [
            (1991, 2000), (2001, 2010), (2011, 2020),
            (2021, 2030), (2031, 2040), (2041, 2050),
        ]

    results = []
    for start, end in decades:
        mask = (years >= start) & (years <= end)
        if mask.any():
            avg = np.mean(growth_rates[mask]) * 100  # convert to percentage
        else:
            avg = np.nan
        results.append({
            "period": f"{start}-{end}",
            "avg_growth_pct": avg,
        })

    return pd.DataFrame(results)


def convergence_half_life(beta: float) -> float:
    """Calculate half-life of convergence in years.

    The gap closes by fraction (1-e^(-β)) per year.
    Half-life = ln(2) / β
    """
    return np.log(2) / beta


def simulate_convergence_path(
    initial_relative_prod: float,
    phi_target: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    us_growth: float = DEFAULT_US_GROWTH,
    n_years: int = 50,
) -> dict[str, np.ndarray]:
    """Simulate a stylized convergence path for a single country.

    Useful for pedagogical demonstrations — uses simple initial conditions.

    Returns dict with 'years', 'relative_productivity', 'annual_growth',
    'momentum_component', 'frontier_component', 'catchup_component'
    """
    years = np.arange(n_years + 1)
    rel_prod = np.empty(n_years + 1)
    rel_prod[0] = initial_relative_prod

    annual_growth = np.empty(n_years)
    momentum_comp = np.empty(n_years)
    frontier_comp = np.empty(n_years)
    catchup_comp = np.empty(n_years)

    prev_growth = us_growth  # assume starting at frontier growth rate

    for t in range(n_years):
        catch_up = beta * (np.log(phi_target) - np.log(rel_prod[t]))
        frontier = (1 - gamma) * us_growth
        momentum = gamma * prev_growth

        growth = momentum + frontier + catch_up

        momentum_comp[t] = momentum
        frontier_comp[t] = frontier
        catchup_comp[t] = catch_up
        annual_growth[t] = growth

        # Relative productivity evolves: the country grows at (us_growth + catch_up) relative to US
        # Since we track relative prod, the net relative growth is: growth - us_growth
        rel_prod[t + 1] = rel_prod[t] * np.exp(growth - us_growth)
        prev_growth = growth

    return {
        "years": years,
        "relative_productivity": rel_prod,
        "annual_growth": annual_growth,
        "momentum_component": momentum_comp,
        "frontier_component": frontier_comp,
        "catchup_component": catchup_comp,
    }
