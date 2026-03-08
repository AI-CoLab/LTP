"""
Data loading utilities for the LTP model.

Loads bundled CSV files or optionally pulls from live APIs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"


def load_steady_state_countries(data_dir: Path | None = None) -> pd.DataFrame:
    """Load steady-state country classifications and relative productivity.

    Returns DataFrame with columns:
        iso3, country_name, relative_productivity, is_steady_state
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "steady_state_countries.csv")


def load_gci_scores(data_dir: Path | None = None) -> pd.DataFrame:
    """Load GCI scores for all countries.

    Returns DataFrame with columns:
        iso3, country_name, gci_score, gci_basic, gci_efficiency, gci_innovation, region
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "gci_scores.csv")


def load_gdp(data_dir: Path | None = None) -> pd.DataFrame:
    """Load historical GDP data (long format).

    Returns DataFrame with columns:
        iso3, country_name, year, gdp_billions_2012ppp
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "imf_gdp.csv")


def load_population(data_dir: Path | None = None) -> pd.DataFrame:
    """Load population data (long format).

    Returns DataFrame with columns:
        iso3, country_name, year, working_age_pop_thousands, total_pop_thousands
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "un_population.csv")


def compute_labour_productivity(
    gdp_df: pd.DataFrame,
    pop_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute labour productivity = GDP / working_age_population.

    Returns DataFrame with columns:
        iso3, country_name, year, gdp, working_age_pop, total_pop,
        labour_productivity, relative_productivity
    """
    # Merge GDP and population data
    merged = pd.merge(
        gdp_df,
        pop_df,
        on=["iso3", "country_name", "year"],
        how="inner",
    )

    # GDP in billions, population in thousands → productivity in $thousands per worker
    # Convert: GDP_billions × 1e9 / (working_age_pop_thousands × 1e3) = GDP_billions / working_age_pop_thousands × 1e6
    merged["labour_productivity"] = (
        merged["gdp_billions_2012ppp"] * 1e6 / merged["working_age_pop_thousands"]
    )

    # Compute relative to US
    us_prod = merged[merged["iso3"] == "USA"][["year", "labour_productivity"]].rename(
        columns={"labour_productivity": "us_productivity"}
    )
    merged = pd.merge(merged, us_prod, on="year", how="left")
    merged["relative_productivity"] = merged["labour_productivity"] / merged["us_productivity"]

    return merged.drop(columns=["us_productivity"])


def get_country_data(
    iso3: str,
    gdp_df: pd.DataFrame,
    pop_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get all data for a single country, sorted by year."""
    prod_df = compute_labour_productivity(gdp_df, pop_df)
    country = prod_df[prod_df["iso3"] == iso3].sort_values("year").reset_index(drop=True)
    return country


def get_model_inputs(data_dir: Path | None = None) -> dict:
    """Load all data needed for the full model.

    Returns dict with keys:
        'steady_state': DataFrame of steady-state countries
        'gci': DataFrame of GCI scores
        'gdp': DataFrame of historical GDP
        'population': DataFrame of population projections
        'productivity': DataFrame with computed productivity
    """
    ss = load_steady_state_countries(data_dir)
    gci = load_gci_scores(data_dir)
    gdp = load_gdp(data_dir)
    pop = load_population(data_dir)
    prod = compute_labour_productivity(gdp, pop)

    return {
        "steady_state": ss,
        "gci": gci,
        "gdp": gdp,
        "population": pop,
        "productivity": prod,
    }
