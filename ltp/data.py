"""
Data loading utilities for the LTP model.

Loads bundled CSV files or optionally pulls from live APIs.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


def load_gdp_live() -> pd.DataFrame:
    """Load GDP data from the live IMF API."""
    from api.imf import fetch_gdp_ppp
    return fetch_gdp_ppp(start_year=1980, end_year=2020)


def load_population_live() -> pd.DataFrame:
    """Load population data from the live UN API."""
    from api.un import fetch_population_projections
    return fetch_population_projections(start_year=1980, end_year=2050)


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


def get_model_inputs(data_dir: Path | None = None, use_live_api: bool = False) -> dict:
    """Load all data needed for the full model.

    Args:
        data_dir: Path to bundled data directory
        use_live_api: If True, fetch GDP and population from live APIs
            (GCI and steady-state always use bundled data)

    Returns dict with keys:
        'steady_state': DataFrame of steady-state countries
        'gci': DataFrame of GCI scores
        'gdp': DataFrame of historical GDP
        'population': DataFrame of population projections
        'productivity': DataFrame with computed productivity
        'source': str indicating data source ('bundled' or 'live_api')
    """
    ss = load_steady_state_countries(data_dir)
    gci = load_gci_scores(data_dir)

    if use_live_api:
        gdp = load_gdp_live()
        pop = load_population_live()
        source = "live_api"
    else:
        gdp = load_gdp(data_dir)
        pop = load_population(data_dir)
        source = "bundled"

    prod = compute_labour_productivity(gdp, pop)

    return {
        "steady_state": ss,
        "gci": gci,
        "gdp": gdp,
        "population": pop,
        "productivity": prod,
        "source": source,
    }


def diagnose_data(data: dict) -> dict:
    """Run diagnostic checks on loaded model data.

    Returns a dict of diagnostic results for display.
    """
    results = {}

    # GDP diagnostics
    gdp = data["gdp"]
    results["gdp"] = {
        "countries": int(gdp["iso3"].nunique()),
        "year_range": f"{int(gdp['year'].min())}–{int(gdp['year'].max())}",
        "total_rows": len(gdp),
        "missing_values": int(gdp["gdp_billions_2012ppp"].isna().sum()),
        "has_usa": "USA" in gdp["iso3"].values,
        "sample_countries": sorted(gdp["iso3"].unique()[:10].tolist()),
    }

    # Population diagnostics
    pop = data["population"]
    results["population"] = {
        "countries": int(pop["iso3"].nunique()),
        "year_range": f"{int(pop['year'].min())}–{int(pop['year'].max())}",
        "total_rows": len(pop),
        "missing_wap": int(pop["working_age_pop_thousands"].isna().sum()),
        "missing_total": int(pop["total_pop_thousands"].isna().sum()),
        "has_projections": int(pop["year"].max()) >= 2050,
    }

    # GCI diagnostics
    gci = data["gci"]
    results["gci"] = {
        "countries": int(gci["iso3"].nunique()),
        "score_range": f"{gci['gci_score'].min():.2f}–{gci['gci_score'].max():.2f}",
        "missing_scores": int(gci["gci_score"].isna().sum()),
    }

    # Steady-state diagnostics
    ss = data["steady_state"]
    results["steady_state"] = {
        "total_countries": len(ss),
        "steady_state_count": int(ss["is_steady_state"].sum()),
        "non_steady_state": int((~ss["is_steady_state"]).sum()),
    }

    # Productivity merge diagnostics
    prod = data["productivity"]
    results["productivity"] = {
        "countries_after_merge": int(prod["iso3"].nunique()),
        "year_range": f"{int(prod['year'].min())}–{int(prod['year'].max())}",
        "missing_productivity": int(prod["labour_productivity"].isna().sum()),
        "us_coverage": int(prod[prod["iso3"] == "USA"]["year"].nunique()),
    }

    # Cross-check: countries in GCI but missing from GDP/pop
    gci_countries = set(gci["iso3"].unique())
    gdp_countries = set(gdp["iso3"].unique())
    pop_countries = set(pop["iso3"].unique())
    prod_countries = set(prod["iso3"].unique())

    results["coverage"] = {
        "in_gci_only": sorted(gci_countries - gdp_countries - pop_countries),
        "in_gci_missing_gdp": sorted(gci_countries - gdp_countries),
        "in_gci_missing_pop": sorted(gci_countries - pop_countries),
        "model_ready": sorted(gci_countries & gdp_countries & pop_countries),
        "model_ready_count": len(gci_countries & gdp_countries & pop_countries),
    }

    # Data source
    results["source"] = data.get("source", "bundled")

    return results
