"""UN Population Division data fetcher.

Fetches working-age and total population projections from the
UN World Population Prospects API.
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# UN Population Division Data Portal API
BASE_URL = "https://population.un.org/dataportalapi/api/v1"


def _fetch_indicator(
    indicator_id: int,
    start_year: int,
    end_year: int,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch a single indicator from the UN Population API."""
    records = []
    page = 1
    while True:
        params = {
            "indicatorIds": indicator_id,
            "startYear": start_year,
            "endYear": end_year,
            "variant": 2,  # Medium variant
            "pagingInHeader": "false",
            "pageSize": 100,
            "pageNumber": page,
        }
        resp = requests.get(f"{BASE_URL}/data/indicators/{indicator_id}",
                            params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            break

        for row in data["data"]:
            if row.get("iso3") and row.get("timeLabel") and row.get("value") is not None:
                records.append({
                    "iso3": row["iso3"],
                    "country_name": row.get("location", row["iso3"]),
                    "year": int(row["timeLabel"]),
                    "value": float(row["value"]),
                })

        # Check for more pages
        if len(data["data"]) < 100:
            break
        page += 1

    return pd.DataFrame(records)


def fetch_population_projections(
    iso3_list: list[str] | None = None,
    start_year: int = 1980,
    end_year: int = 2050,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch population projections from UN Population Division.

    Returns DataFrame with columns matching bundled CSV:
        iso3, country_name, year, working_age_pop_thousands, total_pop_thousands

    Indicator IDs:
        47 = Total population (thousands)
        50 = Population aged 15-64 (thousands) - working age
    """
    # Fetch total population (indicator 47)
    logger.info("Fetching total population from UN API...")
    total_pop = _fetch_indicator(47, start_year, end_year, timeout)
    total_pop = total_pop.rename(columns={"value": "total_pop_thousands"})

    # Fetch working-age population (indicator 50: age 15-64)
    logger.info("Fetching working-age population from UN API...")
    wap = _fetch_indicator(50, start_year, end_year, timeout)
    wap = wap.rename(columns={"value": "working_age_pop_thousands"})

    if total_pop.empty or wap.empty:
        raise ValueError("No population data returned from UN API")

    # Merge
    merged = pd.merge(
        total_pop[["iso3", "country_name", "year", "total_pop_thousands"]],
        wap[["iso3", "year", "working_age_pop_thousands"]],
        on=["iso3", "year"],
        how="inner",
    )

    if iso3_list:
        merged = merged[merged["iso3"].isin(iso3_list)]

    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)
    logger.info("Fetched population data for %d countries from UN API", merged["iso3"].nunique())
    return merged
