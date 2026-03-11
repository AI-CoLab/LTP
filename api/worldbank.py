"""World Bank / WEF GCI data fetcher.

Note: The Global Competitiveness Index (GCI) is published by the World Economic
Forum, not the World Bank. There is no public API for GCI scores — the bundled
CSV data is the only source. This module provides a validation/comparison fetch
from World Bank indicators as a cross-check.
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WB_API = "https://api.worldbank.org/v2"


def fetch_gdp_per_capita_ppp(
    iso3_list: list[str] | None = None,
    start_year: int = 2010,
    end_year: int = 2020,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch GDP per capita PPP from World Bank as a cross-check.

    Indicator: NY.GDP.PCAP.PP.KD (GDP per capita, PPP, constant 2017 international $)

    This is NOT used by the model directly — it's for diagnostics/validation only.
    """
    indicator = "NY.GDP.PCAP.PP.KD"
    country_str = ";".join(iso3_list) if iso3_list else "all"

    records = []
    page = 1
    while True:
        url = (
            f"{WB_API}/country/{country_str}/indicator/{indicator}"
            f"?date={start_year}:{end_year}&format=json&per_page=1000&page={page}"
        )
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if len(data) < 2 or not data[1]:
            break

        for entry in data[1]:
            if entry.get("value") is not None:
                records.append({
                    "iso3": entry["countryiso3code"],
                    "country_name": entry["country"]["value"],
                    "year": int(entry["date"]),
                    "gdp_per_capita_ppp": float(entry["value"]),
                })

        # Check pagination
        meta = data[0]
        if page >= meta.get("pages", 1):
            break
        page += 1

    if not records:
        raise ValueError("No data returned from World Bank API")

    df = pd.DataFrame(records).sort_values(["iso3", "year"]).reset_index(drop=True)
    logger.info("Fetched WB GDP/capita for %d countries", df["iso3"].nunique())
    return df


def fetch_gci_scores(iso3_list: list[str] | None = None):
    """GCI scores are NOT available via public API.

    The WEF Global Competitiveness Index must use bundled CSV data.
    This is by design — WEF does not provide a public API.
    """
    raise NotImplementedError(
        "GCI scores are published by the World Economic Forum and are not "
        "available via public API. Use bundled CSV data (ltp.data.load_gci_scores)."
    )
