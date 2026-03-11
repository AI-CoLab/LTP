"""IMF World Economic Outlook API data fetcher.

Fetches GDP PPP data from the IMF DataMapper API.
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# IMF DataMapper API for WEO data
# PPPGDP = GDP based on PPP (current international $, billions)
BASE_URL = "https://www.imf.org/external/datamapper/api/v1"


def fetch_gdp_ppp(
    iso3_list: list[str] | None = None,
    start_year: int = 1980,
    end_year: int = 2020,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch GDP in PPP terms from IMF WEO DataMapper API.

    Returns DataFrame with columns matching bundled CSV:
        iso3, country_name, year, gdp_billions_2012ppp

    Note: IMF API returns current international $, not constant 2012 PPP.
    The values will differ from the bundled data which uses 2012 PPP base.
    """
    # Fetch PPPGDP indicator (GDP, current prices, PPP, billions)
    url = f"{BASE_URL}/PPPGDP"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data or "PPPGDP" not in data["values"]:
        raise ValueError("Unexpected IMF API response format")

    gdp_data = data["values"]["PPPGDP"]

    # Fetch country names
    countries_url = f"{BASE_URL}/countries"
    countries_resp = requests.get(countries_url, timeout=timeout)
    countries_resp.raise_for_status()
    countries_data = countries_resp.json()

    country_names = {}
    if "countries" in countries_data:
        for code, info in countries_data["countries"].items():
            if isinstance(info, dict) and "label" in info:
                country_names[code] = info["label"]

    records = []
    for iso3, year_values in gdp_data.items():
        if iso3_list and iso3 not in iso3_list:
            continue
        if not isinstance(year_values, dict):
            continue

        name = country_names.get(iso3, iso3)
        for year_str, value in year_values.items():
            try:
                year = int(year_str)
            except (ValueError, TypeError):
                continue

            if start_year <= year <= end_year and value is not None:
                records.append({
                    "iso3": iso3,
                    "country_name": name,
                    "year": year,
                    "gdp_billions_2012ppp": float(value),
                })

    if not records:
        raise ValueError("No GDP data returned from IMF API")

    df = pd.DataFrame(records).sort_values(["iso3", "year"]).reset_index(drop=True)
    logger.info("Fetched GDP data for %d countries from IMF API", df["iso3"].nunique())
    return df
