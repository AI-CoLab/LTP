"""IMF World Economic Outlook API data fetcher (stub)."""


def fetch_gdp_ppp(iso3_list: list[str], start_year: int = 1980, end_year: int = 2020):
    """Fetch GDP in PPP terms from IMF WEO database.

    Not yet implemented — use bundled CSV data instead.
    """
    raise NotImplementedError(
        "Live IMF data fetching is not yet implemented. "
        "Use ltp.data.load_gdp() for bundled CSV data."
    )
