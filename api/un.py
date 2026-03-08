"""UN Population Division data fetcher (stub)."""


def fetch_population_projections(iso3_list: list[str], start_year: int = 2020, end_year: int = 2050):
    """Fetch population projections from UN Population Division.

    Not yet implemented — use bundled CSV data instead.
    """
    raise NotImplementedError(
        "Live UN data fetching is not yet implemented. "
        "Use ltp.data.load_population() for bundled CSV data."
    )
