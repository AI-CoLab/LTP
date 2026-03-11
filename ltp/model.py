"""
Full model pipeline: Hubbard & Sharma (2016) GDP projection model.

Orchestrates the complete projection workflow:
1. Load data
2. Compute historical productivity
3. Kernel regression → estimate φᵢ for transitioning countries
4. Project productivity year-by-year
5. Compute GDP and per-capita GDP
"""

import numpy as np
import pandas as pd

from .kernel import kernel_estimate, find_optimal_bandwidth
from .convergence import (
    us_productivity_path,
    project_country_productivity,
    project_gdp,
    project_gdp_per_capita,
    decompose_growth,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_US_GROWTH,
)
from .data import get_model_inputs


class HubbardSharmaModel:
    """Full Hubbard & Sharma GDP projection model."""

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        us_growth: float = DEFAULT_US_GROWTH,
        bandwidth: float | None = None,
        projection_start: int = 2021,
        projection_end: int = 2050,
    ):
        self.beta = beta
        self.gamma = gamma
        self.us_growth = us_growth
        self.bandwidth = bandwidth  # None = auto-compute
        self.projection_start = projection_start
        self.projection_end = projection_end
        self.n_proj_years = projection_end - projection_start + 1

        # Populated by fit()
        self.data = None
        self.ss_gci = None
        self.ss_phi = None
        self.optimal_bandwidth = None
        self.phi_estimates = None
        self.projections = None

    def fit(self, data_dir=None, gci_overrides: dict | None = None, use_live_api: bool = False):
        """Run the full model pipeline.

        Args:
            data_dir: Optional path to data directory
            gci_overrides: Dict of {iso3: new_gci_score} for scenario analysis.
                Countries with overridden GCI are treated as non-steady-state
                so the convergence equation applies to their new target.
            use_live_api: If True, fetch GDP/population from live APIs
        """
        # 1. Load data
        self.data = get_model_inputs(data_dir, use_live_api=use_live_api)
        ss_df = self.data["steady_state"].copy()
        gci_df = self.data["gci"].copy()
        prod_df = self.data["productivity"]
        pop_df = self.data["population"]

        # Apply GCI overrides for scenario analysis
        # Countries with overridden GCI are forced to non-steady-state
        # so β > 0 applies and they converge to the new kernel-estimated φ
        override_codes = set()
        if gci_overrides:
            for iso3, new_gci in gci_overrides.items():
                mask = gci_df["iso3"] == iso3
                if mask.any():
                    gci_df.loc[mask, "gci_score"] = new_gci
                    override_codes.add(iso3)
            # Remove overridden countries from the steady-state set
            ss_df.loc[ss_df["iso3"].isin(override_codes), "is_steady_state"] = False

        # 2. Get steady-state countries' GCI and φ
        ss_countries = ss_df[ss_df["is_steady_state"] == True]
        ss_with_gci = pd.merge(ss_countries, gci_df[["iso3", "gci_score"]], on="iso3", how="inner")
        self.ss_gci = ss_with_gci["gci_score"].values
        self.ss_phi = ss_with_gci["relative_productivity"].values

        # 3. Kernel regression - find optimal bandwidth
        if self.bandwidth is None:
            self.optimal_bandwidth = find_optimal_bandwidth(self.ss_gci, self.ss_phi)
        else:
            self.optimal_bandwidth = self.bandwidth

        # 4. Estimate φ for all countries
        all_countries = gci_df[["iso3", "country_name", "gci_score"]].copy()
        all_countries["phi_estimated"] = kernel_estimate(
            self.ss_gci, self.ss_phi, all_countries["gci_score"].values,
            self.optimal_bandwidth,
        )

        # For steady-state countries, use their actual relative productivity
        for _, row in ss_countries.iterrows():
            mask = all_countries["iso3"] == row["iso3"]
            if mask.any():
                all_countries.loc[mask, "phi_estimated"] = row["relative_productivity"]

        all_countries["is_steady_state"] = all_countries["iso3"].isin(
            ss_countries["iso3"].values
        )
        self.phi_estimates = all_countries

        # 5. Project US productivity path
        us_data = prod_df[prod_df["iso3"] == "USA"].sort_values("year")
        us_base_year = us_data[us_data["year"] == self.projection_start - 1]
        if us_base_year.empty:
            us_base_year = us_data.iloc[-1:]

        us_prod_base = us_base_year["labour_productivity"].values[0]

        # Get US historical growth for momentum
        us_hist = us_data[us_data["year"] <= self.projection_start - 1]["labour_productivity"].values
        if len(us_hist) >= 2:
            us_hist_growth = np.log(us_hist[-1] / us_hist[-2])
        else:
            us_hist_growth = self.us_growth

        us_prod_path = us_productivity_path(
            us_prod_base, self.us_growth, self.n_proj_years,
            self.gamma, us_hist_growth,
        )

        # 6. Project each country
        proj_years = np.arange(self.projection_start, self.projection_end + 1)
        results = []

        for _, country_row in all_countries.iterrows():
            iso3 = country_row["iso3"]
            country_name = country_row["country_name"]
            phi = country_row["phi_estimated"]
            is_ss = country_row["is_steady_state"]

            # Get historical productivity (need at least 2 years)
            c_prod = prod_df[
                (prod_df["iso3"] == iso3) & (prod_df["year"] <= self.projection_start - 1)
            ].sort_values("year")

            if len(c_prod) < 2:
                continue

            prod_history = c_prod["labour_productivity"].values

            # Project productivity
            projected_prod = project_country_productivity(
                prod_history, us_prod_path, phi,
                self.beta, self.gamma, is_ss,
            )

            # Get population projections
            c_pop = pop_df[
                (pop_df["iso3"] == iso3) &
                (pop_df["year"] >= self.projection_start - 1) &
                (pop_df["year"] <= self.projection_end)
            ].sort_values("year")

            if len(c_pop) < self.n_proj_years:
                continue

            wap = c_pop["working_age_pop_thousands"].values * 1e3  # to actual count
            total_pop = c_pop["total_pop_thousands"].values * 1e3

            # Compute GDP
            gdp = project_gdp(projected_prod, wap)
            gdp_per_capita = project_gdp_per_capita(gdp, total_pop)

            # Store yearly projections
            # Include historical base year + projection years
            years_out = np.arange(self.projection_start - 1, self.projection_end + 1)
            for i, yr in enumerate(years_out):
                if i < len(projected_prod):
                    results.append({
                        "iso3": iso3,
                        "country_name": country_name,
                        "year": yr,
                        "labour_productivity": projected_prod[i],
                        "working_age_pop": wap[i] if i < len(wap) else np.nan,
                        "total_pop": total_pop[i] if i < len(total_pop) else np.nan,
                        "gdp_billions": gdp[i] / 1e9 if i < len(gdp) else np.nan,
                        "gdp_per_capita": gdp_per_capita[i] if i < len(gdp_per_capita) else np.nan,
                        "relative_productivity": (
                            projected_prod[i] / us_prod_path[i] if i < len(us_prod_path) else np.nan
                        ),
                        "phi": phi,
                        "is_steady_state": is_ss,
                        "is_projection": yr >= self.projection_start,
                    })

        self.projections = pd.DataFrame(results)
        return self

    def get_country_projection(self, iso3: str) -> pd.DataFrame:
        """Get projection results for a single country."""
        if self.projections is None:
            raise RuntimeError("Call fit() first")
        return self.projections[self.projections["iso3"] == iso3].copy()

    def get_top_economies(self, year: int = 2050, n: int = 7) -> pd.DataFrame:
        """Get the N largest economies by GDP in a given year."""
        if self.projections is None:
            raise RuntimeError("Call fit() first")
        year_data = self.projections[self.projections["year"] == year]
        return year_data.nlargest(n, "gdp_billions")

    def get_gdp_trajectories(self, iso3_list: list[str]) -> pd.DataFrame:
        """Get GDP trajectories for selected countries."""
        if self.projections is None:
            raise RuntimeError("Call fit() first")
        return self.projections[self.projections["iso3"].isin(iso3_list)].copy()

    def get_regional_shares(self, region_mapping: dict[str, list[str]]) -> pd.DataFrame:
        """Compute regional GDP shares over time.

        Args:
            region_mapping: Dict of {region_name: [iso3_codes]}
        """
        if self.projections is None:
            raise RuntimeError("Call fit() first")

        records = []
        for year in self.projections["year"].unique():
            year_data = self.projections[self.projections["year"] == year]
            total_gdp = year_data["gdp_billions"].sum()

            for region, codes in region_mapping.items():
                region_gdp = year_data[year_data["iso3"].isin(codes)]["gdp_billions"].sum()
                records.append({
                    "year": year,
                    "region": region,
                    "gdp_billions": region_gdp,
                    "share_pct": (region_gdp / total_gdp * 100) if total_gdp > 0 else 0,
                })

        return pd.DataFrame(records)

    def get_growth_decomposition(self, iso3: str) -> pd.DataFrame:
        """Get decade-average growth decomposition for a country."""
        if self.projections is None:
            raise RuntimeError("Call fit() first")

        country = self.projections[self.projections["iso3"] == iso3].sort_values("year")
        if len(country) < 2:
            return pd.DataFrame()

        # Also include historical data
        hist = self.data["productivity"]
        hist_country = hist[hist["iso3"] == iso3].sort_values("year")

        # Combine historical + projected productivity
        all_prod = pd.concat([
            hist_country[["year", "labour_productivity"]],
            country[country["is_projection"]][["year", "labour_productivity"]],
        ]).drop_duplicates("year").sort_values("year")

        # Get combined population
        pop = self.data["population"]
        c_pop = pop[pop["iso3"] == iso3].sort_values("year")

        merged = pd.merge(all_prod, c_pop[["year", "working_age_pop_thousands"]], on="year")
        merged["working_age_pop"] = merged["working_age_pop_thousands"] * 1e3

        if len(merged) < 2:
            return pd.DataFrame()

        decomp = decompose_growth(
            merged["labour_productivity"].values,
            merged["working_age_pop"].values,
        )

        years = merged["year"].values[1:]
        decades = [
            (1991, 2000), (2001, 2010), (2011, 2020),
            (2021, 2030), (2031, 2040), (2041, 2050),
        ]

        records = []
        for start, end in decades:
            mask = (years >= start) & (years <= end)
            if mask.any():
                records.append({
                    "period": f"{start}-{end}",
                    "gdp_growth": np.mean(decomp["gdp_growth"][mask]) * 100,
                    "productivity_growth": np.mean(decomp["productivity_growth"][mask]) * 100,
                    "workforce_growth": np.mean(decomp["workforce_growth"][mask]) * 100,
                })

        return pd.DataFrame(records)
