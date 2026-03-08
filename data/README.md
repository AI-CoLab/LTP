# Data Files

Bundled CSV data files for the Hubbard & Sharma (2016) GDP projection model.

## Files

- **steady_state_countries.csv**: Classification of 140 countries as steady-state (55) or transitioning (85), with relative productivity levels
- **gci_scores.csv**: Global Competitiveness Index scores from WEF 2015-16 Report for 140 countries
- **imf_gdp.csv**: Historical GDP in billions of 2012 PPP USD (1980-2020), based on IMF WEO data
- **un_population.csv**: Working-age (15-64) and total population in thousands (1980-2050), based on UN medium-variant projections

## Data Sources

- **GDP**: IMF World Economic Outlook, October 2015
- **GCI**: World Economic Forum Global Competitiveness Report 2015-16
- **Population**: United Nations Population Division, 2015 Revision (medium variant)
- **Steady-state classification**: Au-Yeung et al. (2013), Australian Treasury Working Paper 2013-02

## Notes

These are representative datasets constructed to match the paper's 2015-era data vintage.
For exact replication, users should source original data from the respective agencies.
The `api/` directory contains optional scripts to pull live data from IMF, World Bank, and UN APIs.
