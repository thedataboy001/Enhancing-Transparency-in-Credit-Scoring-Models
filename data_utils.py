import pandas as pd
import re

def convert_age_to_years(age_str):
    """
    Convert an age string in the format 'X years Y months' to a float representing total years.
    For example, '3 years 6 months' becomes 3.5.
    """
    
    if pd.isna(age_str):
        return None

    # Make it safe and consistent
    s = str(age_str).lower().strip()

    years = 0
    months = 0

    year_match = re.search(r'(\d+)\s*year', s)
    month_match = re.search(r'(\d+)\s*month', s)

    if year_match:
        years = int(year_match.group(1))
    if month_match:
        months = int(month_match.group(1))

    return round(years + (months / 12), 2)
