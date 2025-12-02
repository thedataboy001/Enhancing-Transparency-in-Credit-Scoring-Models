import warnings
warnings.filterwarnings("ignore")

from typing import List 
import pandas as pd

def load_data(file_path: str, drop_columns: List[str], 
              target_column: str) -> pd.DataFrame:

    """Load data from a CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True, axis=1)
    df[target_column] = df[target_column].map({"Standard": 2, "Poor": 1, "Good": 0})

    return df
   
   