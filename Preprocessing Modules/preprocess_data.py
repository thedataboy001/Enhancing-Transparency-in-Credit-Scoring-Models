import pandas as pd
import data_utils as du
from utils_and_constants import (
    DROP_COLNAME, TARGET_COLUMN, RAW_DATASET, 
    PROCESSED_DATASET, CATEGORICAL_COLS, NUMERICAL_COLS
)

def preprocess_data(df) -> pd.DataFrame:
    
    # Convert 'Credit_History_Age' to numerical format (in years)

    df['Credit_History_Age'] = df['Credit_History_Age'].apply(du.convert_age_to_years)

    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filtering negative and extreme values in Age column

    df = df[(df["Age"] > 0) & (df["Age"] < 60)].reset_index(drop=True)

    # Filtering negative values in Delay_from_due_date column

    df = df[df["Delay_from_due_date"] >= 0].reset_index(drop=True)


    df["Credit_Mix"] = df["Credit_Mix"].map({'Bad': 'Bad', 'Good': 'Good', 
                                             'Standard': 'Standard','_': 'NH'})
    
    df["Payment_Behaviour"] = df["Payment_Behaviour"].map({'Low_spent_Small_value_payments': 'LSSP',
    'Low_spent_Large_value_payments': 'LSLP', 'High_spent_Small_value_payments': 'HSSP',
    'High_spent_Large_value_payments': 'HSLP', '!@9#%8': 'NPB'})

    return df