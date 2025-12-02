import warnings
warnings.filterwarnings("ignore")

from typing import List 
import pandas as pd
import data_utils as du
from utils_and_constants import (
    DROP_COLNAME, TARGET_COLUMN, RAW_DATASET, 
    PROCESSED_DATASET, CATEGORICAL_COLS, NUMERICAL_COLS
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(file_path: str, drop_columns: List[str], 
              target_column: str) -> pd.DataFrame:

    """Load data from a CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True, axis=1)
    df[target_column] = df[target_column].map({"Standard": 2, "Poor": 1, "Good": 0})

    return df

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


def transformed_data(df: pd.DataFrame, target_column: str, 
                     categorical_cols: List[str], numerical_cols: List[str]) ->  pd.DataFrame:

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X = df.drop(columns=target_column, axis=1)

    X_preprocessed = preprocessor.fit_transform(X)

    # Get the preprocessed feature names
    feature_names = preprocessor.get_feature_names_out()
    
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names, index=df.index)

    return X_preprocessed_df


def main():
    # Load raw data
    df = load_data(
        file_path=RAW_DATASET, drop_columns=DROP_COLNAME, target_column=TARGET_COLUMN
    )

    # Preprocess data
    df = preprocess_data(df)

    # transformed X data
    X_processed = transformed_data(
        df=df, target_column=TARGET_COLUMN, 
        categorical_cols=CATEGORICAL_COLS, numerical_cols=NUMERICAL_COLS
    )

    # processed dataset
    Credit_Score = df[TARGET_COLUMN].astype(int)

    df_processed = pd.concat([X_processed, Credit_Score], axis=1)

    # Save processed data
    df_processed.to_csv(PROCESSED_DATASET, index=False)

if __name__ == "__main__":
    main()