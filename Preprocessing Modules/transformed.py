import warnings
warnings.filterwarnings("ignore")

from typing import List 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


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