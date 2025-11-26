import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import data_utils as du


def preprocess_data(df):

    # Remove unnecessary features
    
    dropfeatures = ['ID', 'Customer_ID', 'Name', 'SSN']

    df.drop(columns=dropfeatures, inplace=True)

    # Convert 'Credit_History_Age' to numerical format (in years)
    
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(du.convert_age_to_years)

    # Handle categorical features
    
    categorical_features = ['Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']

    # Identify numeric features
    numeric_features = [col for col in df.columns if col not in categorical_features]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    
    # Filtering negative and extreme values in Age column

    df = df[(df["Age"] > 0) & (df["Age"] < 60)].reset_index(drop=True)

    # Filtering negative values in Delay_from_due_date column

    df = df[df["Delay_from_due_date"] >= 0].reset_index(drop=True)

    return df


# if __name__ == '__main__':
#     # Provide a small script entrypoint for quick local testing
#     data = pd.read_csv('data/labelled_data.csv')
#     df = preprocess_data(data)
#     print('Preview of preprocessed data:')
#     print(df.head())
