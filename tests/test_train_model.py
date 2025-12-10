import pandas as pd
from train_model import load_data
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN

def test_load_data(tmp_path):
    # Create a small dummy dataset
    df = pd.DataFrame({
        "age": [25, 30, 45],
        "income": [50000, 60000, 80000],
        TARGET_COLUMN: [0, 1, 0]
    })

    # Write it to a temporary CSV
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    # Run the function
    X, y = load_data(file_path)

    # Assertions
    # y should equal the target column
    pd.testing.assert_series_equal(y, df[TARGET_COLUMN])

    # X should match df without the target column
    pd.testing.assert_frame_equal(X, df.drop(columns=[TARGET_COLUMN]))

    # Ensure the shapes match what we expect
    assert X.shape[1] == len(df.columns) - 1
    assert len(y) == len(df)