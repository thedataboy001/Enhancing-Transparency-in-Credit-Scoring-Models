import json
import pandas as pd
from sklearn.model_selection import train_test_split
from metrics_and_plots import plot_confusion_matrix, save_metrics
from rf_model import train_random_forest, evaluate_random_forest 
from xgb_model import train_xgboost , evaluate_xgboost
from dt_model import train_decision_tree , evaluate_decision_tree
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y

def main():
    X, y = load_data(PROCESSED_DATASET)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train the rf_model using the training set
    rf_model = train_random_forest(X_train, y_train)
    
    # Calculate test set metrics
    rf_model_metrics = evaluate_random_forest(rf_model, X_test, y_test)

    print("====================Random Forest Test Set Metrics==================")
    print(json.dumps(rf_model_metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(rf_model_metrics, model_name="rf_model")
    plot_confusion_matrix(rf_model, X_test, y_test, model_name="rf_model")


    # Train the xgb_model using the training set
    xgb_model = train_xgboost(X_train, y_train)
    
    # Calculate test set metrics
    xgb_model_metrics = evaluate_xgboost(xgb_model, X_test, y_test)
    print("====================XGBoost Test Set Metrics==================")
    print(json.dumps(xgb_model_metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(xgb_model_metrics, model_name="xgb_model")
    plot_confusion_matrix(xgb_model, X_test, y_test, model_name="xgb_model")

    # Train the dt_model using the training set
    dt_model = train_decision_tree(X_train, y_train)
    # Calculate test set metrics
    dt_model_metrics = evaluate_decision_tree(dt_model, X_test, y_test)
    print("====================Decision Tree Test Set Metrics==================")
    print(json.dumps(dt_model_metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(dt_model_metrics, model_name="dt_model")
    plot_confusion_matrix(dt_model, X_test, y_test, model_name="dt_model")

if __name__ == "__main__":
    main()