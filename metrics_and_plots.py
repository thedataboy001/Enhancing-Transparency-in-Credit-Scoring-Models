import json

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(f"{model_name}_confusion_matrix.png")


def save_metrics(metrics, model_name: str):
    with open(f"{model_name}_metrics.json", "w") as fp:
        json.dump(metrics, fp)
