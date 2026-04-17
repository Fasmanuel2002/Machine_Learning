from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, accuracy_score
from typing import Tuple
import numpy as np

def compute_metrics_multi_label(p : Tuple[list, list]) -> dict:
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels , y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)

    recall_ma = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision_ma = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1_ma = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {
        "accuracy": accuracy,
        "recall_micro": recall,
        "precision_micro": precision,
        "f1_micro": f1,
        "mcc": mcc,
        "recall_macro": recall_ma,
        "precision_macro": precision_ma,
        "f1_macro": f1_ma
    }