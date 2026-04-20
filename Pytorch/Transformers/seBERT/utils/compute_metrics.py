from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, accuracy_score
from typing import Tuple, Dict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.special import expit
import torch

def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred


    if isinstance(predictions, tuple):
        logits_worth = predictions[0]
        logits_enough = predictions[1]
    else:
        logits_worth = predictions[:, 0]
        logits_enough = predictions[:, 1]

    if isinstance(labels, tuple):
        y_worth = labels[0]
        y_enough = labels[1]
    else:
        y_worth = labels[:, 0]
        y_enough = labels[:, 1]

    print("=============TYPE =======================================")
    logits_worth = torch.tensor(logits_worth)
    logits_enough = torch.tensor(logits_enough)
    y_worth = torch.tensor(y_worth)
    y_enough = torch.tensor(y_enough)

    probs_worth = expit(logits_worth)


    probs_enough = expit(logits_enough)

    pred_worth = (probs_worth >= 0.5)

    pred_enough = (probs_enough >= 0.5)

    auroc_enough = roc_auc_score(y_enough, probs_enough)

    auprc_enough = average_precision_score(y_enough, probs_enough)

    f1_enough = f1_score(y_enough, pred_enough)

    macro_f1_enough = f1_score(y_enough, pred_enough, average="macro")

    auroc_worth_auto = roc_auc_score(y_worth, probs_worth)

    auprc_worth_auto = average_precision_score(y_worth, probs_worth)

    f1_worth_auto = f1_score(y_worth, pred_worth)

    macro_f1_worth_auto = f1_score(y_worth, pred_worth, average="macro")


    return {
        "auroc_enough_info": auroc_enough,
        "auprc_enough_info": auprc_enough,
        "f1_enough_info": f1_enough,
        "macro_f1_enough_info": macro_f1_enough,
        "auroc_worth_auto": auroc_worth_auto,
        "auprc_worth_auto": auprc_worth_auto,
        "f1_worth_auto": f1_worth_auto,
        "macro_f1_worth_auto": macro_f1_worth_auto
    }
    


