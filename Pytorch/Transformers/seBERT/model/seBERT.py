import os

import torch
from src.classic_ml.dataset.Dataset import Dataset
from src.classic_ml.utils.compute_metrics import compute_metrics_multi_label
import pandas as pd
import numpy as np
from typing import List
from transformers import BertModel

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, output_channels)
        )

    def forward(self, x):
        return self.layers(x)
    

class seBERT_MultiTask(nn.Module):
    def __init__(self, encoder, dropout_prob=0.1):
        super(seBERT_MultiTask, self).__init__()
        self.encoder = encoder

        self.hidden_size = self.encoder.config.hidden_size
        
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(self.dropout_prob)

        self.head_worth_automating = nn.Linear(self.hidden_size, 1)
        
        self.head_enough_information = nn.Linear(self.hidden_size, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None, labels_worth_automating = None, labels_enough_information = None):
        output_encoder = self.encoder(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        # Representación del token [CLS]
        cls_output  = output_encoder.last_hidden_state[:, 0, :] 

        cls_output  = self.dropout(cls_output)

        logits_worth_automating = self.head_worth_automating(cls_output).squeeze(-1)

        logits_enough_information = self.head_enough_information(cls_output).squeeze(-1)

        loss = None
        if labels_worth_automating is not None and labels_enough_information is not None:
            loss_worth_automating = self.loss_fn(logits_worth_automating, labels_worth_automating.float())
            loss_enough_information = self.loss_fn(logits_enough_information, labels_enough_information.float())
            loss = loss_worth_automating + loss_enough_information
        
        return {
            "loss": loss,
            "logits_worth_automating": logits_worth_automating,
            "logits_enough_information": logits_enough_information
        }