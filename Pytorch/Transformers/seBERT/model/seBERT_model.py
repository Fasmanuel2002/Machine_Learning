from importlib.resources import path
import os

import torch
from src.classic_ml.dataset.Dataset import Dataset
from src.classic_ml.utils.compute_metrics import compute_metrics
from src.classic_ml.models.seBert import seBERT_MultiTask
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from transformers import TrainingArguments, Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class seBERT(BaseEstimator, ClassifierMixin):
    def __init__(self, checkpoints_dir = "../checkpoints/", batch_size : int = 16):
        self.trainer = None
        self.checkpoints_dir = checkpoints_dir
        self.model = seBERT_MultiTask(BertModel.from_pretrained("../src/classic_ml/models/seBERT")).to(device)
        self.tokenizer = BertTokenizer.from_pretrained("../src/classic_ml/models/seBERT",do_lower_case=True)
        self.batch_size = batch_size
        self.max_length = 128

    def fit(self, X , y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        X_train_tokens = self.tokenizer(X_train, padding=True, truncation=True, max_length=self.max_length)
        
        X_val_tokens = self.tokenizer(X_val, padding=True, truncation=True, max_length=self.max_length)
        
        y_label_worth_automating_train = y_train["worth_automating"].values
        y_label_enough_information_train = y_train["enough_information"].values
        
        
        y_label_worth_automating_val = y_val["worth_automating"].values
        y_label_enough_information_val = y_val["enough_information"].values

        train_dataset = Dataset(X_train_tokens, y_label_worth_automating_train, y_label_enough_information_train ) 
        val_dataset = Dataset(X_val_tokens, y_label_worth_automating_val, y_label_enough_information_val)
        

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        training_args = TrainingArguments(
            output_dir=self.checkpoints_dir,
            num_train_epochs=4,
            logging_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=10,
            label_names=["labels_worth_automating", "labels_enough_information"],
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            weight_decay=0.1,
            bf16=True

        )
        self.trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,   
        )

        print(self.trainer.train())
        self.save_new_model("../checkpoints/final_model")  # ← Aquí
        return self
    
    def predict_proba(self, X , y = None)  -> Tuple[list, list]:
        y_probs_worth_automating = []
        y_probs_enough = []
        self.trainer.model.eval()
        with torch.no_grad():
            for _, X_row in enumerate(X):
                inputs = self.tokenizer(X_row, padding =True, truncation = True, max_length = self.max_length, return_tensors = "pt").to(device)
                outputs = self.trainer.model(**inputs)
                logits_worth_automating = torch.sigmoid(outputs["logits_worth_automating"])
                logits_enough_information = torch.sigmoid(outputs["logits_enough_information"])
                y_probs_worth_automating.append(logits_worth_automating.item())
                y_probs_enough.append(logits_enough_information.item())

        return [y_probs_worth_automating, y_probs_enough]
    
    def predict(self, X , y = None) -> Tuple[list, list]:
        """Predict is evaluation of the model"""
        y_probs_worth_automating, y_probs_enough = self.predict_proba(X, y=y)
        
        y_pred_worth_automating, y_pred_enough = [], []
        for y_prob_worth_automating, y_prob_enough in zip(y_probs_worth_automating, y_probs_enough):
            y_pred_worth_automating.append(int(y_prob_worth_automating >= 0.5))
            y_pred_enough.append(int(y_prob_enough >= 0.5))

        return y_pred_worth_automating, y_pred_enough
    
    def save_new_model(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        state_dict = {
            k: v.detach().cpu().contiguous()
            for k, v in self.trainer.model.state_dict().items()
        }

        torch.save(state_dict, os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)