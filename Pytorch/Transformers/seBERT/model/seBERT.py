import os

import torch
from src.classic_ml.dataset.Dataset import Dataset
from src.classic_ml.utils.compute_metrics import compute_metrics_multi_label
import pandas as pd
import numpy as np
from typing import List

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class seBERT(BaseEstimator, ClassifierMixin):
    def __init__(self, checkpoints_dir = "../checkpoints/", batch_size : int = 16):
        self.trainer = None
        self.checkpoints_dir = checkpoints_dir
        self.model = BertForSequenceClassification.from_pretrained("../src/classic_ml/models/seBERT",num_labels=3, local_files_only=True).to(device)
        self.tokenizer = BertTokenizer.from_pretrained("../src/classic_ml/models/seBERT",do_lower_case=True)
        self.batch_size = batch_size
        self.max_length = 128

    def fit(self, X , y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        X_train_tokens = self.tokenizer(X_train, padding=True, truncation=True, max_length=self.max_length)
        
        X_val_tokens = self.tokenizer(X_val, padding=True, truncation=True, max_length=self.max_length)

        train_dataset = Dataset(X_train_tokens, y_train) 
        val_dataset = Dataset(X_val_tokens, y_val)
        
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        training_args = TrainingArguments(
            output_dir=self.checkpoints_dir,
            num_train_epochs=5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_multi_label
        )
        print(self.trainer.train())
        return self
    
    def predict_proba(self, X , y = None)  -> List[np.ndarray]:
        y_probs = []
        self.trainer.model.eval()
        with torch.no_grad():
            for _, X_row in enumerate(X):
                inputs = self.tokenizer(X_row, padding =True, truncation = True, max_length = self.max_length, return_tensors = "pt").to(device)
                outputs = self.trainer.model(**inputs)
                probs = outputs[0].softmax(dim=1).cpu().detach().numpy()
                y_probs.append(probs)
        
        return y_probs
    def predict(self, X , y = None) -> List[np.ndarray]:
        """Predict is evaluation of the model"""
        y_probs = self.predict_proba(X, y=y)
        y_pred = []
        for y_prob in y_probs:
            y_pred.append(y_prob.argmax())
        
        return y_pred
    
    def save_new_mode(self, path : str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.model.save_pretrained(path)