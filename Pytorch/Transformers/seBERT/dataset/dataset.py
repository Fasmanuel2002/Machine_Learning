
import torch
from torch.utils.data import Dataset



class Dataset(Dataset):
    """A standard Torch Dataset for BERT-style data"""
    def __init__(self, encodings, labels_worth_automating, labels_enough_information):
        self.encodings = encodings
        self.labels_worth_automating = labels_worth_automating
        self.labels_enough_information = labels_enough_information

    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels_worth_automating"] = torch.tensor(self.labels_worth_automating[idx], dtype=torch.float32)
        item["labels_enough_information"] = torch.tensor(self.labels_enough_information[idx], dtype=torch.float32)
        return item
        

    def __len__(self):
        return len(self.encodings["input_ids"])
    
