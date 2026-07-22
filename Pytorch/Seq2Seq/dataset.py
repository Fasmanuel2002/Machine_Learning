from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence


class BraindumpDataset(Dataset):
    def __init__(self, df):
        self.X = df['input_ids'].tolist()
        self.Y = df['target_ids'].tolist()
        
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)
    

def split_dataset(dataset : BraindumpDataset, batch_size : int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]: 
    """
    Split the dataset into three DataLoaders:
    - Training set: 80% of the full dataset
    - Validation set: 10% of the full dataset
    - Test set: 10% of the full dataset
    """
    
    generator = torch.Generator().manual_seed(42)
    
    dataset_size = len(dataset)
    
    train_size = int(0.80 * dataset_size)
    
    val_size = int(0.10 * dataset_size)
    
    test_size = dataset_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset=dataset, lengths=[train_size, val_size, test_size],generator=generator)
    
    #TRAIN SET
    train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate # type: ignore
    )
    #VALIDATION SET
    validation_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate # type: ignore
    )

    #TEST SET
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate # type: ignore
    )
    
    return train_loader, validation_loader, test_loader


def custom_collate(data : BraindumpDataset):
    
    inputs, targets = zip(*data)
    
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0) # type: ignore
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0) # type: ignore
    
    return { #(6)
        'tokenized_input': inputs_padded,
        'label': targets_padded
    }
    