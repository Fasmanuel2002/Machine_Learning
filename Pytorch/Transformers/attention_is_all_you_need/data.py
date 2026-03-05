from typing import Tuple
from torch.utils.data import random_split, DataLoader
import torch

def split_data_Train_Val_Test(batch_size : int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:


    generator = torch.Generator().manual_seed(42)
    
    with open("train.en", encoding="utf8") as f:
        english_data = f.read().splitlines()

    with open("train.de", encoding="utf8") as f:
        german_data = f.read().splitlines()

    print(len(english_data), len(german_data))
    
    assert len(english_data) == len(german_data), "Dataset mismatch" 
    
    dataset_pairs = list(zip(english_data, german_data))
    
    data_set_size = len(dataset_pairs)
    
    train_size = int(0.80 * data_set_size)
    
    val_size = int(0.10 * data_set_size)
    
    test_size = data_set_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset=dataset_pairs,
                                                   lengths=[train_size, val_size, test_size],
                                                   generator=generator)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader