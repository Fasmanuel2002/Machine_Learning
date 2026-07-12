from torch.utils.data import Dataset, DataLoader
import torch

class BraindumpDataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = [torch.tensor(seq, dtype=torch.long) for seq in X]
        self.Y = [torch.tensor(seq, dtype=torch.long) for seq in Y]
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]