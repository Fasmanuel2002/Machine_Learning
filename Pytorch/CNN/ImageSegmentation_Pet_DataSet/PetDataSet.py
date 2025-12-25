from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
from typing import Tuple

class PetDataSet(Dataset):
    def __init__(self, root_path, limit=None) -> None:
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([root_path + "./images/" + i for i in os.listdir(root_path + "/images/")])[:self.limit]
        self.masks = sorted([root_path + "./trimaps/" + i for i in os.listdir(root_path + "/trimaps/")])[:self.limit]
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        if self.limit is None:
            self.limit = len(self.images)
        
        
    def __getitem__(self, index) -> Tuple:
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")    
        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return min(len(self.images), self.limit)
    
    