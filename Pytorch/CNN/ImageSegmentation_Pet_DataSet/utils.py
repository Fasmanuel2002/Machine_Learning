import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


def extract_photo(filename: str, number_to_Show : int):
    path = '' 
    image_path = os.path.join(path, filename)
    image_list_orig = os.listdir(image_path)
    image_list = [image_path + i for i in image_list_orig]
    img = Image.open(image_list[number_to_Show])
    return img
        
        
    