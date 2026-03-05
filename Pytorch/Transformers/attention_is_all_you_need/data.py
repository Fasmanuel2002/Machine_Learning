from typing import Tuple, List
from torch.utils.data import random_split, DataLoader
import torch
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import os


def load_data() -> Tuple[List[str]]:
    with open("train.en", encoding="utf8") as f:
        english_data = f.read().splitlines()

    with open("train.de", encoding="utf8") as f:
        german_data = f.read().splitlines()
    
    assert len(english_data) == len(german_data), "Dataset mismatch" 

    return english_data, german_data


def load_tokenizers() -> Tuple:
    try:
        spacy_german = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_german = spacy.load("de_core_news_sm")
    try:    
        spacy_english = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_english = spacy.load("en_core_web_sm")
        
    return spacy_english, spacy_german
    

def tokenize(text : str, tokinizer ) -> list:
    return [tok.text for tok in tokinizer.tokenizer(text)]

def yield_tokens(data_iter, tokinizer, index):
    for from_to_tuple in data_iter:
        yield tokinizer(from_to_tuple[index])
        
    
def build_vocabulary(spacy_de , spacy_en):
    
    generator = torch.Generator().manual_seed(42)
    
    english_data, german_data = load_data()
    
    def tokenize_de(text):
        return tokenize(text, spacy_de)
    
    def tokenize_en(text):
        return tokenize(text, spacy_en)
    
    
    dataset_pairs = list(zip(english_data, german_data))
    
    data_set_size = len(dataset_pairs)
    
    train_size = int(0.80 * data_set_size)
    
    val_size = int(0.10 * data_set_size)
    
    test_size = data_set_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset=dataset_pairs,
                                                   lengths=[train_size, val_size, test_size],
                                                   generator=generator)
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train_data , tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )
    
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train_data , tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )
    
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    
    return vocab_src, vocab_tgt
    