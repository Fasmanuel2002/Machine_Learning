from typing import Tuple

from data import tokenize, load_data
import torch
from torch.nn.functional import pad
from torch import Tensor
from torch.utils.data import random_split, DataLoader
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
device = torch.device("cude" if torch.cuda.is_available() else "cpu")

def collate_batch(batch, 
                  src_pipeline,
                  tgt_pipeline,
                  src_vocab,
                  tgt_vocab,
                  device,
                  max_padding : int = 128,
                  pad_id : int = 2) -> Tuple[Tensor, Tensor]:
    """
    This function prepares a batch of sentence pairs for the Transformer.
    1) Create tensors for special tokens:
    <s> (beginning of sentence) and </s> (end of sentence).
    2) Tokenize each sentence and convert tokens to vocabulary indices.
    3) Add <s> at the beginning and </s> at the end of each sequence.
    4) Apply padding so that all sequences in the batch have the same length
    (max_padding), using pad_id.
    5) Stack the sequences into tensors and return them with shape:

   (batch_size, max_padding)
    """
    begin_sentence_id  = torch.Tensor([0], device=device ) # # <s> token id, begin-sentence token id
    eos_id = torch.Tensor([1], device=device) #</s> token id, end-sentence token id
    src_list, tgt_list = [], []
    
    for (_src, _tgt) in batch:
       # Convert the source sentence into a tensor of token ids and add special tokens.
        # Steps:
        # 1) Tokenize the sentence (_src) -> ["I", "love", "transformers"]
        # 2) Convert tokens to vocabulary indices -> [4, 5, 6]
        # 3) Convert to a PyTorch tensor
        # 4) Concatenate <s> (beginning of sentence) at the start and </s> (end of sentence) at the end
        # Result example: ["<s>", "I", "love", "transformers", "</s>"] -> [0, 4, 5, 6, 1]
        
        processed_src = torch.cat([begin_sentence_id,torch.tensor(
            src_vocab(src_pipeline(_src)),
            dtype=torch.float,
            device=device),eos_id,], 0, )
        
        processed_tgt = torch.cat([
            begin_sentence_id,
            torch.tensor(
                tgt_vocab(tgt_pipeline(_tgt)),
                dtype=torch.float,
                device=device
            ),
            eos_id,
        ], 
        0,
        )

        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id
            )
        )
        
        tgt_list.append((
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
                )
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt) # (Batch_size , maximum_padding)

def create_dataloader(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size : int = 12000,
    max_padding : int = 128,
    is_distributed : bool = True) -> Tuple[DataLoader, DataLoader]:
    
    generator = torch.Generator().manual_seed(42)

    def tokenize_de(text):
        return tokenize(text, spacy_de)
    def tokenize_en(text):
        return tokenize(text, spacy_en)
    
    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )
        
    english_data, german_data = load_data()
        
    dataset_pairs = list(zip(english_data, german_data))
    
    data_set_size = len(dataset_pairs)
    train_size = int(0.80  * data_set_size)
    val_size = int(0.10  * data_set_size)
    test_size = data_set_size - train_size - val_size
    
    train_iter, valid_iter, test_iter  = random_split(dataset=dataset_pairs,
                                                   lengths=[train_size, val_size, test_size],
                                                   generator=generator)
    
    train_iter_map = to_map_style_dataset(
        train_iter
    )
    
    valid_iter_map = to_map_style_dataset(
        valid_iter
    )
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    
    valid_sampler = (
        DistributedSampler(valid_iter) if is_distributed else None
    )
    
    train_data_loader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    
    valid_data_loader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn
    )
    
    return train_data_loader, valid_data_loader