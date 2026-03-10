from typing import Tuple

from data import *
import torch
from torch.nn.functional import pad
from torch import Tensor

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