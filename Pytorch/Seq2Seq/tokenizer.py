import re
import pandas    
"""
Core job of the tokenizer

Break text into manageable pieces (tokens).
Map each token to a numerical ID (so models can understand it).
Convert numbers back to text when needed.

Encoding: The text is broken down into sub-word tokens and mapped to unique IDs.
Decoding: Those IDs are converted back into the original text.
"""

class Tokenizer():
    def __init__(self, corpus) -> None:
        
        corpus = corpus.lower()
        
        # Dictionaries for mapping words <-> token ids
        self.text_to_token_ids = {}
        self.token_ids_to_token = {}
        
        # Split corpus into words and punctuation while keeping punctuation as separate tokens
        # Example: "Amazing!" → ["Amazing", "!"]
        words = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)
        
        #Remove empty strings and build vocabulary (unique words/punctuations)
        vocab = sorted(set([word.strip() for word in words if word.strip() != '']))
        
        # Add special tokens
        vocab.append("<unk>")
        vocab.append("<|endoftext|>")
        
        
        #Create the mappings
        # word -> token_id
        self.text_to_token_ids = {word: i for i, word in enumerate(vocab)}

        
        # token_id -> word
        self.token_ids_to_token = {i: word for word, i in self.text_to_token_ids.items()}

        # Save the unknown token ID for fallback
        self.unk_token_id = self.text_to_token_ids["<unk>"]

    
    def encode(self, text):
        """
        Converts text into token IDs.
        Steps:
        1. Split the text into words + punctuation (keeping punctuation as tokens).
        2. Map each word/punctuation to its token ID using our dictionary.
        """
        text = text.lower()
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        words = [word.strip() for word in words if word.strip() != '']
        
        token_ids = [
            self.text_to_token_ids.get(word, self.unk_token_id)
            for word in words if word != ''
        ]
        return token_ids        
    
    def decode(self, token_ids):
        """
        Converts token IDs back into text.
        Steps:
        1. Convert each token ID into its corresponding word/punctuation.
        2. Join them back together into a single string.
        """
        
    
        words = [self.token_ids_to_token.get(token, "<unk>") for token in token_ids]
        
        return " ".join(words)
        
    def get_vocab_size(self):
        return len(self.text_to_token_ids)