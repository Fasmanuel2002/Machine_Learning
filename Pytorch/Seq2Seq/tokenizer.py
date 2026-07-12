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
        #Define the special Tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>" #Unknown tokens
        self.SOS_TOKEN = "<SOS>" #Start of the Sequence
        self.EOS_TOKEN = "<EOS>" #End of the Sequence
        
        special_tokens = special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        corpus = corpus.lower()
        
        # Dictionaries for mapping words <-> token ids
        self.text_to_token_ids = {}
        self.token_ids_to_token = {}
        
        # Split corpus into words and punctuation while keeping punctuation as separate tokens
        # Example: "Amazing!" → ["Amazing", "!"]
        words = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)
        
        #Remove empty strings and build vocabulary (unique words/punctuations)
        vocab = sorted(set([word.strip() for word in words if word.strip() != '']))
        
        final_vocab = special_tokens + vocab
        
        #Create the mappings
        # word -> token_id
        self.text_to_token_ids = {unique_word : i for i, unique_word in enumerate(final_vocab)}
        
        # token_id -> word
        self.token_ids_to_token = {i : unique_word for i, unique_word in enumerate(final_vocab)}
    
    def encode(self, text, add_special_tokens : bool = False):
        """
        Converts text into token IDs.
        Steps:
        1. Split the text into words + punctuation (keeping punctuation as tokens).
        2. Map each word/punctuation to its token ID using our dictionary.
        """
        text = text.lower()
        words = re.split(r'([,.:;?_!"()\']|--)', text)
        words = [word.strip() for word in words if word.strip() != '']
        
        token_ids = [self.text_to_token_ids.get(word, self.text_to_token_ids[self.UNK_TOKEN]) for word in words if word != '']
        
        if add_special_tokens:
            #Addid the Start of the Sentence and the final of the sentence
            token_ids = [self.text_to_token_ids[self.SOS_TOKEN]] + token_ids + [self.text_to_token_ids[self.EOS_TOKEN]]
        return token_ids
    
    def decode(self, tokens_ids):
        """
        Converts token IDs back into text.
        Steps:
        1. Convert each token ID into its corresponding word/punctuation.
        2. Join them back together into a single string.
        """
        
        words = [self.token_ids_to_token[token] for token in tokens_ids 
                 if token not in [self.text_to_token_ids[self.PAD_TOKEN], 
                                  self.text_to_token_ids[self.SOS_TOKEN], 
                                  self.text_to_token_ids[self.EOS_TOKEN]]]
        
        return " ".join(words)
        
    def get_vocab_size(self):
        return len(self.text_to_token_ids)