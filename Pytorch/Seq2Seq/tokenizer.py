import re

class Tokenizer():
    def __init__(self, corpus) -> None:
        corpus = corpus.lower()
        
        
        words = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)
        vocab = sorted(set([word.strip() for word in words if word.strip() != '']))
        
        
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        vocab = self.special_tokens + vocab
        
        
        self.text_to_token_ids = {word: i for i, word in enumerate(vocab)}
        self.token_ids_to_token = {i: word for word, i in self.text_to_token_ids.items()}

        
        self.pad_token_id = self.text_to_token_ids["<pad>"]
        self.sos_token_id = self.text_to_token_ids["<sos>"]
        self.eos_token_id = self.text_to_token_ids["<eos>"]
        self.unk_token_id = self.text_to_token_ids["<unk>"]

    def encode(self, text):
        text = text.lower()
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        token_ids = [
            self.text_to_token_ids.get(word.strip(), self.unk_token_id)
            for word in words if word.strip() != ''
        ]
        
        
        token_ids.append(self.eos_token_id)
        
        return token_ids        
    
    def decode(self, token_ids):
        # 1. Blindaje definitivo: si es un tensor, lo aplanamos a 1D y lo pasamos a lista
        if hasattr(token_ids, "flatten"):
            token_ids = token_ids.flatten().tolist()
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
            
        words = []
        for token in token_ids:
            # Si encontramos el Fin de Secuencia, cortamos
            if token == self.eos_token_id:
                break
                
            # Si no es un token especial de relleno o inicio, lo añadimos
            if token not in [self.pad_token_id, self.sos_token_id]:
                words.append(self.token_ids_to_token.get(token, "<unk>"))
                
        raw_text = " ".join(words)
        clean_text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', raw_text)
        return clean_text
        
    def get_vocab_size(self):
        return len(self.text_to_token_ids)