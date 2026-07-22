from torch import nn

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self):
        ...


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
    def forward(self):
            ...


class Seq2Seq(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
    def forward(self):
            ...

