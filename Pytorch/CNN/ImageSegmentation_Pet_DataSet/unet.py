import torch.nn as nn
import torchvision

class UNET(nn.Module):
    def __init__(self, n_classes : int = 21):
        super().__init__()
        
        #Encoder1
        self.e11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.e13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, padding=2)
        
        
        
        #Encoder2 
        self.e21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.e23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, padding=2)
        
        #Encoder3
        self.e31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.e33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.maxpool12 = nn.MaxPool2d(kernel_size=2, padding=2)
        
        
        #Decoder1 
        self.upconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.d11 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.d13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        #Decoder2 
        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.d21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.d23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        #Decoder3
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.d31 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.d33 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        
        out_conv2d = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1)
        
    def forward(self, x):
        ...
        
        
        
        
        
        
        
        
        