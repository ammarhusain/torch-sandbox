import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SmallAutoEncoder(nn.Module):
  """
  Simple & small auto encoder built for MNIST images (28,28)
  Most sizes are hardcoded in there and produce a 2 class output. 
  Convolution blocks couls perhaps be more generalized eventually.
  """
  def __init__(self):
    super().__init__()

    self.name = "SmallAutoEncoder"
    
    self.n_classes = 2
    
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),\
      # why is padding so large?
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 8, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    ) # [1, 28,28] -> [8,14,14]
    
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 16, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    ) # [8,14,14] -> [16,7,7]
    
    self.conv_block_3 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1),
      nn.ReLU(inplace=True),      
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    ) # [16,7,7] -> [32,4,4] 
    
    self.conv_block_4 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),      
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    ) # [32,4,4] -> [64,2,2] 
    
    self.up_conv_block = nn.Sequential(
      nn.ConvTranspose2d(in_channels= 64, out_channels=32, kernel_size=5, stride=1), 
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(in_channels= 32, out_channels=16, kernel_size=5, stride=1),  
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(in_channels= 16, out_channels=8, kernel_size=5, stride=1), 
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(in_channels= 8, out_channels=4, kernel_size=5, stride=1), 
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(in_channels= 4, out_channels=2, kernel_size=5, stride=1), 
    ) #[64,2,2] -> [2, 22, 22]
                                                                 
  def forward(self, x):
    # Encode
    conv1 = self.conv_block_1(x)
    conv2 = self.conv_block_2(conv1)
    conv3 = self.conv_block_3(conv2)
    conv4 = self.conv_block_4(conv3)

    # Decode
    up_conv1 = self.up_conv_block(conv4)
    # project back into input dimension
    out = F.interpolate(input=up_conv1, size=x.size()[2:] )

    return out
  