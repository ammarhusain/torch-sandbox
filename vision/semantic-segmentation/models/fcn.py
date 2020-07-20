import torch.nn as nn
import torch.nn.functional as F

class FCN32s(nn.Module):
  def __init__(self, n_classes, learned_bilinear=False):
    #super(FCN32s, self).__init__()
    super().__init__()

    self.learned_bilinear = learned_bilinear
    self.n_classes = n_classes
    
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100),\
      # why is padding so large?
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
    
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
    
    self.conv_block_3 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.ReLU(inplace=True),      
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )    
    
    self.conv_block_4 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),      
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )

    self.conv_block_5 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),      
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
    
    self.classifier = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7), # collapses into fully connected layer
      nn.ReLU(inplace=True),
      nn.Dropout2d(),
      nn.Conv2d(4096, 4096, 1), # fully connected layer
      nn.ReLU(inplace=True),
      nn.Dropout2d(),
      nn.Conv2d(4096, self.n_classes, 1)
    )
    
  def forward(self, x):
    conv1 = self.conv_block_1(x)
    conv2 = self.conv_block_2(conv1)
    conv3 = self.conv_block_2(conv2)
    conv4 = self.conv_block_2(conv3)
    conv5 = self.conv_block_2(conv4)
    
    score = self.classifier(conv5)
    
    orig_W = x.size()[2:] 
    out = F.interpolate(input=score, size=orig_W)

    return out
  
  def vgg16_init(vgg16):
    """
    AH TODO implement this
    """
    return None