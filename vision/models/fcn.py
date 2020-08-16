import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FCN32s(nn.Module):
  def __init__(self, n_classes, pretrained=True):
    super().__init__()

    self.name = "FCN32s"
    
    # self.learned_bilinear = learned_bilinear
    self.n_classes = n_classes
    
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),\
      # why is padding so large? REMOVED!! Check consequence
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
    
    self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
    self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        
    # Initialize with a pretrained backbone if requested
    if pretrained:
      vgg16 = models.vgg16(pretrained=True)
      self.vgg16_init_pretrained(vgg16)
    
  def forward(self, x):
    conv1 = self.conv_block_1(x)
    conv2 = self.conv_block_2(conv1)
    conv3 = self.conv_block_3(conv2)
    conv4 = self.conv_block_4(conv3)
    conv5 = self.conv_block_5(conv4)
    
    score = self.classifier(conv5)
    
    score_pool4 = self.score_pool4(conv4)
    score = F.interpolate(input=score, size=score_pool4.size()[2:])    
    score += score_pool4
    
    score_pool3 = self.score_pool3(conv3)
    score = F.interpolate(input=score, size=score_pool3.size()[2:])    
    score += score_pool3

    out = F.interpolate(input=score, size=x.size()[2:] )
    
    return out
  
  def vgg16_init_pretrained(self, vgg16):
    """
    Use the weights from a torchvision pretrained VGG model to get things bootstrapped.
    """

    blocks = list(self.conv_block_1.children()) +\
             list(self.conv_block_2.children()) +\
             list(self.conv_block_3.children()) +\
             list(self.conv_block_4.children()) +\
             list(self.conv_block_5.children())
    
    pretrained_feature_backbone = list(vgg16.features.children())
    for pretrained_elem, local_elem in zip(pretrained_feature_backbone, blocks):
      if isinstance(pretrained_elem, nn.Conv2d) and isinstance(local_elem, nn.Conv2d):
        assert pretrained_elem.weight.size() == local_elem.weight.size()
        assert pretrained_elem.bias.size() == local_elem.bias.size()
        local_elem.weight.data = pretrained_elem.weight.data
        local_elem.bias.data = pretrained_elem.bias.data
        
    # Copy the 2 fullyconnected linear layers from the classifier
    self.classifier[0].weight.data = vgg16.classifier[0].weight.data.view(self.classifier[0].weight.data.size())
    self.classifier[0].bias.data = vgg16.classifier[0].bias.data.view(self.classifier[0].bias.data.size())
    self.classifier[3].weight.data = vgg16.classifier[3].weight.data.view(self.classifier[3].weight.data.size())
    self.classifier[3].bias.data = vgg16.classifier[3].bias.data.view(self.classifier[3].bias.data.size())
    return