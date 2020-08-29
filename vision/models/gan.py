import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiscriminatorNet(nn.Module):
  """
  A three hidden-layer discriminative network. Hard coded for 28x28 MNIST images.
  """
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    n_features = 28*28 # 784
    
    self.hidden_1 = nn.Sequential(
      nn.Linear(n_features, 1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden_2 = nn.Sequential(
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden_3 = nn.Sequential(
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.out = nn.Sequential(
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    h1 = self.hidden_1(x) # [N, 1024]
    h2 = self.hidden_2(h1) 
    h3 = self.hidden_3(h2) 
    out = self.out(h3)
    return out
    

class GeneratorNet(nn.Module):
  """
  Three hidden-layer generative net. The output layer will have a TanH activation function,
  which maps the resulting values to [-1,1] which is what we normalize the MNIST images to.
  """
  def __init__(self, noise_dim=100):
    super(GeneratorNet, self).__init__()
    n_out = 28*28
    
    self.hidden_1 = nn.Sequential(
      nn.Linear(noise_dim, 256),
      nn.LeakyReLU(0.2),
    )
    self.hidden_2 = nn.Sequential(
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
    )
    self.hidden_3 = nn.Sequential(
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
    )
    self.out = nn.Sequential(
      nn.Linear(1024, n_out),
      nn.Tanh()
    )  
    
  def forward(self, x):
    h1 = self.hidden_1(x) # [N, 256]
    h2 = self.hidden_2(h1) 
    h3 = self.hidden_3(h2) 
    out = self.out(h3)
    return out
  
  
# ------------------------------------------------------------ #
"""
  Auxiliary Classifier GAN (AC_GAN)
    
  https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/#:~:text=The%20Auxiliary%20Classifier%20GAN%2C%20or,than%20receive%20it%20as%20input.
"""
  
class ACDiscriminatorNet(nn.Module):
  """
  A three hidden-layer discriminative network. Hard coded for 28x28 MNIST images.
  """
  def __init__(self):
    super(ACDiscriminatorNet, self).__init__()
    n_features = 28*28 # 784    
    n_classes = 10
    
    self.hidden_1 = nn.Sequential(
      nn.Linear(n_features, 1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden_2 = nn.Sequential(
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.hidden_3 = nn.Sequential(
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )
    self.out_1 = nn.Sequential(
      nn.Linear(256, 1),
    )
    self.out_2 = nn.Sequential(
      nn.Linear(256, n_classes),
    )

  def forward(self, x):
    h1 = self.hidden_1(x) # [N, 1024]
    h2 = self.hidden_2(h1) 
    h3 = self.hidden_3(h2) 
    out_1 = self.out_1(h3)
    out_2 = self.out_2(h3)
    return out_1, out_2
  
class ACGeneratorNet(nn.Module):
  """
  Three hidden-layer generative net. The output layer will have a TanH activation function,
  which maps the resulting values to [-1,1] which is what we normalize the MNIST images to.
  """
  def __init__(self, noise_dim=100):
    super(ACGeneratorNet, self).__init__()
    n_out = 28*28
    embedding_dim=30
    n_classes = 10
    self.embedding = nn.Embedding(n_classes, embedding_dim)
    
    self.hidden_1 = nn.Sequential(
      nn.Linear(embedding_dim + noise_dim, 256),
      nn.LeakyReLU(0.2),
    )
    self.hidden_2 = nn.Sequential(
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
    )
    self.hidden_3 = nn.Sequential(
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
    )
    self.out = nn.Sequential(
      nn.Linear(1024, n_out),
      nn.Tanh()
    )  
    
  def forward(self, x, lbl):
    # x: [batch_sz, noise_dim]
    emb = self.embedding(lbl).squeeze() # [batch_sz, emb_dim]
    # concatenate embedding with noise
    x = torch.cat((x, emb), 1)

    h1 = self.hidden_1(x) # [N, 256]
    h2 = self.hidden_2(h1) 
    h3 = self.hidden_3(h2) 
    out = self.out(h3)
    return out

  
##################################
## Deep Convolutional GAN (DC-GAN)
##################################

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class DCDiscriminatorNet(nn.Module):
  """
  The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function.
  """
  def __init__(self):
    super(DCDiscriminatorNet, self).__init__()
    # input: # [batch_sz, 3, 64, 64]
    self.conv_1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64,
        kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ) # [batch_sz, 64, 32, 32]
    self.conv_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128,
        kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ) # [batch_sz, 128, 16, 16]
    self.conv_3 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=256,
        kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ) # [batch_sz, 256, 8, 8]
    self.conv_4 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=512,
        kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ) # [batch_sz, 512, 4, 4]
    self.out = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=1,
        kernel_size=4, stride=1, padding=0, bias=False),
      nn.Sigmoid()
    ) # [batch_sz, 1, 1, 1]
    
    self.apply(weights_init)
    
  def forward(self, x):
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = self.conv_3(x)
    x = self.conv_4(x)
    x = self.out(x)    
    return x
  
  
class DCGeneratorNet(nn.Module):
  def __init__(self, noise_dim=100):
    super(DCGeneratorNet,self).__init__()
    self.hidden_1 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=noise_dim,
                        out_channels=512,
                        kernel_size=4, stride=1, 
                        padding=0, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(True)
    ) # [batch_sz, 512, 4, 4]
    self.hidden_2 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=512,
                        out_channels=256,
                        kernel_size=4, stride=2, 
                        padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True)
    ) # [batch_sz, 256, 8, 8]
    self.hidden_3 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=256,
                        out_channels=128,
                        kernel_size=4, stride=2, 
                        padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True)
    ) # [batch_sz, 128, 16, 16]
    self.hidden_4 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=128,
                        out_channels=64,
                        kernel_size=4, stride=2, 
                        padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True)
    ) # [batch_sz, 64, 32, 32]
    self.out = nn.Sequential(
      nn.ConvTranspose2d(in_channels=64,
                        out_channels=3,
                        kernel_size=4, stride=2, 
                        padding=1, bias=False),
      nn.Tanh()
    ) # [batch_sz, 32, 64, 64]
    
    self.apply(weights_init)
    
  def forward(self, x):
    x = self.hidden_1(x)
    x = self.hidden_2(x)
    x = self.hidden_3(x)
    x = self.hidden_4(x)
    x = self.out(x)    
    return x