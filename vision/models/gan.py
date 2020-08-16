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
    n_out = 1
    
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
      nn.Linear(256, n_out),
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
    n_noise = noise_dim
    n_out = 28*28
    
    self.hidden_1 = nn.Sequential(
      nn.Linear(n_noise, 256),
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