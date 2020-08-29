import os
import torch
import numpy as np

import random
from torch.utils import data
import torchvision.datasets
from torchvision import transforms

class FashionMNISTDataLoader(data.Dataset):
  """FashionMNISTDataLoader
    
      Generates Fashion-MNIST images.
  """
  package_path = os.path.dirname(os.path.abspath(__file__))
  def __init__(self, split="training"):
    self.name = "mnist"
    self.n_classes = 10
    self.img_size = (28,28)    
    compose = torchvision.transforms.Compose([transforms.ToTensor(),\
                   transforms.Normalize(mean=.5, std=.5) # range [-1,1]
                  ])
    self.torch_data = torchvision.datasets.FashionMNIST(self.package_path+"/data", train=(split=="training"), \
                                                 transform=compose, target_transform=None, download=True)
      
    # some dummy image mean.
    self.img_mean = np.array([122.5])
    
    print(f"Generating {len(self.torch_data)} images")
    
  def __len__(self):
    return len(self.torch_data)
  
  def __getitem__(self, idx):
    img, label = self.torch_data[idx]
    return img, label
  
  def get_viz(self, img, label):
    return img.squeeze().numpy(), label.squeeze().numpy()

  
class NoisyFashionMNISTDataLoader(FashionMNISTDataLoader):
  """
  Adds random noise to the images for the auto encoder to denoise
  """
  def __init__(self, split="training"):
    super().__init__(split)
    
  def __getitem__(self, idx):
    img, _ = super().__getitem__(idx)
    label = img > 0
    
    # Add noise (in range 0-1) to half the pixels in the image.
    # sample each pixel with probability of 50%
    noisy_pixels = torch.bernoulli(torch.ones(img.shape[1:])*0.5)
    noise = torch.empty(img.shape[1:]).uniform_(0, 1)
    img = torch.clamp(img + noisy_pixels*noise, min=-1.0, max=1.0)
    return img, label.squeeze().long() # remove the channel dimension and convert to long int type
