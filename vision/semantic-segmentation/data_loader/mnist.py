import os
import torch
import numpy as np

import random
from torch.utils import data
import torchvision.datasets

class MNISTDataLoader(data.Dataset):
  """MNISTDataLoader
    
      Generates MNIST images.
  """
  package_path = os.path.dirname(os.path.abspath(__file__))
  def __init__(self, split="training"):
    self.name = "mnist"
    self.n_classes = 10
    self.img_size = (28,28)    
    self.torch_data = torchvision.datasets.MNIST(self.package_path+"/data", train=(split=="training"), \
                                                 transform=None, target_transform=None, download=False)
      
    # some dummy image mean.
    self.img_mean = np.array([122.5])
    
    print(f"Generating {len(self.torch_data)} images")
    
  def __len__(self):
    return len(self.torch_data)
  
  def __getitem__(self, idx):
    img, label = self.torch_data[idx]
    img = np.array(img,dtype=np.float64)
    img -= self.img_mean
    img = img / 255.0
    # Reshape to get channels as dim=1 of batched tensor since that is what Conv2d Pytorch needs.
    # (N)CHW
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img, label
  
  def get_viz(self, img, label):
    return img.squeeze().numpy(), label.squeeze().numpy()

  
class NoisyMNISTDataLoader(MNISTDataLoader):
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
    img = torch.clamp(img + noisy_pixels*noise, min=0.0, max=1.0)
    return img, label.squeeze().long() # remove the channel dimension and convert to long int type
    