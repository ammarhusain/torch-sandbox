import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

class CelebAFacesDataLoader(data.Dataset):
  """CelebAFacesDataLoader
    
      Generates CelebAFacesDataLoader images.
      http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  """
  package_path = os.path.dirname(os.path.abspath(__file__))
  def __init__(self):
    self.name = "celeb-a-faces"
    #self.n_classes = 10
    self.img_size = (64,64)    
    compose = transforms.Compose([\
                                transforms.Resize(self.img_size),\
                                transforms.CenterCrop(self.img_size),\
                                transforms.ToTensor(),\
                                transforms.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)) # range [-1,1]
                  ])
    
    self.torch_data = datasets.ImageFolder(self.package_path+"/data/Celeb-A-Faces",\
                                                       transform=compose)
    
    print(f"Generating {len(self.torch_data)} images")
    
  def __len__(self):
    return len(self.torch_data)
  
  def __getitem__(self, idx):
    img, label = self.torch_data[idx]
    return img, label