import os
import torch
import numpy as np

import random
from torch.utils import data

class SimulatedDataLoader(data.Dataset):
  """SimulatedDataLoader
    
      Generates random shapes
  """
  package_path = os.path.dirname(os.path.abspath(__file__))
  def __init__(self, split="training", n_imgs = 2000, img_size=(512,512), n_classes = 10):
    self.name = "simulated"
    self.n_classes = n_classes
    self.img_size = img_size
    self.n_imgs = n_imgs
    if split == "validation":
      self.n_imgs = 400
      
    # some dummy image mean.
    self.img_mean = np.array([122.5, 122.5, 122.5])
  
    # weight the background 10x less in the loss fn.
    self.class_imbalance_weight = torch.tensor([1.0] +  9*[10.0]) 

    self.shapes = []
    for i in range(n_imgs):
      self.shapes.append([4*[SimulatedDataLoader.get_random_location(*self.img_size)]])
    
    print(f"Generating {self.n_imgs} images")
    
  def __len__(self):
    return self.n_imgs
  
  def __getitem__(self, idx):
    img = np.zeros((*self.img_size,3),dtype=np.float64)
    annt = np.zeros(self.img_size,dtype=np.float64)

    img, annt = SimulatedDataLoader.add_plus(img, annt, *SimulatedDataLoader.get_random_location(*self.img_size))
    img, annt = SimulatedDataLoader.add_circle(img, annt, *SimulatedDataLoader.get_random_location(*self.img_size), fill = True)
    img, annt = SimulatedDataLoader.add_circle(img, annt, *SimulatedDataLoader.get_random_location(*self.img_size))

    return self.preprocess_image(img), torch.from_numpy(annt).long()
  
  def preprocess_image(self, img):
    ## Transform image to standard size, mean substraction & normalize.
    # convert to numpy array
    img = np.array(img,dtype=np.float64)
    # reverse channels: RGB -> BGR
    img = img[:,:,::-1]
    img -= self.img_mean
    img = img/ 255.0

    # Reshape to get channels as dim=1 of batched tensor since that is what Conv2d Pytorch needs.
    # (N)HWC -> (N)CHW
    img = img.transpose(2,0,1)
    return torch.from_numpy(img).float()
  
  def get_viz(self, img, annt):
    img = img.numpy().transpose(1,2,0) + (self.img_mean/255.0)
    annt = np.uint8(annt + 100)
    return img, annt
  
  @staticmethod
  def get_random_location(width, height):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.20))

    return (x, y, size)
  
  @staticmethod
  def add_plus(img, annt, x, y, size):
    s = int(size / 2)
    img[x-1:x+1,y-s:y+s, 1] = 255
    annt[x-1:x+1,y-s:y+s] = 1
    img[x-s:x+s,y-1:y+1, 1] = 255
    annt[x-s:x+s,y-1:y+1] = 1
    return img, annt

  @staticmethod
  def add_circle(img, annt, x, y, size, fill=False):
    xx, yy = np.mgrid[:img.shape[0], :img.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.zeros(annt.shape)
    new_arr = np.logical_or(new_arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))
    img[new_arr] = [255,0,0] if fill == True else [0,0,255]
    annt[new_arr] = 2 if fill == True else 3
    return img, annt