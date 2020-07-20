import os
import torch
import glob
import numpy as np
from PIL import Image

from torch.utils import data

class MITSceneParsingLoader(data.Dataset):
  """MITSceneParsingLoader
    
    http://sceneparsing.csail.mit.edu/
    
    This class can also be extended to load data for places challenge:
    https://github.com/CSAILVision/placeschallenge/tree/master/sceneparsing
  """
  def __init__(self, split="training", is_transform=False,
              img_size=(512,512), augmentations=None):
    self.n_classes = 151
    self.img_size = img_size
    self.mean = np.array([104.00699, 116.66877, 122.67892])
    self.files = {}
    self.augmentations = augmentations
    self.is_transform = is_transform
    
    self.images_path = os.path.join("data_loader/data/ADEChallengeData2016", "images", split)
    self.annotations_path = os.path.join("data_loader/data/ADEChallengeData2016", "annotations", split)

    self.files = glob.glob(self.images_path + "/*.jpg")
    
    # AH cut down to 512 files
    self.files = self.files[0:512]
    
    if not self.files:
      raise Exception("No files for split=[%s] found in %s" % (split, self.images_path))
      
    print(f"Found {len(self.files)} {split} images")

    
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    img = Image.open(self.files[index])
    # AH tmp resize
    img = img.resize(self.img_size)
    # convert to numpy array
    img = np.array(img,dtype=np.uint8)
    
    annt_path = os.path.join(self.annotations_path, os.path.basename(self.files[index])[:-4] + ".png")
    annt = Image.open(annt_path)
    # AH tmp resize
    annt = annt.resize(self.img_size)
    annt = np.array(annt,dtype=np.uint8)

    if self.augmentations is not None:
      img, annt = self.augmentations(img, annt)
      
    if self.is_transform:
      img, annt = self.transform(img, annt)
    return img, 
  
  def transform(self, img, annt):
    """
    AH TODO:
    """
    return img, annt