import os
import torch
import glob
import numpy as np
import PIL
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
    self.img_mean = np.array([104.00699, 116.66877, 122.67892])
    self.files = {}
    self.augmentations = augmentations
    self.is_transform = is_transform
    
    self.images_path = os.path.join("data_loader/data/ADEChallengeData2016", "images", split)
    self.annotations_path = os.path.join("data_loader/data/ADEChallengeData2016", "annotations", split)

    self.files = glob.glob(self.images_path + "/*.jpg")
    
    # AH cut down to 512 files
    #self.files = self.files[0:512]
    
    if not self.files:
      raise Exception("No files for split=[%s] found in %s" % (split, self.images_path))
      
    print(f"Found {len(self.files)} {split} images")

    
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    img = Image.open(self.files[index]).convert("RGB")
    annt_path = os.path.join(self.annotations_path, os.path.basename(self.files[index])[:-4] + ".png")
    annt = Image.open(annt_path)
    
    #AH todo
    if self.augmentations is not None:
      img, annt = self.augmentations(img, annt)    
    
    ## Transform image to standard size, mean substraction & normalize.
    img = img.resize(self.img_size)
    # convert to numpy array
    img = np.array(img,dtype=np.float64)
    # reverse channels: RGB -> BGR
    if len(img.shape) != 3:
      print(f"YELLL i {index} ... {img.shape}")
    img = img[:,:,::-1]
    img -= self.img_mean
    img = img/ 255.0
    # Reshape to get channels as dim=1 of batched tensor since that is what Conv2d Pytorch needs.
    # (N)HWC -> (N)CHW
    img = img.transpose(2,0,1)
    
    ## Tranform annotations
    annt_orig_np = np.array(annt, dtype=np.uint8)
    classes = np.unique(annt_orig_np)

    annt = annt.resize(self.img_size, PIL.Image.NEAREST)
    annt_down_np = np.array(annt, dtype=np.uint8)

    if not np.all(classes == np.unique(annt_down_np)):
      print("WARN: resizing labels yielded fewer classes")
    if not np.all(np.unique(annt_down_np) < self.n_classes):
      raise ValueError("Segmentation map contained invalid class values")

    img_t = torch.from_numpy(img).float()
    annt_t = torch.from_numpy(annt_down_np).long()

    return img_t, annt_t
  
  def transform(self, img, annt):
    """
    AH TODO:
    """
    return img, annt