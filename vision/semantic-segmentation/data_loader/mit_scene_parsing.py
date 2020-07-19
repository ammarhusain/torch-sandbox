import os
import torch


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
    self.split = split
    self.images_path = os.path.join("data_loader/data", "images", split)
    self.annotations_path = os.path.join("data_loader/data", "annotations", split)

    self.files[split] =    
    
    
      
    
