import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy2d(nn.Module):
  def __init__(self, dim=1):
    super().__init__()
    self.dim = dim
    self.criterion = nn.NLLLoss()
  def forward(self, input, target):
    return -1 * self.criterion(F.log_softmax(input, dim=self.dim), target)
  