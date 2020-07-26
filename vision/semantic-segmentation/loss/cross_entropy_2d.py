import torch.nn as nn

class CrossEntropy2d(nn.Module):
  def __init__(self, dim=1):
    super().__init__()
    self.criterion = nn.NLLLoss()
    self.softmax = nn.LogSoftmax(dim=1)
  def forward(self, input, target):
    return self.criterion(self.softmax(input), target)
  