import torch.nn as nn

class CrossEntropy2d(nn.Module):
  def __init__(self, dim=1, ignore_idx=-100, weight=None):
    super().__init__()

    print(f"Loss weight: {weight}")

    if weight is not None:
      self.criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_idx)
    else:
      self.criterion = nn.NLLLoss(ignore_index=ignore_idx)
    self.softmax = nn.LogSoftmax(dim=1)
  def forward(self, input, target):
    return self.criterion(self.softmax(input), target)
  