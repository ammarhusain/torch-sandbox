import numpy as np
#AH: TODO reimplement this class from scratch
class MetricsComp:
  def __init__(self, n_cls):
    self.n_classes = n_cls
    self.confusion_matrix = np.zeros((n_cls, n_cls))
    
  def _fast_hist(self, label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

  def update(self, label_trues, label_preds):
    for lt, lp in zip(label_trues, label_preds):
      self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

  def get_results(self):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = self.confusion_matrix
    acc = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) #TMP/ (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(self.n_classes), iu))
    return ({"Overall Acc: ": acc,
              "FreqW Acc : ": fwavacc,
              "Mean IoU : ": mean_iu,
            }, cls_iu)

  def reset(self):
    self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageComp:
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
    
def EncodeColor(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb