from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

def get_optimizer(optimizer_name):
  return {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
  }[optimizer_name]