from models.fcn import FCN32s

def get_model(arch_name, n_classes):
  if arch_name == "fcn32s":
    return FCN32s(n_classes)
  else:
    return None
  