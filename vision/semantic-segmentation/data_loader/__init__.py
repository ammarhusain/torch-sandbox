from data_loader.mit_scene_parsing import MITSceneParsingLoader

def get_loader(name):
  return {
    "mit_sceneparsing_benchmark": MITSceneParsingLoader
  }[name]