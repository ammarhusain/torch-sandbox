from data_loader.mit_scene_parsing import MITSceneParsingLoader
from data_loader.simulated_data import SimulatedDataLoader

def get_loader(name):
  return {
    "mit_sceneparsing_benchmark": MITSceneParsingLoader,
    "simulated": SimulatedDataLoader,
  }[name]