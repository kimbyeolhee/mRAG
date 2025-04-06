import hydra
from multiprocess import set_start_method
import os
import json

if 'CONFIG' in  os.environ:
    CONFIG = os.environ["CONFIG"]
else:
    CONFIG= 'main'

@hydra.main(config_path="config", config_name=CONFIG, version_base="1.2")
def main(config):
    print("Config: ", config)

    from modules.mrag import mRAG
    mrag = mRAG(**config, config=config)

    if 'dataset_split' in config:
        dataset_split = config['dataset_split']
    else:
        dataset_split = 'dev'

    mrag.eval(dataset_split=dataset_split)

if __name__ == "__main__":
    set_start_method("spawn")
    main()