import torchfile
from glob import glob 
import os 
import json
from tqdm import tqdm 
import numpy as np 


DATASET_DIR = "../LS3D-W"

def load_ann_paths():
    annotations = glob(DATASET_DIR + "/**/**/**/*.t7")
    annotations += glob(DATASET_DIR + "/**/*.t7")

    print("loaded %d annotations"%len(annotations))

    return annotations

if __name__ == "__main__":
    anns = load_ann_paths()

    for ann_path in tqdm(anns):
        ann = torchfile.load(ann_path)
        js = json.dumps(ann.tolist())

        save_path = ann_path[:-2] + "json"
        #print(save_path, ann_path)
        
        with open(save_path, "w") as f:
            f.write(js)
        