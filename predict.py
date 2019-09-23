from tensorflow.contrib import predictor
from glob import glob 
import os 
import cv2 
import numpy as np
from tqdm import tqdm 

from utils import get_landmarks

def load_model_predictor():
    export_dir = "exported/saved_model/1569206590"
    predict_fn = predictor.from_saved_model(export_dir)
    return predict_fn

def load_images():
    images_dir = "../web/" #"/home/az/Documents/LS3D-W/300W-Testset-3D/"
    image_paths = glob(images_dir+"*.jpg")    
    return image_paths

if __name__ == "__main__":
    image_paths  = load_images()
    predict_fn = load_model_predictor()

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256,256))
        predictions = predict_fn({"image": img, "name": img_path })

        heatmaps = predictions['heatmap']

        pts = get_landmarks(heatmaps[-1][0])
        for pt in pts:
            cv2.circle(img, (int(pt[1]), int(pt[0])), 2, (0, 255, 0), -1, cv2.LINE_AA)

        
        for heatmap in heatmaps:
            heatmap  = np.sum(heatmap[0], axis=2)
            # heatmap = (heatmap / -255).astype(np.uint8)
            heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
            heatmap = cv2.resize(heatmap, (256, 256))
            cv2.imshow("hmap", heatmap)

            cv2.imshow("result", img)
            cv2.waitKey(0)
