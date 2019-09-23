import numpy as np
import cv2 

from config import * 

def get_landmarks(heatmap):
    pts = []
    heatmap = np.moveaxis(heatmap, -1, 0)
    for i,heatmap in enumerate(heatmap):   
        
        heatmap = cv2.resize(heatmap, (IMG_DIM, IMG_DIM))   
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        # if 28 < i < 35:
        #     cv2.imshow("pt%d"%i, heatmap)
        

        pt = np.unravel_index(heatmap.argmax(), heatmap.shape)
        pt = pt[0], pt[1]
        pts.append(pt)
    
    return pts