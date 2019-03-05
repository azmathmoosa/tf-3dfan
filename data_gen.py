import tensorflow as tf 
from glob import glob 
import cv2
import numpy as np 
import os 
import json 
import imgaug as ia
from imgaug import augmenters as iaa


IMG_DIM = 256
TRAIN_PATHS = [
    "../LS3D-W/300VW-3D/Trainset/**/",
    "../LS3D-W/300VW-3D/CatA/**/",
    "../LS3D-W/300VW-3D/CatB/**/",
    "../LS3D-W/300VW-3D/CatC/**/",
    "../LS3D-W/AFLW2000-3D-Reannotated",
    "../LS3D-W/Menpo-3D"
]
EVAL_PATHS = [
    "../LS3D-W/300W-Testset-3D"
]

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([    
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0    
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    sometimes(iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -5 to +5 percent (per axis)
            rotate=(-30, 30), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
])


def generator(purpose='train'):
    img_paths, label_paths = get_img_label_paths(purpose)

    for i,l in zip(img_paths, label_paths):        
        if os.path.exists(i) and os.path.exists(l):
            try:
                yield process_face(i, l)
            except Exception as e:
                print(e, i, l)
        

def get_img_label_paths(purpose='train'):    

    if purpose == 'train':
        dataset_paths = TRAIN_PATHS
    else:
        dataset_paths = EVAL_PATHS

    img_paths = []
    for path in dataset_paths:
        img_paths += glob(path + "/*.jpg")
        img_paths += glob(path + "/*.png")
    
    label_paths = [path[:-3]+'json' for path in img_paths]
    return img_paths, label_paths



def process_face(image_path, label_path):
    img = cv2.imread(image_path)
    parts_xy = load_parts_xy(label_path)

    xmin = np.min(parts_xy[:,0])
    xmax = np.max(parts_xy[:,0])
    ymin = np.min(parts_xy[:,1])
    ymax = np.max(parts_xy[:,1])

    margin_p = 0.2
    margin = (xmax - xmin)*margin_p
    xmin = int(xmin - margin/2)
    xmax = int(xmax + margin/2)
    ymin = int(ymin - margin/2)
    ymax = int(ymax + margin/2)

    parts_xy[:, 0] -= xmin 
    parts_xy[:, 1] -= ymin

    face = img[ymin:ymax, xmin:xmax] 

    face, parts_xy = resize_crop(face, parts_xy)
    face = face[:,:,::-1]
    
    gtmap = generate_gtmap(parts_xy, 2.2, face.shape[0])

    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    face = seq_det.augment_image(face)
    gtmap = seq_det.augment_heatmaps(ia.HeatmapsOnImage(gtmap, shape=face.shape)).get_arr()

    face = face/255.

    gtmaps = [gtmap for i in range(2)]

    return face, gtmaps, image_path


def load_parts_xy(ann_path):
    with open(ann_path, "r") as js:
        arr = json.loads(js.read())

    parts = np.array([[item[0],item[1]] for item in arr])
    return parts 

def resize_crop(crop, parts_xy):
    h,w = crop.shape[:2]
    
    parts_xy[:, 0] /= w 
    parts_xy[:, 1] /= h

    crop = cv2.resize(crop, (IMG_DIM, IMG_DIM))
    parts_xy[:, 0] *= IMG_DIM
    parts_xy[:, 1] *= IMG_DIM


    return crop, parts_xy

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def generate_gtmap(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres, outres, npart), dtype=np.float32)
    for i in range(npart):
        visibility = 1 #joints[i, 2]
        if visibility > 0:
            gtmap[:, :, i] = draw_labelmap(gtmap[:, :, i], joints[i, :], sigma)

    gtmap = cv2.resize(gtmap, (64, 64))
    return gtmap


if __name__ == "__main__":
    for x,y,path in generator('train'):
        cv2.imshow("face", x[:,:,::-1])
        cv2.imshow("heatmaps", np.sum(y[0], axis=2))
        cv2.waitKey(100)