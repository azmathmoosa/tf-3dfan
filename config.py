IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
IMG_DIM = IMG_WIDTH
HG_STACK = 2
HM_DIM = 64
EXPORT_DIR = "exported/"
MODEL_DIR = "./train-tf-fan-mse-9"
TRAIN_MAX_STEPS = 100*100000
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TRAIN_EPOCHS = 100

TRAIN_PATHS = [
    "/home/az/Documents/LS3D-W/Menpo-3D"
    "/home/az/Documents/LS3D-W/300VW-3D/CatA/**/",
    "/home/az/Documents/LS3D-W/300VW-3D/CatB/**/",    
    "/home/az/Documents/LS3D-W/300VW-3D/CatC/**/",
    "/home/az/Documents/LS3D-W/300VW-3D/Trainset/**/",
    "/home/az/Documents/LS3D-W/AFLW2000-3D-Reannotated",
    # "/home/az/Documents/LS3D-W/300W-Testset-3D",
    
]
EVAL_PATHS = [
    # "/home/az/Documents/LS3D-W/300VW-3D/CatA/**/",
    # "/home/az/Documents/LS3D-W/300VW-3D/CatB/**/",
    # "/home/az/Documents/LS3D-W/300VW-3D/CatC/**/"
    # "/home/az/Documents/LS3D-W/Menpo-3D"    
    "/home/az/Documents/LS3D-W/300W-Testset-3D",
]
PREDICT_PATHS = [
    "/mnt/secondary/Work/face-rec/data/3dfan/web"
]