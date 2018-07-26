import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
from config import Config
import matplotlib.pyplot as plt


import utils
import model as modellib
import visualize
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class ShapesConfig(Config):
    """
    为数据集添加训练配置
    继承基类Config
    """
    NAME = "shapes" # 该配置类的识别符

    #Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1 # GPU数量
    IMAGES_PER_GPU = 1# 单GPU上处理图片数(这里我们构造的数据集图片小，可以多处理几张)

    # 分类种类数目 (包括背景)
    NUM_CLASSES = 81 # background + 3 shapes

    # 使用小图片可以更快的训练
    IMAGE_MIN_DIM = 476 # 图片的小边长
    IMAGE_MAX_DIM = 640 # 图片的大边长



config = ShapesConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

file_names = next(os.walk(IMAGE_DIR))[1]
image = skimage.io.imread(os.path.join(IMAGE_DIR, "12283150_12d37e6389_z.jpg"))#random.choice(file_names))

# Run detection

results = model.detect([image], verbose=1)
# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])