import cv2
import os
import sys
sys.path.append("./Mask_RCNN/")
import random
import tensorflow as tf
import copy
import math
import numpy as np
import skimage.io
from keras.utils import plot_model
import matplotlib
from Mask_RCNN.config import Config
import matplotlib.pyplot as plt
import multiprocessing
import Mask_RCNN.utils
import Mask_RCNN.model as modellib
from Mask_RCNN.config import Config
from skimage import img_as_float
from skimage.transform import rescale, resize
import visualize
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('video_inpath','../input/raw_data/023/023_video.avi','the path of video')
tf.app.flags.DEFINE_string('txt_outpath','./023_mask.txt','the path of result')
tf.app.flags.DEFINE_string('model_path','./mask_rcnn_coco.h5','the path of model')

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

def LocationFeature(result,width,height,cell_num =3):
    result = result[0]
    # nObject = len(result['class_ids'])
    interestID = [class_names.index('person'), class_names.index('car'), class_names.index('traffic light'), class_names.index('stop sign')]
    featureVector = cell_num * cell_num * len(interestID)
    featureVector = np.zeros(shape=(featureVector, 1), dtype=np.float32)
    # print(nObject)
    idIndex = [ind for ind,it in enumerate(result['class_ids']) if it in interestID]
    idClass = [interestID.index(it) for it in result['class_ids'][idIndex]]
    result['rois'] = result['rois'].astype(np.float32)
    idLocal = result['rois'][idIndex]
    idLocal[:, 0] = 0.5 * (idLocal[:, 2] + idLocal[:, 0]) / height
    idLocal[:, 1] = 0.5 * (idLocal[:, 3] + idLocal[:, 1]) / width
    idCenter = idLocal[:, (0, 1)]
    idCenter = np.floor(cell_num * idCenter)
    idCenter = cell_num * idCenter[:, 0] + idCenter[:, 1]
    idScore = result['scores'][idIndex]
    objLoc = np.asanyarray(idClass) * cell_num * cell_num + idCenter
    for loc, score in zip(objLoc, idScore):
       featureVector[int(loc)] = max(featureVector[int(loc)], score)
    return featureVector

class TriConfig(Config):
    NAME = "TRI"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    IMAGE_MIN_DIM = 512#512
    IMAGE_MAX_DIM = 768#768

def LoadModel(modelPath):
    # ROOT_DIR = os.getcwd()
    # COCO_MODEL_PATH = os.path.join(ROOT_DIR,"Mask_RCNN","mask_rcnn_coco.h5")
    config = TriConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
    model.load_weights(modelPath, by_name=True)
    return model

def MaskFeature(image,cell_num = 5, model=None ):
    if model is None:
        model = LoadModel()
    [height, width, _] = image.shape
    results = model.detect([image], verbose=0)
    featureMap = LocationFeature(copy.deepcopy(results), width, height, cell_num=cell_num)
    # featureMap = featureMap.reshape([-1, cell_num, cell_num])
    return results, featureMap

def main(unused_argv ):
    videoPath = FLAGS.video_inpath
    txtPath = FLAGS.txt_outpath
    modelPath = FLAGS.model_path
    txtFile = open(txtPath,"w+")

    videoObj = cv2.VideoCapture(videoPath)
    cell_num = 3
    if videoObj.isOpened() is False:
        print('Can not open Video')
        return
    ret, frame = videoObj.read()
    [height, width, _] = frame.shape
    model = LoadModel(modelPath)
    index = 0
    print(videoPath + 'is processing')
    while frame is not None:
        index += 1
        #print()
        #if index > 20:
        #   break
        results, featureMap = MaskFeature(frame,cell_num=cell_num,model=model)
        featureLine = ''
        featureLine = '%d' % index
        for i in range(len(featureMap)):
            featureLine += ',%.3f'% featureMap[i]
        featureLine += '\n'
        #print(featureLine)
        txtFile.writelines(featureLine)
        ret, frame = videoObj.read()
        # results = model.detect([frame], verbose=1)
        # r = copy.deepcopy(results[0])
        # featureMap = LocationFeature(results, width, height,cell_num=cell_num)
        #featureMap = featureMap.reshape([-1, cell_num, cell_num])
        #print(featureMap)
        #r = results[0]
        #visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
        #                           class_names, r['scores'])
    del model
    txtFile.close()
    videoObj.release()
    


if __name__ == "__main__":
    tf.app.run()



#MaskDetect('../input/raw_data/023/023_video.avi' , [1,2,3,4,5,6,7,8,9],'./mask_feature.txt')
#test()