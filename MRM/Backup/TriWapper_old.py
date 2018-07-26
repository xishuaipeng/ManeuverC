
import cv2
import os
import sys
sys.path.append("./Mask_RCNN/")
import random
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
import visualize
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

def LocationFeature(result,width,height,ceilNum=3):
    result=result[0]
    # nObject = len(result['class_ids'])
    # ceilNum = 5
    interestID = [class_names.index('person'),class_names.index('car'),class_names.index('traffic light'),class_names.index('stop sign')]
    featureVector = ceilNum*ceilNum*len(interestID)
    featureVector = np.zeros(shape=(featureVector, 1), dtype=np.float32)
    # print(nObject)
    result['rois'] = result['rois'].astype(np.float32)
    idLocal = result['rois']
    idLocal[:, 0] = 0.5 * (idLocal[:, 2] + idLocal[:, 0]) / height
    idLocal[:, 1] = 0.5 * (idLocal[:, 3] + idLocal[:, 1]) / width
    idCenter = idLocal[:, (0, 1)]
    idCenter = np.round(ceilNum - 1 / idCenter, 0)
    idCenter = ceilNum * idCenter[:, 0] + idCenter[:, 1]
    for lIndex in range(len(idCenter)):
        try:
            idIndex = interestID.index(result['class_ids'][lIndex])
        except:
            continue
        else:
            Mloc = int(idIndex * ceilNum * ceilNum + idCenter[lIndex])
            featureVector[Mloc] = max(featureVector[Mloc], result['scores'][lIndex])
    # for ID in interestID:
    #     idIndex = np.where(result['class_ids'] == ID)
    #     if len(idIndex[0]) == 0:
    #         continue
    #     idScore = result['scores'][idIndex]
    #     idLocal = result['rois'][idIndex]
    #     idLocal[:,0] = 0.5 * (idLocal[:, 2] + idLocal[:,0])/height
    #     idLocal[:, 1] = 0.5 * (idLocal[:, 3] + idLocal[:, 1]) / width
    #     idCenter = idLocal[:, (0, 1)]
    #     idCenter = np.round(ceilNum - 1 / idCenter, 0)
    #     idCenter = ceilNum * idCenter[:, 0]  + idCenter[:, 1]
    #     for lIndex in range(len(idCenter)):
    #         Mloc = int(interestID.index(ID)*ceilNum*ceilNum + idCenter[lIndex])
    #         featureVector[Mloc] = max(featureVector[Mloc], idScore[lIndex])
    return featureVector









class TriConfig(Config):
    NAME = "TRI"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 768

def MaskDetect(videoPath , franeIndex, outfile):

    videoObj = cv2.VideoCapture(videoPath)
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if videoObj.isOpened() is False:
        print('Can not open Video')
        return
    ret, frame = videoObj.read()
    [height, width, _] = frame.shape
    portrait = False
    maxL = max(height, width)
    if height == maxL:
        portrait = True
    minL = min(height, width)
    radio = maxL/minL
    minL = 64*round(minL/2**6)
    maxL = minL * radio
    maxL = 64*round(maxL/2**6)
    if portrait:
        height = maxL
        width = minL
    else:
        height = minL
        width = maxL
    # print(height,width)
    config = TriConfig()
    # config.IMAGE_SHAPE = np.array([maxL, maxL, 3])
    # config.IMAGE_MIN_DIM = minL
    # config.IMAGE_MAX_DIM = maxL
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
     # plot_model(model.keras_model, to_file='model.png')
    cell_num = 3
    index = 0
    while (videoObj.isOpened()):
        ret, frame = videoObj.read()
        index += 1
        if index < 27000:
            continue
        frame = cv2.resize(frame, (width, height))
        results = model.detect([frame], verbose=1)
        r = copy.deepcopy(results[0])
        print(r)
        featureMap = LocationFeature(results, width, height,ceilNum=cell_num)
        featureMap = featureMap.reshape([-1,cell_num,cell_num])
        print(featureMap)
        visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                   class_names, r['scores'])
        # print(results)
        # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    del model
    videoObj.release()
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main(args)
video_id = '023'
video_path = '../' + 'input/raw_data/'+ video_id + '/' + video_id +'_video.avi'
print(video_path)
MaskDetect(video_path , [1,2,3,4,5,6,7,8,9],'./mask_feature.txt')