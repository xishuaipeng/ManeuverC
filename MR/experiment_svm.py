import tensorflow as tf
import os
import numpy as np
from  Dataset import data_factory
from Encoder import encoder_factory
from Decoder import decoder_factory
from  tqdm import tqdm
import cv2
from sklearn.svm import SVR,SVC

def train(data_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
    data_dir = '..\..\Out',\
    im_shape = [299,299,3],\
    time_step = 31):
    speedup_dir = os.path.join(data_dir,'speedup')
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=True)
    batch_features = []
    batch_label = []
    while True:
        evt_data, restart, evt_index = train_data.next(shuffle = True, unit = 'evt')
        event_np = os.path.join(speedup_dir,'%d_%s_vec.npy'%(evt_index, 'Inception_v3'))
        if os.path.isfile(event_np):
                batch_data = np.load(event_np).item()
                batch_features.extend( batch_data['data'])
                batch_label.extend(batch_data['label'])
        if restart:
            break
    last_feature = [ frame_feature[-1]   for frame_feature in batch_features]
    SVM_classer = SVC(kernel='linear',random_state=2 )
    SVM_classer.fit(last_feature,batch_label)
    y_ = SVM_classer.predict(last_feature)
    correct_num =len( [x  for x, y in zip(y_, batch_label) if x==y])
    print('train accuracy: %f'% (correct_num/ len(batch_label)))
    return SVM_classer
            
        
            

def test(SVM_classer, data_list = ['ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'],\
data_dir = '..\..\Out',\
im_shape = [299,299,3],\
time_step = 31):
    speedup_dir = os.path.join(data_dir,'speedup')
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list,  quiet=True)
    batch_features = []
    batch_label = []
    while True:
        evt_data, restart, evt_index = train_data.next(shuffle = True, unit = 'evt')
        event_np = os.path.join(speedup_dir,'%d_%s_vec.npy'%(evt_index, 'Inception_v3'))
        if os.path.isfile(event_np):
                batch_data = np.load(event_np).item()
                batch_features.extend( batch_data['data'])
                batch_label.extend(batch_data['label'])
        if restart:
            break
    last_feature = [ frame_feature[-1]   for frame_feature in batch_features]
    y_ = SVM_classer.predict(last_feature)
    correct_num =len( [x  for x, y in zip(y_, batch_label) if x==y])
    print('test accuracy: %f'% (correct_num/ len(batch_label)))
    return SVM_classer
    
        





classfer = train()
test(classfer)
# test()
