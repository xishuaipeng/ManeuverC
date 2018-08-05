import tensorflow as tf
import os
import numpy as np
from  Dataset import data_factory
from Encoder import encoder_factory
from  tqdm import tqdm
import cv2


def data_process(im_shape=0):
    img_str = tf.placeholder(dtype = tf.string,name='img_path')
    frame = tf.image.decode_jpeg(img_str)
    if im_shape != 0:
        frame = tf.image.resize_images(frame, [im_shape[0],im_shape[1]])
    frame = (frame - 128.0 )/128.0
    return img_str, frame

def gene_or_load_feature(img_path,mat_path, sess, img_str, frame, _input, _feature, _train):
    if os.path.isfile(mat_path) is False:
        fimg = open(img_path,'rb').read() 
        fimg = sess.run(frame, feed_dict={img_str: fimg})
        feature = sess.run(_feature, feed_dict={_input: [fimg], _train: False})
        feature = np.squeeze(feature)
        np.save(mat_path, feature)
    else:
        feature = np.load(mat_path)
    return feature 

def feature_extraction(data_list, data_dir = '..\..\Out', im_shape = [640,480,3], sufix = 'map', subset = 'train'):
    sppedup_dir = os.path.join(data_dir,'speedup2', subset)
    if os.path.isdir(sppedup_dir) is False:
        os.makedirs(sppedup_dir)
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=False)
    with tf.Session() as sess:
        _img_str, _frame = data_process(im_shape)
        encoder= encoder_factory('Inception_v3')
        if sufix == 'map':
            _input, _feature, _train = encoder.last_map([None, im_shape[0],im_shape[1],im_shape[2]])
        else:
            _input, _feature, _train = encoder.last_feature([None, im_shape[0],im_shape[1],im_shape[2]])
        sess.run(tf.global_variables_initializer())
        encoder.load_model(sess)
        while True:
            evt_data, restart, evt_index, unique_id = train_data.next(shuffle=False,unit='evt')
            event_np = os.path.join(sppedup_dir, '%s_%s_%s.npy'%(unique_id, encoder.model_id, sufix))
            print('%d: %s'%(evt_index, evt_data[0]['id']))
            event_front_features = []
            event_driver_features = []
            event_signal = []
            seq_label =[]
            early_time =[]
            early_distance =[]
            front_img_path = []
            driver_img_path = []
            event_id = []
            evt_label = -1
            for  seq_data in tqdm(evt_data):
                seq_f_features = []
                seq_d_features = []
                # process image
                for fimg_path, dimg_path in zip(seq_data['fimg'],seq_data['dimg']) :
                    img_dir = os.path.dirname(fimg_path)
                    mat_path = os.path.join(img_dir, '%s_%s'%( encoder.model_id, sufix)) 
                    if os.path.isdir(mat_path) is False:
                        os.makedirs(mat_path)
                    fmat_path , dmat_path = list(map( lambda x : os.path.join(mat_path, '%s.npy'%os.path.basename(x)[0:-4] ), [fimg_path , dimg_path]))
                    feature = gene_or_load_feature(fimg_path,fmat_path ,sess, _img_str, _frame , _input, _feature, _train)
                    seq_f_features.append(feature)
                    feature = gene_or_load_feature(dimg_path , dmat_path,sess, _img_str,_frame, _input, _feature, _train)
                    seq_d_features.append(feature)
                event_front_features.append(np.stack(seq_f_features))
                event_driver_features.append(np.stack(seq_d_features))
                # process other information
                seq_label.append(seq_data['seq_label'])
                early_time.append(seq_data['early_time'])
                early_distance.append(seq_data['early_distance'])
                front_img_path.append(seq_data['fimg'])
                driver_img_path .append(seq_data['dimg'])
                event_id.append(seq_data['id'])
                event_signal.append(seq_data['signal'])
                if evt_label==-1:
                    evt_label = seq_data['evt_label']
                else:
                    assert(evt_label == seq_data['evt_label'])
                
            np.save(event_np,{'id':event_id, 'front_feature':event_front_features, 'driver_features':event_driver_features,'signal':event_signal,\
            'seq_label':seq_label, 'early_time':early_time,'early_distance':early_distance,\
            'event_label': evt_label, 'front_img_path': front_img_path,  'driver_img_path': driver_img_path })


feature_extraction(data_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
data_dir = '..\..\Out',\
im_shape = [640,480,3],\
sufix = 'map', subset = 'train')


feature_extraction(data_list = ['ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'],\
data_dir = '..\..\Out',\
im_shape = [640,480,3],\
sufix = 'map', subset = 'test')
