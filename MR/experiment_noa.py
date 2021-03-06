import tensorflow as tf
import os
import numpy as np
from  Dataset import data_factory
from Encoder import encoder_factory
from Decoder import decoder_factory
from  tqdm import tqdm
import cv2

def data_process(im_shape=0):
    img_str = tf.placeholder(dtype = tf.string,name='img_path')
    frame = tf.image.decode_jpeg(img_str)
    if im_shape != 0:
        frame = tf.image.resize_images(frame, [im_shape[0],im_shape[1]])
    frame = (frame - 128.0 )/128.0
    return img_str, frame

def easy_train(data_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
    data_dir = '..\..\Out',\
    im_shape = [299,299,3],\
    time_step = 31):
    speedup_dir = os.path.join(data_dir,'speedup')
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=False)
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder= encoder_factory('Inception_v3')
        decoder = decoder_factory('LSTM')
        _input, _feature, _train = encoder.last_feature([None, im_shape[0],im_shape[1],im_shape[2]])
        _decoder_input = tf.placeholder(dtype = tf.float32, name = 'decoder_input', shape=[None, time_step, 2048])
        _truth, _loss, _train_op, _pred = decoder( _decoder_input, n_hidden = 400, n_class=5, learning_rate=1e-3)
        sess.run(tf.global_variables_initializer())
        encoder.load_model(sess)
        saver = tf.train.Saver()
        graph_writer = tf.summary.FileWriter( 'D:/xishuaip/TRI/Research/Graph',sess.graph)
        # sess.run(graph_writer)
        for epoch in range(2000):
            saver.save(sess, '..\..\Backup\M81ckpt',global_step = epoch)
            batch_features = []
            batch_label = []
            for i in range(200):
                evt_data, restart, evt_index = train_data.next(shuffle = True, unit = 'evt')
                # evt_index = 2
                event_np = os.path.join(speedup_dir,'%d_%s_vec.npy'%(evt_index, encoder.model_id))
                if os.path.isfile(event_np):
                    batch_data = np.load(event_np).item()
                    batch_features.extend( batch_data['data'])
                    batch_label.extend(batch_data['label'])
            # batch_label = batch_label -1
            loss,pred, _ = sess.run(( _loss, _pred, _train_op), feed_dict={_decoder_input: batch_features, _truth: batch_label})
            # print(init_state.shape)
            total_correct = len([x for x, y in zip(pred, batch_label) if x==y])
            print('%d: mean loss : %f, accuracy: %f'%(epoch, loss, total_correct/len(batch_label)))
        # test()
        







# ,'ID001_T011','ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'

def train(data_list = ['ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'],\
data_dir = '..\..\Out',\

im_shape = [299,299,3],\
time_step = 31):
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=False)
    speedup_dir = os.path.join(data_dir,'speedup')
    if os.path.isdir(speedup_dir) is False:
        os.makedirs(speedup_dir)
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder= encoder_factory('Inception_v3')
        decoder = decoder_factory('LSTM')
        _input, _feature, _train = encoder.last_feature([None, im_shape[0],im_shape[1],im_shape[2]])
        _decoder_input = tf.placeholder(dtype = tf.float32, name = 'decoder_input', shape=[None, time_step, 2048])
        _truth, _loss, _train_op, _pred = decoder( _decoder_input, n_hidden = 100, n_class=5, learning_rate=1e-2)
        sess.run(tf.global_variables_initializer())
        encoder.load_model(sess)
        saver = tf.train.Saver()
        for epoch in range(100):
            saver.save(sess, '..\..\Backup\M802.ckpt',global_step = epoch)
            total_loss = 0.0
            total_num = 0.0
            total_correct = 0.0
            total_seq = 0.0
            while True:
                evt_data, restart, evt_index = train_data.next(shuffle=True,unit='evt')
                batch_features = []
                batch_label = []
                event_np = os.path.join(speedup_dir, '%d_%s_vec.npy'%(evt_index, encoder.model_id))
                if os.path.isfile(event_np):
                    batch_data = np.load(event_np).item()
                    batch_features = batch_data['data']
                    batch_label = batch_data['label']
                else:
                    print(evt_data[0]['mat_path'])
                    for  seq_data in tqdm(evt_data):
                        label = seq_data['label']
                        batch_label.extend([label])
                        seq_features = []
                        for fimg_path in seq_data['fimg']:
                            fimg = open(fimg_path,'rb').read() 
                            fimg = sess.run(frame, feed_dict={img_str: fimg})
                            feature = sess.run(_feature, feed_dict={_input: [fimg], _train: False})
                            seq_features.append(np.squeeze(feature))
                        batch_features.append(np.stack(seq_features))
                    np.save(event_np,{'data':batch_features, 'label':batch_label})

                # batch_features = np.asanyarray(batch_features).reshape(batch, time_step, -1)
                
                loss,pred, _ = sess.run(( _loss, _pred, _train_op), feed_dict={_decoder_input: batch_features, _truth: batch_label})
                total_loss = total_loss + loss
                total_num = total_num + 1
                total_seq = total_seq + len(batch_label)
                total_correct = total_correct + len([x for x, y in zip(pred, batch_label) if x==y])
                if restart:
                    break

            print('%d: mean loss : %f, accuracy: %f'%(epoch, total_loss/total_num, total_correct/total_seq))
            # test()

            

def test(data_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
data_dir = '..\..\Out',\
im_shape = [299,299,3],\
time_step = 31):
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=True)
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder= encoder_factory('Inception_v3')
        decoder = decoder_factory('LSTM')
        _input, _feature, _train = encoder.last_feature([None, im_shape[0],im_shape[1],im_shape[2]])
        _decoder_input = tf.placeholder(dtype = tf.float32, name = 'decoder_input', shape=[None, time_step, 2048])
        _truth, _loss, _train_op,  _pred = decoder( _decoder_input, n_hidden = 100, n_class=5)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,  '..\..\Backup\M802.ckpt-24')
        total_correct = 0.0
        total_num = 0.0  
        while True:
            evt_data, restart, evt_index = train_data.next(shuffle=False,unit='evt')
            batch_features = []
            batch_label = []
            event_np = os.path.join(data_dir, '%d.npy'%evt_index)
            # print(evt_index)
            if (evt_index in [71,96]):
                continue
            if os.path.isfile(event_np):
                batch_data = np.load(event_np).item()
                batch_features = batch_data['data']
                batch_label = batch_data['label']

            else:
                print(evt_data[0]['mat_path'])
                for  seq_data in tqdm(evt_data):
                    label = seq_data['label']
                    batch_label.extend([label])
                    seq_features = []
                    for fimg_path in seq_data['fimg']:
                        fimg = open(fimg_path,'rb').read() 
                        fimg = sess.run(frame, feed_dict={img_str: fimg})
                        feature = sess.run(_feature, feed_dict={_input: [fimg], _train: False})
                        seq_features.append(np.squeeze(feature))
                    batch_features.append(np.stack(seq_features))
                np.save(event_np,{'data':batch_features, 'label':batch_label})

            pred = sess.run(( _pred), feed_dict={_decoder_input: batch_features})
            correct_num = len([x for x, y in zip(pred, batch_label) if x==y])
            total_correct = total_correct + correct_num
            total_num = total_num + len(batch_label)
            if restart:
                break

        print('accurate : %f'%( total_correct/total_num))
        






easy_train()
# test()
