import tensorflow as tf
import os
import numpy as np
from  Dataset import data_factory
from Encoder.Model_Zoo import Model_Zoo
from Train.Train_Op import Train_Op
import tqdm
import cv2

def data_process(im_shape=0):
    img_str = tf.placeholder(dtype = tf.string,name='img_path')
    frame = tf.image.decode_jpeg(img_str)
    if im_shape != 0:
        frame = tf.image.resize_images(frame, [im_shape[0],im_shape[1]])
    frame = (frame - 128.0 )/128.0
    return img_str, frame


def train():

    data_list = ['ID001_T001', 'ID001_T002', ]# ,
    data_dir = '..\..\Out'
    im_shape = [480,640,3]
    train_data = data_factory('TRI', data_dir=data_dir  , data_list= data_list, quiet=True)
    model_factory  = Model_Zoo()
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder, decoder = model_factory('Inception_v3')
        T =31
        input_layer, feature_layer, is_train = encoder.back_bone([1, im_shape[0],im_shape[1],im_shape[2]])
        [_,w,h,d] = feature_layer.shape
        feature_input_layer = tf.placeholder(tf.float32,shape = [T,w,h,d], name='CNN_Features')
        pred_layer ,gtruth_layer ,attention_layer  = decoder.build( feature_input_layer,  n_hidenn = 100, n_class = 5 )
        loss, train_op = Train_Op().build('Ssoftmax','AdamOptimizer',1e-3,pred_layer,gtruth_layer)
        sess.run(tf.global_variables_initializer())
        encoder.load_model(sess)
        saver = tf.train.Saver()
        for epoch in range(1):
            saver.save(sess, '..\..\Backup\M730.ckpt',global_step = epoch)
            mean_loss = 0.0
            total_case = 0.0
            while(True):
                seq_data, restart = train_data.next(shuffle=False)
                seq_data = seq_data[0]
                label = seq_data['label']
                seq_fimg = []
                for fimg, dimg  in zip(seq_data['fimg'], seq_data['dimg']):
                    [fimg,dimg] = list(map(lambda f: open(f,'rb').read(), [fimg,dimg]))
                    fimg = sess.run(frame, feed_dict={img_str: fimg})
                    # dimg = sess.run(frame, feed_dict={img_str: dimg})
                    fimg = sess.run(feature_layer, feed_dict={input_layer: [fimg], is_train: False})
                    seq_fimg.extend(fimg)
                seq_fimg = np.stack(seq_fimg)
                _loss, _ = sess.run(( loss, train_op), feed_dict={feature_input_layer: seq_fimg, gtruth_layer: [label]})
                mean_loss = mean_loss+ _loss
                total_case =total_case + 1
                if restart:
                    break

            print('mean loss : %f'%(mean_loss/total_case))

def test():
    data_list = ['ID001_T001', 'ID001_T002', ]# ,
    data_dir = '..\..\Out'
    im_shape = [480,640,3]
    train_data = data_factory('TRI', data_dir=data_dir  , data_list= data_list, quiet=True)
    model_factory  = Model_Zoo()
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder, decoder = model_factory('Inception_v3')
        T =31
        input_layer, feature_layer, is_train = encoder.back_bone([1, im_shape[0],im_shape[1],im_shape[2]])
        [_,w,h,d] = feature_layer.shape
        feature_input_layer = tf.placeholder(tf.float32,shape = [T,w,h,d], name='CNN_Features')
        pred_layer ,gtruth_layer ,attention_layer  = decoder.build( feature_input_layer,  n_hidenn = 100, n_class = 5 )
        saver = tf.train.Saver()
        saver.restore(sess, '..\..\Backup_2\M726-38')
        correct = 0
        total = 0
        while(True):
            seq_data, restart = train_data.next(unit = 'event',shuffle=False)
            label_correct = []
            for i, seq in tqdm.tqdm(enumerate(seq_data) ):
                label = seq['label']
                seq_fimg = []
                for fimg, dimg  in zip(seq['fimg'], seq['dimg']):
                    [fimg,dimg] = list(map(lambda f: open(f,'rb').read(), [fimg,dimg]))
                    fimg = sess.run(frame, feed_dict={img_str: fimg})
                    ffeature = sess.run(feature_layer, feed_dict={input_layer: [fimg], is_train: False})
                    seq_fimg.extend(ffeature)
                seq_fimg = np.stack(seq_fimg)
                predict, attention = sess.run((pred_layer,attention_layer), feed_dict={feature_input_layer: seq_fimg})
                # out_img = 
                # attention = np.asanyarray(attention)
                [_, f_row, f_col,_] = ffeature.shape
                for index, f_path in enumerate(seq['fimg']) :
                    fimg = cv2.imread(f_path)
                    att_map = attention[index].reshape(f_row,f_col)
                    att_map = cv2.resize(att_map,(im_shape[1], im_shape[0]))
                    # att_mask = att_map -0.5
                    att_mask =  att_map * 0.5
                    out_img = 0.5* fimg + np.expand_dims(att_mask,-1).astype(np.float)  * fimg
                    out_img = out_img.astype(np.uint8)
                    cv2.imwrite(f_path.replace('Out','Att'), out_img )
                predict = np.argmax(predict)
                total = total + 1
                if predict == label:
                    correct = correct + 1
                    label_correct.extend([1])
                else:
                    label_correct.extend([0])
            print(label_correct) 
            if restart:
                break
            
        print('accurate: %f'%(correct/total))


        




# train()
test()
