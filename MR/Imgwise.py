import tensorflow as tf
import numpy as np
from  Dataset.Tri_data import Tri_data
from Encoder.Model_Zoo import Model_Zoo
from Train.Train_Op import Train_Op
import cv2

def data_process(im_shape=0):
    img_str = tf.placeholder(dtype = tf.string,name='img_path')
    frame = tf.image.decode_jpeg(img_str)
    if im_shape != 0:
        frame = tf.image.resize_images(frame, [im_shape[0],im_shape[1]])
    frame = (frame - 128.0 )/128.0
    return img_str, frame


def main():

    data_list = ['ID001_T001', 'ID001_T002']# ,
    data_dir = 'D:\\xishuaip\\TriOut'
    im_shape = [480,640,3]
    train_data = Tri_data(data_dir , data_list, quiet=True)
    train_data.encoder()
    model_factory  = Model_Zoo()
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder, decoder = model_factory('Inception_v3')
        T =31
        input_layer, feature_layer = encoder.back_bone([1, im_shape[0],im_shape[1],im_shape[2]])
        [_,w,h,d] = feature_layer.shape
        feature_input_layer = tf.placeholder(tf.float32,shape = [T,w,h,d], name='CNN_Features')
        pred_layer ,gtruth_layer ,attention_layer  = decoder.build( feature_input_layer,  n_hidenn = 100, n_class = 5 )
        loss, train_op = Train_Op().build('Ssoftmax','AdamOptimizer',1e-3,pred_layer,gtruth_layer)
        encoder.load_model(sess)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(500):
            saver.save(sess, '../../Backup/M726',global_step = epoch)
            mean_loss = 0.0
            total_case = 0.0
            while(True):
                seq_data, end_epoch = train_data.next(shuffle=True)
                if end_epoch:
                    break
                label = seq_data['label']
                seq_fimg = []
                for fimg, dimg  in zip(seq_data['fimg'], seq_data['dimg']):
                    [fimg,dimg] = list(map(lambda f: open(f,'rb').read(), [fimg,dimg]))
                    fimg = sess.run(frame, feed_dict={img_str: fimg})
                    # dimg = sess.run(frame, feed_dict={img_str: dimg})
                    fimg = sess.run(feature_layer, feed_dict={input_layer: [fimg]})
                    seq_fimg.extend(fimg)
                seq_fimg = np.stack(seq_fimg)
                _loss, _ = sess.run(( loss, train_op), feed_dict={feature_input_layer: seq_fimg, gtruth_layer: [label]})
                mean_loss = mean_loss+ _loss
                total_case =total_case + 1

            print('mean loss : %f'%(mean_loss/total_case))





main()
