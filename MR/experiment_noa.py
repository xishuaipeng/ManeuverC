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


def train():
    data_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010']# ,
    data_dir = '..\..\Out'
    im_shape = [299,299,3]
    time_step = 31
    train_data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=True)
    with tf.Session() as sess:
        img_str, frame = data_process(im_shape)
        encoder= encoder_factory('Inception_v3')
        decoder = decoder_factory('LSTM')
        _input, _feature, _train = encoder.last_feature([None, im_shape[0],im_shape[1],im_shape[2]])
        _decoder_input = tf.placeholder(dtype = tf.float32, name = 'decoder_input', shape=[None, time_step, 2048])
        _truth, _loss, _train_op = decoder( _decoder_input, n_hidden = 100, n_class=5)
        sess.run(tf.global_variables_initializer())
        encoder.load_model(sess)
        saver = tf.train.Saver()

        for epoch in range(100):
            saver.save(sess, '..\..\Backup\M801.ckpt',global_step = epoch)
            total_loss = 0.0
            total_num = 0.0
            
            while True:
                evt_data, restart, evt_index = train_data.next(shuffle=True,unit='evt')
                batch_features = []
                batch_label = []
                event_np = os.path.join(data_dir, '%d.npy'%evt_index)
     
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
                loss, _ = sess.run(( _loss, _train_op), feed_dict={_decoder_input: batch_features, _truth: batch_label})
                total_loss = total_loss + loss
                total_num = total_num + 1
                if restart:
                    break

            print('%d: mean loss : %f'%(epoch, total_loss/total_num))

                






                
                # seq_fimg = []
                # for fimg, dimg  in zip(seq_data['fimg'], seq_data['dimg']):
                #     [fimg,dimg] = list(map(lambda f: open(f,'rb').read(), [fimg,dimg]))
                #     fimg = sess.run(frame, feed_dict={img_str: fimg})
                #     # dimg = sess.run(frame, feed_dict={img_str: dimg})
                #     fimg = sess.run(feature_layer, feed_dict={input_layer: [fimg], is_train: False})
                #     seq_fimg.extend(fimg)
                # seq_fimg = np.stack(seq_fimg)
                # _loss, _ = sess.run(( loss, train_op), feed_dict={feature_input_layer: seq_fimg, gtruth_layer: [label]})
                # mean_loss = mean_loss+ _loss
                # total_case =total_case + 1
                # if restart:
                #     break

            

# def test():
#     data_list = ['ID001_T001', 'ID001_T002', ]# ,
#     data_dir = '..\..\Out'
#     im_shape = [480,640,3]
#     train_data = data_factory('TRI', data_dir=data_dir  , data_list= data_list, quiet=True)
#     model_factory  = Model_Zoo()
#     with tf.Session() as sess:
#         img_str, frame = data_process(im_shape)
#         encoder, decoder = model_factory('Inception_v3')
#         T =31
#         input_layer, feature_layer, is_train = encoder.back_bone([1, im_shape[0],im_shape[1],im_shape[2]])
#         [_,w,h,d] = feature_layer.shape
#         feature_input_layer = tf.placeholder(tf.float32,shape = [T,w,h,d], name='CNN_Features')
#         pred_layer ,gtruth_layer ,attention_layer  = decoder.build( feature_input_layer,  n_hidenn = 100, n_class = 5 )
#         saver = tf.train.Saver()
#         saver.restore(sess, '..\..\Backup_2\M726-38')
#         correct = 0
#         total = 0
#         while(True):
#             seq_data, restart = train_data.next(unit = 'event',shuffle=False)
#             label_correct = []
#             for i, seq in tqdm.tqdm(enumerate(seq_data) ):
#                 label = seq['label']
#                 seq_fimg = []
#                 for fimg, dimg  in zip(seq['fimg'], seq['dimg']):
#                     [fimg,dimg] = list(map(lambda f: open(f,'rb').read(), [fimg,dimg]))
#                     fimg = sess.run(frame, feed_dict={img_str: fimg})
#                     ffeature = sess.run(feature_layer, feed_dict={input_layer: [fimg], is_train: False})
#                     seq_fimg.extend(ffeature)
#                 seq_fimg = np.stack(seq_fimg)
#                 predict, attention = sess.run((pred_layer,attention_layer), feed_dict={feature_input_layer: seq_fimg})
#                 # out_img = 
#                 # attention = np.asanyarray(attention)
#                 [_, f_row, f_col,_] = ffeature.shape
#                 for index, f_path in enumerate(seq['fimg']) :
#                     fimg = cv2.imread(f_path)
#                     att_map = attention[index].reshape(f_row,f_col)
#                     att_map = cv2.resize(att_map,(im_shape[1], im_shape[0]))
#                     # att_mask = att_map -0.5
#                     att_mask =  att_map * 0.5
#                     out_img = 0.5* fimg + np.expand_dims(att_mask,-1).astype(np.float)  * fimg
#                     out_img = out_img.astype(np.uint8)
#                     cv2.imwrite(f_path.replace('Out','Att'), out_img )
#                 predict = np.argmax(predict)
#                 total = total + 1
#                 if predict == label:
#                     correct = correct + 1
#                     label_correct.extend([1])
#                 else:
#                     label_correct.extend([0])
#             print(label_correct) 
#             if restart:
#                 break
            
#         print('accurate: %f'%(correct/total))


        




train()
# test()
