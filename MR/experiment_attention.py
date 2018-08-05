import tensorflow as tf
import os
import numpy as np
from  Dataset import data_factory
from Encoder import encoder_factory
from Decoder import decoder_factory
from  tqdm import tqdm
from tensorflow.python.ops import array_ops
import cv2
from sklearn.svm import SVR,SVC

class AttentionRNN:
    def __init__(self):
        self.is_train= False
        self.is_build = False
        self.h2x_layer = None
        self.att_layer  = None
        self.rnn_layer  = None
        self.logits_layer = None
        self.ground_truth_layer = None
        self.optimizer = None
        # self.pre_process= None
        # pass
        
    def _build(self, n_channel, n_hidden, n_class, learn_rate):
        # n_bath, n_time, n_size, n_channel = array_ops.shape(inputs)
        if self.is_build is False:
            with tf.variable_scope('Global'):
                self.is_train = tf.placeholder(dtype=tf.bool,shape = [], name='is_train')
                self. ground_truth_layer  =  tf.placeholder(shape=[None], dtype = tf.int32, name = 'ground_truth')
            with tf.variable_scope('Attention'):
                self.h2x_layer = tf.layers.Dense(n_channel)
                self.att_layer = tf.layers.Dense(1)
            with tf. variable_scope('RNN'):
                self.rnn_layer = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            with tf. variable_scope('LOGITS'):
                self. logits_layer  =  tf.layers.Dense(n_class)
            with tf.variable_scope('Loss'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
            self.is_build = True

    def __call__(self, inputs, n_hidden, n_class, learn_rate= 1e-3):
        # with tf.variable_scope('Net'):
        n_bath, _, _, n_channel = array_ops.shape(inputs)
        inputs = tf.layers.batch_normalization(inputs, training=self.is_train)
        self._build(n_channel, n_hidden, n_class, learn_rate)
        inputs = tf.unstack(inputs, axis = 1)
        rnn_hs = []
        att_maps= []
        for index, input in enumerate(inputs):
            if index == 0:
                init_state = self.rnn_layer.zero_state(n_bath,  dtype=tf.float32) 
                state = init_state
                h = state[0]
            h2x_op = self.h2x_layer(h)
            fuse_op = input * tf.expand_dims(h2x_op,1) 
            att_map = tf.nn.softmax(tf.squeeze(self.att_layer(fuse_op))) 
            att_feature = tf.reduce_sum(tf.expand_dims(att_map,-1) * input, axis=1)
            h, state =  self.rnn_layer(att_feature, state = state)
            rnn_hs.append(h)
            att_maps.append(att_map)
        rnn_hs, att_maps = list(map( lambda x:  tf.transpose(tf.stack(x), perm = [1,0,2]), [rnn_hs, att_maps]))

        logits = self.logits_layer(rnn_hs[:,-1,:])
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.ground_truth))
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = self.optimizer.minimize(loss)

        return self.ground_truth_layer, loss, train_op , logits, (self.is_train, att_maps)

def load_data(data_dir, data_list):
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
    return batch_features, batch_label


def stage_recorder(att_map, resize_shape, save_dir = 'D:'):
     n_batch, n_time, n_size = att_map.shape
    for ib, batch in enumerate( range(n_batch)):
        for it, time in  enumerate( range(n_time)):
            map = att_map[batch, time, :]
            map_img = np.asanyarray(map*255.0).astype(np.uint8)
            map_reimg = cv2.resize(map_img, resize_shape)
            img_path = os.path.join(save_dir, '%d_%d.jpg'%(ib,it))
            cv2. imwrite(img_path, map_reimg)


def event_eve(data_dir, data_list, sess, _pred, _istrain, score_threshold):
    speedup_dir = os.path.join(data_dir,'speedup')
    data = data_factory('TRI', data_dir=data_dir , data_list= data_list, quiet=True)
    coorect = 0.0
    incorrect = 0.0
    while True:
        evt_data, restart, evt_index = train_data.next(shuffle = True, unit = 'evt')
        event_np = os.path.join(speedup_dir,'%d_%s_vec.npy'%(evt_index, 'Inception_v3'))
        if os.path.isfile(event_np):
                event_data_label = np.load(event_np).item()
                event_data = event_data_label['data']
                event_label = event_data_label['label']
                event_max_oc_label = max(event_label,key=event_label.count)
                pred = sess.run( _pred, feed_dict={ _inputs:event_data,  _istrain: False })
                seq_index = [(index, np.argmax(score))  for index, score in enumerate(pred)  if np.max(score)>=score_threshold & np.argmax(score) > 0  ]
                if seq_index[0][1] == event_max_oc_label:
                    coorect = coorect + 1
                else:
                    incorrect = incorrect +1
        if restart:
            break
    print('Event accuracy : %f'%(coorect/(coorect + incorrect)))


    
def train_test( train_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
    test_list = ['ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'],\
    data_dir = '..\..\Out',\
    im_shape = [640,480,3],\
    time_step = 31):
    train_features, train_label = load_data(data_dir,train_list )
    test_features, test_label = load_data(data_dir,test_list )
    epoches = 3000
    net  = AttentionRNN()
    _inputs = tf.placeholder(shape=[None, time_step, 2048], dtype=tf.float32, name ='lstm_inputs')
    _ground_truth, _loss, _train_op, _pred ,(_istrain, _att_map) = net(_inputs, 100,5, 1e-3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            loss, _, pred, att_map = sess.run((_loss, _train_op, _pred, _att_map), feed_dict={_ground_truth: train_label, _inputs:train_features, _istrain: True })
            correct_num =len( [x  for x, y in zip(pred, train_label) if x==y])
            print('%d: loss: %f,  accuracy: %f'% (epoch, loss, correct_num/ len(train_label)))
            if epoch % 100==0:
                stage_recorder(att_map, (im_shape[0], im_shape[1]))

        pred = sess.run( _pred, feed_dict={ _inputs:test_features,  _istrain: False })
        correct_num =len( [x  for x, y in zip(pred, test_label) if x==y])
        print('test:  accuracy: %f'% (  correct_num/ len(test_label)))
    return sess
            
        
    

train_test()

# test()
