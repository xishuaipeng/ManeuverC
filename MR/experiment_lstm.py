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
class rnn:
    def __init__(self):
        self. rnn_cell  = None
        self.logits_cell = None

        self.ground_truth = None
        self.is_train= False





class Decoder:
    def __init__(self):
        self. rnn_cell  = None
        self.logits_cell = None
        self.ground_truth = None
        self.is_train= False
        pass

    def _lstm(self,  n_hidden, n_class):
        if self.rnn_cell is None:
            with tf.variable_scope('RNN'):
                self.is_train = tf.placeholder(dtype=tf.bool,shape = [], name='is_train')
                self. rnn_cell  = tf.contrib.rnn.GRUCell(n_hidden, activation = tf.nn.tanh)
                # BasicLSTMCell

        if self.logits_cell is None:
            with tf.variable_scope('Classifier'):
                self. logits_cell  =  tf.layers.Dense(n_class)#, activation=tf.sigmoid

        if self.ground_truth is None:
            with tf.variable_scope('Ground_Truth'):
                self. ground_truth  =  tf.placeholder(shape=[None], dtype = tf.int32, name = 'ground_truth')

    def __call__(self, inputs, n_hidden, n_class, learn_rate= 1e-3):
        n_bath = array_ops.shape(inputs)[0]
        self._lstm(n_hidden, n_class)
        inputs = tf.layers.batch_normalization(inputs, training=self.is_train)
        outs , state = tf.nn.dynamic_rnn(self.rnn_cell, inputs, dtype= tf.float32 )
        logits = self.logits_cell(outs[:,-1,:])
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.ground_truth))
        optimizer = tf.train.AdamOptimizer(learn_rate)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss)
        return self.ground_truth, loss, train_op , tf.argmax(logits, 1), self.is_train

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



    
def train_test( train_list = ['ID001_T001', 'ID001_T002', 'ID001_T003','ID001_T004','ID001_T009', 'ID001_T010'],\
    test_list = ['ID001_T012','ID001_T013','ID001_T014','ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'],\
    data_dir = '..\..\Out',\
    im_shape = [299,299,3],\
    time_step = 31):
    train_features, train_label = load_data(data_dir,train_list )
    test_features, test_label = load_data(data_dir,test_list )
    epoches = 5000
    net  = lstm()
    _inputs = tf.placeholder(shape=[None, time_step, 2048], dtype=tf.float32, name ='lstm_inputs')
    _ground_truth, _loss, _train_op, _pred ,_istrain = net(_inputs, 100,5, 1e-3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            loss, _, pred = sess.run((_loss, _train_op, _pred), feed_dict={_ground_truth: train_label, _inputs:train_features, _istrain: True })
            correct_num =len( [x  for x, y in zip(pred, train_label) if x==y])
            print('%d: loss: %f,  accuracy: %f'% (epoch, loss, correct_num/ len(train_label)))
        pred = sess.run( _pred, feed_dict={ _inputs:test_features,  _istrain: False })
        correct_num =len( [x  for x, y in zip(pred, test_label) if x==y])
        print('test:  accuracy: %f'% (  correct_num/ len(test_label)))

            
        
    

train_test()

# test()
