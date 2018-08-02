import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
from tensorflow.contrib.layers.python.layers import layers as layers_lib

class Decoder:
    def __init__(self):
        pass
    def __call__(self):
        pass

class LSTM(Decoder):
    def __init__(self ):
        self.end_points = {}
        self.model_id = 'LSTM'
        self.model_path = ''
 
    def __call__(self, inputs, n_hidden = 100, n_class=1 ,learning_rate=1e-3):
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.learning_rate = learning_rate
        # inputs = tf.unstack(inputs, axis= 0 )
        with tf.variable_scope('Decoder'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            outs, state = tf.nn.dynamic_rnn( lstm_cell,inputs,dtype=tf.float32)
        self.end_points['out'] =  outs
        self.end_points['state'] =  state
        return self._softmax()

    def _softmax(self):
        outs =  self.end_points['out']
        with tf.variable_scope('LSTM_Softmax'):
            weight = tf.get_variable(shape= [self.n_hidden, self.n_class], name= 'w', dtype =tf.float32)
            bias = tf.get_variable(shape= [ self.n_class], name= 'b', dtype =tf.float32)
            logits =tf.matmul(outs[:,-1,:], weight) + bias
        with tf.variable_scope('LSTM_Loss'):
            ground_truth = tf.placeholder(shape=[None,], dtype=tf.int32)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = ground_truth))
            optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
            train_op =optimizer.minimize(loss)
        return ground_truth, loss, train_op





    
    

def decoder_factory(name = 'LSTM'):
    decoder = None
    if name == 'LSTM':
        decoder= LSTM()
    assert (decoder!=None)
    return decoder