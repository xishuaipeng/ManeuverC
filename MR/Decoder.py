import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import array_ops
weight_initializer = tf.contrib.layers.xavier_initializer()
const_initializer = tf.constant_initializer(0.0)
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
        self.lstm_build = False
        self.softmax_build = False
        self.loss_build = False


    def _lstm_cell(self):
        with tf.variable_scope('Decoder', reuse =self.lstm_build ):
            [ib,it, ic] = self.input_shape
            lc = ic.value/2
            if self.lstm_build is False:
                # lstm_cell_two = tf.contrib.rnn.BasicLSTMCell(int(n_hidden))
                self.trans_input = tf.layers.Dense(lc, activation= tf.nn.relu,name = 'transfer_input')
                self.bn_input = tf.layers.BatchNormalization(name = 'bn_input')
                self.multi_cell  = tf.contrib.rnn.BasicLSTMCell(int(self.n_hidden))
                self.lstm_build = True

    def _softmax_layer(self):
        if self.softmax_build is False:
            with tf.variable_scope('LSTM_Softmax'):
                self.logistic = tf.layers.Dense(self.n_class,activation= tf.nn.sigmoid ,name = 'Logistic')
            self.softmax_build = True

            
    def _loss_layer(self):
        if self.loss_build is False:
            with tf.variable_scope('LSTM_Loss'):
                self.ground_truth = tf.placeholder(shape=[None,], dtype=tf.int32)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate)
            self.loss_build = True

 
    def __call__(self, inputs, n_hidden = 100, n_class=1 ,learning_rate=1e-3):
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.input_shape = inputs.shape
        self.learning_rate = learning_rate
        self._lstm_cell()
        inputs = self.trans_input(inputs)
        inputs = self.bn_input(inputs)
        outs, state  = tf.nn.dynamic_rnn( self.multi_cell, inputs, dtype=tf.float32)
            # if state == outs:
            #     print('same')
        self.end_points['out'] =  tf.reduce_mean(outs, axis=1)
        self.end_points['state'] =  state
        return self._softmax()

    def _softmax(self):
        outs =  self.end_points['out']
        self._softmax_layer()
        logits =  self.logistic(outs)
        self._loss_layer()
        with tf.variable_scope('LSTM_Loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = self.ground_truth))
            grad_vars = self.optimizer.compute_gradients(loss)
            grad_vars = [(grad, var) for grad, var in grad_vars if var not in ['InceptionV3'] ]
            grads = [grad for grad, var in grad_vars]
            vars = [var for grad, var in grad_vars]
            clipped_grads, _  = tf.clip_by_global_norm(grads, 5 )
            train_op = self.optimizer.apply_gradients(zip(clipped_grads, vars))




            # =self.optimizer.minimize(loss)
        return self.ground_truth, loss, train_op, tf.argmax(logits,1)



class Attention_LSTM(Decoder):
    def __init__(self ):
        self.end_points = {}
        self.model_id = 'Attention_LSTM'
        self.model_path = ''
        self.att_build = False
        self.rnn_build  = False
        self.init_build = False

    def _init_state(self,inputs, n_hidden):
        [ib,it, iw, ih, ic] = inputs.shape
        inputs = tf.reshape(inputs, shape = [-1, it, iw*ih*ic])
        inputs = tf.reduce_mean(inputs, axis = 1)
        with tf.variable_scope('init_state'):
            if self.init_build is False:
                ws = tf. get_variable(name = 'sweight', shape = [ic*iw*ih, n_hidden], dtype =tf.float32)
                bs= tf. get_variable(name = 'sbias', shape = [ n_hidden], dtype =tf.float32)
                wh = tf. get_variable(name = 'hweight', shape = [ic*iw*ih, n_hidden], dtype =tf.float32)
                bh= tf. get_variable(name = 'hbias', shape = [ n_hidden], dtype =tf.float32)
            c =  tf.matmul(inputs, ws) + bs
            h =  tf.matmul(inputs, wh) + bh
        return h,c



             

        
    def _attention(self, inputs, h):
        [ib,il,ic] = inputs.shape
        [sb,sc] = h.shape
        # assert ib==sb
        with tf.variable_scope('Attention'):
            #b,l,c
            if self.att_build == False:
                self.attention_wsi = tf.get_variable(shape=[sc, ic], dtype = tf.float32, name='wsi')
                self.attention_wc1 =  tf.get_variable(shape=[ ic, 1], dtype = tf.float32, name='wc1')
                self.att_build = True
            # transfer state to input
            ts = tf.matmul(h,self.attention_wsi)#sb, ic
            #add two
            m =  tf.nn.relu(inputs + tf.expand_dims(ts, 1))#sb, il, ic
            #
            m = tf.reshape(m,[-1, ic])
            m = tf.squeeze(tf.matmul(m, self.attention_wc1))
            m = tf.reshape(m, [-1, il])
            s =  tf.nn.softmax(m)#sb,il
            # s = tf.reshape(s, [sb,il])
            out = tf.reduce_sum(inputs* tf.expand_dims(s,-1) , axis= 1)#sb, il,c-> sb,c
            return s, out
    
    def __call__(self, inputs, n_hidden = 100, n_class=1 ,learning_rate=1e-3):
        [batch, time, w, h, c] = inputs.shape
        batch = batch
        l = w.value * h.value
        c = c.value
        time = time.value
        init_state =self._init_state(inputs, n_hidden)
        inputs = tf.reshape(inputs, [-1,time, l, c])
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.learning_rate = learning_rate
        inputs = tf.unstack(inputs, axis = 1)
        if self.rnn_build is False:
            self._lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            self.rnn_build = True
        outs = []
        maps = []
        for t, x in enumerate(inputs) :#b,1,l,c
            if t == 0:
                state = init_state
                h = init_state[0]
            att_map, att_feature = self._attention(x, h)
            h, state = self._lstm_cell(att_feature,state)
            outs.extend([h]) 
            maps.extend([att_map])
        self.end_points['out'] =  tf.stack(outs, axis=1)
        self.end_points['map'] =  tf.stack(maps, axis=1)
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
        return ground_truth, loss, train_op, tf.argmax(logits,1)


def decoder_factory(name = 'LSTM'):
    decoder = None
    if name == 'LSTM':
        decoder= LSTM()
    elif name == 'ATT_LSTM':
        decoder = Attention_LSTM()
    assert (decoder!=None)
    return decoder