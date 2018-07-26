import tensorflow as tf
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet

class Attention:
    def __init__(self):
        pass
    def build(self,input_shape, n_class):
        pass
    def load_model(self,model_path):
        pass
    def back_bone(self,input_shape, model_path ):
        pass
    
class Classical_Attention(Attention):
    def __init__(self ):
        self.end_points = {}
        self.n_hidden = 0
        self.n_class = 0
        self.model_id = 'Classical_Net'
        self.model_path = ''
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.rnn_layer = tf.nn.rnn_cell.BasicLSTMCell
 
    def _get_initialization(self, features, name):
        with tf.variable_scope(name):
            features_mean = tf.reduce_mean(features, 1)
            features_mean = tf.reshape(features_mean, [-1, self.T, self.D])
            features_mean = tf.reduce_mean(features_mean, 1)

            w_h = tf.get_variable('w_h', [self.D, self.n_hidden], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.n_hidden], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.n_hidden], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.n_hidden], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h
            
    def _attention_layer(self,features, h, reuse):
        with tf.variable_scope('attention_layer', reuse=reuse):
            features_flat = tf.reshape(features, [-1, self.D])
            w_h = tf.get_variable('w_h', [self.n_hidden, self.D], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.D], initializer=self.const_initializer)
            w_f = tf.get_variable('w_f', [self.D, self.D], initializer=self.weight_initializer)
            b_f = tf.get_variable('b_f', [self.D], initializer=self.const_initializer)
            h_att = tf.matmul(h, w_h)  + b_h
            f_att = tf.matmul(features_flat, w_f)  + b_f

            w = tf.get_variable('w', [self.D, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D, 1], initializer=self.weight_initializer)

            hf_att = tf.nn.relu(h_att + f_att)
            out_att = tf.reshape(tf.matmul(tf.reshape(hf_att, [-1, self.D]), w), [-1, self.L])
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            return context, alpha
    

    # decoding of lstm
    def _decode_rnn(self, h, context, dropout, reuse):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.n_hidden, self.n_class], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.n_class], initializer=self.const_initializer)


            w_out = tf.get_variable('w_out', [self.n_class, self.n_class], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.n_class], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.n_class], initializer=self.weight_initializer)
            h_logits += tf.matmul(context, w_ctx2out)
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out

            return out_logits

    def build(self, feature_layer , n_hidenn = 10, n_class = 1 ):
        self.n_class = n_class
        self.n_hidden = n_hidenn
        label_layer = tf.placeholder(tf.int32, [None,])
        [T,W,H,C] = feature_layer.shape
        self.T = T
        self.L = W*H
        self.D = C
        feature_layer = tf.reshape(feature_layer,[-1,self.L,self.D])
        rnn_cell = self.rnn_layer(num_units = n_hidenn)
        c, h = self._get_initialization(features = feature_layer, name='rnn-cell')
        alpha_list = []
        for t in range(self.T):  # each t of lstm-layers process
            frame_feature = tf.slice(feature_layer, [t, 0, 0], [ 1, self.L,self.D])
            context, alpha = self._attention_layer(frame_feature, h, reuse=(t != 0))
            alpha_list.append(alpha)

            with tf.variable_scope("RNN", reuse=(t != 0)):
                _, (c, h) = rnn_cell(context, (c, h))
        logits = self._decode_rnn(h, context, dropout=True, reuse=False)
        self.end_points['logits'] = logits
        self.end_points['attention'] = alpha_list
        self.end_points['gtruth'] = label_layer
        return self.end_points['logits'],self.end_points['gtruth'] ,self.end_points['attention'] 
