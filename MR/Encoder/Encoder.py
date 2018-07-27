import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
from tensorflow.contrib.layers.python.layers import layers as layers_lib

class Encoder:
    def __init__(self):
        pass
    def build(self,input_shape, n_class):
        pass
    def load_model(self,model_path):
        pass
    def back_bone(self,input_shape, model_path ):
        pass
    
class Resnet_v2_101(Encoder):
    def __init__(self ):
        self.end_points = []
        self.model_id = 'Rest_v2_101'
        self.model_path = ''
 
    def build(self, input_shape = [None, 299, 299, 3], n_class=1 ):
        with tf.Graph().as_default() as graph:
            inputs = tf.placeholder(tf.float32,shape = input_shape, name='Inputs')
            is_train = tf.placeholder(tf.bool, [], 'is_train')
            #X = inputs
            arg_scope = slimNet.resnet_v2.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = slimNet.resnet_v2.resnet_v2_101(inputs, n_class, is_training= is_train,reuse =None)
        self.end_points = end_points

    
    def load_model(self, sess, model_path = '../../Model/inception_v3.ckpt'):
        sess = tf.Session()
        self.model_path = model_path
        try:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            print('restore all variations')
        except:
            try:
                varlist = [var for var in tf.model_variables() if
                           'logits' not in var.op.name]
                saver = tf.train.Saver(var_list=varlist)
                saver.restore(sess, self.model_path)
                print('restore part variations')
            except:
                print('Failed to restore model from %s' % self.model_path)

    def back_bone(self,input_shape):
        self.build(input_shape, 1)
        return self, self.end_points['inputs'], self.end_points['']



class Inception_v3(Encoder):
    def __init__(self ):
        self.end_points = []
        self.model_id = 'Inception_v3'
        self.model_path = ''
 
    def build(self, input_shape = [None, 299, 299, 3],n_class=1 ):
        inputs = tf.placeholder(tf.float32,shape = input_shape, name='Inputs')
        arg_scope = slimNet.inception.inception_v3_arg_scope()
        is_train = tf.placeholder(tf.bool,shape=[], name='is_train_BN')
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training =is_train):
                _, end_points = slimNet.inception.inception_v3_base(inputs)
        self.end_points = end_points
        self.end_points['Inputs'] = inputs
        self.end_points['is_train']  = is_train

    
    def load_model(self, sess, model_path = '../../Model/inception_v3.ckpt'):
        sess = tf.Session()
        self.model_path = model_path
        try:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            print('restore all variations')
        except:
            try:
                varlist = [var for var in tf.model_variables() if 'Logits' not in var.op.name and 'Aux' not in var.op.name]
                saver = tf.train.Saver(var_list=varlist)
                saver.restore(sess, self.model_path)
                print('restore part variations')
            except:
                print('Failed to restore model from %s' % self.model_path)

    def back_bone(self,input_shape ):
        self.build(input_shape)
        # [n,w,h,c] = self.end_points['Mixed_7c']
        return self.end_points['Inputs'], self.end_points['Mixed_7c'], self.end_points['is_train']