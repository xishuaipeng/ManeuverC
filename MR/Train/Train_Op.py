import tensorflow as tf 

class Train_Op():
    def __init__(self):
        self.id = 'train_op'

    def _loss_op(self, loss_name):
        if loss_name == 'Ssoftmax':
            loss_fun = tf.nn.sparse_softmax_cross_entropy_with_logits
        return loss_fun   

    def _optimizer_op(self,op_name, learning_rate=1e-4):
        if op_name == 'AdamOptimizer':
            op_fun = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return op_fun

    def build(self, loss_name, optimizer_name, learning_rate, logits_layer, label_layer):
        with tf.name_scope('optimizer'):
            loss_fun  = self._loss_op('Ssoftmax')
            ptimizer = self._optimizer_op('AdamOptimizer', learning_rate=learning_rate)
            loss = tf.reduce_sum(loss_fun(logits = logits_layer, labels = label_layer))
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = ptimizer.apply_gradients(grads_and_vars=grads_and_vars)
            return loss, train_op

