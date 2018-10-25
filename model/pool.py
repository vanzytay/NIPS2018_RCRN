import tensorflow as tf
from .fo_pool import *

class RCRNpooling(tf.nn.rnn_cell.RNNCell):
    """ Handles the pooling using two BiRNNs as gates
    """

    def __init__(self, out_fmaps, pool_type,
                    initializer=None, in_dim=None):
        self.__pool_type = pool_type
        self.__out_fmaps = out_fmaps
        if(initializer is None):
            initialzier = tf.orthogonal_initializer()

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, feats + [gates]]
        """
        pool_type = self.__pool_type
        with tf.variable_scope(scope or "QRNN-{}-pooling".format(pool_type)):
            if pool_type == 'f':
                # extract Z activations and F gate activations
                Z, F = tf.split(inputs, 2, 1)
                # return the dynamic average pooling
                output = tf.multiply(F, state) + tf.multiply(tf.subtract(1., F), Z)
                return output, output
            elif pool_type == 'fo':
                # extract Z, F gate and O gate
                Z, F, O = tf.split(inputs, 3, 1)
                new_state = tf.multiply(F, state) + tf.multiply(tf.subtract(1., F), Z)
                output = tf.multiply(O, new_state)
                return output, new_state
            elif pool_type == 'ifo':
                # extract Z, I gate, F gate, and O gate
                Z, I, F, O = tf.split(inputs, 4, 1)
                new_state = tf.multiply(F, state) + tf.multiply(I, Z)
                output = tf.multiply(O, new_state)
                return output, new_state
