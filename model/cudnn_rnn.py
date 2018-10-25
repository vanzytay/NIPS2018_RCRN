from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class cudnn_rnn:
    """ Universal cudnn_rnn class
    Supports both LSTM and GRU

    Variational dropout is optional
    """

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                    is_train=None, scope=None, init=None, rnn_type='',
                    direction='bidirectional'):
        if(init is None):
            rnn_init = tf.random_normal_initializer(stddev=0.1)
        else:
            rnn_init = init
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.num_units = num_units
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.direction=direction
        self.num_params = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            if('LSTM' in rnn_type):
                gru_fw = tf.contrib.cudnn_rnn.CudnnLSTM(
                    1, num_units, kernel_initializer=rnn_init)
                if(self.direction=='bidirectional'):
                    gru_bw = tf.contrib.cudnn_rnn.CudnnLSTM(
                        1, num_units, kernel_initializer=rnn_init)
                else:
                    gru_bw = None
            else:
                gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                    1, num_units, kernel_initializer=rnn_init)
                if(self.direction=='bidirectional'):
                    gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                        1, num_units, kernel_initializer=rnn_init)
                else:
                    gru_bw = None

            self.grus.append((gru_fw, gru_bw, ))

    def __call__(self, inputs, seq_len, batch_size=None,
                is_train=None, concat_layers=True,
                var_drop=1, train_init=0):
        # batch_size = inputs.get_shape().as_list()[0]
        batch_size = tf.shape(inputs)[0]
        outputs = [tf.transpose(inputs, [1, 0, 2])]

        for layer in range(self.num_layers):
            if(train_init):
                init_fw = tf.tile(tf.Variable(
                    tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                if(self.direction=='bidirectional'):
                    init_bw = tf.tile(tf.Variable(
                        tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                else:
                    init_bw = None
            else:
                init_fw = tf.tile(tf.zeros([1, 1, self.num_units]),
                                [1, batch_size, 1])
                if(self.direction=='bidirectional'):
                    init_bw = tf.tile(tf.zeros([1, 1, self.num_units]),
                                    [1, batch_size, 1])
                else:
                    init_bw = None
            if(var_drop==1):
                mask_fw = dropout(tf.ones([1, batch_size, self.input_size],
                                    dtype=tf.float32),
                                  keep_prob=self.keep_prob, is_train=self.is_train)
                output_fw = outputs[-1] * mask_fw
                if(self.direction=='bidirectional'):
                    mask_bw = dropout(tf.ones([1, batch_size, self.input_size],
                                        dtype=tf.float32),
                                      keep_prob=self.keep_prob, is_train=self.is_train)
                    output_bw = outputs[-1] * mask_bw
            else:
                output_fw = outputs[-1]
                output_fw = dropout(output_fw,
                                keep_prob=self.keep_prob,
                                is_train=self.is_train)
                if(self.direction=='bidirectional'):
                    output_bw = outputs[-1]
                    output_bw = dropout(output_bw,
                                    keep_prob=self.keep_prob,
                                    is_train=self.is_train)
            gru_fw, gru_bw = self.grus[layer]
            if('LSTM' in self.rnn_type):
                init_state1 = (init_fw, init_fw)
                init_state2 = (init_bw, init_bw)
            else:
                init_state1 = (init_fw,)
                init_state2 = (init_bw,)

            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    output_fw, initial_state=init_state1)
                self.num_params += gru_fw.canonical_weight_shapes

            out = out_fw

            if(self.direction=='bidirectional'):
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        output_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = gru_bw(inputs_bw, initial_state=init_state2)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out = tf.concat([out, out_bw], 2)
                self.num_params += gru_bw.canonical_weight_shapes
            outputs.append(out)
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        counter = 0
        for t in self.num_params:
            counter += t[0] * t[1]
        print('Cudnn Parameters={}'.format(counter))

        return res
