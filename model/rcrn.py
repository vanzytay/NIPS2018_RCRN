from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .pool import *
from .fo_pool import *
from .cudnn_rnn import *

def RCRN(embed, lengths,
        initializer=None, name='', reuse=None,
        dropout=None, is_train=None, var_drop=0, dim=None,
        direction='bidirectional', rnn_type='LSTM',
        cell_dropout=None,
        fuse_kernel=0, train_init=0):
    """ Implementation of RCRN encoders

    Args:
        embed: tensor sequence of bsz x seq_len x dim
        lengths: tensor of [bsz] with actual lengths
        initializer: tensorflow initializer
        name: give it a name!
        reuse: whether to reuse vars
        dropout: tensor scalar. Pass it via feed_dict
        is_train: tensor bool, whether training or not
        var_drop: whether to use variational dropout
        dim: int size of the output dim (if not, uses input dim)
        direction: 'bidirectional' or 'unidirectional'
        rnn_type: the rnn type of internal cell
        cell_dropout: bool, whether to use dropout during recurrence
        fuse_kernel: int, 1 to use fast cuda ops and 0 not to
        train_init: whether starting state is zero or trainable parameters
    """

    if(dim is None):
        dim = embed.get_shape().as_list()[2]

    dim2 = dim
    if(direction=='bidirectional'):
        dim2 = dim2 * 2

    batch_size = tf.shape(embed)[0]
    if(train_init):
        initial_state = tf.tile(tf.Variable(
            tf.zeros([1, dim2])), [batch_size, 1])
    else:
        initial_state = tf.tile(
            tf.zeros([1, dim2]), [batch_size, 1])

    d = dim
    bsz = batch_size

    with tf.variable_scope("main_rnn", reuse=reuse):
        main_rnn = cudnn_rnn(num_layers=1, num_units=d,
                    batch_size=bsz,
                    input_size=embed.get_shape().as_list()[-1],
                    keep_prob=dropout,
                    is_train=is_train,
                    direction=direction,
                    rnn_type=rnn_type,
                    init=initializer
                    )
        proj_embed = main_rnn(embed,
                        seq_len=lengths,
                        var_drop=var_drop,
                        train_init=train_init
                        )
    with tf.variable_scope("fg_rnn", reuse=reuse):
        forget_rnn = cudnn_rnn(num_layers=1, num_units=d,
                    batch_size=bsz,
                    input_size=embed.get_shape().as_list()[-1],
                    keep_prob=dropout,
                    direction=direction,
                    is_train=is_train,
                    rnn_type=rnn_type,
                    init=initializer)
        forget_gate = forget_rnn(embed, seq_len=lengths,
                                var_drop=var_drop,
                                train_init=train_init
                                )
    with tf.variable_scope("og_rnn", reuse=reuse):
        output_rnn = cudnn_rnn(num_layers=1, num_units=d,
                    batch_size=bsz,
                    input_size=embed.get_shape().as_list()[-1],
                    keep_prob=dropout,
                    direction=direction,
                    is_train=is_train,
                    rnn_type=rnn_type,
                    init=initializer)
        output_gate = output_rnn(embed, seq_len=lengths,
                                var_drop=var_drop,
                                train_init=train_init
                                )

    # forget_gate = gate
    pooling = RCRNpooling(dim2, 'fo')
    if(cell_dropout is not None and cell_dropout<1.0):
        print("Adding dropout")
        pooling = tf.contrib.rnn.DropoutWrapper(pooling,
                            output_keep_prob=cell_dropout)
    initial_state = pooling.zero_state(tf.shape(embed)[0],
                                        tf.float32)
    output_gate = tf.nn.sigmoid(output_gate)
    forget_gate = tf.nn.sigmoid(forget_gate)

    if(fuse_kernel==1):
        print("Using Cuda-level Fused Kernel Optimization")
        with tf.name_scope("FoPool"):
            c = fo_pool(proj_embed, forget_gate,
                        initial_state=initial_state,
                        time_major=0)
        embed = c * output_gate
    else:
        stack_input = tf.concat([proj_embed,
                                    forget_gate, output_gate], 2)
        embed, _ = tf.nn.dynamic_rnn(pooling, stack_input,
                                initial_state=initial_state,
                                sequence_length=tf.cast(
                                            lengths,tf.int32))


    return embed, tf.reduce_sum(embed, 1)
