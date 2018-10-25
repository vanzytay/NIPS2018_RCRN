import tensorflow as tf
import os

custom_op_path = os.path.dirname(os.path.abspath(__file__)) + '/qrnn_lib.so'
# print("Loading Custom Op from {}".format(custom_op_path))
qrnn_lib = tf.load_op_library(custom_op_path)

time_major_fo_pool_unsliced = qrnn_lib.time_major_fo_pool
time_major_bwd_fo_pool = qrnn_lib.time_major_bwd_fo_pool

batch_major_fo_pool_unsliced = qrnn_lib.batch_major_fo_pool
batch_major_bwd_fo_pool = qrnn_lib.batch_major_bwd_fo_pool

@tf.RegisterGradient("TimeMajorFoPool")
def _fo_pool_grad(op, grad):
    return time_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0],
                                  forget=op.inputs[1], gh=grad)

@tf.RegisterGradient("BatchMajorFoPool")
def _fo_pool_grad(op, grad):
    return batch_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0],
                                   forget=op.inputs[1], gh=grad)


def fo_pool(x, forget, initial_state=None, time_major=False):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        x: Tensor, input values in [Batch, Time, Channels] format,
           float32 or double
           or [Time, Batch, Channels] if time_major
        forget: Tensor, input values in [Batch, Time, Channels] format,
           float32 or double. Usually in the range 0-1.
           or [Time, Batch, Channels] if time_major
        initial_state: Tensor, initial hidden state values in [Batch, Channels] format,
           float32 or double.
    Returns:
        Tensor: fo_pooled output, [Batch, Time, Channels] format
                or [Time, Batch, Channels] if time_major
    """
    if initial_state is None:
        initial_state = tf.zeros((tf.shape(x)[1] if time_major else tf.shape(x)[0],
                                  tf.shape(x)[2]), dtype=tf.dtype)
    if time_major:
        return time_major_fo_pool_unsliced(x, forget, initial_state)[1:]
    else:
        return batch_major_fo_pool_unsliced(x, forget, initial_state)[:, 1:]
