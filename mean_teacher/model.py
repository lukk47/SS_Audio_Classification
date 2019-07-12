import tensorflow as tf
from tensorflow.contrib import slim
from .framework import assert_shape
from . import weight_norm as wn
from . import nn


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          name=None):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):

            net = inputs
            assert_shape(net, [None, 32, 32, 3])

            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, 32, 32, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(tf.convert_to_tensor(translate),
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 16, 16, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 8, 8, 256])

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            assert_shape(net, [None, 6, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 10, init=is_initialization)

            assert_shape(primary_logits, [None, 10])
            
            return primary_logits


def audio(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          is_initialization=False,
          name=None,
          bg_noise=False,
          bg_noise_level=0,
          bg_noise_input=None):
    with tf.name_scope(name, "audio"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,nn.bg_noise_layer,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):
            #pylint: disable=no-value-for-parameter
            net = inputs
            net = tf.cond(tf.convert_to_tensor(bg_noise),
                          lambda: nn.bg_noise_layer(net, bg_noise_level=bg_noise_level, bg_noise_input=bg_noise_input),
                          lambda: net)
            net = tf.cond(tf.convert_to_tensor(normalize_input),
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(tf.convert_to_tensor(translate),
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')

            net = slim.flatten(net)

            primary_logits = wn.fully_connected(net, 30, init=is_initialization)
            
            return primary_logits
