from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#sys.modules[__name__].__dict__.clear()

import tensorflow as tf
#import tensorlayer as tl
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

tf.reset_default_graph()

parser = argparse.ArgumentParser()

parser.add_argument("--mode", choices=["train", "test"])

parser.add_argument("--input_dir_all", help="path to folder containing images")
parser.add_argument("--output_dir_all", help="where to put output files")
parser.add_argument("--cv_info_dir", default=None, help="directory contains cross validation set ups")
parser.add_argument("--task_No", default=None, help="number of task, 1 means t1, 2 means t2, 3 means multi")
parser.add_argument("--desired_l1_loss", default=0.005, help="desired l1 loss for early stop the training < max_epochs ")


#parser.add_argument("--checkpoint", help="the dir contains the last trained model. for continuing training.")

parser.add_argument("--seed", type=int)
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])


a = parser.parse_args()


a.scale_size=512
EPS = 1e-12
CROP_SIZE = 512



# ORig: -------------------------
#Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
#Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
# End ORig: 
# Moha: -------------------------
Examples = collections.namedtuple("Examples", "paths, inputs, targets_1,targets_2, count, steps_per_epoch")

Model = collections.namedtuple("Model", "outputs_1, outputs_2, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train,gen_loss,discrim_loss_fake,discrim_loss_real, gen_loss_dice, gen_loss_jaccard, gen_loss_Tversky")
# end MOha

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    # Moha: based on code from tensorlayer
    """Soft dice (Sorensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    """
    
    output=deprocess(output)
    target=deprocess(target)
    
    output=tf.to_float(tf.to_int32(output > 0.5))
    target=tf.to_float(tf.to_int32(target > 0.5))    
    
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def tversky_loss(labels, predictions, alpha=0.3, beta=0.7, smooth=1e-10):
    # Moha: based on code from: https://analysiscenter.github.io/radio/_modules/radio/models/tf/losses.html#tversky_loss
    """ Tversky loss function.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containing target mask.
    predictions : tf.Tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing tversky loss.
    """
    
    labels =deprocess(labels)
    predictions =deprocess(predictions)
    
    labels = tf.contrib.layers.flatten(labels)
    predictions = tf.contrib.layers.flatten(predictions)
            
    labels=tf.to_float(tf.to_int32(labels > 0.5))
    predictions=tf.to_float(tf.to_int32(predictions > 0.5))
    
    
    truepos = tf.reduce_sum(labels * predictions)
    fp_and_fn = (alpha * tf.reduce_sum(predictions * (1 - labels))
                 + beta * tf.reduce_sum((1 - predictions) * labels))

    return (truepos + smooth) / (truepos + smooth + fp_and_fn)



def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    print('encoder:')
    print(generator_inputs.shape)
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)
        print(output.shape)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)
            print(output.shape)

    print('decoder:')
    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            print(output.shape)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)
        print(output.shape)

    return layers[-1]





def create_model_MT(inputs, targets_1, targets_2):

    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []
        print('discriminator:')

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
        print(input.shape)
        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)
            print(convolved.shape)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)
                print(convolved.shape)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)
            print(output.shape)

        return layers[-1]


    targets=tf.concat([targets_1, targets_2],axis=3)

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1 
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        discrim_loss_real=tf.reduce_mean(-(tf.log(predict_real + EPS)))
        discrim_loss_fake=tf.reduce_mean(-(tf.log(1 - predict_fake + EPS)))
    
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))        
        gen_loss_dice=1 - dice_coe(outputs, targets,loss_type='sorensen')
        gen_loss_jaccard=1 - dice_coe(outputs, targets,loss_type='jaccard')      
        
        gen_loss_Tversky=tf.abs(1-tversky_loss(targets,outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight 
    

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    
    # orig: ----------------------
    #update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    # end orig
    # Moha: ----------------------
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1,
                               gen_loss,discrim_loss_real,discrim_loss_fake, gen_loss_jaccard, 
                               gen_loss_dice,gen_loss_Tversky])

    outputs_1=outputs[:,:,:,:3]
    outputs_2=outputs[:,:,:,3:]
    # End Moha
    
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    
    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        
        discrim_loss=ema.average(discrim_loss),        
        discrim_grads_and_vars=discrim_grads_and_vars,
                
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),    
        gen_grads_and_vars=gen_grads_and_vars,


        train=tf.group(update_losses, incr_global_step, gen_train),
        # Noha: -----------------
        outputs_1=outputs_1,
        outputs_2=outputs_2,
        gen_loss=ema.average(gen_loss),
        discrim_loss_fake=ema.average(discrim_loss_fake),
        discrim_loss_real=ema.average(discrim_loss_real),        
        gen_loss_jaccard=ema.average(gen_loss_jaccard),
        gen_loss_dice=ema.average(gen_loss_dice),
        gen_loss_Tversky=ema.average(gen_loss_Tversky)
        # End Moha
    )


# def create_model(inputs, targets):
#     with tf.variable_scope("generator"):
#         out_channels = int(targets.get_shape()[-1])
#         outputs = create_generator(inputs, out_channels)

#     # create two copies of discriminator, one for real pairs and one for fake pairs
#     # they share the same underlying variables
#     with tf.name_scope("real_discriminator"):
#         with tf.variable_scope("discriminator"):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_real = create_discriminator(inputs, targets)

#     with tf.name_scope("fake_discriminator"):
#         with tf.variable_scope("discriminator", reuse=True):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_fake = create_discriminator(inputs, outputs)

#     with tf.name_scope("discriminator_loss"):
#         # minimizing -tf.log will try to get inputs to 1
#         # predict_real => 1 
#         # predict_fake => 0
#         discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
#         discrim_loss_real=tf.reduce_mean(-(tf.log(predict_real + EPS)))
#         discrim_loss_fake=tf.reduce_mean(-(tf.log(1 - predict_fake + EPS)))
    
#     with tf.name_scope("generator_loss"):
#         # predict_fake => 1
#         # abs(targets - outputs) => 0
#         gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
#         gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))        
#         gen_loss_dice=1 - dice_coe(outputs, targets,loss_type='sorensen')
#         gen_loss_jaccard=1 - dice_coe(outputs, targets,loss_type='jaccard')      
        
#         gen_loss_Tversky=tf.abs(1-tversky_loss(targets,outputs))
#         gen_loss = 100*(gen_loss_Tversky) + gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight 
    

#     with tf.name_scope("discriminator_train"):
#         discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
#         discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#         discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
#         discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

#     with tf.name_scope("generator_train"):
#         with tf.control_dependencies([discrim_train]):
#             gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#             gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#             gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#             gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

#     ema = tf.train.ExponentialMovingAverage(decay=0.99)
    
#     # orig: ----------------------
#     #update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
#     # end orig
#     # Moha: ----------------------
#     update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1,
#                                gen_loss,discrim_loss_real,discrim_loss_fake, gen_loss_jaccard, 
#                                gen_loss_dice,gen_loss_Tversky])
#     # End Moha
    
#     global_step = tf.train.get_or_create_global_step()
#     incr_global_step = tf.assign(global_step, global_step+1)

    
#     return Model(
#         predict_real=predict_real,
#         predict_fake=predict_fake,
        
#         discrim_loss=ema.average(discrim_loss),        
#         discrim_grads_and_vars=discrim_grads_and_vars,
                
#         gen_loss_GAN=ema.average(gen_loss_GAN),
#         gen_loss_L1=ema.average(gen_loss_L1),    
#         gen_grads_and_vars=gen_grads_and_vars,

#         outputs=outputs,
#         train=tf.group(update_losses, incr_global_step, gen_train),
#         # Noha: -----------------
#         gen_loss=ema.average(gen_loss),
#         discrim_loss_fake=ema.average(discrim_loss_fake),
#         discrim_loss_real=ema.average(discrim_loss_real),        
#         gen_loss_jaccard=ema.average(gen_loss_jaccard),
#         gen_loss_dice=ema.average(gen_loss_dice),
#         gen_loss_Tversky=ema.average(gen_loss_Tversky)
#         # End Moha
#     )


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])


        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//3,:])
        b_images = preprocess(raw_input[:,width//3:2*width//3,:])
        c_images = preprocess(raw_input[:,2*width//3:,:])
        
        

    if a.which_direction == "AtoB":
        inputs, targets_1, targets_2 = [a_images, b_images, c_images]
        #targets_2=c_images
    elif a.which_direction == "BtoA":
        #inputs, targets_1, targets_2 = [b_images, a_images]
        print("Error:::: just use AtoB direction")
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_1_images = transform(targets_1)
        target_2_images = transform(targets_2)
#        target_images=tf.concat([target_1_images,target_2_images],axis=3)

    paths_batch, inputs_batch, targets_1_batch, targets_2_batch = tf.train.batch([paths, input_images, target_1_images, target_2_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets_1=targets_1_batch,
        targets_2=targets_2_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )




def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs_1","targets_1", "outputs_2","targets_2"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        #table construction
        index.write("<th>name</th><th>input</th><th>output_1</th><th>target_1</th><th>output_2</th><th>target_2</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs_1", "targets_1", "outputs_2","targets_2"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)
        # Images that are represented using floating point values are expected to have values in the range [0,1)
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model_MT(examples.inputs, examples.targets_1, examples.targets_2)


    inputs = deprocess(examples.inputs)
    targets_1 = deprocess(examples.targets_1)
    targets_2 = deprocess(examples.targets_2)
    outputs_1 = deprocess(model.outputs_1)
    outputs_2 = deprocess(model.outputs_2)
    #print(outputs_2.shape)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets_1 = convert(targets_1)
        converted_targets_2 = convert(targets_2)

    with tf.name_scope("convert_outputs"):
        converted_outputs_1 = convert(outputs_1)
        converted_outputs_2 = convert(outputs_2)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets_1": tf.map_fn(tf.image.encode_png, converted_targets_1, dtype=tf.string, name="target_1_pngs"),
            "targets_2": tf.map_fn(tf.image.encode_png, converted_targets_2, dtype=tf.string, name="target_2_pngs"),
            "outputs_1": tf.map_fn(tf.image.encode_png, converted_outputs_1, dtype=tf.string, name="output_1_pngs"),
            "outputs_2": tf.map_fn(tf.image.encode_png, converted_outputs_2, dtype=tf.string, name="output_2_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets_1", converted_targets_1)
        tf.summary.image("targets_2", converted_targets_2)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs_1", converted_outputs_1)
        tf.summary.image("outputs_2", converted_outputs_2)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    
    # Moha: --------------
    tf.summary.scalar("discriminator_loss_real", model.discrim_loss_real)
    tf.summary.scalar("discriminator_loss_fake", model.discrim_loss_fake)
    tf.summary.scalar("generator_loss_GAN_total", model.gen_loss)
    tf.summary.scalar("generator_loss_GAN_dice", model.gen_loss_dice)
    tf.summary.scalar("generator_loss_GAN_jaccard", model.gen_loss_jaccard)    
    tf.summary.scalar("generator_loss_GAN_Tversky", model.gen_loss_Tversky)    

    
    # End MOha

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("#############################")
            print("loading model from checkpoint")
            print("#############################")
            try:
                checkpoint = tf.train.latest_checkpoint(a.checkpoint)
                saver.restore(sess, checkpoint)
            except:
                print("loading was unsuccessful")
                print("#############################")

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    # Moha: --------------------
                    fetches["discrim_loss_real"] = model.discrim_loss_real
                    fetches["discrim_loss_fake"] = model.discrim_loss_fake                    
                    fetches["gen_loss_total"] = model.gen_loss
                    fetches["gen_loss_dice"]= model.gen_loss_dice
                    fetches["gen_loss_jaccard"]= model.gen_loss_jaccard    
                    fetches["gen_loss_Tversky"]= model.gen_loss_Tversky    

                    # End MOha
                    


                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print('***********************************************')
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))                    

                    print("gen_loss_GAN -----------", results["gen_loss_GAN"])
                    print("gen_loss_L1 -----------", results["gen_loss_L1"])
                    print("discrim_loss_total -----------", results["discrim_loss"])

                    print("discrim_loss_real -----------", results["discrim_loss_real"])
                    print("discrim_loss_fake -----------", results["discrim_loss_fake"])
                    print("gen_loss_total -----------", results["gen_loss_total"])
                    
                    print("gen_loss_dice -----------", results["gen_loss_dice"])
                    print("gen_loss_jaccard -----------", results["gen_loss_jaccard"])
                    
                    print("gen_loss_Tversky -----------", results["gen_loss_Tversky"])
                    
                    
                    if results["gen_loss_L1"] < float(a.desired_l1_loss):
                        print("###################")                              
                        print("Reached desired error")
                        print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                        print("saving model")
                        saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                        break

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


def CombineImages(ImageListNames, task_No, input_dir_all):
    try:
        os.mkdir(write_to_dir)
    except:
        print('destination folder is already exist')
        shutil.rmtree(write_to_dir)
        os.mkdir(write_to_dir)


    for filename in ImageListNames:
        
        try:
            filename=filename.replace('\n','')
        except:
            no=0
            
        Image = io.imread(a.input_dir_all+'/Inputs/'+ filename)
        Size=Image.shape
        if len(Size)==2:
            Image_=np.zeros(shape=[Image.shape[0],Image.shape[1],3], dtype='uint8')
            Image_[:,:,0]=Image
            Image_[:,:,1]=Image
            Image_[:,:,2]=Image
            Image=Image_ 

        if task_No==1:
            
            print('1 is not acceptable for single task')
            sys.exit('1 is not acceptable for single task')
            
#            Image_t1=io.imread(a.input_dir_all+'/Targets_'+str(task_No) +'/'+ filename)
#            Size=Image_t1.shape
#            if len(Size)==2:
#                Image_=np.zeros(shape=[Image_t1.shape[0],Image_t1.shape[1],3], dtype='uint8')
#                Image_[:,:,0]=Image_t1
#                Image_[:,:,1]=Image_t1
#                Image_[:,:,2]=Image_t1
#                Image_t1=Image_
#            Image_combined=np.zeros(shape=[Image.shape[0],2*Image.shape[1],3], dtype='uint8')
#            Image_combined[:,:CROP_SIZE,:]=Image
#            Image_combined[:,CROP_SIZE:,:]=Image_t1

        if task_No==2:        
            print('2 is not acceptable for single task')
            sys.exit('2 is not acceptable for single task')
            
#            Image_t2=io.imread(a.input_dir_all+'/Targets_'+str(task_No) +'/'+ filename)
#            Size=Image_t2.shape
#            if len(Size)==2:
#                Image_=np.zeros(shape=[Image_t2.shape[0],Image_t2.shape[1],3], dtype='uint8')
#                Image_[:,:,0]=Image_t2
#                Image_[:,:,1]=Image_t2
#                Image_[:,:,2]=Image_t2
#                Image_t2=Image_ 
#            Image_combined=np.zeros(shape=[Image.shape[0],2*Image.shape[1],3], dtype='uint8')
#            Image_combined[:,:CROP_SIZE,:]=Image
#            Image_combined[:,CROP_SIZE:,:]=Image_t2

        if task_No==3:        
            Image_t1=io.imread(a.input_dir_all+'/Targets_1'+'/'+ filename)
            Image_t2=io.imread(a.input_dir_all+'/Targets_2'+'/'+ filename)
            Size=Image_t1.shape
            if len(Size)==2:
                Image_=np.zeros(shape=[Image_t1.shape[0],Image_t1.shape[1],3], dtype='uint8')
                Image_[:,:,0]=Image_t1
                Image_[:,:,1]=Image_t1
                Image_[:,:,2]=Image_t1
                Image_t1=Image_
            Size=Image_t2.shape
            if len(Size)==2:
                Image_=np.zeros(shape=[Image_t2.shape[0],Image_t2.shape[1],3], dtype='uint8')
                Image_[:,:,0]=Image_t2
                Image_[:,:,1]=Image_t2
                Image_[:,:,2]=Image_t2
                Image_t2=Image_                                                 
            Image_combined=np.zeros(shape=[Image.shape[0],3*Image.shape[1],3], dtype='uint8')
            Image_combined[:,:CROP_SIZE,:]=Image
            Image_combined[:,CROP_SIZE:2*CROP_SIZE,:]=Image_t1            
            Image_combined[:,2*CROP_SIZE:,:]=Image_t2            
                
    
        
        io.imsave(write_to_dir+filename, Image_combined)


from skimage import io
from scipy import misc
import shutil
import sys

#a.input_dir_all='../ImageData'
#a.cv_info_dir='../CV_info'
#a.output_dir_all='../Outputs'
#a.task_No='3'
#a.max_epochs=2000
#a.desired_l1_loss=0.05
#a.batch_size=10


CvDirs = glob.glob(os.path.join(a.cv_info_dir, "*"))

for cv in range(1,len(CvDirs)+1): # #################

    
    print('#####################')
    print('### cv:',str(cv),'##################')
    print('#####################')
    trainfile=a.cv_info_dir+'/set_'+str(cv)+'/train.txt'
    testfile=a.cv_info_dir+'/set_'+str(cv)+'/test.txt'
    
    
    text_file = open(trainfile)
    list_train = text_file.readlines()
    text_file.close()
    text_file = open(testfile)
    list_test = text_file.readlines()
    text_file.close()
    
    
    a.mode='train'
    print('############',a.mode)
    a.checkpoint=a.output_dir_all+'/pix2pix_MT/Models_MT'+'/set_'+str(cv)
    tf.reset_default_graph()
    write_to_dir = a.input_dir_all + "/Temp_CombinedImages/"
    CombineImages(list_train, int(a.task_No), a.input_dir_all)
    a.input_dir=write_to_dir
    a.output_dir=a.output_dir_all+'/pix2pix_MT/Models_MT'+'/set_'+str(cv)
    main()
    
    a.mode='test'
    print('############',a.mode)
    tf.reset_default_graph()
    shutil.rmtree(write_to_dir)
    CombineImages(list_test, int(a.task_No), a.input_dir_all)
    a.checkpoint=a.output_dir_all+'/pix2pix_MT/Models_MT'+'/set_'+str(cv)
    a.input_dir=write_to_dir
    a.output_dir=a.output_dir_all+'/pix2pix_MT/Results_MT'+'/set_'+str(cv)
    try:
        shutil.rmtree(a.output_dir)
    except:
        no=1
    main()
    try:
        shutil.rmtree(a.input_dir)
    except:
        no=1
