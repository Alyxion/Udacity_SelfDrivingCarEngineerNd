#
# Project 12 of Udacity's Self-Driving Car Engineer Nanodegree
#
# Submission of December 26th - Copyright (c) by Michael Ikemann
#

#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import os
import cv2

# -------- Definitions --------

# Checkpoint storage directory
storage_dir = 'checkpoints'
# Checkpoint file prefix
prefix = 'semseg'
# Defines if training is enabled
do_train = True
# Defines the checkpoint to continue from (with or without additional training). Set to zero to don't load any file
best_checkpoint = 0
# Defines the full file name of the best checkpoint file
checkpoint_fn = './{}/{}_{}.ckpt'.format(storage_dir, prefix, best_checkpoint)
# Hyper parameters, dropout of 0.5, learning_rate of 13-4, 200 epochs seem fine, still not converged after 200,
# batchSize of 16 reaches GPUs memory limit locally
hyperParameters = dict(keepProb=0.5, learningRate=1e-4, epochs=200, batchSize=16, l2NormFac=1e-3)

# Comment me out to directly execute the pretrained model
# do_train = False
# best_checkpoint = 195

# -------- End of Definitions --------

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # We import VGG16 and use theirs layers 7, 4 and 3
    #
    # - Layer 7 is it's most inner one, directly before the flattening / classification and has still 1/32 of the
    #   original size.
    # - Between Layer 4 and 7 there is only a single additional size reduction, so Layer 4 has half of the size of
    #   Layer 7 of 1/16
    # - Layer 3 is the most left reasonably classifiable one and has 1/8 of the size of the original image
    #
    # Each layer is first convoluted via a 1x1 convolution from it's original size to the count of classes, basically
    # teaching this convolutional layer to map from feature to class.
    #
    # Then layer 7 is resized to the size of layer 3 and combined with it. Afterwards this combination is deconvluted to
    # the size of layer 3 and combined with this as well, each time via tf.add.
    # Now the combination of the 3 layers is again deconvoluted to the size of the original and trained vs the ground
    # truth which as well has a dimension of num_classes.
    #
    # After the training process you can convert an input image of let's say 256x256x3 (RGB) to an output image
    # of 256x256xNumber_Of_Classes and so calculate the likeliness if a single pixel is part of for example the street
    # or the background.
    #
    # The original output is potentially slightly noise but can be filtered afterwards and exclude too undense outliers.
    #
    # For more details see https://arxiv.org/pdf/1605.06211.pdf
    #
    # Another additional paper in this context is this one https://arxiv.org/pdf/1708.02551.pdf where you can get an
    # idea of how to separate single car instances afterwards based upon the work of the first pure segmentation step:
    # https://arxiv.org/pdf/1708.02551.pdf

    # Layer 7 is the most inner layer, overall reduced by factor 32
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))
    # Classify layer 4 via 1x1 convolution, original 1/16 of original
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))
    # Layer 3 is shrinked by 1/8 of the original, classify pixels via 1x1 convolutions
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))

    # Deconvolute layer 7 from 1/32 to 1/16 via stepping of
    out = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))
    # We can add deconvoluted 1/16 layer 7 and 1/16 layer 4
    out = tf.add(out, layer4_1x1)
    # Deconvolute from 1/16 to 1/8
    out = tf.layers.conv2d_transpose(out, num_classes, 4, 2, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))
    # Add layers (4+7) of size 1/8 to layer 3 of size 1/8
    out = tf.add(out, layer3_1x1)
    # Scale combined layer by factor 8 to be back to original size (more or less)
    out = tf.layers.conv2d_transpose(out, num_classes, 16, 8, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(hyperParameters['l2NormFac']))

    return out


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, start_epoch = 0, model_saver = None, model_prefix = '', model_dir = ''):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param start_epoch: In case the training was continued this contains the start epoch, otherwise it's zero
    :param model_saver: The safer with which the model is stored
    :param model_prefix: The prefix of the weights backups filename
    :param model_dir: The model target directory
    """
    # TODO: Implement function

    best_loss = float('inf')

    # for all epochs
    for epoch in range(epochs):
        # fetch next batch
        batches = get_batches_fn(batch_size)
        # for each image/label combination
        for image, label in batches:
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label,
                                                                          keep_prob: hyperParameters['keepProb'],
                                                                          learning_rate: hyperParameters['learningRate']})

        if start_epoch!=0:
            real_epoch = start_epoch+epoch+1
        else:
            real_epoch = epoch

        print("Epoch : {}/{} - Loss: = {:.2f}".format(real_epoch, epochs+start_epoch, loss))

        # store new checkpoint if it's loss is lower than the last epoch one's
        if loss<best_loss and model_prefix!='':
            out_fn = os.path.join(model_dir, "{}_{}.ckpt".format(model_prefix, real_epoch))
            print("Training loss decreased from {} to {} - storing checkpoint in {}".format(best_loss, loss, out_fn))
            best_loss = loss;
            model_saver.save(sess, out_fn)

tests.test_train_nn(train_nn)

def run():
    print("running")
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        out = layers(vgg_layer3_out=layer3_out, vgg_layer4_out=layer4_out, vgg_layer7_out=layer7_out,
                     num_classes=num_classes)
        label = tf.placeholder(tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(out, label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())
        model_saver = tf.train.Saver()

        # load best checkpoint if one was defined
        if best_checkpoint != 0:
            model_saver.restore(sess, checkpoint_fn)

        # TODO: Train NN using the train_nn function
        if do_train:
            train_nn(sess, hyperParameters['epochs'], hyperParameters['batchSize'], get_batches_fn, train_op,
                     cross_entropy_loss, image_input, label, keep_prob, learning_rate, best_checkpoint, model_saver, prefix, storage_dir)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()