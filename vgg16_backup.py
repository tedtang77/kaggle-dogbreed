from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

#[Added by Ted]
import math

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
#[Deprecated]# from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
from keras import backend as K
K.set_image_dim_ordering('th')
# http://forums.fast.ai/t/keras-2-released/1956/22
# http://forums.fast.ai/t/error-when-trying-to-fit-vgg-expected-lambda-input-2-to-have-shape-none-3-224-224-but-got-array-with-shape-16-224-224-3/3386/4
# K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')  # Ex: (3, 224, 224)

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
vgg_dropout = 0.5

def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16():
    """
        The VGG 16 Imagenet model
    """


    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.dropout = vgg_dropout
        self.create()
        self.get_classes()


    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer.
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            #[Deprecated]# model.add(Convolution2D(filters, 3, 3, activation='relu'))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def set_dropout(self, dropout=0.): 
        """
           Set new dropout to change the weights of all dense layers because of dropout changle
           
           Argss:
               dropout: The new dropout prabability (1 - keep_prov_new)
        """
        # scale = ( keep_prob_prev / keep_prob_new )
        scale = (1 - self.dropout) / (1 - dropout)
        
        # process_weights: proc_wgts
        layers = self.model.layers
        # Set new dropout of Dense Layers
        for layer in layers:
            if type(layer) is Dense: layer.set_weights( [o * scale for o in layer.get_weights()] )


    def create(self):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        
        self.set_dropout(dropout=0.)


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        
        #[Ted] Train multiple layers
        layers = model.layers
        # Get the index of the first dense layer...
        first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
        print("train from first dense layer: {}".format(first_dense_idx))        
        
        #...and set this and all subsequent layers to trainable
        for layer in layers[first_dense_idx:]: layer.trainable=True
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        # http://forums.fast.ai/t/keras-2-released/1956/22
        #batches.nb_class = batches.num_class
  
        self.ft(batches.num_classes)  #[deprecated]# self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices)) # get a list of all the class labels
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices and update model.classes
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.001):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        """
            Trains the model for a fixed number of epochs (iterations on a dataset).
            See Keras documentation: https://keras.io/models/model/
        """
        #[Deprecated]# self.model.fit(trn, labels, nb_epoch=nb_epoch,
        #                    validation_data=(val, val_labels), batch_size=batch_size)
        self.model.fit(trn, labels, epochs=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit(self, batches, val_batches, nb_epoch=1):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        # http://forums.fast.ai/t/keras-2-released/1956/22
        batches.nb_class = batches.num_classes
        batches.nb_sample = batches.samples
        val_batches.nb_class = val_batches.num_classes
        val_batches.nb_sample = val_batches.samples
        
        #[deprecated]# self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
        #                  validation_data=val_batches, nb_val_samples=val_batches.nb_sample)
        

        # see https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes:
        # and: https://keras.io/models/sequential/#sequential-model-methods
        # steps_per_epoch: 
        # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting
        # the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch
        # size.
        
        self.model.fit_generator(batches, 
                             steps_per_epoch=int(math.ceil(batches.samples/batches.batch_size)),
                             epochs=nb_epoch,
                             validation_data=val_batches, 
                             validation_steps=int(math.ceil(val_batches.samples/val_batches.batch_size)))



    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        # When you read Keras documentation (which you should read very often) https://keras.io/models/model/#predict_generator , predict_generator returns “labels / classes” but Kaggle wants you to predict “probability”. So get_batches() takes extra argument class_mode so that after putting class_mode=None to get_batches() in test(…) I can get “probability” from predict_generator instead of “ “labels / classes” as what I need.
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        
        # http://forums.fast.ai/t/keras-2-released/1956/22
        test_batches.nb_class = test_batches.num_classes
        test_batches.nb_sample = test_batches.samples
        
        #[deprecated]# return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
        # return test_batches, self.model.predict_generator(test_batches, test_batches.samples)
        return test_batches, self.model.predict_generator(test_batches, steps=int(math.ceil(test_batches.samples/test_batches.batch_size)))

