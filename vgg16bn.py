import os, json, math
from glob import glob
import numpy as np

from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import image

from keras import backend as K
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first') # Ex: (3, 224, 224)

# vgg_mean (in BGR)
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
vgg_dropout = 0.5


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model
        (VGG-16 and VGG-19 were trained using Caffe, and Caffe uses OpenCV to load images which uses BGR by default, so both VGG models are expecting BGR images.)
        
        Args:
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:,::-1] # reverse axis RGB into BGR


class Vgg16BN():
    """
        The VGG 16 Imagenet model with Batch Normalization for the Dense Layers
    """
    
    
    def __init__(self, size=(224, 224), include_top=True):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.dropout = vgg_dropout
        self.create(size, include_top)
        self.get_classes()
        
        
        
    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it's not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
        
    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the Vgg16BN model
            
            Args:
                imgs (np.ndarray) : An array of N images (N x width x height x channels).
                details (boolean) : whether print details or not
                
            Returns:
                preds (np.array)  : Highest confidence value of the predictions for each image
                idxs (np.ndarray) : Class index of the predictions with the max confidence
                classes (list)    : Class labels of the predictions with the max confidence            
        """
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs(i)] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes
    
    
    def ConvBlock(self, layers, filters):
        """
            Adds specific number of ZeroPadding and Convolution Layers to the model,
            and a MaxPooling layer to the end.
            
            Args:
                layers (int):    The number of zero padded convolution layers to 
                                 be added to the model
                filters (int):   The number of convolution filters to be created 
                                 for each layer
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(filters, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
                      
    def FCBlock(self, name=None):
        """
            Adds a fully connected layer of 4096 neurons to the model with
            a dropout of 0.5 (default Vgg dropout)
        """
        model = self.model
        model.add(Dense(4096, activation='relu', name=name))
        model.add(BatchNormalization())
        model.add(Dropout(vgg_dropout))

    
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
    
    
    def get_split_models(self):
        """
            Gets the two models spliting convolution model and dense model at Flatten layer
            
            Returns:
                conv_model: the model constructing by Vgg convolution layers ending at the last MaxPooling2D layer 
                fc_model: the model constructing by Vgg dense layers starting at Flatten layer
            
        """
        flatten_idx = [idx for idx, layer in enumerate(self.model.layers) if type(layer)==Flatten][0]
        
        conv_model = Sequential(self.model.layers[:flatten_idx])
        conv_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        fc_model = Sequential([ 
            Flatten(input_shape=conv_model.layers[-1].output_shape[1:]) 
        ])
        for layer in self.model.layers[flatten_idx+1:]: 
            fc_model.add(layer)
        fc_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return conv_model, fc_model
   

    def create(self, size, include_top):
        """
            Creates the Vgg16BN architecture and loads the pretrained weights
            
            Args:
                size (tuple(int)): (height, weight) of input image size. default: (224, 224)
                include_top (boolean): whether includes the top dense layers or only the convolution layers
        """
        if size != (224, 224):
            include_top = False
                      
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)
        
        if not include_top:
            fname = 'vgg16_bn_conv.h5'
            model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
            return
        
        model.add(Flatten())
        self.FCBlock(name='FC1')
        self.FCBlock(name='FC2')
        model.add(Dense(1000, activation='softmax', name='softmax'))
        
        fname = 'vgg16_bn.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        
        self.set_dropout(dropout=0.)
        
                      
    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Take the path to a directory, and generates batches of augmented/normalized data. 
            Yields batches indefinitely, in an indefinte loop.
            
            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224,224), 
                class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)
      
    
    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (Fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn 
            weights for the last layer in subsequent training.
            
            Args:
                num: Number of neurons of the last layer
        """
        model = self.model
        model.pop()
        for layer in model.layers:  layer.trainable = False
            
        # make the model trainable for all FC layers
        layers = model.layers
        last_conv_idx = [idx for idx, layer in enumerate(layers) if type(layer) is Conv2D][-1]
        for conv_layer in layers[:last_conv_idx+1]: conv_layer.trainable = False
        for fc_layer in layers[last_conv_idx+1:]: fc_layer.trainable = True
        
        model.add(Dense(num, activation='softmax'))
        self.compile()
    
    
    def finetune(self, batches):
        """
            Modifies the original VGG16BN network architecture and update self.classes for new training data
            
            Args:
                batches : a keras.preprocessing.image.ImageDataGenerator object.
                          See definition of get_batches()
        """
        self.ft(batches.num_classes)
                      
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}
        classes = list(iter(batches.class_indices))  # Get a list of all the class labels    
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
                      
    
    def fit_data(self, trn, label, val, val_label, epochs=1, batch_size=64, verbose=2):
        """
            Trains the model for a fixed number of epochs (iterations on a dataset).
            See Keras documentation: https://keras.io/models/model/#fit
        """
        self.model.fit(trn, label, batch_size=batch_size, epochs=epochs, 
                       validation_data=(val, val_label), verbose=verbose)
                      
    
    def fit(self, batches, val_batches, epochs=1, verbose=2):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/#fit_generator
        """
        self.model.fit_generator(batches, 
                                 steps_per_epoch=int(math.ceil(batches.samples/batches.batch_size)),
                                 epochs=epochs,
                                 validation_data=val_batches,
                                 validation_steps=int(math.ceil(val_batches.samples/val_batches.batch_size)),
                                 verbose=verbose)
    
                      
    def test(self, path, batch_size=8, verbose=0):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch
            
            See Keras documentation: https://keras.io/models/model/#predict_generator
            
            Args:
                path (string) :  Path to the target directory. It should contain 
                                one subdirectory per class.
                batch_size (int) : The number of images to be considered in each batch.
            
            Returns:
                test_batches 
                numpy array(s) of predictions for the test batches.
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        
        return test_batches, self.model.predict_generator(test_batches,
                                                          steps=int(math.ceil(test_batches.samples/test_batches.batch_size)),
                                                          verbose=verbose)