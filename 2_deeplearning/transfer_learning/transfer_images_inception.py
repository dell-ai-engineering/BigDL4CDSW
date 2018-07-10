%cd "/home/cdsw/2_deeplearning"
# coding: utf-8

# # Transfer Learning: Image Classification
# 
# Here we are going to do some transfer learning on an image classification problem.
# 
# 

# In[1]:
from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')

# Let's make sure that we have our sc instantiated
sc


# In[3]:


#Import all the required packages

import numpy as np
import pandas as pd

from os import listdir
from os.path import join, basename
import struct
import json
from scipy import misc
import datetime as dt

from bigdl.nn.layer import *
from optparse import OptionParser
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.nn.initialization_method import *
from transformer import *
from imagenet import *
from transformer import Resize

get_ipython().magic(u'matplotlib inline')


# # Getting the dataset

# You will need to download the data
# 
# ```bash
# cd /location/to/your/bigdl-tutorials/notebooks
# curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
# tar xzf flower_photos.tgz
# ```
# 

# Do a bit of cleanup of the data
# 
# ```bash
# rm LICENSE.txt
# ```

# In[4]:


# initializing BigDL engine
init_engine()


# In[6]:


# paths for datasets, saving checkpoints 

DATA_PATH = "/home/cdsw/tmp/flower_photos/"
checkpoint_path = "/home/cdsw/tmp/flower_photos/checkpoints"


# In[7]:


def get_inception_data(folder, file_type="image", data_type="train", normalize=255.0):
    """
    Builds the entire network using Inception architecture  
    
    :param class_num: number of categories of classification
    :return: entire model architecture 
    """
    #Getting the path of our data
    path = os.path.join(folder, data_type)
    if "seq" == file_type:
        #return imagenet.read_seq_file(sc, path, normalize) #-- incase if we are trying to read the orig imagenet data
        return read_seq_file(sc, path, normalize)
    elif "image" == file_type:
        #return imagenet.read_local(sc, path, normalize)
        return read_local(sc, path, normalize)


# In[8]:


# helper func to read the files from disk
def read_local(sc, folder, normalize=255.0, has_label=True):
    """
    Read images from local directory
    :param sc: spark context
    :param folder: local directory
    :param normalize: normalization value
    :param has_label: whether the image folder contains label
    :return: RDD of sample
    """
    # read directory, create image paths list
    image_paths = read_local_path(folder, has_label)
    # print "BEFORE PARALLELIZATION: ", image_paths
    # create rdd
    image_paths_rdd = sc.parallelize(image_paths)
    # print image_paths_rdd
    feature_label_rdd = image_paths_rdd.map(lambda path_label: (misc.imread(path_label[0]), np.array(path_label[1])))         .map(lambda img_label:
             (Resize(256, 256)(img_label[0]), img_label[1])) \
        .map(lambda feature_label:
             (((feature_label[0] & 0xff) / normalize).astype("float32"), feature_label[1]))
    # print "feature_label_rdd", feature_label_rdd
    return feature_label_rdd


# In[9]:


'''
Reading the training and validation data and perform pre-processing 
'''


# the image size expected by the model
image_size = 224

# image transformer, used for pre-processing the train images 
train_transformer = Transformer([Crop(image_size, image_size),
                                  Flip(0.5),
                                  ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                  TransposeToTensor(False)])

# reading the traning data
train_data = get_inception_data(DATA_PATH, "image", "train").map(
                lambda features_label: (train_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))

print('train_data: ' + str(train_data.count()))

# validation data transformer 
val_transformer = Transformer([Crop(image_size, image_size, "center"),
                                Flip(0.5),
                                ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                TransposeToTensor(False)])

#reading the validation data
val_data = get_inception_data(DATA_PATH, "image", "val").map(
                lambda features_label: (val_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))

print('val_data: ' + str(val_data.count()))


# In[15]:


def scala_T(input_T):
    """
    Helper function for building Inception layers. Transforms a list of numbers to a dictionary with ascending keys 
    and 0 appended to the front. Ignores dictionary inputs. 
    
    :param input_T: either list or dict
    :return: dictionary with ascending keys and 0 appended to front {0: 0, 1: realdata_1, 2: realdata_2, ...}
    """    
    if type(input_T) is list:
        # insert 0 into first index spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_T)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_T


# In[16]:


def Inception_Layer_v1(input_size, config, name_prefix=""):
    """
    Builds the inception-v1 submodule, a local network, that is stacked in the entire architecture when building
    the full model.  
    
    :param input_size: dimensions of input coming into the local network
    :param config: ?
    :param name_prefix: string naming the layers of the particular local network
    :return: concat container object with all of the Sequential layers' ouput concatenated depthwise
    """        
    
    '''
    Concat is a container who concatenates the output of it's submodules along the provided dimension: all submodules 
    take the same inputs, and their output is concatenated.
    '''
    concat = Concat(2)
    
    """
    In the above code, we first create a container Sequential. Then add the layers into the container one by one. The 
    order of the layers in the model is same with the insertion order. 
    
    """
    conv1 = Sequential()
    
    #Adding layes to the conv1 model we jus created
    
    #SpatialConvolution is a module that applies a 2D convolution over an input image.
    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1).set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1).set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1).set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    
    
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size,config[3][1], 1, 1, 1, 1).set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2).set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    
    
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1).set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat


# In[17]:


def Inception_v1_NoAuxClassifier(class_num):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False).set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, scala_T([scala_T([64]), scala_T(
         [96, 128]), scala_T([16, 32]), scala_T([32])]), "inception_3a/"))
    model.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T(
         [128, 192]), scala_T([32, 96]), scala_T([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T(
         [96, 208]), scala_T([16, 48]), scala_T([64])]), "inception_4a/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T(
         [112, 224]), scala_T([24, 64]), scala_T([64])]), "inception_4b/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T(
         [128, 256]), scala_T([24, 64]), scala_T([64])]), "inception_4c/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T(
         [144, 288]), scala_T([32, 64]), scala_T([64])]), "inception_4d/"))
    model.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_5a/"))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([384]), scala_T(
         [192, 384]), scala_T([48, 128]), scala_T([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num).set_name("loss3/classifier_flowers"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model


# In[18]:


#providing the no of classes in the dataset to model (5 for flowers)
classNum = 5

# Instantiating the model the model
# inception_model = Inception_v1(classNum)  #-- main inception-v1 model
inception_model = Inception_v1_NoAuxClassifier(classNum)


# In[19]:


# path, names of the downlaoded pre-trained caffe models
caffe_prototxt = 'bvlc_googlenet.prototxt'
caffe_model = 'bvlc_googlenet.caffemodel'

# loading the weights to the BigDL inception model, EXCEPT the weights for the last fc layer (classification layer)
model = Model.load_caffe(inception_model, caffe_prototxt, caffe_model, match_all=False, bigdl_type="float")


# In[20]:


'''
Reading the training and validation data and perform pre-processing 
'''


# the image size expected by the model
image_size = 224

# image transformer, used for pre-processing the train images 
train_transformer = Transformer([Crop(image_size, image_size),
                                  Flip(0.5),
                                  ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                  TransposeToTensor(False)])

# reading the traning data
train_data = get_inception_data(DATA_PATH, "image", "train").map(
                lambda features_label: (train_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))


# validation data transformer 
val_transformer = Transformer([Crop(image_size, image_size, "center"),
                                Flip(0.5),
                                ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                TransposeToTensor(False)])

#reading the validation data
val_data = get_inception_data(DATA_PATH, "image", "val").map(
                lambda features_label: (val_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))


# In[21]:


# training the model


# parameters for 
batch_size = 16
no_epochs = 2

# Optimizer
optimizer = Optimizer(
                model=model,
                training_rdd=train_data,
                #optim_method=Adam(learningrate=0.002),
                optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002),
                criterion=ClassNLLCriterion(),
                end_trigger=MaxEpoch(no_epochs),
                batch_size=batch_size
            )

# setting checkpoints
optimizer.set_checkpoint(EveryEpoch(), checkpoint_path, isOverWrite=False)

# setting validation parameters 
optimizer.set_validation( batch_size=batch_size,
                          val_rdd=val_data,
                          trigger=EveryEpoch(),
                          val_method=[Top1Accuracy()])


# In[22]:


# Log the training process to measure loss/accuracy, can be 
app_name= 'inception-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/inception_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/inception_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print "saving logs to ",app_name


# In[23]:


# Boot training process
# ERROR: not enough java heap space, too little RAM issue
get_ipython().magic(u'pylab inline')
trained_model = optimizer.optimize()
print "Optimization Done."

