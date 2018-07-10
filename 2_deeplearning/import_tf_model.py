# __Importing a Tensorflow Frozen  Model__
# ----------------------------------

##### TODO : use bigdl instead of keras image processing
##### TODO : push images to hdfs and run inference with results to hdfs/hive/sparkstream/kafka/...
 
%cd "/home/cdsw/2_deeplearning"
from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')

import operator
from pprint import pprint
import time
from IPython.display import Image
from IPython.display import display, HTML
from bigdl.util.common import *
from bigdl.nn.layer import *
#
#import tensorflow  as tf

import keras
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

import utils

init_engine()

##### 1. __define location of Inceptionv3 frozen model,the graph input/output names and labels.__
inputs = ['input']
outputs = ['InceptionV3/Predictions/Reshape_1']
tf_model = '/home/cdsw/2_deeplearning/resources/frozen_models/inception_v3_2016_08_28_frozen.pb'
tf_labels = "/home/cdsw/2_deeplearning/resources/frozen_models/imagenet_slim_labels.txt"

##### 2. __Load the tensorflow model as an BigDL model__
model = Model.load_tensorflow(tf_model, inputs, outputs, bigdl_type="float")

##### 3. __(Optional) save it in a folder so that it can be viewed using tensorboard __
#model.save_graph_topology("/home/cdsw/tmp/bigdl_summaries/")
#utils.start_tensorboard("/home/cdsw/tmp/bigdl_summaries/")
##### should be able to select tensorboard from the cdsw project menu then graphs from the tensorboard menu
#Image('resources/images/bigdl_tensorflow_imported_graphview.png')


with open(tf_labels,"r") as lbls:
  label_lines = [line.rstrip() for line in lbls]
testset = {'lemon':'resources/lemon.jpg','squirrel':'resources/squirrel.jpg',"basketball game":"resources/basketball_game.jpg"}
predicted =[]
for (lbl,imgpath) in testset.iteritems():
  test_image = image.load_img(imgpath,target_size=(299,299))
  test_image =  image.img_to_array(test_image)
  x = np.expand_dims(test_image, axis=0)
  x = preprocess_input(x)
  res = model.predict(x)
  reslbl = { label_lines[k]:v for k,v in  enumerate(res[0]) }
  top5 = sorted(reslbl.items(), key=operator.itemgetter(1),reverse=True)[:5]
  predicted.append([lbl]+[top5])
  
utils.html_table(predicted)