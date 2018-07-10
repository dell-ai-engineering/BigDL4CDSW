# Handwritten Digit Classfication using Convolutional Neural Network

# The tutorial presented will again tackle the MNIST digit classification problem for the purpose of demonstrating quantization of our model.

from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')
%cd "/home/cdsw/2_deeplearning"
import os
import matplotlib
# matplotlib.use('Agg')
get_ipython().magic(u'pylab inline')

import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from utils import get_mnist,generate_summaries
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from pyspark import SparkContext
sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("yarn").set("spark.driver.memory","8g"))

init_engine()

# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.
mnist_path = "resources/datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)

print train_data.count()
print test_data.count()

# Let's create the LeNet-5 model.

# Create a LeNet model
def build_model(class_num):
    model = Sequential()
    model.add(Reshape([1, 28, 28]))
    model.add(SpatialConvolution(1, 6, 5, 5).set_name('conv1'))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2).set_name('pool1'))
    model.add(Tanh())
    model.add(SpatialConvolution(6, 12, 5, 5).set_name('conv2'))
    model.add(SpatialMaxPooling(2, 2, 2, 2).set_name('pool2'))
    model.add(Reshape([12 * 4 * 4]))
    model.add(Linear(12 * 4 * 4, 100).set_name('fc1'))
    model.add(Tanh())
    model.add(Linear(100, class_num).set_name('score'))
    model.add(LogSoftMax())
    return model
lenet_model = build_model(10)

# Create an Optimizer

optimizer = Optimizer(
    model=lenet_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=0.4, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(5),
    batch_size=2048)

# Set the validation logic
optimizer.set_validation(
    batch_size=2048,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

# generate summaries and start tensortboard if needed
(train_summary, val_summary) = generate_summaries('/home/cdsw/tmp/bigdl_summaries', 'cnn')
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)


# Train the network. Wait some time till it finished.. Voila! You've got a trained model

trained_model = optimizer.optimize()

def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return l[0] - 1


# Let's first take a few traing samples and check the labels


# label-1 to restore the original label.
print "Ground Truth labels:" 
print ', '.join([str(map_groundtruth_label(s.label.to_ndarray())) for s in train_data.take(8)])
imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in train_data.take(8)]),cmap='gray'); plt.axis('off')


# Now, let's see the prediction results on test data by our trained model.

predictions = trained_model.predict(test_data)
#imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in test_data.take(8)]),cmap='gray'); plt.axis('off')
print "Trained Model Predictions"
print 'Ground Truth labels:'
print ', '.join(str(map_groundtruth_label(s.label.to_ndarray())) for s in test_data.take(8))
print 'Predicted labels:'
print ', '.join(str(map_predict_label(s)) for s in predictions.take(8))

quant_model = trained_model.quantize()

predictions = quant_model.predict(test_data)
#imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in test_data.take(8)]),cmap='gray'); plt.axis('off')
print "Quantized Model Predictions"
print 'Ground Truth labels:'
print ', '.join(str(map_groundtruth_label(s.label.to_ndarray())) for s in test_data.take(8))
print 'Predicted labels:'
print ', '.join(str(map_predict_label(s)) for s in predictions.take(8))

def timeStamped(fname, fmt='{fname}-%Y-%m-%d-%H-%M-%S.bigdl'):
  return dt.datetime.now().strftime(fmt).format(fname=fname)

trained_name = timeStamped('trained_model')
quant_name = timeStamped('quant_model')

trained_model.saveModel("/tmp/" + trained_name)    
quant_model.saveModel("/tmp/" + quant_name)

trainedinfo = os.stat("/tmp/" + trained_name)
trainedsize = trainedinfo.st_size

quantinfo = os.stat("/tmp/" + quant_name)
quantsize = quantinfo.st_size

quantdiff = trainedsize - quantsize

print "Quantized Model is",quantsize,"bytes. The original model is",trainedsize,"bytes. You saved",quantdiff,"bytes by using a quantized model."