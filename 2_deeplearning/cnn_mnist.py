# Handwritten Digit Classfication using Convolutional Neural Network
# The tutorial presented will again tackle the MNIST digit classification problem. 
# We will build a Convolutional Neural Network which is vastly used in many different applications. 
# CNN are networks with loops in them, allowing information to persist. Take a look at [this great blog](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) to gain insights of CNN.

%cd "/home/cdsw/2_deeplearning"
from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')
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


# 1. Train the network
# First, we should get and store MNIST into RDD of Sample.
# Note: *edit the "mnist_path" accordingly. If the "mnist_path" directory does not consist of the mnist data, 
# mnist.read_data_sets method will download the dataset directly to the directory*.

# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.
mnist_path = "resources/datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)

print train_data.count()
print test_data.count()


# 2. Model creation

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


# 3. Optimizer setup and training
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


# * Train the network. Wait some time till it finished.. Voila! You've got a trained model


trained_model = optimizer.optimize()

# ## 4. Predication on test data

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
imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in test_data.take(8)]),cmap='gray'); plt.axis('off')

print 'Ground Truth labels:'
print ', '.join(str(map_groundtruth_label(s.label.to_ndarray())) for s in test_data.take(8))
print 'Predicted labels:'
print ', '.join(str(map_predict_label(s)) for s in predictions.take(8))

# 5. Model inspection

# Now look at the parameter shapes. The parameters are exposed as a dict, and can be retrieved using model.parameters().

# The param shapes typically have the form (batch_number?, output_channels, input_channels, filter_height, filter_width) 
# (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).

params = trained_model.parameters()

#batch num, output_dim, input_dim, spacial_dim
for layer_name, param in params.iteritems():
    print layer_name,param['weight'].shape,param['bias'].shape


# 6. Weight visualiztion

# Then let's demonstrate how to visualize the weights of convolutional layers in the model.

#vis_square is borrowed from caffe example
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  
    plt.imshow(data,cmap='gray'); plt.axis('off')


filters_conv1 = params['conv1']['weight']

filters_conv1[0,0,0]

vis_square(np.squeeze(filters_conv1, axis=(0,)).reshape(1*6,5,5))


# the parameters are a list of [weights, biases]
filters_conv2 = params['conv2']['weight']

vis_square(np.squeeze(filters_conv2, axis=(0,)).reshape(12*6,5,5))


# ## 7. Loss visualization

loss = np.array(train_summary.read_scalar("Loss"))
top1 = np.array(val_summary.read_scalar("Top1Accuracy"))

def plotLoss():
  plt.figure(figsize = (12,12))
  plt.subplot(2,1,1)
  plt.plot(loss[:,0],loss[:,1],label='loss')
  plt.xlim(0,loss.shape[0]+10)
  plt.grid(True)
  plt.title("loss")
  
def plotAccuracy():
  plt.subplot(2,1,2)
  plt.plot(top1[:,0],top1[:,1],label='top1')
  plt.xlim(0,loss.shape[0]+10)
  plt.title("top1 accuracy")
  plt.grid(True)

plotLoss()
plotAccuracy()
