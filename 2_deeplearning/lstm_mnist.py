# Handwritten Digit Classfication using LSTM

# The tutorial presented will again tackle the MNIST digit classification problem. 
# We will build a Long short-term memory model to help us classify the digits. 
# Take a look at [this wonderful blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to gain insights of LSTM.

%cd "/home/cdsw/2_deeplearning"
from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')


import matplotlib
#matplotlib.use('Agg')
get_ipython().magic(u'pylab inline')

import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
import matplotlib.pyplot as plt
from utils import get_mnist, generate_summaries
from pyspark import SparkContext
from matplotlib.pyplot import imshow
sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("yarn").set("spark.driver.memory","8g"))

init_engine()


# ## 1. Load MNIST dataset

# First, we should get and store MNIST into RDD of Sample.

# Note: *edit the "mnist_path" accordingly. If the "mnist_path" directory does not consist of the mnist data, 
# mnist.read_data_sets method will download the dataset directly to the directory*.

# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.
mnist_path = "resources/datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)

train_data = train_data.map(lambda s: Sample.from_ndarray(np.resize(s.features[0].to_ndarray(), (28, 28)), s.label.to_ndarray()))
test_data = test_data.map(lambda s: Sample.from_ndarray(np.resize(s.features[0].to_ndarray(), (28, 28)), s.label.to_ndarray()))
print train_data.count()
print test_data.count()


# 2. Hyperparamter setup

# Parameters
batch_size = 64

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


# 3. Model creation

# Let's define our LSTM model here. Rather than adding a RNNcell in the RNN tutorial, now we add LSTM cell into our Recurrent container.

def build_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(LSTM(input_size, hidden_size))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Linear(hidden_size, output_size))
    return model
rnn_model = build_model(n_input, n_hidden, n_classes)


# 4. Optimizer setup and training

# Let's create an optimizer for training. As presented in the code, we are trying to optimize a [CrossEntropyCriterion](https://bigdl-project.github.io/master/#APIGuide/Losses/#crossentropycriterion) and use [Adam](https://bigdl-project.github.io/master/#APIGuide/Optimizers/Optim-Methods/#adam) to update the weights. Also in order to enable visualization support, we need to [generate summary info in BigDL](https://bigdl-project.github.io/master/#ProgrammingGuide/visualization/).

# Create an Optimizer

#criterion = TimeDistributedCriterion(CrossEntropyCriterion())
criterion = CrossEntropyCriterion()
optimizer = Optimizer(
    model=rnn_model,
    training_rdd=train_data,
    criterion=criterion,
    optim_method=Adam(),
    end_trigger=MaxEpoch(5),
    batch_size=batch_size)

# Set the validation logic
optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

# generate summaries and start tensortboard if needed
(train_summary, val_summary) = generate_summaries('/home/cdsw/tmp/bigdl_summaries', 'lstm')
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
# In[6]:


trained_model = optimizer.optimize()

# Loss visualization
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


# 5. Prediction on test data

def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return int(l[0] - 1)


predictions = trained_model.predict(test_data)
imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in test_data.take(8)]),cmap='gray'); plt.axis('off')
print 'Ground Truth labels:'
print ', '.join(str(map_groundtruth_label(s.label.to_ndarray())) for s in test_data.take(8))
print 'Predicted labels:'
print ', '.join(str(map_predict_label(s)) for s in predictions.take(8))
