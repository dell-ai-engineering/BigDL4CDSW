
# Using an auto encoder on MNIST handwritten digits.

# In this tutorial, we are going to learn how to compress handwritten digit images from MNIST dataset 
# and reconstruct them by using [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)　algorithm. 
# The autoencoder model is mainly composed of an ***encoder*** and a ***decoder***. As you can see in the following diagram
# (*credit to this* [blog](https://blog.keras.io/building-autoencoders-in-keras.html)) of the model architecture, 
# after the original input is fed through the encoder and decoder layer, the target output is expected to be the same 
# as the input although there could be lost information actually after reconstruction***, which is the key point of autoencoder. 
# It's a very basic and limited data compression algorithm but still good enough for practical applications such as data denoising 
# and dimensionality reduction for data visualization.

# ![model_architecture](jumpstart/images/autoencoder/autoencoder_schema.jpg)

# Without further ado, let's finish some imports and setups before the experiment and then delve into the code illustration.

%cd "/home/cdsw/2_deeplearning"

from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')
import matplotlib
# matplotlib.use('Agg')

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from utils import get_mnist, generate_summaries
from pyspark import SparkContext
from matplotlib.pyplot import imshow

sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("yarn").set("spark.driver.memory","8g"))

init_engine()


# 1. Load MNIST dataset
# Read the training data and test data from our designated dataset path by using `get_mnist` method. It also pre-processes 
# the data by standardizing the image pixel values for better neural network performance.
# Please edit the "mnist_path" accordingly. If the "mnist_path" directory does not consist of the mnist data, 
# the `get_mnist`  method will download the dataset directly to the directory.

# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.
mnist_path = "/home/cdsw/tmp/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)
train_data = train_data.map(lambda sample:
            Sample.from_ndarray(np.resize(sample.features[0].to_ndarray(), (28*28,)), np.resize(sample.features[0].to_ndarray(), (28*28,))))
test_data = test_data.map(lambda sample:
            Sample.from_ndarray(np.resize(sample.features[0].to_ndarray(), (28*28,)), np.resize(sample.features[0].to_ndarray(), (28*28,))))

print train_data.count()
print test_data.count()


# 2. Model setup

# Specify the initial hyperparameters prior to the training.

# Parameters
training_epochs = 10
batch_size = 128
display_step = 1

# Network Parameters
n_hidden = 32
n_input = 784 # MNIST data input (img shape: 28*28)


# 3. Model creation

# By introducing non-lineararity into our autoencoder model, the activation layer ***ReLU*** and ***Sigmoid*** 
# are added into encoder and decoder respectively.

# Create Model

def build_autoencoder(n_input, n_hidden):
    # Initialize a sequential container
    model = Sequential()

    # encoder
    model.add(Linear(n_input, n_hidden))
    model.add(ReLU())
    # decoder
    model.add(Linear(n_hidden, n_input))
    model.add(Sigmoid())
    
    return model
model = build_autoencoder(n_input, n_hidden)


# 4. Optimizer setup
# Create an Optimizer

optimizer = Optimizer(
    model=model,
    training_rdd=train_data,
    criterion=MSECriterion(),
    optim_method=Adam(),
    end_trigger=MaxEpoch(2),
    batch_size=batch_size)


# generate summaries and start tensortboard if needed
(train_summary, val_summary) = generate_summaries('/home/cdsw/tmp/bigdl_summaries', 'autoencoder')
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)


#Train model
trained_model = optimizer.optimize()


# 5. Loss visualization
# Let's draw the performance curve during optimization.

loss = np.array(train_summary.read_scalar("Loss"))

plt.figure(figsize = (12,12))
plt.plot(loss[:,0],loss[:,1],label='loss')
plt.xlim(0,loss.shape[0]+10)
plt.grid(True)
plt.title("loss")


# 6. Prediction on test dataset
# Then we test our autoencoder from the previous loaded dataset, compress and reconstruct the digit images 
# then compare the results with the original inputs, which are also our target outputs. We are going to use 
# only 10 examples to demonstrate our created and trained autoencoder is working.


(images, labels) = mnist.read_data_sets(mnist_path, "test")
examples_to_show = 10
examples = trained_model.predict(test_data).take(examples_to_show)
f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(images[i], (28, 28)))
    a[1][i].imshow(np.reshape(examples[i], (28, 28)))

