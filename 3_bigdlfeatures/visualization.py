%cd "/home/cdsw/2_deeplearning"
import matplotlib
matplotlib.use('Agg')
%pylab inline

import datetime as dt
import tempfile

import sys
print(os.path.abspath(os.path.join('../2_deeplearning/resources')))
sys.path.append(os.path.abspath(os.path.join('../2_deeplearning/resources')))
from utils import get_mnist

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
from IPython.display import Markdown, display

sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[4]").set("spark.driver.memory","2g"))

init_engine()

# Get and store MNIST into RDD of Sample, please edit the "mnist_path" accordingly.
mnist_path = "datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)

print train_data.count()
print test_data.count()

# Create a LeNet model
def build_model(class_num):
    input = Reshape([1, 28, 28]).set_name("reshape_1_28_28")()
    conv1 = SpatialConvolution(1, 6, 5, 5).set_name("conv1_5x5")(input)
    tanh1 = Tanh().set_name("tanh1")(conv1)
    pool1 = SpatialMaxPooling(2, 2, 2, 2).set_name("pool1_2x2")(tanh1)
    tanh2 = Tanh().set_name("tanh2")(pool1)
    conv2 = SpatialConvolution(6, 12, 5, 5).set_name("conv2_5x5")(tanh2)
    pool2 = SpatialMaxPooling(2, 2, 2, 2).set_name("pool2_2x2")(conv2)
    reshape = Reshape([12 * 4 * 4]).set_name("reshape_192")(pool2)
    fc1 = Linear(12 * 4 * 4, 100).set_name("fc_192_100")(reshape)
    tanh3 = Tanh().set_name("tanh3")(fc1)
    fc2 = Linear(100, class_num).set_name("fc_100_" + str(class_num))(tanh3)
    output = LogSoftMax().set_name("classifier")(fc2)

    model = Model(input, output)
    return model
lenet_model = build_model(10)

# Create an Optimizer

optimizer = Optimizer(
    model=lenet_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=0.4, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(5),
    batch_size=256)

# Set the validation logic
optimizer.set_validation(
    batch_size=256,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy(), Loss()]
)

app_name='lenet-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
# create TrainSummary

(train_summary,val_summary) = generate_summaries('/tmp/bigdl_summaries', app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)

# Boot training process
trained_model = optimizer.optimize()
print "Optimization Done."

