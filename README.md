# DellEMC Ready Solution for Aritificial Intelligence - Machine Learning with Hadoop - Jumpstart Kit
Go [HERE](https://github.com/dell-ai-engineering/bigdlengine4cdsw) to download and build the DellEMC customized engine for BigDL on Cloudera Data Science Workbench
### Deployment Validation
#### Test your working environment [Validation Test](validate_infrastruture.py)
#### Download the frozen model files for examples from the workbench terminal access menu:
```
wget https://github.com/dell-ai-engineering/BigDL4CDSW/releases/download/1.0/bigdl_resnet-50_imagenet_0.4.0.model -P 2_deeplearning/resources/
wget https://github.com/dell-ai-engineering/BigDL4CDSW/releases/download/1.0/inception_v3_2016_08_28_frozen.pb -P 2_deeplearning/resources/frozen_models/
```

### Spark Basics
#### 1. [Spark RDD](./1_sparkbasics/rdd.py)
#### 2. [Spark Dataframe](./1_sparkbasics/dataframe.py)
#### 3. [Spark SQL](./1_sparkbasics/spark_sql.py)

### Deep Learning with BigDL
We use BigDL to run through mnist dataset using different models for each one and generate metrics allowing tensorbaord to view them
#### 1. [Logistic Regression](./2_deeplearning/mnist_lr.py) 
#### 2. [Feedforward Neural Network](./2_deeplearning/mnist_deep_feed_forward_neural_network.py )
#### 3. [Convolutional Neural Network](./2_deeplearning/cnn_images.py )
#### 4. [Recurrent Neural Network](./2_deeplearning/mnist_rnn.py)
#### 5. [Bidirectional Recurrent Neural Network](./2_deeplearning/mnist_birnn.py)
#### 6. [LSTM](./2_deeplearning/lstm_images.py)
#### 7. [Auto-encoder](./2_deeplearning/mnist_autoencoder.py)

### BigDL Features
#### 1. [Visualization](./3_bigdlfeatures/visualization.py)
#### 1. [Visualization](./3_bigdlfeatures/quantization.py)

### References :
#### 1. [DellEmc Ready Bundle For Machine Learning ](https://www.dellemc.com/en-us/solutions/data-analytics/machine-learning/index.htm )
#### 2. [Cloudera Data Science Workbench Documentation ](https://www.cloudera.com/documentation/data-science-workbench/latest.html )
#### 3. [Intel BigDL Documnentation ](https://bigdl-project.github.io/0.5.0/)
#### 4. [ChestXNet Original Paper ](https://stanfordmlgroup.github.io/projects/chexnet/)






