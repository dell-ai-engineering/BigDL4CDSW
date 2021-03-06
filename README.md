# DellEMC Ready Solution for Aritificial Intelligence - Machine Learning with Hadoop - Jumpstart Kit
Go [HERE](https://github.com/dell-ai-engineering/bigdlengine4cdsw) to download and build the DellEMC customized engine for BigDL on Cloudera Data Science Workbench
### Deployment Validation
#### Validate your CDSW working environment [Validation Test](validate_infrastruture.py)
#### Download the frozen model files for examples from the workbench terminal access menu:
```
wget https://github.com/dell-ai-engineering/BigDL4CDSW/releases/download/1.0/bigdl_resnet-50_imagenet_0.4.0.model -P 2_deeplearning/resources/
wget https://github.com/dell-ai-engineering/BigDL4CDSW/releases/download/1.0/inception_v3_2016_08_28_frozen.pb -P 2_deeplearning/resources/frozen_models/
```

### Spark Basics
#### 1. [Spark RDD](./1_sparkbasics/1_rdd.py)
#### 2. [Spark Dataframe](./1_sparkbasics/2_dataframe.py)
#### 3. [Spark SQL](./1_sparkbasics/3_spark_sql.py)

### Deep Learning with BigDL
We use BigDL to run through mnist dataset using different models for each one and generate metrics allowing tensorbaord to view them
#### 1. [Logistic Regression](./2_deeplearning/lr_mnist.py) 
#### 2. [Feedforward Neural Network](./2_deeplearning/deep_feed_forward_mnist.py )
#### 3. [Convolutional Neural Network](./2_deeplearning/cnn_mnist.py )
#### 4. [Recurrent Neural Network](./2_deeplearning/rnn_mnist.py)
#### 5. [Bidirectional Recurrent Neural Network](./2_deeplearning/birnn_mnist.py)
#### 6. [LSTM](./2_deeplearning/lstm_mnist.py)
#### 7. [Auto-encoder](./2_deeplearning/autoencoder_mnist.py)

### BigDL Features
#### 1. [Visualization](./3_bigdlfeatures/visualization.py)
#### 2. [Quantization](./3_bigdlfeatures/quantization.py)

### References :
#### 1. [DellEMC Ready Bundle For Machine Learning ](https://www.dellemc.com/en-us/solutions/data-analytics/machine-learning/index.htm )
#### 2. [Cloudera Data Science Workbench Documentation ](https://www.cloudera.com/documentation/data-science-workbench/latest.html )
#### 3. [Intel BigDL Documnentation ](https://bigdl-project.github.io/0.5.0/)
#### 4. [ChestXNet Original Paper ](https://stanfordmlgroup.github.io/projects/chexnet/)






