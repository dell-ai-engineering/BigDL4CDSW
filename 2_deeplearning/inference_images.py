"""
This model is a resnet model loaded from the bigdl modelzoo.

"""
%cd "/home/cdsw/2_deeplearning"
from sys import path
sys.path.append('/home/cdsw/2_deeplearning/resources')
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.transform.vision.image import *
import csv
import os
height=224
width=224
sparkConf = create_spark_conf().setMaster("local[2]").setAppName("Inference Example")
sc = get_spark_context(sparkConf)
init_engine()
model=Model.loadModel("resources/bigdl_resnet-50_imagenet_0.4.0.model") #the .model file downloaded from the model zoo
frame=ImageFrame.read("resources/inference/") #The images created by using ImageFrame to read from folder
transformer = Resize(height,width) 
transformed=transformer(frame) #Applying the transformation
result=model.predict_class(np.asarray(transformed.get_image())) #getting the predicted results.
result=result.tolist()
with open('resources/imagenet_classname.txt', 'rb') as f:
  reader = csv.reader(f)
  class_list = list(reader)
files = os.listdir("resources/inference")
class_result = []
for r in result:
  class_result.append(class_list[r])
print(zip(files,class_result))

