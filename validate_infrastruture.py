import os
import os.path

from IPython.display import display, HTML

def html_log(message,tag="H1", color="black",center=False):
  if center:
    display(HTML('<{tag}> <center> <font color="{color}"> {message}</font></center></{tag}>'.format(tag=tag, message=message,color=color)))
  else:
      display(HTML('<{tag}> <font color="{color}"> {message}</font></{tag}>'.format(tag=tag, message=message,color=color)))
      

def html_table(data):
  display(HTML(
    '<table style="border: 1px solid black" ><tr>{}</tr></table>'.format(
        '</tr><tr style="border: 1px solid black">'.join(
            '<td style="border: 1px solid black">{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
 ))

  

html_log("Checking BigDL Environment",center=True)

def check_env(env_var):
  if env_var not in os.environ:
    display(HTML('<H2 <font color="red">{0} Environment variable not set </font></H2>'.format(env_var)))
    return False
  env_paths = os.environ.get(env_var).split(':')
  if not all([ os.path.isfile(p) for p in env_paths ] ):
    display(HTML("<H3> <font color=\"red\"> {0} Environment set ,but one of the paths not present</font></H2>".format(env_var)))
    return False
  else:
    html_log("Succesfully checked for {0}".format(env_var), 'p', 'gree')
    #display(HTML("<H3> <font color='green'> </font></H3>".format(env_var)))
    print "{}=={}".format(env_var, os.environ.get(env_var))
    return True
  
for bigdl_var in ['BigDL_JAR_PATH', 'PYTHONPATH']: 
  check_env(bigdl_var)
 
try:
  from bigdl.util.common import *
  from bigdl.nn.layer import *
  import bigdl.version
except:
  html_log('Unable to import BigDL Libary', 'p', 'red')
else:
  html_log(" BigDL Python Library Imported",'p','gree')
  #display(HTML('<H3> <font color="green"> BigDL Python Library Imported </font></H3>'))

try: 
  from pyspark import SparkContext
  sc = SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[*]"))
except:
  html_log('Unable to open get a spark context', 'p', 'red')
  #display(HTML('<H3> <font color="red">Unable to open Spark context </font></H3>'))
else:
  html_log('Got a spark context handle', 'p', 'gree')
  #display(HTML('<H3> <font color="green">Spark Context created </font></H3>'))
  html_table(sc._conf.getAll())

try:
  init_engine() # prepare the bigdl environment 
except:
  html_log('Unable to Initialize BigDL Engine', 'p', 'red')
else:
  html_log('BigDL Engine initialized , Good to go ....', 'p', 'gree')
  print "BigDL Version : {} ".format(bigdl.version.__version__)

;


