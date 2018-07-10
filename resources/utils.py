import datetime as dt
import subprocess
import shlex

import psutil

import numpy as np
from bigdl.util import common
from bigdl.dataset import mnist
from bigdl.optim.optimizer import *
from IPython.display import display, HTML

def get_mnist(sc, mnist_path):
    # target is start from 0,
    (train_images, train_labels) = mnist.read_data_sets(mnist_path, "train")
    (test_images, test_labels) = mnist.read_data_sets(mnist_path, "test")
    training_mean = np.mean(train_images)
    training_std = np.std(train_images)
    rdd_train_images = sc.parallelize(train_images)
    rdd_train_labels = sc.parallelize(train_labels)
    rdd_test_images = sc.parallelize(test_images)
    rdd_test_labels = sc.parallelize(test_labels)

    rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda (features, label):
                                        common.Sample.from_ndarray(
                                        (features - training_mean) / training_std,
                                        label + 1))
    rdd_test_sample = rdd_test_images.zip(rdd_test_labels).map(lambda (features, label):
                                        common.Sample.from_ndarray(
                                        (features - training_mean) / training_std,
                                        label + 1))
    return (rdd_train_sample, rdd_test_sample)


def generate_summaries(folder, appname):
  folder = '/tmp/bigdl_summaries'
  public_port = int(os.environ.get('CDSW_PUBLIC_PORT'))
  tb_listening = False
  app_name=appname + '-' +dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_summary = TrainSummary(log_dir=folder, app_name=app_name)
  train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
  val_summary = ValidationSummary(log_dir=folder, app_name=app_name)
  # check if tensorboard already listening to public port
  try:
    (p,pid)  = [(p, p.pid)  for p in psutil.process_iter() if p.name() == 'tensorboard'][0]
    tb_listening = any([x for x in p.connections() if x.laddr.port == public_port and x.status == 'LISTEN'])
  except:
    pass
  if not tb_listening:
    #start tensorboard if not already done and point to summary folder
    print ("starting tesorboard on CDSW public port ...")
    tbc = shlex.split('tensorboard --logdir={summary_folder} --port {cdsw_public_port}'.format(summary_folder=folder,cdsw_public_port = public_port))
    with open('/tmp/tb_Err','wb') as tberr:
      p = subprocess.Popen(tbc,stderr = tberr)
  else:
    print("tensorboard already listening on CDSW public port ")

  return (train_summary ,val_summary)


def html_table(data):
  display(HTML(
    '<table style="border: 1px solid black" ><tr>{}</tr></table>'.format(
        '</tr><tr style="border: 1px solid black">'.join(
            '<td style="border: 1px solid black">{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
 ))