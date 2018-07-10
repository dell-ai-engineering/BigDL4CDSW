PYSPARK_PYTHON=./venv.zip/venv/bin/python spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python \
    --master yarn \
    --deploy-mode cluster\
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --py-files ${PYTHON_API_PATH} \
    --archives ${VENV_HOME}/venv.zip \
    --conf spark.driver.extraClassPath=bigdl-SPARK_2.2-0.5.0-jar-with-dependencies.jar \
    --conf spark.executor.extraClassPath=bigdl-SPARK_2.2-0.5.0-jar-with-dependencies.jar \
    $@
