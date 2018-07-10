# Spark SQL provides a special type of collection of data called DataFrame which is organized into named columns, 
# built on top of RDD. It is conceptually equivalent to a table in a relational database or a data frame in R/Python, 
# but with richer optimizations. DataFrames can be constructed from a wide array of sources such as: 
# structured data files, tables in Hive, external databases, or existing RDDs. 
# In the following example, we will show that how to create a Dateframe for a JSON dataset.

from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession.builder \
         .appName("Spark_Basics") \
         .getOrCreate()

# Create a Dataframe from a JSON
# Defines a Python list storing one JSON object.
json_strings = ['{"name":"Bob","address":{"city":"Los Angeles","state":"California"}}', ]
# Defines an RDD from the Python list.
peopleRDD = sc.parallelize(json_strings)
# Creates an DataFrame from an RDD[String].
people = spark.read.json(peopleRDD)
people.show()

# In this example we show how to search through the log file for the number of error messages using Dataframe.
from pyspark.sql import Row

text_data = sc.parallelize(["MYSQL ERROR 1\n","MYSQL ERROR 2\n","MYSQL\n"])
# Creates a DataFrame having a single column named "line"
df = text_data.map(lambda r: Row(r)).toDF(["line"])
# Counts ERRORs
errors = df.filter(df["line"].like("%ERROR%"))
# Counts all the errors
errors.count()