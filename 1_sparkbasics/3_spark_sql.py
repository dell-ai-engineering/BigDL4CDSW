# Spark SQL is a Spark module that makes it both easier and more efficient to load and query 
# for structured and semistructred data. We can interact with Spark SQL with SQL and regular Python/Java/Scala code.
# Internally, Spark SQL uses extra information to optimize the performance of the processing. 
# In following example we will show that how to run SQL queries using Spark SQL.

# At first we need to contruct a DataFrame for a JSON dataset.

from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession.builder \
         .appName("Spark_Basics") \
         .getOrCreate()

json_strings = ['{"name":"Bob","address":{"city":"Los Angeles","state":"California"}}', 
               '{"name":"Adam","address":{"city":"Seattle","state":"Washington"}}']

peopleRDD = sc.parallelize(json_strings)
people = spark.read.json(peopleRDD)
people.show()

# Now we register the DataFrame as a SQL temporary view using the funtion *sql* 
# which returns the result as a *DataFrame*, and then we can run SQL queries.

people.createOrReplaceTempView("people")

sqlDF = spark.sql("SELECT * FROM people").filter(people['name']=="Adam")
sqlDF.show()