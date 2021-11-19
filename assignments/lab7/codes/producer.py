from __future__ import absolute_import
from time import sleep
from kafka import KafkaProducer
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import json

# brokers = 'localhost:9092'
brokers = '10.150.0.2:9092'
topic = 'irispred'
data_path = "iris.csv"

# Each row is encoded to utf-8 format before publishing
producer = KafkaProducer(bootstrap_servers=[brokers],
                         value_serializer=lambda x: 
                         x.encode('utf-8') 
                         )

spark = SparkSession \
    .builder \
    .appName("IrisPub") \
    .getOrCreate()

# Define schema
schema = StructType() \
  .add("sepal_length", FloatType()) \
  .add("sepal_width", FloatType()) \
  .add("petal_length", FloatType()) \
  .add("petal_width", FloatType()) \
  .add("species", StringType())

# Read csv file with headers
iris_data = spark.read.csv(data_path,header=True,schema=schema)

cnt = 0
# Publish each row to the topic. 
for row in iris_data.toJSON().collect():
    producer.send(topic, value=row)
    sleep(1)
    # # For testing
    # cnt += 1
    # if cnt > 3:
    #     break
