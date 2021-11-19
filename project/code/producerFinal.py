from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from time import sleep
import json
import os

data_path = "gs://bdlp/test.csv"
data_path2 = "."
command = "gsutil cp -r " + data_path + " " + data_path2
os.system(command)

FILE_PATH = "test.csv"
TOPIC_NAME = "test"
HOST = "10.128.0.16:9092"

spark = SparkSession.builder \
  .appName("DataStreamPublisher") \
  .getOrCreate()

df = spark.read.csv(FILE_PATH, inferSchema=True, header=True)
df_to_json = df.toJSON().collect()

producer = KafkaProducer(bootstrap_servers=HOST)
print("Sending messages to topic:", TOPIC_NAME)
for elem in df_to_json:
  message = elem.encode('utf-8')
  producer.send(TOPIC_NAME, message)
  sleep(5)
