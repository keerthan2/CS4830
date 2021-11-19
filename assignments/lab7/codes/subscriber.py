import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import split,decode,substring
import pyspark.sql.functions as f
from pyspark.sql.types import *

# brokers = 'localhost:9092'
brokers = '10.150.0.2:9092'
topic = 'irispred'
model_path = "gs://bdl_7/pipeline_model"

spark = SparkSession \
        .builder \
        .appName("IrisSub") \
        .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", brokers) \
  .option("subscribe", topic) \
  .load()

schema = StructType() \
  .add("sepal_length", FloatType()) \
  .add("sepal_width", FloatType()) \
  .add("petal_length", FloatType()) \
  .add("petal_width", FloatType()) \
  .add("species", StringType())

# Decode and extract dataframe from json
df = df.select(f.from_json(f.decode(df.value, 'utf-8'), schema=schema).alias("input"))
df = df.select("input.*") # Select only from input

# drop sepal_width feature (refer to lab5 for reasoning)
df_d = df.drop('sepal_width')

# Load model
from pyspark.ml import PipelineModel
model = PipelineModel.load(model_path)
pred_df = model.transform(df_d)
pred_df = pred_df.select(pred_df.pred_species, pred_df.species, pred_df.prediction, pred_df.label)

def batch_func(df, epoch):
  """
  Function for computing accuracy and displaying it in console along with predicted and true labels
  """
  from pyspark.sql import Row
  from pyspark.ml.evaluation import MulticlassClassificationEvaluator
  evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')
  acc = evaluator.evaluate(df)*100 
  acc_row = Row(ba=acc)
  acc_df = spark.createDataFrame([acc_row])
  acc_col = f"Batch {epoch} Accuracy"
  acc_df = acc_df.withColumnRenamed('ba',acc_col) # Rename to include batch number 
  pred_lab_df = df.select(df.pred_species, df.species)
  col1_name = f"Batch {epoch} predicted species" # Rename to include batch number
  col2_name = f"Batch {epoch} true species" # Rename to include batch number
  pred_lab_df = pred_lab_df.withColumnRenamed('pred_species',col1_name)
  pred_lab_df = pred_lab_df.withColumnRenamed('species',col2_name)
  pred_lab_df.write.format("console").save()
  acc_df.write.format("console").save()

# When using foreachBatch, the console does not print batch number by default. Therefore we include batch number
# in columns of the dataframes that is printed in the console (refer to function batch_func)
query = pred_df \
        .writeStream \
        .option("truncate",False) \
        .foreachBatch(batch_func) \
        .start() \

# For testing if producer works fine
"""
query = df \
        .writeStream \
        .format("console") \
        .option("truncate",False) \
        .start()
"""
query.awaitTermination()

