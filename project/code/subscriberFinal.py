from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.pipeline import PipelineModel
import pyspark.sql.functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
from datetime import datetime
from pyspark.sql import Row


TOPIC_NAME = "test"
HOST = '10.128.0.16:9092'
MODEL_PATH = "gs://bdlp/rfsavedModelFinal" 

spark = SparkSession.builder \
	.appName("DataStreamReader") \
	.getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

schema = StructType() \
  .add("Summons Number", LongType()) \
  .add("Plate ID", StringType()) \
  .add("Registration State", StringType()) \
  .add("Plate Type", StringType()) \
  .add("Issue Date", StringType()) \
  .add("Violation Code", IntegerType()) \
  .add("Vehicle Body Type", StringType()) \
  .add("Vehicle Make", StringType()) \
  .add("Issuing Agency", StringType()) \
  .add("Street Code1", IntegerType()) \
  .add("Street Code2", IntegerType()) \
  .add("Street Code3", IntegerType()) \
  .add("Vehicle Expiration Date", StringType()) \
  .add("Issuer Code", IntegerType()) \
  .add("Issuer Command", StringType()) \
  .add("Issuer Squad", StringType()) \
  .add("Violation Time", StringType()) \
  .add("Time First Observed", StringType()) \
  .add("Violation_County", StringType()) \
  .add("Violation In Front Of Or Opposite", StringType()) \
  .add("House Number", StringType()) \
  .add("Street Name", StringType()) \
  .add("Intersecting Street", StringType()) \
  .add("Date First Observed", StringType()) \
  .add("Law Section", IntegerType()) \
  .add("Sub Division", StringType()) \
  .add("Violation Legal Code", StringType()) \
  .add("Days Parking In Effect", StringType()) \
  .add("From Hours In Effect", StringType()) \
  .add("To Hours In Effect", StringType()) \
  .add("Vehicle Color", StringType()) \
  .add("Unregistered Vehicle?", IntegerType()) \
  .add("Vehicle Year", IntegerType()) \
  .add("Meter Number", StringType()) \
  .add("Feet From Curb", IntegerType()) \
  .add("Violation Post Code", StringType()) \
  .add("Violation Description", StringType()) \
  .add("No Standing or Stopping Violation", StringType()) \
  .add("Hydrant Violation", StringType()) \
  .add("Double Parking Violation", StringType()) \
  .add("Latitude", StringType()) \
  .add("Longitude", StringType()) \
  .add("Community Board", StringType()) \
  .add("Community Council", StringType()) \
  .add("Census Tract", StringType()) \
  .add("BIN", StringType()) \
  .add("BBL", StringType()) \
  .add("NTA", StringType()) 

df = spark \
	.readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", HOST) \
  .option("subscribe", TOPIC_NAME) \
  .load()

def preprocessing(df):
    # Dropping columns with many null values
    to_drop = ['Intersecting Street', 'Time First Observed', 'Time First Observed', 'Violation Legal Code', 'Unregistered Vehicle?', 'Meter Number', 'No Standing or Stopping Violation', 'Hydrant Violation', 'Double Parking Violation', 'Latitude', 'Longitude', 'Community Board', 'Community Council', 'Census Tract', 'BIN', 'BBL', 'NTA']
    df = df.drop(*to_drop)


    # Dropping duplicate rows from Summons Number as its the primary key
    df = df.dropDuplicates(subset=['Summons Number'])
    
    # Converting date from StringType to DateType
    func =  F.udf(lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
    df = df.withColumn('Issue_Date2', func(F.col('Issue Date')))
    df = df.drop('Issue Date')
    df = df.withColumnRenamed('Issue_Date2','Issue Date')

    # Deriving new columns for day, month and year (IntegerType) from the Issue Date column
    df = df.withColumn("Issue Day", F.dayofweek(F.col("Issue Date"))).withColumn("Issue Month", F.month(F.col("Issue Date"))).withColumn("Issue Year", F.year(F.col("Issue Date")))
    
    # Filtering rows with Issue year from 2013 to 2017 
    df = df.filter((F.col("Issue Year") > 2012) & (F.col("Issue Year") < 2018))

    # Dividing Violation Time into 6 bins
    def bins(x):
        default_ = 3
        hr = x[:2]
        period = x[-1].upper()

        if hr in ['00','01','02','03','12'] and period == 'A':
            return 1
        elif hr in ['04','05','06','07'] and period == 'A':
            return 2
        elif hr in ['08','09','10','11'] and period == 'A':
            return 3
        elif hr in ['12','00','01','02','03'] and period == 'P':
            return 4
        elif hr in ['04','05','06','07'] and period == 'P':
            return 5
        elif hr in ['08','09','10','11'] and period == 'P':
            return 6
        else:
            return default_

    bin_udf = F.udf(bins, IntegerType())
    df = df.withColumn("Time bin", bin_udf(F.col("Violation Time")))

    # Dropping columns which seem to be irrelevant to affect Violation County
    to_drop = ['Summons Number', 'Plate ID', 'Violation Code', 'Vehicle Expiration Date', 'Sub Division', 'House Number', 'Street Name', 'Date First Observed', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Color', 'Days Parking In Effect', 'Violation Post Code', 'Violation Description', 'Vehicle Year', 'Feet From Curb', 'Issue Date', 'Violation Time']
    df = df.drop(*to_drop)

    default_values = {
        'Registration State': 'NY',
        'Plate Type': 'PAS',
        'Issue Month': 6,
        'Issue Year': 2015,
        'Vehicle Body Type': 'SUBN',
        'Vehicle Make': 'FORD',
        'Issuing Agency': 'T',
        'Street Code1': 0,
        'Street Code2': 0,
        'Street Code3': 0,
        'Issuer Command': 'T103',
        'Issuer Squad': 'A',
        'Violation In Front Of Or Opposite': 'F'
    }

    df = df.na.fill(default_values)
    return df

df1 = df.select(F.from_json(F.decode(F.col("value"), 'utf-8'), schema).alias("data")).select("data.*")
df1 = preprocessing(df1)

model = PipelineModel.load(MODEL_PATH)

df2 = model.transform(df1)

df2 = df2.select(['predictedLabel','Violation_County', 'prediction', 'indexedLabel'])


def write(df, epoch):
    start_time = time.time()
    
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol="indexedLabel",metricName="accuracy")
    acc = evaluator_acc.evaluate(df)*100 

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol="indexedLabel",metricName="f1")
    f1 = evaluator_f1.evaluate(df)

    pred_lab_df = df.select(['predictedLabel','Violation_County'])
    col1_name = f"Batch {epoch} predicted violation county" # Rename to include batch number
    col2_name = f"Batch {epoch} true violation county" # Rename to include batch number
    pred_lab_df = pred_lab_df.withColumnRenamed('predictedLabel',col1_name)
    pred_lab_df = pred_lab_df.withColumnRenamed('Violation_County',col2_name)
    
    end_time = time.time()

    print(' ')
    pred_lab_df.write.format("console").save()
    print(f"--> Batch {epoch} Processing Time: {end_time-start_time: .5f} secs")
    print(f"--> Batch {epoch} Accuracy: {acc: .5f} %")
    print(f"--> Batch {epoch} F1-score: {f1: .5f}")

query = df2 \
        .writeStream \
        .format("console") \
        .foreachBatch(write) \
        .option("truncate",False) \
        .start()

query.awaitTermination()