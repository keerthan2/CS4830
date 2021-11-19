from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, IndexToString

spark = SparkSession.builder.getOrCreate()

FILE_PATH = 'gs://bdl2021_final_project/nyc_tickets_train.csv'

df = spark.read.format('csv').option("header", "true").option("inferSchema", "true").load(FILE_PATH)

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

df = preprocessing(df)

str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType) and f.name != 'Violation_County']

indexed_features = [col + "_idx" for col in str_cols]
features = list((set(df.columns) - set(['Violation_County'])) - set(str_cols)) + indexed_features

# Label encoding of class variable
labelIndexer = StringIndexer(inputCol="Violation_County", outputCol="indexedLabel").fit(df)

# Indexing of categorical features
featureIndexers = [StringIndexer(inputCol=col, outputCol=col_idx, handleInvalid="keep").fit(df) for col, col_idx in zip(str_cols, indexed_features)]

assembler = VectorAssembler(inputCols=features, outputCol="features")

# Inverse Indexing of Label
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

stages = [labelIndexer, *featureIndexers, assembler, labelConverter]

def trainModel(model, stages, save_path):
    stages.insert(-1, model)
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(training_data)
    model.save(save_path)
    return model

save_path_lr = "gs://bdlp/lrsavedModelFinal"
save_path_rf = "gs://bdlp/rfsavedModelFinal"

# Train test split
training_data, test_data = df.randomSplit([0.8, 0.2], seed=7)

# Training a logistic regression model 
lr = LogisticRegression(regParam=0.4, elasticNetParam=0.7, labelCol="indexedLabel", featuresCol="features")
lrModel = trainModel(lr, stages, save_path_lr)

# Training a random forest model
rf = RandomForestClassifier(numTrees=500, maxBins=10000, labelCol="indexedLabel", featuresCol="features")
rfModel = trainModel(rf, stages, save_path_rf)

# Make predictions.
def prediction(model):
    trainPrediction = model.transform(training_data)
    testPrediction = model.transform(test_data)

    evaluatorF1 = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="f1")

    evaluatorAcc = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

    trainAccuracy = evaluatorAcc.evaluate(trainPrediction)
    testAccuracy = evaluatorAcc.evaluate(testPrediction)

    trainF1 = evaluatorF1.evaluate(trainPrediction)
    testF1 = evaluatorF1.evaluate(testPrediction)
    
    print(f"Train F1: {trainF1:.5f}\nTest F1: {testF1:.5f}")
    print(f"Train accuracy: {trainAccuracy:.5f}\nTest Accuracy: {testAccuracy:.5f}")
    
print("Performance on Logistic Regression")
prediction(lrModel)
print("")
print("Performance on Random Forest")
prediction(rfModel)