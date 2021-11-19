"""
Script for running data exploration and feature analysis/engineering
"""

from __future__ import print_function
import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import os

sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')

iris_data = spark.read.format("bigquery").option("table", "irisdataset.iris_data_input").load() 

iris_data.createOrReplaceTempView("iris_data")

# Set the species column as the target

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

# Preprocessing
feature_cols = ["sepal_length","sepal_width", "petal_length", "petal_width"]
iris_df = iris_data.select(iris_data['sepal_length'].cast("float"),iris_data['sepal_width'].cast("float"),
                       iris_data['petal_length'].cast("float"), iris_data['petal_width'].cast("float"),
                       iris_data['species'])
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="species", outputCol="label")
iris_df_indexed = indexer.fit(iris_df).transform(iris_df)
iris_df_indexed = iris_df_indexed.drop('species')

## Data Exploration

# Count nans, null 
from pyspark.sql.functions import isnan, isnull, when, count, col
iris_df_indexed.select([count(when(isnan(c), c)).alias(c) for c in iris_df_indexed.columns]).show()
iris_df_indexed.select([count(when(isnull(c), c)).alias(c) for c in iris_df_indexed.columns]).show()

feature_cols = ["sepal_length","sepal_width", "petal_length", "petal_width"]
iris_df_indexed.describe(*feature_cols).show()

# Plotting histogram of labels
from pyspark.ml.feature import IndexToString
labelReverse = IndexToString().setInputCol("label").setOutputCol("species")
tmp_df = labelReverse.transform(iris_df_indexed).drop("label")
tmp_df = tmp_df.toPandas()
fig = plt.figure(figsize=(5,5))
plt.hist(tmp_df['species'])
plt.show()
fig.savefig("class_balance.png")
plt.close()
os.system('gsutil cp class_balance.png gs://bdl_5/plots/')

# Plotting histogram of features
iris_pd_df = iris_df_indexed.drop('label').toPandas()
cols = iris_pd_df.columns
fig = plt.figure(figsize=(8,8))
for i,feat in enumerate(cols):
  plt.subplot(2,2,i+1)
  plt.hist(iris_pd_df[feat])
  plt.xlabel(feat)
fig.savefig("features_hist.png")
plt.close()
os.system('gsutil cp features_hist.png gs://bdl_5/plots/')


import matplotlib.pyplot as plt
import seaborn as sns

# Box plot to detect outliers
iris_pd_df = iris_df_indexed.toPandas()
fig = plt.figure(figsize=(10,8))
for i,feat in enumerate(feature_cols):
  plt.subplot(2,2,i+1)
  sns.boxplot(x=iris_pd_df[feat])
  plt.xlabel(feat)
fig.savefig("box_plots.png")
plt.close()
os.system('gsutil cp box_plots.png gs://bdl_5/plots/')

## Feature Importance check
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

training_data, test_data = iris_df_indexed.randomSplit([0.8,0.2], seed=0)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')

assembler = VectorAssembler(inputCols = ["sepal_length",
                                        "sepal_width", 
                                        "petal_length", 
                                        "petal_width"], 
                            outputCol = "features")

rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="label", seed=42)
pipe = Pipeline(stages = [assembler, rf])
paramGrid = ParamGridBuilder().addGrid(rf.numTrees,[3,50,100]).addGrid(rf.maxDepth,[10,20,30]).addGrid(rf.impurity,['gini','entropy']).build()
crossval = CrossValidator(estimator=pipe, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)
cvModel = crossval.fit(training_data)

bestModel = cvModel.bestModel.stages[1]
impfeats = bestModel.featureImportances
impdic = {}
for idx,feat in enumerate(feature_cols):
  impdic[feat] = impfeats[idx]
fig = plt.figure()
plt.bar(range(len(impdic)), list(impdic.values()), align='center')
plt.xticks(range(len(impdic)), list(impdic.keys()))
plt.ylabel("Feature Importance")
fig.savefig("feature_importance.png")
plt.close()
os.system('gsutil cp feature_importance.png gs://bdl_5/plots/')

"""
gs://bdl_5/data_exp_and_feature_eng.py
gs://spark-lib/bigquery/spark-bigquery-latest.jar
"""