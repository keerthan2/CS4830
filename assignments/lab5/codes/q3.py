"""
Script for running experiments on preprocessing and model selection
"""

from __future__ import print_function
import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler


sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')

iris_data = spark.read.format("bigquery").option("table", "irisdataset.iris_data_input").load()

iris_data.createOrReplaceTempView("iris_data")

# Preprocessing
feature_cols = ["sepal_length","sepal_width", "petal_length", "petal_width"]
iris_df = iris_data.select(iris_data['sepal_length'].cast("float"),iris_data['sepal_width'].cast("float"),
                       iris_data['petal_length'].cast("float"), iris_data['petal_width'].cast("float"),
                       iris_data['species'])
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="species", outputCol="label")
iris_df_indexed = indexer.fit(iris_df).transform(iris_df)
iris_df_indexed = iris_df_indexed.drop('species')



### "Sepal width removal
iris_feat_rem = iris_df_indexed.drop('sepal_width')

### Outlier removal
iris_pd_df = iris_df_indexed.toPandas()
Q1 = iris_pd_df.quantile(0.25)
Q3 = iris_pd_df.quantile(0.75)
IQR = Q3 - Q1
# we remove the data point even if one of the features lies outside the box. That is 
# why we use .any(axis=1) where axis=1 corresponds to the features axis
iris_pd_filt = iris_pd_df[~((iris_pd_df < (Q1 - 1.5 * IQR)) |(iris_pd_df > (Q3 + 1.5 * IQR))).any(axis=1)]
iris_df_filt = spark.createDataFrame(iris_pd_filt)

### Creating pipeline of assembler + minmax

from pyspark.ml.feature import MinMaxScaler

# training_data, test_data = iris_df_filt.randomSplit([0.8,0.2], seed=0) # With outlier removal
# training_data, test_data = iris_df_indexed.randomSplit([0.8,0.2], seed=0) # Without outlier removal and feature removal
training_data, test_data = iris_feat_rem.randomSplit([0.8,0.2], seed=0) # With feature removal

# assembler for without feature removal case
# assembler = VectorAssembler(inputCols = ["sepal_length",
#                                         "sepal_width", 
#                                         "petal_length", 
#                                         "petal_width"], 
#                             outputCol = "features")

# assembler and feature_cols for feature removal case
assembler = VectorAssembler(inputCols = ["sepal_length",
                                        "petal_length", 
                                        "petal_width"], 
                            outputCol = "features")
feature_cols = ["sepal_length", "petal_length", "petal_width"]

## MinMax scaling
scaler = MinMaxScaler(inputCol='features', outputCol="features_scaled")
preprocess_pipeline = Pipeline(stages = [assembler, scaler])

train = preprocess_pipeline.fit(training_data).transform(training_data)
test = preprocess_pipeline.fit(test_data).transform(test_data)
training_data = train.drop("sepal_length", "petal_length", "petal_width",'features').withColumnRenamed("features_scaled", "features")
test_data = test.drop("sepal_length", "petal_length", "petal_width",'features').withColumnRenamed("features_scaled", "features")

## Logistic regression
from pyspark.ml.classification import LogisticRegression

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')

lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)
pipe = lr

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a grid of multiple values of the hyper-parameter regParam
paramGrid = ParamGridBuilder().addGrid(lr.regParam,[0.01,0.03,0.1]).addGrid(lr.elasticNetParam,[0.5,0.8,1]).build()

#Create a CrossValidator Object
crossval = CrossValidator(estimator=pipe, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)

#Train the model with the CrossValidator Object 
cvModel = crossval.fit(training_data)
train_pred_LR = cvModel.transform(training_data)
test_pred_LR = cvModel.transform(test_data)
bestModel = cvModel.bestModel
# print(f"Reg param for the best model: {bestModel._java_obj.getRegParam()}")

## Random forest
from pyspark.ml.classification import RandomForestClassifier

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')

rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="label", seed=42)
pipe = rf

# pipe = Pipeline(stages = [assembler, lr])
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a grid of multiple values of the hyper-parameter regParam
paramGrid = ParamGridBuilder().addGrid(rf.numTrees,[3,5,10]).addGrid(rf.maxDepth,[10,20,30]).addGrid(rf.impurity,['gini','entropy']).build()

#Create a CrossValidator Object
crossval = CrossValidator(estimator=pipe, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)

#Train the model with the CrossValidator Object 
cvModel = crossval.fit(training_data)
train_pred_RF = cvModel.transform(training_data)
test_pred_RF = cvModel.transform(test_data)
bestModel = cvModel.bestModel
# print(f"Number of trees for the best model: {bestModel._java_obj.getNumTrees()}")
# print(f"Depth of best model: {bestModel._java_obj.getMaxDepth()}")
# print(f"Impurity of best model: {bestModel._java_obj.getImpurity()}")

## Naive Bayes
from pyspark.ml.classification import NaiveBayes
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')
nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label")
pipe = nb

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a grid of multiple values of the hyper-parameter regParam
paramGrid = ParamGridBuilder().addGrid(nb.smoothing,[0.5,1,10]).build()

#Create a CrossValidator Object
crossval = CrossValidator(estimator=pipe, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)

#Train the model with the CrossValidator Object 
cvModel = crossval.fit(training_data)

train_pred_NB = cvModel.transform(training_data)
test_pred_NB = cvModel.transform(test_data)

## Decision Trees
from pyspark.ml.classification import DecisionTreeClassifier

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label',metricName='accuracy')
dt = DecisionTreeClassifier(labelCol="label")
pipe = dt

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a grid of multiple values of the hyper-parameter regParam
paramGrid = ParamGridBuilder().addGrid(dt.impurity,['gini','entropy']).build()

#Create a CrossValidator Object
crossval = CrossValidator(estimator=pipe, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator,
                          numFolds=3)

#Train the model with the CrossValidator Object 
cvModel = crossval.fit(training_data)

train_pred_DT = cvModel.transform(training_data)
test_pred_DT = cvModel.transform(test_data)

print("LR: Accuracy score on train set =",evaluator.evaluate(train_pred_LR)*100," %")
print("LR: Accuracy score on test set = ",evaluator.evaluate(test_pred_LR)*100," %")
print("RF: Accuracy score on train set =",evaluator.evaluate(train_pred_RF)*100," %")
print("RF: Accuracy score on test set = ",evaluator.evaluate(test_pred_RF)*100," %")
print("NB: Accuracy score on train set =",evaluator.evaluate(train_pred_NB)*100," %")
print("NB: Accuracy score on test set = ",evaluator.evaluate(test_pred_NB)*100," %")
print("DT: Accuracy score on train set =",evaluator.evaluate(train_pred_DT)*100," %")
print("DT: Accuracy score on test set = ",evaluator.evaluate(test_pred_DT)*100," %")


"""
gs://bdl_5/q3.py
gs://spark-lib/bigquery/spark-bigquery-latest.jar
"""

