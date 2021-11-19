from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from datetime import datetime
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()

# Exploration subset file path
FILE_PATH = 'gs://bdlp/test300k.csv'

df = spark.read.format('csv').option("header", "true").option("inferSchema", "true").load(FILE_PATH)

print("Total Number of Rows in Exploration Set")
print(df.count())

print("Summons Number Frequency Distribution")
df.groupBy('Summons Number').count().orderBy('count', ascending = False).show()

# It is clear that the summons number is the primary key, given very few duplicate occurences. Therefore, duplicate occurences must mean duplicate entries
print("Example of a Duplicated entry")
df.filter(F.col("Summons Number") == '1367826652').show()

# Exploration of finally retained columns
retained_columns_final = [
    'Registration State',
    'Plate Type',
    'Issue Date',
    'Violation Time',
    'Vehicle Body Type',
    'Vehicle Make',
    'Issuing Agency',
    'Street Code1',
    'Street Code2',
    'Street Code3',
    'Issuer Command',
    'Issuer Squad',
    'Violation In Front Of Or Opposite'
]

# Collect frequency distributions of each
df_list = []
print("Finding groupbycounts for all retained feature columns:")
for retcol in retained_columns_final:
    df_list.append(df.groupBy(retcol).count().orderBy('count', ascending = False))

for idx, df_item in enumerate(df_list):
    print(f"For the column {retained_columns_final[idx]}, top 10 frequency of values:")
    df_item.show(10)

print("Label class Violation_County distribution:")
df.groupBy('Violation_County').count().orderBy('count', ascending = False).show(10)

from pyspark.sql.types import *
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
df = df.withColumn("Violation Time bin", bin_udf(F.col("Violation Time")))

print("Violation Time Bin distribution (after binning from Violation Time)")
df.groupBy('Violation Time bin').count().orderBy('count', ascending = False).show(10)

func =  F.udf(lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
df = df.withColumn('Issue_Date2', func(F.col('Issue Date')))
df = df.drop('Issue Date')
df = df.withColumnRenamed('Issue_Date2','Issue Date')

df = df.withColumn("Issue Day", F.dayofweek(F.col("Issue Date"))).withColumn("Issue Month", F.month(F.col("Issue Date"))).withColumn("Issue Year", F.year(F.col("Issue Date")))

print("Issue Date distributions of Day, Month, Year after separation of Date:")

df.groupBy('Issue Day').count().orderBy('count', ascending = False).show(10)

df.groupBy('Issue Month').count().orderBy('count', ascending = False).show(10)

df.groupBy('Issue Day').count().orderBy('count', ascending = False).show(10)

df.groupBy('Issue Year').count().orderBy('count', ascending = False).show(10)

