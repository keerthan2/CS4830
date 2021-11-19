import pyspark
import os
import shutil
import sys


def split_to_bins(x):
  if x>=0 and x<6:
    return ("0-6",1)
  elif x>=0 and x<12:
    return ("6-12",1) 
  elif x>=12 and x<18:
    return ("12-18",1)
  else:
    return ("18-24",1)


input_path = sys.argv[1]
output_dir_path = sys.argv[2]

if os.path.exists(output_dir_path):
  shutil.rmtree(output_dir_path)


sc = pyspark.SparkContext.getOrCreate()
lines = sc.textFile(input_path)
header = lines.first() 
lines = lines.filter(lambda row: row != header)
out = lines.map(lambda line: int(line.split(' ')[1].split(':')[0]))
out = out.map(split_to_bins)
out = out.reduceByKey(lambda x,y: x+y)
out.coalesce(1).saveAsTextFile(output_dir_path) 