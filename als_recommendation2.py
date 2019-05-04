import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.window import Window 
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import sys
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

	

conf = SparkConf().setAppName('Alternating Least Squares')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


file = sys.argv[1]
# read the review data
review_df = sqlContext.read.json(file)
#review_df = review_df.select("reviewerID", "overall", "asin").orderBy('reviewerID', ascending=True)
review_df.createOrReplaceTempView("review_table")

# create an index table of asin->id and reviewerID->id
asin_df = sqlContext.sql("SELECT DISTINCT asin FROM review_table")
w = Window.orderBy("asin") 
asin_index_df = asin_df.withColumn("index", F.row_number().over(w))
asin_index_df.createOrReplaceTempView("asin_index_table")


review_df = sqlContext.sql("SELECT DISTINCT reviewerID FROM review_table")
w = Window.orderBy("reviewerID") 
review_index_df = review_df.withColumn("index", F.row_number().over(w))
review_index_df.createOrReplaceTempView("review_index_table")

# make a new review table with only (reviewerID, reviewer_index, asin, asin_index, overall)
summary_df = sqlContext.sql \
("SELECT review_table.reviewerID, review_index_table.index AS reviewer_index, review_table.asin, asin_index_table.index AS asin_index, review_table.overall \
 FROM review_table, asin_index_table, review_index_table WHERE review_table.asin = asin_index_table.asin AND review_table.reviewerID = review_index_table.reviewerID")
summary_df.createOrReplaceTempView("summary_table")


ratings = summary_df.select("reviewer_index", "asin_index", "overall").rdd.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))

rank = 20
numIterations = 15
model = ALS.train(ratings, rank, numIterations)


testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("MSE="+str(MSE))


X = model.userFeatures()
Y = model.productFeatures()

X.saveAsTextFile("userFeatures")
Y.saveAsTextFile("productFeatures")

Xdf = X.toDF()
Xdf = Xdf.withColumnRenamed("_1", "reviewer_index")
Xdf = Xdf.withColumnRenamed("_2", "reviewer_latent_vector")
Xdf.createOrReplaceTempView("X_table")

Ydf = Y.toDF()
Ydf = Ydf.withColumnRenamed("_1", "asin_index")
Ydf = Ydf.withColumnRenamed("_2", "asin_latent_vector")
Ydf.createOrReplaceTempView("Y_table")


XY_df = sqlContext.sql("SELECT a.reviewer_index, b.asin_index, c.asin, a.reviewer_latent_vector, b.asin_latent_vector, c.overall \
					FROM X_table AS a, Y_table AS b, summary_table as c \
					WHERE a.reviewer_index = c.reviewer_index AND b.asin_index = c.asin_index")

XY_df.createOrReplaceTempView("XY_table")
XY_df.rdd.map(lambda x:(x[3], x[4], x[5])).saveAsTextFile("train")










