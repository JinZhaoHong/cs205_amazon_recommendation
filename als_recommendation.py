import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.window import Window 
from pyspark.sql import functions as F


def main():
	### Spark Collaborative filtering using Alternative Least Square
	### http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf

	conf = SparkConf().setMaster('local[*]').setAppName('Review Processor')
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)

	# Some hyperparameters
	k = 10 # The latent dimension

	# read the review data
	review_df = sqlContext.read.json('reviews_Video_Games_5.json')
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


	# count how many products and how many users
	num_distinct_asins_df = sqlContext.sql("SELECT COUNT(asin) as num_distinct_asins FROM asin_index_table")
	num_distinct_reviewerIDs_df = sqlContext.sql("SELECT COUNT(reviewerID) as num_distinct_reviewerIDs FROM review_index_table")
	total_asin_count = int(num_distinct_asins_df.first()[0])
	total_reviewerID_count = int(num_distinct_reviewerIDs_df.first()[0])


	# create X and Y matrix. X (k, n) is the user matrix. Y (k, m) is the item matrix
	# X : RDD((index, x1), . . . , (index, xn))
	# Y : RDD((index, y1), . . . , (index, ym))
	X = sc.parallelize([(i+1, np.random.normal(0, 1, k).tolist()) for i in range(total_reviewerID_count)])
	Y = sc.parallelize([(i+1, np.random.normal(0, 1, k).tolist()) for i in range(total_asin_count)])

	
	# make a new review table with only (reviewerID, reviewer_index, asin, asin_index, overall)
	review_with_index_df = sqlContext.sql \
	("SELECT review_table.reviewerID, review_index_table.index AS reviewer_index, review_table.asin, asin_index_table.index AS asin_index, review_table.overall \
	 FROM review_table, asin_index_table, review_index_table WHERE review_table.asin = asin_index_table.asin AND review_table.reviewerID = review_index_table.reviewerID")

	# Ratings : RDD((u, i, rui), . . .)
	ratings_rdd = review_with_index_df.select("reviewer_index", "asin_index", "overall").rdd

	# First map RDD, (i, (u, rui))



if __name__ == "__main__": 
	main()
