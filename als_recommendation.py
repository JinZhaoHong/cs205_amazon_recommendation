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
	lambda_ = 0.9 # The regulizer.
	epochs = 1 # The number of iterations

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
	X = sc.parallelize([(i+1, np.random.uniform(0, 1, k).tolist()) for i in range(total_reviewerID_count)]).cache()
	Y = sc.parallelize([(i+1, np.random.uniform(0, 1, k).tolist()) for i in range(total_asin_count)]).cache()

	#print(sc.parallelize(X.top(100)).cartesian(sc.parallelize(Y.top(100))).top(1))
	#return

	
	# make a new review table with only (reviewerID, reviewer_index, asin, asin_index, overall)
	review_with_index_df = sqlContext.sql \
	("SELECT review_table.reviewerID, review_index_table.index AS reviewer_index, review_table.asin, asin_index_table.index AS asin_index, review_table.overall \
	 FROM review_table, asin_index_table, review_index_table WHERE review_table.asin = asin_index_table.asin AND review_table.reviewerID = review_index_table.reviewerID")

	# Ratings : RDD((u, i, rui), . . .)
	ratings_rdd = review_with_index_df.select("reviewer_index", "asin_index", "overall").rdd

	# First map RDD, (i, (u, rui)). Cached for optimization.
	ratings_rdd_itemkey = ratings_rdd.map(lambda x: (x[0], (x[1], x[2]))).cache()

	# First map RDD, (u, (i, rui)). Cached for optimization.
	ratings_rdd_userkey = ratings_rdd.map(lambda x: (x[0], (x[1], x[2]))).cache()

	def outer_product1(a):
		# a = (i, (y, (u, r)))
		# return (u, (i, r, y, yy^T))
		i = a[0]
		y = a[1][0]
		u = a[1][1][0]
		r = a[1][1][1]

		y = np.array(y)
		y.shape = (k, 1)

		return (u, (i, r, y, y.dot(y.T)))

	def outer_product2(a):
		# x = (u, (x, (i, r)))
		# return (i, (u, r, x, xx^T))
		u = a[0]
		x = a[1][0]
		i = a[1][1][0]
		r = a[1][1][1]

		x = np.array(x)
		x.shape = (k, 1)

		return (i, (u, r, x, x.dot(x.T)))

	for i in range(epochs):
		# Alternative Least Squares

		######## First Update X ########
		# Join Ratings with Y factors using key i (items)
		Y_join_ratings_rdd = Y.join(ratings_rdd_itemkey)


		# Map to compute yiyi^T and change key to u (user)
		# ReduceByKey u (user) to compute sum yiyi^T and invert
		# return (u, (i, r, y, yy^T))
		Y_join_ratings_rdd = Y_join_ratings_rdd.map(outer_product1)
		# return (u, inv sum yy^T)
		sum_yyT_rdd_inv = Y_join_ratings_rdd \
		.map(lambda x: (x[0], x[1][3])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: (x[0], np.linalg.inv(x[1] + lambda_ * np.eye(k))))

		# Another ReduceByKey to compute sum(r_ui * y_i)
		sum_ry_rdd = Y_join_ratings_rdd \
		.map(lambda x: (x[0], x[1][1] * x[1][2])) \
		.reduceByKey(lambda x, y: x + y)
		
		# Join yyT_rdd and ry_rdd to calculate the update for x_u, where u is the key.
		# update X
		X = sum_yyT_rdd_inv.join(sum_ry_rdd).map(lambda x: (x[0], x[1][0].dot(x[1][1]))).cache()

		######## Second Update Y ########

		# Join Ratings with X factors using key u (users)
		X_join_ratings_rdd = X.join(ratings_rdd_userkey)

		# Map to compute xixi^T and change key to i (items)
		# ReduceByKey i (item) to compute sum xixi^T and invert
		# return (i, (u, r, x, xx^T))
		X_join_ratings_rdd = X_join_ratings_rdd.map(outer_product2)

		# (i, inv sum xx^T)
		sum_xxT_rdd_inv = X_join_ratings_rdd \
		.map(lambda x: (x[0], x[1][3])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: (x[0], np.linalg.inv(x[1] + lambda_ * np.eye(k))))

		# Another ReduceByKey to compute sum(r_ui * x_u)
		sum_rx_rdd = X_join_ratings_rdd \
		.map(lambda x: (x[0], x[1][1] * x[1][2])) \
		.reduceByKey(lambda x, y: x + y)

		# Join xxT_rdd and rx_rdd to calculate the update for y_i, where i is the item.
		Y = sum_xxT_rdd_inv.join(sum_rx_rdd).map(lambda x: (x[0], x[1][0].dot(x[1][1]))).cache()


	X.saveAsTextFile("X")
	Y.saveAsTextFile("Y")
	# Generate movie recommendations for (user, movie) pairs. 
	# print(sc.parallelize(X.top(2)).top(2))
	# Select 100 users and 100 movies to make recommendations
	X_100_join_Y_100_rdd = sc.parallelize(X.top(100)).map(lambda x : (x[0], x[1].tolist())).cartesian(sc.parallelize(Y.top(100)).map(lambda x : (x[0], x[1].tolist())))

	X_100_join_Y_100_rdd = X_100_join_Y_100_rdd.map(lambda x: (x[0][0], x[1][0], np.array(x[0][1]).T.dot(np.array(x[1][1]))[0][0]))

	X_100_join_Y_100_rdd.repartition(10).saveAsTextFile("Predictions")


if __name__ == "__main__": 
	main()
