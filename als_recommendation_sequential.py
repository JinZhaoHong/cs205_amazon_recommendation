import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.window import Window 
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import sys




def main():
	### Spark Collaborative filtering using Alternative Least Square
	### http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf

	conf = SparkConf().setMaster('local').setAppName('Alternating Least Squares Sequential')
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)

	# Some hyperparameters
	k = 20 # The latent dimension
	lambda_ = 1 # The regularizer.
	epochs = 5 # The number of iterations

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
	summary_df = sqlContext.sql \
	("SELECT review_table.reviewerID, review_index_table.index AS reviewer_index, review_table.asin, asin_index_table.index AS asin_index, review_table.overall \
	 FROM review_table, asin_index_table, review_index_table WHERE review_table.asin = asin_index_table.asin AND review_table.reviewerID = review_index_table.reviewerID")
	summary_df.createOrReplaceTempView("summary_table")

	# Ratings : RDD((u, i, rui), . . .)
	ratings_rdd = summary_df.select("reviewer_index", "asin_index", "overall").rdd

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
	# Select 100 users and 100 movies to make recommendations

	X_100_join_Y_100_rdd = sc.parallelize(X.top(100)).map(lambda x : (x[0], x[1].tolist())).cartesian(sc.parallelize(Y.top(100)).map(lambda x : (x[0], x[1].tolist())))

	X_100_join_Y_100_rdd = X_100_join_Y_100_rdd.map(lambda x: (x[0][0], x[1][0], np.array(x[0][1]).T.dot(np.array(x[1][1]))[0][0].item()))


	X_100_join_Y_100_df = X_100_join_Y_100_rdd.toDF()
	X_100_join_Y_100_df = X_100_join_Y_100_df.withColumnRenamed("_1", "reviewer_index") \
	.withColumnRenamed("_2", "asin_index") \
	.withColumnRenamed("_3", "predicted_overall")
	X_100_join_Y_100_df.createOrReplaceTempView("X_100_join_Y_100_table")

	X_100_join_Y_100_df = sqlContext.sql("SELECT a.reviewer_index, a.asin_index, b.asin, a.predicted_overall \
										FROM X_100_join_Y_100_table AS a, asin_index_table AS b \
										WHERE a.asin_index = b.index")

	X_100_join_Y_100_df.write.option("header", "true").csv("Prediction100")


	######## Evaluate Training Mean Absolute Error Score Using the Existing Ratings ########
	Xdf = X.map(lambda x : (x[0], x[1].tolist())).toDF()
	# Assigning column names directly in toDF() throws exception.
	Xdf = Xdf.withColumnRenamed("_1", "reviewer_index")
	Xdf = Xdf.withColumnRenamed("_2", "reviewer_latent_vector")
	Xdf.createOrReplaceTempView("X_table")

	Ydf = Y.map(lambda x : (x[0], x[1].tolist())).toDF()
	Ydf = Ydf.withColumnRenamed("_1", "asin_index")
	Ydf = Ydf.withColumnRenamed("_2", "asin_latent_vector")
	Ydf.createOrReplaceTempView("Y_table")

	XY_df = sqlContext.sql("SELECT a.reviewer_index, b.asin_index, c.asin, a.reviewer_latent_vector, b.asin_latent_vector, c.overall \
						FROM X_table AS a, Y_table AS b, summary_table as c \
						WHERE a.reviewer_index = c.reviewer_index AND b.asin_index = c.asin_index")

	XY_df.createOrReplaceTempView("XY_table")
	XY_df.rdd.map(lambda x:(x[3], x[4], x[5])).saveAsTextFile("XY_train")



	XY_prediction_df = XY_df.rdd.map(lambda x : (x[0], x[1], np.array(x[3]).T.dot(np.array(x[4]))[0][0].item())).toDF()

	XY_prediction_df = XY_prediction_df.withColumnRenamed("_1", "reviewer_index") \
	.withColumnRenamed("_2", "asin_index") \
	.withColumnRenamed("_3", "predicted_overall")

	XY_prediction_df.createOrReplaceTempView("XY_prediction_table")

	XY_df = sqlContext.sql("SELECT a.reviewer_index, a.asin_index, a.asin, a.overall, b.predicted_overall \
						FROM XY_table AS a, XY_prediction_table AS b \
						WHERE a.reviewer_index = b.reviewer_index AND a.asin_index = b.asin_index")

	XY_df.createOrReplaceTempView("XY_table")

	XY_df.write.option("header", "true").csv("XY")

	error, count = XY_df.rdd.map(lambda x: (np.abs(float(x[3])-float(x[4])), 1)).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
	rmse = float(error) / float(count)
	print("Mean Absolute Error="+str(rmse))


if __name__ == "__main__": 
	main()
