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

	# read the review data
	review_df = sqlContext.read.json('reviews_Video_Games_5.json')
	#review_df = review_df.select("reviewerID", "overall", "asin").orderBy('reviewerID', ascending=True)
	review_df.createOrReplaceTempView("review_table")


	# create an index table of asin->id and reviewerID->id
	asin_df = sqlContext.sql("SELECT DISTINCT asin FROM review_table")
	w = Window.orderBy("asin") 
	asin_index_df = asin_df.withColumn("index", F.row_number().over(w)) # This is one indexing
	asin_index_df.createOrReplaceTempView("asin_index_table")


	review_df = sqlContext.sql("SELECT DISTINCT reviewerID FROM review_table")
	w = Window.orderBy("reviewerID") 
	review_index_df = review_df.withColumn("index", F.row_number().over(w)) # This is one indexing
	review_index_df.createOrReplaceTempView("review_index_table")


	# count how many products and how many users
	num_distinct_asins_df = sqlContext.sql("SELECT COUNT(asin) as num_distinct_asins FROM asin_index_table")
	num_distinct_reviewerIDs_df = sqlContext.sql("SELECT COUNT(reviewerID) as num_distinct_reviewerIDs FROM review_index_table")

	#num_distinct_asins_df.write.option("header", "true").csv("total_asin_count")
	#num_distinct_reviewerIDs_df.write.option("header", "true").csv("total_reviewerID_count")
	total_asin_count = int(num_distinct_asins_df.first()[0])
	total_reviewerID_count = int(num_distinct_reviewerIDs_df.first()[0])

	
	# make a new review table with only (reviewerID, reviewer_index, asin, asin_index, overall)
	review_with_index_df = sqlContext.sql \
	("SELECT review_table.reviewerID, review_index_table.index AS reviewer_index, review_table.asin, asin_index_table.index AS asin_index, review_table.overall \
	 FROM review_table, asin_index_table, review_index_table WHERE review_table.asin = asin_index_table.asin AND review_table.reviewerID = review_index_table.reviewerID \
	 ORDER BY review_table.reviewerID")


	# reduce by key on user to collect the set of reviews done by the user.
	rdd = review_with_index_df.rdd.map(lambda x: (x[1], [(x[3], x[4])])).reduceByKey(lambda a, b : a + b)


	# using a flat map to create the data matrix
	def map_function(x):
		# x has the format (reviewer_index, [(asin_index1, overall1), (asin_index2, overall2)])
		row = []
		# First make all reviews 0
		for i in range(total_asin_count):
			row.append(0)
		# The fill in actual user reviews
		reviews = x[1]
		for review in reviews:
			index = int(review[0]) - 1 # Make it 0 indexing.
			rating = int(review[1])
			row[index] = rating
		return (x[0], row)

	rdd = rdd.map(map_function)

	# Register the data matrix as temp table
	df_matrix = sqlContext.createDataFrame(rdd).toDF("reviewer_index", "ratings")
	df_matrix.createOrReplaceTempView("review_matrix_table")

	# Calculate the pairwise similarity score for a particular user and filter out the top 10 
	# First, use the first user in the rdd as an example user
	sample_user = rdd.top(1)[0]

	def cosine_similarity(x):
		# This part can be distributed using python multiprocessing
		a, b = np.array(x[1]), np.array(sample_user[1])
		cosine_similarity = a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b))
		return (cosine_similarity, (x[0], sample_user[0]))

	rdd = rdd.map(cosine_similarity).sortByKey(False).map(lambda x: (x[0].item(), x[1][0], x[1][1]))

	df_similarity = sqlContext.createDataFrame(rdd).toDF("similarity", "reviewer_index1", "reviewer_index2")
	df_similarity.createOrReplaceTempView("reviewer_similarity_table")

	df_similarity.show()

	# Fetch the 

	#review_with_index_df.repartition(1).write.option("header", "false").csv("processed_reviews")



if __name__ == "__main__": 
	main()
