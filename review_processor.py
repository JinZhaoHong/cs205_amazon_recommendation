import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.window import Window 
from pyspark.sql import functions as F


def main():
	### Spark Collaborative filtering using Alternative Least Square
	### http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf

	conf = SparkConf().setAppName('Review Processor')
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)

	# read the review data
	review_df = sqlContext.read.json('Musical_Instruments_5.json')
	#review_df = review_df.select("reviewerID", "overall", "asin").orderBy('reviewerID', ascending=True)
	review_df.createOrReplaceTempView("review_table")



	# create an index table of asin->id and reviewerID->id
	asin_df = sqlContext.sql("SELECT DISTINCT asin FROM review_table")
	w = Window.orderBy("asin") 
	asin_index_df = asin_df.withColumn("index", F.row_number().over(w)) # This is one indexing
	asin_index_df.createOrReplaceTempView("asin_index_table")

	#asin_df.coalesce(1).write.format('json').save('asin_df.json')


	review_df = sqlContext.sql("SELECT DISTINCT reviewerID FROM review_table")
	w = Window.orderBy("reviewerID") 
	review_index_df = review_df.withColumn("index", F.row_number().over(w)) # This is one indexing
	review_index_df.createOrReplaceTempView("review_index_table")

	#review_index_df.coalesce(1).write.format('json').save('review_index_df.json')

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

	#df_matrix.coalesce(1).write.format('json').save('df_matrix.json')
	# Calculate the pairwise similarity score for a particular user and filter out the top 10 
	# First, use the first user in the rdd as an example user
	sample_user = rdd.top(1)[0]

	def cosine_similarity(x):
		# This part can be distributed using python multiprocessing
		a, b = np.array(x[1]), np.array(sample_user[1])
		cosine_similarity = a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b))
		return (cosine_similarity, (x[0], sample_user[0]))

	#rdd = rdd.map(cosine_similarity).sortByKey(False).map(lambda x: (x[0].item(), x[1][0], x[1][1]))

	#df_similarity = sqlContext.createDataFrame(rdd).toDF("similarity", "reviewer_index1", "reviewer_index2").sort("reviewer_index1", ascending = True)
	#df_similarity.createOrReplaceTempView("reviewer_similarity_table")



	
	#top_reviewer = list(top10_ratings.select("reviewer_index").collect())

	rdd_asin = review_with_index_df.rdd.map(lambda x: (x[3], [(x[1], x[4])])).reduceByKey(lambda a, b : a + b)

	def map_function_asin(x):
		# x has the format (asin_index, [(reviewer_index1, overall1), (reviewer_index2, overall2)])
		row = []
		# First make all reviews 0
		for i in range(total_reviewerID_count):
			row.append(0)
		# The fill in actual user reviews
		#reviews = 

		for review in x[1]:

			index = int(review[0]) - 1 #find the index in the list
			rating = int(review[1])
			row[index] = rating
		return (x[0], row)

	rdd_asin = rdd_asin.map(map_function_asin)

	df_matrix_asin = sqlContext.createDataFrame(rdd_asin).toDF("asin_index", "ratings").sort("asin_index", ascending = True)
	df_matrix_asin.createOrReplaceTempView("review_matrix_table_asin")

	#df_matrix_asin.show()
	ratings_array = np.array(df_matrix.select("ratings").collect()).squeeze()
	#print(ratings_array[1], "dsds")

	def similar(x):
		a = np.array(x[1]).flatten()
		row = []
		for i in range(total_reviewerID_count):

			#b = ratings_array[i]
			b = ratings_array[i]
			cosine_similarity = float(a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b)))
			row.append(cosine_similarity)
			#row.append(1)
		return (x[0], row)

	rdd_sim = rdd.map(similar)
	df_matrix_sim= sqlContext.createDataFrame(rdd_sim).toDF("reviewer_index", "Similarity").sort("reviewer_index", ascending = True)

	#df_matrix_asin.show()
	
	#del ratings_array

	ratings_by_asin = np.array(df_matrix_asin.select("ratings").collect()).squeeze()


	def g_score(x):
		similarity = np.array(x[1]).flatten()
		s = np.sum(similarity)
		row = []
		for i in range(total_asin_count):
			ratings_asin = ratings_by_asin[i]
			score = float(np.dot(similarity, ratings_asin)/s)
			row.append(score)

		#print(row, "sdsdf")
		return (x[0], row)

	rdd_score = rdd_sim.map(g_score)
	df_g_score = sqlContext.createDataFrame(rdd_score).toDF("reviewer_index", "g_score_vector").sort("reviewer_index", ascending = True)

	df_g_score.show()



	'''

	for i in range(total_asin_count):
		r = df_matrix_asin.filter(df_matrix_asin.asin_index == i).first()[1]

			
		row.append(np.dot(r, S)/np.sum(S))

	print(row, "edsdsd")
	#review_with_index_df_asin.show()

	def g_score(x):
		#df_temp = np.array(df_similarity.filter("similrity != review_id").select("similarity").collect())
		#ratings = np.array(x[1])
		review_id = x[0]
		#S = np.array(df_similarity.filter("reviewer_index1 == review_id").sort("reviewer_index2", ascending=True).select("similarity").collect()).flatten()
		
		row = []
		S = np.array((top10_cross_similarity.select("similarity").collect())).flatten()

		for i in range(total_asin_count):
			r = df_matrix_asin.filter("asin_index" == i).first()[1]

			
			row.append(np.dot(r, S)/np.sum(S))
			
		return (review_id, row)



	# Fetch the 	
	rdd_g_score = df_matrix.rdd.map(g_score)
	df_g_score = sqlContext.createDataFrame(rdd_g_score).toDF("reviewer_id", "g_score_vector")

	#df_g_score.coalesce(1).write.format('json').save('df_g_score.json')
	#review_with_index_df.repartition(1).write.option("header", "false").csv("processed_reviews")

'''

if __name__ == "__main__": 
	main()