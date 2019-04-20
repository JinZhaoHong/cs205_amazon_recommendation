import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.window import Window 
from pyspark.sql import functions as F


def main():

	conf = SparkConf().setMaster('local[*]').setAppName('Review Processor')
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)

	# Step 1, read the review data
	review_df = sqlContext.read.json('reviews_Video_Games_5.json')
	#review_df = review_df.select("reviewerID", "overall", "asin").orderBy('reviewerID', ascending=True)
	review_df.createOrReplaceTempView("review_table")

	# Step 2, create an index table of asin->id
	asin_df = sqlContext.sql("SELECT DISTINCT asin FROM review_table ORDER BY asin ASC")
	w = Window.orderBy("asin") 
	index_df = asin_df.withColumn("index", F.row_number().over(w))
	index_df.createOrReplaceTempView("asin_index_table")

	# Step 3, make a new review table with only (reviewerID, asin, index, overall)
	review_with_index_df = sqlContext.sql \
	("SELECT review_table.reviewerID, review_table.asin, asin_index_table.index, review_table.overall \
	 FROM review_table, asin_index_table WHERE review_table.asin = asin_index_table.asin ORDER BY review_table.reviewerID")

	# Step 4, create a table that stores the number of distinct asins.
	num_distinct_asins_df = sqlContext.sql("SELECT COUNT(asin) as num_distinct_asins FROM asin_index_table")
	total_asin_count_broadcast = int(num_distinct_asins_df.first()[0])
	#num_distinct_asins_df.createOrReplaceTempView("num_distinct_asins_table")

	# Step 5, reduce by key on user to collect the set of reviews done by the user.
	rdd = review_with_index_df.rdd.map(lambda x: (x[0], [(x[1], x[2], x[3])])).reduceByKey(lambda a, b : a + b)


	# Step 6, using a flat map to create the data matrix
	"""
	def flat_map_function(x):
		# x has the format (reviewerID, [(asin1, index1, overall1), (asin2, index2, overall2)])
		total_asin_count = total_asin_count_broadcast
		row = []
		row.append(x[0])
		# First make all reviews 0
		for i in range(total_asin_count):
			row.append(0)
		# The fill in actual user reviews
		reviews = x[1]
		for review in reviews:
			index = int(review[1])
			rating = int(review[2])
			row[index] = rating
		return row
		"""

	#print(rdd.flatMap(flat_map_function).top(2))

	rdd.saveAsTextFile("output")


if __name__ == "__main__": 
	main()
