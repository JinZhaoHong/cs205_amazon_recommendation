# cs205_amazon_recommendation

# Steps:
1. Download one of the review dataset from http://jmcauley.ucsd.edu/data/amazon/index.html

2. replace the review dataset name in review_processor.py

3. spark-submit review_processor.py

4. Results will be stored in processed_reviews, total_asin_count, total_reviewerID_count three folders

5. open mapper.py and reducer.py and change the total_asin_count.

6. Run python mapper.py | reducer.py to obtain the data matrix
