#!/usr/bin/python

import sys
import re
import csv
import numpy as np

num_distinct_asins = 10672 # Get this number from review_processor.py

previous = None
cum_data = []

for line in sys.stdin:
    userId, movieId, rating = line.split( '\t' )

    if userId != previous:
        if previous is not None:

            array = np.zeros(num_distinct_asins)
            for data in cum_data:
                # MoiveId is one indexing, so convert it to zero indexing.
                movieId, rating = int(data[0]) - 1, float(data[1])
                array[movieId] = rating

            print(array.tolist())

        previous = userId
        cum_data = []
    
    cum_data.append((movieId, rating))

print(previous + '\t' + str(cum_mass / count))