#!/usr/bin/python

import sys
import re
import csv

with open('processed_reviews/result.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        userId, movieId, rating = row[1], row[3], row[4]
        print(userId + "\t" + movieId + "\t" + rating)