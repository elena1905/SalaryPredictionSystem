'''
# Created on Mar 12, 2013
#
# @author: Wenchong
#
# Python Version 3.0
#
# This file implements functions used in knn and decisiontree.
'''


from random import random


'''
# ========================================================================
# get data from data source file and convert the data to proper format
# ========================================================================
'''
def getInputData(datafile):
    # get data by line and split each line by comma
    data = [line.split(',') for line in open(datafile)]
    
    # format original data
    for i in range(0,len(data)):
        # strip the space from the left side of variables from index 2 to 15
        for j in range(2, len(data[i])):
            data[i][j] = data[i][j].lstrip(' ')
        
        # if the value exist, convert the numeric data type from string to integer
        if not(data[i][1] == '?'):
            data[i][1] = int(data[i][1])
        if not(data[i][3] == '?'):
            data[i][3] = int(data[i][3])
        if not(data[i][5] == '?'):
            data[i][5] = int(data[i][5])
        if not(data[i][11] == '?'):
            data[i][11] = int(data[i][11])
        if not(data[i][12] == '?'):
            data[i][12] = int(data[i][12])
        if not(data[i][13] == '?'):
            data[i][13] = int(data[i][13])
        
        # strip the '\n' character from the right side of the target variable
        data[i][15] = data[i][15].rstrip('\n')
    
    return data



'''
# ========================================================================
# divide the source data into training set and test set
# ========================================================================
'''
def divideData(data, test = 0.05):
    trainingSet = []
    testSet = []
    
    # split data into two sets
    for row in data:
        # random.random() generates 0 <= float number < 1.0
        if random() < test:
            # testSet is test% of the total data set
            testSet.append(row)
        else:
            trainingSet.append(row)
    
    return trainingSet, testSet


