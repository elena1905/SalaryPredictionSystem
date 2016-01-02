'''
# Created on Mar 12, 2013
#
# @author: Wenchong
#
# Python Version 3.0
#
# This file implements knn class.
'''


import math
from functions.function import getInputData


'''
# ========================================================================
# knn class for calculating knn distances, generating knn estimates and
# storing predictions to solutions file
# ========================================================================
'''
class knn:
    my_data = []
    
    ''' construct knn class and normalize numeric features '''
    def __init__(self, data):
        self.my_data = data
        
        # normalize all numeric features
        for i in range(0, len(data)):
            self.my_data[i][1] = self.normalize(data[i], 1, 17, 90)
            self.my_data[i][3] = self.normalize(data[i], 3, 12285, 1484705)
            self.my_data[i][5] = self.normalize(data[i], 5, 1, 16)
            self.my_data[i][11] = self.normalize(data[i], 11, 0, 99999)
            self.my_data[i][12] = self.normalize(data[i], 12, 0, 4356)
            self.my_data[i][13] = self.normalize(data[i], 13, 1, 99)
    
    ''' normalize numeric features '''
    def normalize(self, row, index, min_val, max_val):
        # if the the value exits, normalize the numeric feature
        if not(row[index] == '?'):
            row[index] = (row[index] - min_val) / (max_val - min_val)
        
        return row[index]
    
    ''' get the distances between query and all records in data set '''
    def getdistances(self, query):
        distancelist = []
        
        # normalize numeric features in query
        num1 = self.normalize(query, 1, 17, 90)
        num2 = self.normalize(query, 3, 12285, 1484705)
        num3 = self.normalize(query, 5, 1, 16)
        num4 = self.normalize(query, 11, 0, 99999)
        num5 = self.normalize(query, 12, 0, 4356)
        num6 = self.normalize(query, 13, 1, 99)
        
        # loop over every item in the data set
        for i in range(len(self.my_data)):
            # compute the numeric distance
            num_dist = 0
            num_dist += (self.my_data[i][1]-num1)**2
            num_dist += (self.my_data[i][3]-num2)**2
            num_dist += (self.my_data[i][5]-num3)**2
            num_dist += (self.my_data[i][11]-num4)**2
            num_dist += (self.my_data[i][12]-num5)**2
            num_dist += (self.my_data[i][13]-num6)**2
            num_dist = math.sqrt(num_dist)
            
            # compute the categorical distances
            cat_dist = 0
            if not(self.my_data[i][2] == query[2]):
                cat_dist += 1
            if not(self.my_data[i][4] == query[4]):
                cat_dist += 1
            if not(self.my_data[i][6] == query[6]):
                cat_dist += 1
            if not(self.my_data[i][7] == query[7]):
                cat_dist += 1
            if not(self.my_data[i][8] == query[8]):
                cat_dist += 1
            if not(self.my_data[i][9] == query[9]):
                cat_dist += 1
            if not(self.my_data[i][10] == query[10]):
                cat_dist += 1
            if not(self.my_data[i][14] == query[14]):
                cat_dist += 1
            cat_dist /= 8
            
            # add the distance and the index
            distance = num_dist + cat_dist
            distancelist.append([distance, self.my_data[i][15]])
        
        # sort by distance
        distancelist.sort()
        
        # return sorted distance list
        return distancelist
    
    ''' get the most frequent class in the top k results and make prediction '''
    def knnestimate(self, query, k = 5):
        # get sorted distances
        dlist = self.getdistances(query)
        targ1_count = 0
        targ2_count = 0
        
        # get the number of different targets
        for i in range(k):
            if dlist[i][1] == '<=50K':
                targ1_count = targ1_count + 1
            else:
                targ2_count = targ2_count + 1
        
        # the prediction is the target with bigger number
        if targ1_count > targ2_count:
            return '<=50K'
        else:
            return '>50K'
    
    ''' generate predictions for queries file and store in solutions file '''
    def generateSolutions(self, queriesFile, solutionsFile):
        # get queries set from queires file
        querySet = getInputData(queriesFile)
        
        # open solutions file for writing
        outfile = open(solutionsFile, 'w')
        
        # loop over each queries set
        for i in range(0, len(querySet)):
            # get prediction for each record in queries set
            prediction = self.knnestimate(querySet[i], k = 5)
            
            # format a string for output
            tempstr = "tst" + str(i + 1) + "," + prediction
            
            # write to file
            print(tempstr, file = outfile)
        
        # close output file
        outfile.close()
'''
# ========================================================================
# End of class knn
# ========================================================================
'''