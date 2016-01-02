'''
# Created on Mar 12, 2013
#
# @author: Wenchong
#
# Python Version 3.0
#
# This file implements cross validation algorithms.
'''


from classes.decisiontree import mdclassify
from functions.function import divideData


'''
# ========================================================================
# Cross Validation
# test the algorithm given by algf to get the accuracy of one test set
# ========================================================================
'''
def testAlgorithm(algf, trainingSet, testSet):
    correct = 0
    targ1_correct = 0
    targ2_correct = 0
    targ1_count = 0
    targ2_count = 0
    
    # construct the algorithm given by algf
    alg = algf(trainingSet)
    
    # for storing the result set of the predictions
    algans = []
    
    
    for row in testSet:
        # Predict salary for every item in testSet
        algans.append(alg.knnestimate(row))
    
    # calculate the number of correct predictions
    for i in range(0, len(algans)):
        if testSet[i][15] == '<=50K':
            # total number of '<=50K' in testSet
            targ1_count += 1
            
            # total number of correct predictions for '<=50K'
            if (algans[i] == testSet[i][15]):
                targ1_correct += 1
        else:
            # total number of correct predictions for '>50K'
            if (algans[i] == testSet[i][15]):
                targ2_correct += 1
    
    # total number of '>50K' in testSet
    targ2_count = len(testSet) - targ1_count
    
    # calculate the percentage of correct predictions
    correct = (targ1_correct / targ1_count + targ2_correct / targ2_count) / 2
    
    # Return mean accuracy of the salary prediction
    return correct



def testAlgorithm2(algf, trainingSet, testSet):
    correct = 0
    targ1_correct = 0
    targ2_correct = 0
    targ1_count = 0
    targ2_count = 0
    
    # construct the algorithm given by algf
    alg = algf(trainingSet)
    
    # for storing the result set of the predictions
    algans = []
    
    for i in range(0, len(testSet)):
        # Predict salary for every item in testSet
        classification = mdclassify(testSet[i], alg)
        
        if ('<=50K' in classification):
            algans.append('<=50K')
        else:
            algans.append('>50K')
        
        # calculate the number of correct predictions
        if testSet[i][15] == '<=50K':
            # total number of '<=50K' in testSet
            targ1_count += 1
            
            # total number of correct predictions for '<=50K'
            if (algans[i] == testSet[i][15]):
                targ1_correct += 1
        else:
            # total number of correct predictions for '>50K'
            if (algans[i] == testSet[i][15]):
                targ2_correct += 1
    
    # total number of '>50K' in testSet
    targ2_count = len(testSet) - targ1_count
    
    # calculate the percentage of correct predictions
    correct = (targ1_correct / targ1_count + targ2_correct / targ2_count) / 2
    
    # Return mean accuracy of the salary prediction
    return correct



'''
# ========================================================================
# cross validate training data by splitting into training set and test set
# return the mean accuracy of the predictions of all the test set
========================================================================
'''
def crossValidate(algf, data, trials = 100, test = 0.1):
    correct = 0.0
    
    # Divide data for trials times
    for i in range(trials):
        # Get training set and test set
        trainingSet, testSet = divideData(data, test)
        
        # Sum up mean accuracy of the prediction of each test set
        correct += testAlgorithm(algf, trainingSet, testSet)
    
    # Return mean accuracy of the prediction of all test set
    return correct / trials



def crossValidate2(algf, data, trials = 100, test = 0.1):
    correct = 0.0
    
    # Divide data for trials times
    for i in range(trials):
        # Get training set and test set
        trainingSet, testSet = divideData(data, test)
        
        # Sum up mean accuracy of the prediction of each test set
        correct += testAlgorithm2(algf, trainingSet, testSet)
    
    # Return mean accuracy of the prediction of all test set
    return correct / trials
'''
# ========================================================================
# End of Cross Validation
# ========================================================================
'''