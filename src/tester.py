'''
# Created on Mar 12, 2013
#
# @author: Wenchong Chen
#
# Python Version 3.0
#
# Artificial Intelligence 2 Module Project:
# Salary Prediction System
# to classify employees' salaries.
#
# Implemented Decision Tree and KNN models.
# Solutions are generated using Dicision Tree.
'''


from classes.knn import knn
from functions.crossvalidate import crossValidate
from functions.function import getInputData


if __name__ == '__main__':
    print ("Start processing...")
    
    ''' load training data '''
    # get training data from data source file
    sourceData = getInputData('./../data/trainingset.txt')
    
    ''' cross validate training data '''
    # split source data to training set and test set randomly by multiple times
    # calculate the mean accuracy
    # select the model that generates the highest accuracy
    
    print ("cross validating training data...")
    
    # knn
    correct = crossValidate(knn, sourceData, 10, 0.05)
    print("knn correct: " + str(correct))
    
    # decision tree
    #correct = crossValidate2(buildtree, sourceData, 10, 0.05)
    #print("knn correct: " + str(correct))
    
    ''' export predictions '''
    # queries data source file
    queriesFile = "./../data/queries.txt"
    
    # file for storing predictions
    solutionsFile = "./../solutions/predictions.txt"
    
    print ("exporting predictions...")
    
    # store the target variables in the solutions file
    # using knn to generate predictions
    alg = knn(sourceData)
    alg.generateSolutions(queriesFile, solutionsFile)
    
    # using decision tree to generate predictions
    # construct the algorithm given by algf
    #alg = decisiontree(sourceData)
    #alg.generateSolutions(queriesFile, solutionsFile)
    
    print ("Finsihed processing!")

''' The End '''