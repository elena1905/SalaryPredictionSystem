'''
# Created on Mar 12, 2013
#
# @author: Wenchong
#
# Python Version 3.0
#
# This file implements decisionnode class, decisiontree class
# and relevant functions.
'''


from functions.function import getInputData


'''
# ========================================================================
# Decision Tree
# ========================================================================
'''
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb

''' divide data into two sets '''
def divideset(rows,column,value):
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function=None
    
    # Numeric: returns true if the value of a column in a row is >= range value
    if isinstance(value,int) or isinstance(value,float):
        split_function=lambda row : row[column]>=value
    # Nominal: returns true if the value of a column in a row is the specified word
    else:
        split_function=lambda row : row[column]==value
    
    # Divide the rows into two sets and return them
    # Above the value
    set1=[row for row in rows if split_function(row)]
    # Below the value
    set2=[row for row in rows if not split_function(row)]
    
    return (set1,set2)

''' get the number of each category '''
def uniquecounts(rows):
    # results is a dictionary
    results = {}
    
    for row in rows:
        # The result is the last column
        # r is the value of the last column in a row
        r = row[len(row)-1]
        
        # Create key-value pair for r if r is not in dictionary
        if r not in results: results[r] = 0
        '''if r not in results.keys(): results[r]=0'''
        
        # Add one to the total count of r in the dictionary
        results[r] += 1
    
    return results

''' calculate gini impurity '''
def giniimpurity(rows):
    total = len(rows)
    
    # counts is a dictionary that looks like:
    # counts = {'Premium': 3, 'Basic': 5, 'None': 7}
    counts = uniquecounts(rows)
    
    imp = 0
    
    '''for k1 in counts.keys()'''
    for k1 in counts:
        # For example, p(Premium): p1 = 3 / (3 + 5 + 7)
        p1 = float(counts[k1]) / total
        for k2 in counts:
            # Skip items having been considered
            if k1 == k2: continue
            
            p2 = float(counts[k2]) / total
            imp += p1*p2
    
    return imp

''' calculate entropy '''
def entropy(rows):
    from math import log
    
    log2 = lambda x:log(x)/log(2)
    
    results = uniquecounts(rows)
    
    # Now calculate the entropy
    ent = 0.0
    
    for r in results.keys():
        p = float(results[r]) / len(rows)
        '''ent = ent - p * log2(p)'''
        ent -= p * log2(p)
    
    return ent

''' build decision tree '''
def buildtree(rows,scoref=entropy):
    if len(rows)==0: return decisionnode()
    
    # (1) Find the best attribute to split on:
    # -----------------------------------------
    # (1.1) Calculate the entropy of the whole group.
    current_score=scoref(rows)
    
    # Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None
    
    # (1.2) Divide up the group by the possible values of
    # each attribute
    '''======== Question: Need to split the last column ======'''
    '''column_count=len(rows[0])'''
    column_count=len(rows[0]) - 1
    
    # col is an index
    for col in range(0,column_count):
        # (1.2.1) Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            # Initialize dictionary key-value pairs for a column of all rows
            column_values[row[col]]=1
        
        # (1.2.2) Now try dividing the rows up for each value
        # in this column
        for value in column_values.keys():
            (set1,set2) = divideset(rows, col, value)
            
            # (1.3) Calculate the Information gain of the new groups.
            # Information gain is the difference between the
            # current entropy and the weighted-average entropy of
            # the two groups.
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            
            # (1.4) Select the attribute with the highest
            # information gain as the one to split on
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
            
            ''' Improved algorithm '''
            '''
            if len(set1) > 0 and len(set2) > 0:
                p = float(len(set1)) / len(rows)
                gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
            '''
            
    # (2) Create the branches
    # -----------------------
    # (2.1) Determines if the branch can be divided further
    # or if it has reached a solid conclusion.
    # A branch stops dividing when the information gain
    # from splitting a node is not more than zero.
    if best_gain > 0:
        # (2.2) If one of the new branches can be divided,
        # the algorithm creates two branches corresponding
        # to true or false for the splitting condition.
        # It does this by recursively calling the algorithm for
        # each branch
        # Build decision tree recursively
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        # The results of the calls on each subset are attached to the
        # True and False branches of the nodes, eventually constructing
        # the entire tree.
        return decisionnode(col = best_criteria[0], value = best_criteria[1], tb = trueBranch, fb = falseBranch)
    else:
        # If the best pair of subsets doesn't have a lower
        # weighted-average entropy than the current set, that
        # branch ends and the counts of all the possible
        # outcomes  are stored.
        return decisionnode(results = uniquecounts(rows))

''' print tree nodes '''
def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results != None:
        print (indent+str(tree.results))
    else:
        # Print the criteria
        print (indent+str(tree.col)+':'+str(tree.value)+'? ')
        
        # Print the branches
        print (indent+'T->'),printtree(tree.tb,indent+'  ')
        print (indent+'F->'),printtree(tree.fb,indent+'  ')

''' classify the target for prediction '''
def classify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
        return classify(observation,branch)

''' prune the tree to overcome overfitting problem '''
def prune(tree,mingain):
    # If the branches aren't leaves, then prune them
    if tree.tb.results==None:
        prune(tree.tb,mingain)
    if tree.fb.results==None:
        prune(tree.fb,mingain)
    
    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results!=None and tree.fb.results!=None:
        # Build a combined dataset
        tb,fb=[],[]
        for v,c in tree.tb.results.items():
            tb+=[[v]]*c
        for v,c in tree.fb.results.items():
            fb+=[[v]]*c
    
        # Test the reduction in entropy
        delta=entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)
        
        if delta<mingain:
            # Merge the branches
            tree.tb,tree.fb=None,None
            tree.results=uniquecounts(tb+fb)

''' improved classification '''
def mdclassify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:                                                                                 
        v=observation[tree.col]
        if v==None:
            # data missing!
            # calculate the results for each branch
            # combine them with their respective weightings
            tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
            tcount=sum(tr.values())
            fcount=sum(fr.values())
            tw=float(tcount)/(tcount+fcount)
            fw=float(fcount)/(tcount+fcount)
            result={}
            for k,v in tr.items(): result[k]=v*tw
            for k,v in fr.items(): result[k]=v*fw      
            return result
        else:
            if isinstance(v,int) or isinstance(v,float):
                if v>=tree.value: branch=tree.tb
                else: branch=tree.fb
            else:
                if v==tree.value: branch=tree.tb
                else: branch=tree.fb
            return mdclassify(observation,branch)



'''
# ========================================================================
# class decisiontree
# ========================================================================
'''
class decisiontree:
    tree = []
    
    ''' construct knn class and normalize numeric features '''
    def __init__(self, data):
        self.tree = buildtree(data)
    
    ''' generate predictions for queries file and store in solutions file '''
    def generateSolutions(self, queriesFile, solutionsFile):
        # get queries set from queires file
        querySet = getInputData(queriesFile)
        
        # open solutions file for writing
        outfile = open(solutionsFile, 'w')
        
        # loop over each queries set
        for i in range(0, len(querySet)):
            # get prediction for each record in queries set
            prediction = mdclassify(querySet[i], self.tree)
            
            if ('<=50K' in prediction):
                prediction = '<=50K'
            else:
                prediction = '>50K'
            
            # format a string for output
            tempstr = "tst" + str(i + 1) + "," + prediction
            
            # write to file
            print(tempstr, file = outfile)
        
        # close output file
        outfile.close()
'''
# ========================================================================
# End of Decision Tree
# ========================================================================
'''