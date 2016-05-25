# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mode
from decisiontreeid3 import ID3Classifier

##Shuffles two lists of equal length and keeps corresponding elements in the same index.
def shuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

class RandomForestClassifier(object):

    def __init__(self, n_trees=100, max_features=np.sqrt, max_depth=20,
        min_split=2, bootstrap=0.8):
    
        self.n_trees = n_trees#The number of decision trees in the forest.
        self.max_features = max_features#number of features
        self.max_depth = max_depth#level of trees
        self.min_split = min_split#minimum number that required for split a node
        self.bootstrap = bootstrap#the fraction of randomly choosen data
        self.forest = []


    #Creates a forest of decision trees using a bootstrap and feature selection 
    def fit(self, X, y):
        
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples*self.bootstrap)
        
        for i in xrange(self.n_trees):
            shuffle(X, y)
            X_subset = X[:n_sub_samples]
            y_subset = y[:n_sub_samples]
            tree = ID3Classifier(self.max_features, self.max_depth,self.min_split)
            tree.feature_selection(X_subset, y_subset)
            self.forest.append(tree)

    #Predict the class of each sample 
    def predict(self, X):
        n_samples = X.shape[0] #shape[0] get the number raw
        n_trees = len(self.forest)
        # ping
        predictions = np.empty([n_trees, n_samples], dtype="S10")# initial array
        for i in xrange(n_trees):
            predictions[i] = self.forest[i].predict(X)
        return mode(predictions)[0][0]

    #get the accuracy of the prediction of X compared to test_y.
    def score(self, X, y):
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in xrange(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1.0
        accuracy = correct/n_samples
        return accuracy


