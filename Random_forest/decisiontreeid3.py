import random
import numpy as np
from collections import Counter
from scipy.stats import mode

#calculate the entropy
def entropy(Y):

    features = Counter(Y)#to statistic the number of features,e.g. ['a','a'',a','b','b'] as ['a':3,'b':2]
    entropy = 0.0
    total = len(Y)
    for y, count_y in features.items():
        probability_y = (count_y/total)
        entropy += -(probability_y)*(np.log2(probability_y))
    return entropy

#calculate information_gain
def information_gain(y, positive, negative):
    return entropy(y) - (entropy(positive)*len(positive)/len(y) + entropy(negative)*len(negative)/len(y))


class ID3Classifier(object):

    #init
    def __init__(self, max_features=lambda x: x, max_depth=10,
                    min_split=2):
     
        self.max_features = max_features# number of features
        self.max_depth = max_depth# level of trees
        self.min_split = min_split#minimum number that required for split a node
        

    #select features without consider feature selection
    def getfeatures(self, X, y):
        n_features = X.shape[1] #shape[1] get the column, number of the features
        feature_indices = random.sample(xrange(n_features), n_features)#get the index of each row, select all features
        
        self.trunk = self.build_tree(X, y, feature_indices, 0)

    #select features considering feature selection
    def feature_selection(self, X, y):
       
        n_features = X.shape[1] #shape[1] get the column
        n_sub_features = int(self.max_features(n_features))
        feature_indices = random.sample(xrange(n_features), n_sub_features)#get the array of index 
        
        self.trunk = self.build_tree(X, y, feature_indices, 0)

    #train a tree recursively
    def build_tree(self, X, y, feature_indices, depth):

        if depth is self.max_depth or len(y) < self.min_split or entropy(y) is 0:
            return mode(y)[0][0]
        
        feature_index, threshold = find_split(X, y, feature_indices)

        X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
        if y_true.shape[0] is 0 or y_false.shape[0] is 0:
            
            return mode(y)[0][0]
        
        branch_true = self.build_tree(X_true, y_true, feature_indices, depth + 1)
        branch_false = self.build_tree(X_false, y_false, feature_indices, depth + 1)

        return Node(feature_index, threshold, branch_true, branch_false)
    
    #Predict the class of each sample in X. 
    def predict(self, X):
        

        num_samples = X.shape[0]
        
        y = np.empty(num_samples, dtype="S10")

        for j in xrange(num_samples):
            node = self.trunk

            while isinstance(node, Node):
                if X[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[j] = node

        return y


    

#Returns the best split rule for a tree node.
def find_split(X, y, feature_indices):
   
    best_gain = 0
    best_feature_index = 0
    best_value = 0
    for feature_index in feature_indices:
        values = sorted(set(X[:, feature_index])) #get the value of features

        for j in xrange(len(values) - 1):
            #threshold = (values[j] + values[j+1])/2
            value = values[j]
            #print threshold
            X_true, y_true, X_false, y_false = split(X, y, feature_index, value)
            gain = information_gain(y, y_true, y_false)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_value = value

    return best_feature_index, best_value

# node in a decision tree with the binary condition xi<threshold
class Node(object):
    
    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false
        

#Splits dataset into subdataset
def split(X, y, feature_index, value):
    

    X_true = []
    y_true = []
    X_false = []
    y_false = []

    for j in xrange(len(y)):
       
        if X[j][feature_index] == value and y[j] == y[0]:
            X_true.append(X[j])
            y_true.append(y[j])
        else:
            X_false.append(X[j])
            y_false.append(y[j])

    X_true = np.array(X_true)
    y_true = np.array(y_true)
    X_false = np.array(X_false)
    y_false = np.array(y_false)

    return X_true, y_true, X_false, y_false




