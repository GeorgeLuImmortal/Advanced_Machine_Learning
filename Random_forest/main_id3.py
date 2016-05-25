from decisiontreeid3 import ID3Classifier
import csv
import numpy as np
from sklearn import cross_validation

def main():
    

    document = open("/Users/lujinghui/Desktop/random_forest/random_forest/tennis.csv", "rb")
    f_reader = csv.DictReader(document)

    Features = []#store features
    Class = []#store classification
    
    #get the dataset
    fieldnames = f_reader.fieldnames
    for row in f_reader:
        XRow = []
        for fieldname in fieldnames:
            if fieldname != "class":
                XRow.append(row[fieldname])
            else:
                Class.append(row[fieldname])
        Features.append(XRow)
        
    #change to np type
    X = np.array(Features)
    y = np.array(Class)

  
    #split the train set, test set using cross validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
     
    max_features=X.shape[1]#the number of features
    max_depth=10#the max depth 
    min_split=2#minimum number that required for split a node
    
    #train the tree
    tree = ID3Classifier(max_features,max_depth,min_split)
    tree.getfeatures(X_train, y_train)
    
    #get the result
    y_predict = tree.predict(X_test)
    
    
    
    #calculate the accuracy
    y_len = len(y_predict)
    correct = 0
    for i in xrange(y_len):
        if y_predict[i] == y_test[i]:
            correct = correct + 1.0
    accuracy = correct/y_len
    print 'The accuracy of this tree in this dataset is',accuracy*100,'%.'
    
    total_accuracy=0
    #y = tree.predict(y_test)
    #print 'The first sample of testset is classified as', y[0], '.'
    for i in range(100): 
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
        max_features=X.shape[1]#the number of features
        max_depth=10#the max depth 
        min_split=2#minimum number that required for split a node

        #train the tree
        tree = ID3Classifier(max_features,max_depth,min_split)
        tree.getfeatures(X_train, y_train)
        #get the result
        y_predict = tree.predict(X_test)
    
        #calculate the accuracy
        y_len = len(y_predict)
        correct = 0
        for i in xrange(y_len):
            if y_predict[i] == y_test[i]:
                correct = correct + 1.0
        accuracy = correct/y_len
        total_accuracy+=accuracy
    print 'The average accuracy of this tree in this dataset is',total_accuracy,'%.'

if __name__ == "__main__":
    main()