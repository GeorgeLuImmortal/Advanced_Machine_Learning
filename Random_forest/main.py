from randomforest import RandomForestClassifier
import csv
from sklearn import cross_validation
import numpy as np

def main():
 
    banks = open("/Users/lujinghui/Desktop/random_forest/random_forest/tennis.csv", "rb")
    f_reader = csv.DictReader(banks)

    features = []#store features
    classification = []#store class
    
    #get the dataset
    fieldnames = f_reader.fieldnames
    for row in f_reader:
        featuresRow = []
        for fieldname in fieldnames:
            if fieldname != "class":
                featuresRow.append(row[fieldname])
                
            else:
                classification.append(row[fieldname])
              
        features.append(featuresRow)
        
    features = np.array(features)
    classification = np.array(classification)
   
    #using cross validation to split dataset
    features_train, features_test, classification_train, classification_test = cross_validation.train_test_split(features, classification)
    
    forest = RandomForestClassifier()
    forest.fit(features_train, classification_train)
    
    
    accuracy = forest.score(features_test, classification_test)
    print 'The accuracy of this random forest is', 100*accuracy, '%.'
    
    total_accuracy=0
    #classifications = forest.predict(features_test)
    #print 'The first sample of the testset is classified as a', classifications[0], '.'
    for i in range(100):
        features_train, features_test, classification_train, classification_test = cross_validation.train_test_split(features, classification)
    
        forest = RandomForestClassifier()
        forest.fit(features_train, classification_train)
    
    
        accuracy = forest.score(features_test, classification_test)
        total_accuracy+=accuracy
    print 'The average accuracy of this random forest is',total_accuracy, '% in this dataset.'

if __name__ == "__main__":
    main()
