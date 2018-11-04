# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:14:42 2018

@author: Amitesh ranjan Srivastava
Decision Tree
"""


# Decision Tree Classification

# Importing the libraries
import numpy as np
import pandas as pd


def decisiontree(proj):
    filename=proj+".csv"
    # Importing the dataset
    dataset = pd.read_csv(filename)
    dataset['bug'] = np.where(dataset['bug']>=1, 1, 0)
    X = dataset.iloc[:, 3:23].values
    #X = dataset.iloc[:, 2:22].values prop-6.csv
    y = dataset.iloc[:,-1].values
    
    
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
    
    print(tp)
    print(fn)
    print(fp)
    print(tn)
    
    
    
    accuracy = ((tp+tn)/(tp+fn+fp+tn))*100
    
    precision = (tp/(tp+fp))*100
    
    recall=(tp/(tp+fn))*100
    
    fmeasure = 2*tp+fn+fp/tp
    
    print("accuracy: ",accuracy)
    print("precision: ",precision )
    print("recall: ",recall)
    print("fmeasure: ",fmeasure)
    listEva=[filename,accuracy,precision,recall,fmeasure]
    return listEva

'''--------------------------------------------Execution------------------------------------------------------------------'''

#Main Method to call the defination
listOutput=[]
'''
listProj=["arc","berek","camel-1.6","e-learning","forrest-0.8","intercafe","ivy-2.0","jedit-4.3","kalkulator","log4j-1.2","lucene-2.4","nieruchomosci","pbeans2","pdftranslator","poi-3.0","prop-6","redaktor","serapion","skarbonka","synapse-1.2","systemdata","szybkafucha","termoproject","tomcat","velocity-1.6","workflow","xalan-2.7","xerces-1.4","zuzel"]
for i in range(len(listProj)):
    listOutput.append(str(decisiontree(listProj[i])))
'''
listProj=decisiontree("forrest-0.8")