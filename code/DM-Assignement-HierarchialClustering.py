# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:14:42 2018

@author: Amitesh ranjan Srivastava
Hierarchical Clustering
"""

# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

#Alogorithm
def HC(proj):
    
    filename=proj+".csv"
    print(filename)
    # Importing the dataset
    dataset = pd.read_csv(filename)
    dataset['bug'] = np.where(dataset['bug']>=1, 1, 0)
    X = dataset.iloc[:, 3:23].values
    #X = dataset.iloc[:, 2:22].values
    Y_training = dataset.iloc[:,-1].values
    y_actual=list(Y_training)
    
    
    # Using the dendrogram to find the optimal number of clusters
    
    sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Features')
    plt.ylabel('Euclidean distances')
    plt.show()
    
    # Fitting Hierarchical Clustering to the dataset
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    y_predicted=list(y_hc)
    '''
    for i in range(len(y_predicted)):
        if y_predicted[i] == 0:
            y_predicted[i]=1
        else:
            y_predicted[i]=0
    '''       
    
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_actual, y_predicted).ravel()
    
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
    listOutput.append(str(HC(listProj[i])))
'''
listProj=HC("forrest-0.8")

