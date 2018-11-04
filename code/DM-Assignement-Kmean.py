"""
Created on Mon Apr 23 12:14:42 2018

@author: Amitesh ranjan Srivastava
K-Means Clustering
"""

# Importing the libraries
import numpy as np
import pandas as pd
import random

def Kmean(proj):
    filename=proj+".csv"
    # Importing the dataset
    dataset = pd.read_csv(filename)
    total=len(dataset)
    no_training_rows=int(total*0.67) # 2/3 Train Data and 1/3 Test Data
    
    dataset['bug'] = np.where(dataset['bug']>=1, 1, 0)
    X_training = dataset.iloc[0:no_training_rows, 3:23].values
    #X_training = dataset.iloc[0:no_training_rows, 2:22].values ---prop-6.csv
    Y_training = dataset.iloc[0:no_training_rows,-1].values
    
    
     # set the number OF CLUSTER randomly
    num_to_select = 2                          
    list_of_random_clusters = random.sample(list(X_training), num_to_select)
    cluster_1 = list_of_random_clusters[0]
    cluster_2 = list_of_random_clusters[1] 
    
    data_for_cluster_1=[]
    data_for_cluster_2=[]
    
    #find the centroid from the gievn centroid list -iteration
    def kmean_calculation(centroid_list):
        count=0
        sqdlist=[]
        for i in range(1,len(centroid_list)+1):
                centroid=list_of_random_clusters[i-1]
                distList=[]
                for j in range(no_training_rows):
                    list_training_data=list(X_training)
                    X_Training_List=list_training_data[j]
                    for k in range(len(X_Training_List)):
                        count= count+(np.square(X_Training_List[k]-centroid[k]))
                    distList.append(int(np.sqrt(count))) 
                    count=0
                sqdlist.append(distList)
        
        
        
        distance_cluster_1_list=sqdlist[0]
        distance_cluster_2_list=sqdlist[1]
        for l in range(len(distance_cluster_1_list)):
            if(distance_cluster_1_list[l]<distance_cluster_2_list[l]):
                data_for_cluster_1.append(list_training_data[l])
            else:
                data_for_cluster_2.append(list_training_data[l])
        
        centroid_mean_1=np.array(data_for_cluster_1).mean(axis=0)
        centroid_mean_2=np.array(data_for_cluster_2).mean(axis=0)
        centroid_list=[]
        centroid_list.append(centroid_mean_1)
        centroid_list.append(centroid_mean_2)
        return centroid_list
    
    # Iterate for 200 times and calulate centroid for 2 clusters 100 times
    for n in range(200):
        list_of_centroids=kmean_calculation(list_of_random_clusters)
        print("Iteration: ",n)
        
    print("final list of centroid for 2 clusters: ",list_of_centroids)
    ''' Cluster_1 = 1 and Cluster_2 = 0'''
    
    
    
    #Test Data Evaluation
    X_testing = dataset.iloc[no_training_rows:total, 3:23].values
    Y_testing = dataset.iloc[no_training_rows:total,-1].values
    no_testing_rows=len(X_testing)
    test_data_for_cluster=[]
    
    #Identify what is the class label for 2 clusters
    def find_class(cluster):
        
        list_c_1=list(cluster)
        l1=list(dataset.index[dataset['wmc']==list_c_1[0]])
        l2=list(dataset.index[dataset['dit']==list_c_1[1]])
        l3=list(dataset.index[dataset['noc']==list_c_1[2]])
        l4=list(dataset.index[dataset['cbo']==list_c_1[3]])
        l5=list(set(l1).intersection(l2))
        l6=list(set(l5).intersection(l3))
        l7=list(set(l6).intersection(l4))
        return int(dataset.iloc[l7[0],-1])
    
    # class label for cluster 1
    class1=find_class(cluster_1)
    
    # class label for cluster 2
    class2=find_class(cluster_2)
    
    #Classify to which cluster the test data belongs
    def kmean_evaluation(centroid_list):
        count=0
        sqdlist=[]
        for i in range(1,len(centroid_list)+1):
                centroid=list_of_random_clusters[i-1]
                distList=[]
                for j in range(no_testing_rows):
                    list_testing_data=list(X_testing)
                    X_Testing_List=list_testing_data[j]
                    for k in range(len(X_Testing_List)):
                        count= count+(np.square(X_Testing_List[k]-centroid[k]))
                    distList.append(int(np.sqrt(count))) 
                    count=0
                sqdlist.append(distList)
                print("length: ",len(distList))
        
        
        
        distance_cluster_1_list=sqdlist[0]
        distance_cluster_2_list=sqdlist[1]
        for l in range(len(distance_cluster_1_list)):
            if(distance_cluster_1_list[l]<distance_cluster_2_list[l]):
                test_data_for_cluster.append(class1)
            else:
                test_data_for_cluster.append(class2)
        return test_data_for_cluster
    
    #Predicted Value
    test_data_predicted_value=kmean_evaluation(list_of_centroids)
    
    #Actual Value
    test_data_actual_value=list(Y_testing)
    
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(test_data_actual_value, test_data_predicted_value).ravel()
    
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

#listProj=["arc","berek","camel-1.6","e-learning","forrest-0.8","intercafe","ivy-2.0","jedit-4.3","kalkulator","log4j-1.2","lucene-2.4","nieruchomosci","pbeans2","pdftranslator","poi-3.0","redaktor","serapion","skarbonka","synapse-1.2","systemdata","szybkafucha","termoproject","tomcat","velocity-1.6","workflow","xalan-2.7","xerces-1.4","zuzel"]
#listProj=["redaktor","serapion","skarbonka","synapse-1.2","systemdata","szybkafucha","termoproject","tomcat","velocity-1.6","workflow","xalan-2.7","xerces-1.4","zuzel"]
#listProj=["prop-6"]
'''
for i in range(len(listProj)):
    listOutput.append(str(Kmean(listProj[i])))
'''
listProj=Kmean("forrest-0.8")
