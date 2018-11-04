"""
Created on Mon Apr 23 12:14:42 2018

@author: Amitesh ranjan Srivastava
Fuzy C-mean Clustering
"""

# Importing the libraries
import numpy as np
import pandas as pd
import random

def Fuzy(proj):
    filename=proj+".csv"
    #Assuming valiue for m
    m=2
    
    # Importing the dataset
    dataset = pd.read_csv(filename)
    total=len(dataset)
    no_training_rows=int(total*0.67) # 2/3 Train Data and 1/3 Test Data
    
    dataset['bug'] = np.where(dataset['bug']>=1, 1, 0)
    X_training = dataset.iloc[0:no_training_rows, 3:23].values
    #X_training = dataset.iloc[0:no_training_rows, 2:22].values ---prop-6.csv
    Y_training = dataset.iloc[0:no_training_rows,-1].values
    
    #No.Of Clusters
    no_of_clusters=2
    
    #Random selection of clusters
    list_of_random_clusters = random.sample(list(X_training), no_of_clusters)
    cluster_1 = list_of_random_clusters[0]
    cluster_2 = list_of_random_clusters[1] 
    
    #Centroid of 2 clusters
    centroid_1=round(np.array(cluster_1).mean(axis=0),2)
    centroid_2=round(np.array(cluster_2).mean(axis=0),2)
    
    #Objects in vector space
    X=[]
    
    X_List=list(X_training)
    for i in range(len(X_List)):
        sum_of_square=0
        X_vector=X_List[i]
        X_vector_list=list(X_vector)
        for j in range(len(X_vector_list)):
            sum_of_square=sum_of_square+np.square(X_vector_list[j])
        X.append(int(np.sqrt(sum_of_square)))
        
        
    """
    Uij= 1/ SUM k=1 (|xi-cj|\|xi-ck|)^2/m-1
    
    centroid- 
    Cj=SUM i=1 to N (Uij*Xi)/SUM i=1 to N (Uij)
    """ 
    def new_centroid_caclculation(centroid1,centroid2):
        list_of_new_centroids=[]
        #Calculate Uij for Cluster 1
        p=int(2/(7-1))
        list_of_U1=[]
        for k in range(len(X)):
            denominator1=round(((np.abs(X[k]-centroid1))/(np.abs(X[k]-centroid1))**p),2)
            denominator2=round(((np.abs(X[k]-centroid1))/(np.abs(X[k]-centroid2))**p),2)
            sumd1d2=int(denominator1+denominator2)
            if sumd1d2 != 0:
                list_of_U1.append(round((1/sumd1d2),4))
            else:
                list_of_U1.append(0)
            
        #Calculate Uij for Cluster 2
        
        list_of_U2=[]
        for k in range(len(X)):
            denominator1=round(((np.abs(X[k]-centroid2))/(np.abs(X[k]-centroid1))**p),2)
            denominator2=round(((np.abs(X[k]-centroid2))/(np.abs(X[k]-centroid2))**p),2)
            sumd1d2=int(denominator1+denominator2)
            if sumd1d2 != 0:
                list_of_U2.append(round((1/sumd1d2),4))
            else:
                list_of_U2.append(0)
        
        #Calculate new centroids
        
        sum_c1_num=0;
        sum_c1_den=0;
        for l in range(len(list_of_U1)):
            sum_c1_num=sum_c1_num+(list_of_U1[l]*X[l])
            sum_c1_den=sum_c1_den+(list_of_U1[l])
        list_of_new_centroids.append(round((sum_c1_num/sum_c1_den),2))
        
        
        sum_c2_num=0;
        sum_c2_den=0;
        for l in range(len(list_of_U2)):
            sum_c2_num=sum_c2_num+(list_of_U2[l]*X[l])
            sum_c2_den=sum_c2_den+(list_of_U2[l])
        list_of_new_centroids.append(round((sum_c2_num/sum_c2_den),2))
        return list_of_new_centroids
    
    c1=centroid_1
    c2=centroid_2
    print("Initial -centroid1: ",c1," -centroid2: ",c2)
    for m in range(200): #200 iterations
        new_centroid_list=new_centroid_caclculation(c1,c2)
        c1=new_centroid_list[0]
        c2=new_centroid_list[1]
        print("iteration:",m," - centroid1: ",c1," -centroid2: ",c2)
        
    
    #Identify which class label is assocaited with each given Clusters
    list_c1_bugs=[] 
    list_c2_bugs=[]
    list_cluster_class=[]
    def identify_class_for_cluster():
        for i in range(len(X_List)):
            mean_x=round(np.array(X_List[i]).mean(axis=0),2)
            a1=np.abs(int(np.square(mean_x)-np.square(c1)))
            a2=np.abs(int(np.square(mean_x)-np.square(c2)))
            e1=np.sqrt(a1)
            e2=np.sqrt(a2)
            if e1<e2:
                class1=dataset.iloc[i,-1]
                if class1 == 1:
                     list_c1_bugs.append(class1)
            else:
                class2=dataset.iloc[i,-1]
                if class2 == 1:
                    list_c2_bugs.append(class2)
                
        len_actual=len(X_List)
        med=int(len_actual/2)
        if len(list_c1_bugs) > 0:
            len_pred=len(list_c1_bugs)
            if len_pred > med:
                list_cluster_class.append(1)
                list_cluster_class.append(0)
                return list_cluster_class
            else:
                list_cluster_class.append(0)
                list_cluster_class.append(1)
                return list_cluster_class
        else:
            len_pred=len(list_c2_bugs)
            if len_pred > med:
                list_cluster_class.append(0)
                list_cluster_class.append(1)
                return list_cluster_class
            else:
                list_cluster_class.append(1)
                list_cluster_class.append(0)
                return list_cluster_class
                
            
    list_class=identify_class_for_cluster()
    class_label_1=list_class[0]
    class_label_2=list_class[1]
    
    
    #Test Data Evaluation
    X_testing = dataset.iloc[no_training_rows:total, 3:23].values
    Y_testing = dataset.iloc[no_training_rows:total,-1].values
    no_testing_rows=len(X_testing)
    test_data_for_cluster=[]
    test_list=list(X_testing)
    def identify_class_for_test_cluster():
        for i in range(len(test_list)):
            mean_x=round(np.array(test_list[i]).mean(axis=0),2)
            a1=np.abs(int(np.square(mean_x)-np.square(c1)))
            a2=np.abs(int(np.square(mean_x)-np.square(c2)))
            e1=np.sqrt(a1)
            e2=np.sqrt(a2)
            if e1<e2:
                test_data_for_cluster.append(class_label_1)
            else:
                test_data_for_cluster.append(class_label_1)
        return test_data_for_cluster
    
    test_data_predicted_value=identify_class_for_test_cluster()
    test_data_actual_value=list(Y_testing)
    
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(test_data_actual_value, test_data_predicted_value).ravel()
    
    
    
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
    listOutput.append(str(Fuzy(listProj[i])))
'''
listProj=Fuzy("forrest-0.8")