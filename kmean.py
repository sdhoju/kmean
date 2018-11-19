# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:42:33 2018

@author: Sameer
"""
print("Start")
no_of_groups=10
K=4

import random
import nltk
from nltk.corpus import stopwords
import re
import pickle
import time
import collections
import math
from sklearn.datasets import fetch_20newsgroups

t0=time.time()



'''Read the dataset 20newsgroups and return list of tuple of data and it's target name'''
def load_data():
    data=[]
    no_of_groups=10
    remove =('headers', 'footers', 'quotes')
    newsgroups_train = fetch_20newsgroups(subset="train",remove =remove,shuffle=True)
    print("Total of %d data read"%len(newsgroups_train.data))
    
    
    '''Getting top 10 newsgroups by convertng the nparray into collection '''
    top_10_newsgroup = [target for target,count in collections.Counter(newsgroups_train.target).most_common(no_of_groups)] 
    
    print("Top 10 newgroups are:")
    for i in range (len(top_10_newsgroup)):
        print("\t"+newsgroups_train.target_names[i])
    
#    data = fetch_20newsgroups(subset="train", categories=categories,remove =remove ) 
    for i in range (len(newsgroups_train.target)):
        if newsgroups_train.target[i] in top_10_newsgroup:
            data.append(newsgroups_train.data[i])
        
    print("After getting data from top 10 newsgroup ,total of %d data is returned"%len(data))
    print()

    return data   
    

'''Returns the list of words in a give message. It removes stopwords and words shorter than 3 characters''' 
def words_from_message(message):
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    words=[]
    message=message.lower()
    try:
        words=[]
        message_split=message.split()
        for w in message_split:
            w = w.strip('\'"?,.')
            val = re.search(r"^[a-zA-Z][a-zA-Z]*$", w)
#            ignore if it is a stop word or les than len 2
            stopWords = set(stopwords.words('english'))
            if(w in stopWords or val is None or len(w)<=2):
                continue
            else:
                w=wordnet_lemmatizer.lemmatize(w)
                words.append(w)
    except:
        pass
    
    return words

'''Create Word features words in from all words'''
def get_word_features(wordlist):
    max_no_of_features=5000
    word_features = collections.Counter(wordlist)
    for k in list(word_features):
        if word_features[k] < 10:
            del word_features[k]
    print("Max number of Features: %d"%max_no_of_features)
    print("Created Word features words in from all words")
    print()
    return word_features.most_common(max_no_of_features)


def extract_features(document):
#    word_features = pickle.load(open("data/word_features.pickle", "rb"))
    document_words = set(document)
    features = []
    for word,cnt in word_features:
        if word in document_words:
            '''USE TD-idf Weight'''
            features.append((1+math.log10(cnt))*math.log10(len(word_features)/cnt))
        else:
            features.append(0)
    features=normalize_vector(features)
    return features

'''Create list of all words in from all the documents'''
def get_all_words(data):
    all_words=[]
    for d in data:  
        all_words.extend(words_from_message(d))
    print("Created list of all words in from all the documents")
    print("Done in  %0.3fs." % (time.time() - t0))
    return all_words



'''Implement of  K mean clustering'''
def Kmean(data,K):
    
    '''Randomly chose the K number of centroids from the existing data '''
    clusters=[]
    centroids=[]

    for i in range(K):
        rand=random.randint(0,len(data)-1)
        centroid=extract_features(words_from_message(data[rand] ))
#        centroid=normalize_vector(centroid)
        centroids.append(centroid)
        clusters.append(i)
    print("%d centroids initialized. "%len(clusters))

   
    '''Get the label closest to centroid'''
    clustered_data=cluster(centroids,clusters)
#    print("Clustering from randon Centroid")
#    print_cluster(clustered_data)
    sse=[]
    for i in range(K):
        sse.append(SSE(clusters[i],centroids[i],clustered_data))
    print("SSE for %d clusters is"%K)
#    print(sse)
    plot(clusters,sse)
   
    '''recompute the centroid'''
    for i in range(K):
        centroids[i]=recompute_centroid(i,centroids[i],clustered_data)

    '''repeat the clustering until the centroid'''
    count=0
    while(clustered_data != cluster(centroids,clusters) and count<20):
        clustered_data=cluster(centroids,clusters) 
        for i in range(len(clusters)):
            centroids[i]=recompute_centroid(i,centroids[i],clustered_data)
        count+=1
     
#    print("Clustering after %d iteration"%count)
#    print_cluster(clustered_data)


    return (clusters,centroids,clustered_data)


'''Cluster the data to closest centroid '''
def cluster(centroids,clusters): 
    clustered_d=[]
    '''Initilize the first distance to first centroid'''
    cos=cosine(centroids[0],all_data_feature[0])
    cluster=clusters[0]
    cluster
    '''Cluster all the data'''
    
    for i in range(len(all_data_feature[:200])):
        D=all_data_feature[i]
        '''Calculate the Cosine to all the centroids and label it to closest cluster '''
        for j in range(len(clusters)):
            if cosine(centroids[j],D) > cos:
                cos=cosine(centroids[j],D)
                cluster=clusters[j]
        clustered_d.append((data[i],cluster,D))
    return clustered_d

    
    
                
'''normalize the vector by dividng it's maginitude'''            
def normalize_vector(Vector):
    normalized_vector=[]
    total=0
    for v in Vector:
        total= total+v**2
    mag=total**(1/2)
    if mag ==0:
        mag=1
    for v in Vector:
        normalized_vector.append(v/mag)
    return normalized_vector
        
'''get the centroid of a cluster as an average '''            
def recompute_centroid(cluster,centroid,clustered_data):
#    all_data_feature = pickle.load(open("data/all_features.pickle", "rb"))
    new_centroid=centroid

    for i in range(len(centroid)):
        for j in range(len(clustered_data)):
            if clustered_data[j][1]==cluster:
                new_centroid[i]=(centroid[i]+clustered_data[j][2][i])/(len(centroid)+1) 
                
#    new_centroid=normalize_vector(new_centroid)
#    print(len(new_centroid))
    return new_centroid    
    
def cosine(A,B):
    cos=0
    for i in range(len(A)):
        cos=cos+ ( A[i]*B[i])
    return cos    
    
def distance(A,B):
    total=0
    for i in range(len(A)):
        total=total+ (A[i] - B[i])**2
    distance = total**(1/2)
    return distance
        
    
def SSE(cluster,centroid,clustered_data):
    sse=0
    for i in range(len(clustered_data)):
        #SSE= sum of square if distance between centroid and X
        if clustered_data[i][1]==cluster:
            sse= sse + distance(centroid, clustered_data[i][2])**2
    return  sse 
    
    
def plot(X,Y):
    import numpy as np
    import matplotlib.pyplot as plt
    y_pos = np.arange(len(X))
    plt.bar(y_pos, Y, align='center', alpha=0.8)
    plt.xticks(y_pos, X)
    plt.ylabel('SSE')
    plt.xlabel('Clusters')
    plt.title('Bar chart for SSE and cluster')
     
    plt.show()
    
    
def print_cluster(clustered_data):
    for d,c,_ in clustered_data:
        print(c,end=" ")
    print("\n\n")
               
def predict(msg):    
    D=extract_features(words_from_message(msg))
    '''Initilize the first distance to first centroid'''
    cos=cosine(centroids[0],all_data_feature[0])
    cluster=clusters[0]
    
    '''Cluster all the data'''
    for i in range(len(all_data_feature[:500])):
        D=all_data_feature[i]
        '''Calculate the distance to all the centroids and label it to closest cluster '''
        for j in range(K):
            if cosine(centroids[j],D) > cos:
                cos=cosine(centroids[j],D)
                cluster=clusters[j]
    return cluster    
    


data = load_data()

#all_words=get_all_words(data[:1000])
#pickle.dump(all_words, open("data/all_words.pickle", "wb"))
all_words = pickle.load(open("data/all_words.pickle", "rb"))
print("Total number of words: %d"%len(all_words))

#word_features = get_word_features(all_words)
#pickle.dump(word_features, open("data/word_features.pickle", "wb"))
#all_data_feature=[]
#for d in data:
#    data_feature=extract_features(words_from_message(d))
##    data_feature=normalize_vector(data_feature)
#    all_data_feature.append(data_feature)
#pickle.dump(all_data_feature, open("data/all_features.pickle", "wb"))



word_features = pickle.load(open("data/word_features.pickle", "rb"))
all_data_feature = pickle.load(open("data/all_features.pickle", "rb"))

'''Kmean pass data and  value of K ''' 
K=4
trained=Kmean(data,K)

clusters=trained[0]
centroids=trained[1]
clustered_data=trained[2]
'''Get SSE for cluster and plot them'''    
sse=[]
for i in range(K):
    sse.append(SSE(clusters[i],centroids[i],clustered_data))
print("SSE for %d clusters is"%K)
#print(sse)
plot(clusters,sse)

#msg="Used in Ward's Method of clustering in the first stage of clustering only the first 2 cells clustered together would increase SSEtotal. For cells described by more than 1 variable this gets a little hairy to figure out, it's a good thing we have computer programs to do this for us. If you are interested in trying to make your own program to perform this procedure I've scoured the internet to find a nice procedure to figure this out. The best"
#predicted_cluster=predict(msg)
#print(predicted_cluster)


print("Done in  %0.3fs." % (time.time() - t0))

















