# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:23:29 2019

@author: fengzhenXu
"""
import os 
import time
import random
import jieba
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import nltk
#词去重
def make_word_set(words_file):
    words_set=[]
    with open(words_file,'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            word=line.strip()
            if len(word)>0 and word not in words_set:
                words_set.append(word)
    return set(words_set)


#文本处理，样本生成过程
def text_processing(folder_path,test_size=0.2):
    folder_list=os.listdir(folder_path)
    data_list=[]
    class_list=[]
    
    
    for folder in folder_list:
        new_folder_path=os.path.join(folder_path,folder)
        files=os.listdir(new_folder_path)
        j=1
        for file in files:
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding="utf-8") as fp:
                raw=fp.read()
            word_cut=jieba.cut(raw,cut_all=False)
            word_list=list(word_cut)
            
            data_list.append(word_list)
            class_list.append(folder)
            j+=1
    '''
    data_class_list=zip(data_list,class_list)
    random.shuffle(data_class_list)
    index=int(len(data_class_list)*test_size)+1
    train_list=data_class_list[index:]
    test_list=data_class_list[:index]
    train_data_list,train_class_list=zip(*train_list)
    test_data_list,test_class_list=zip(*test_list)
    '''
    train_data_list, test_data_list, train_class_list, test_class_list = sklearn.model_selection.train_test_split(data_list, class_list, test_size=test_size)
    all_words_dict={}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word]+=1
            else:
                all_words_dict[word]=1
    all_words_tuple_list=sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
    all_words_list=[word[0] for word in all_words_tuple_list]
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list

def words_dict(all_words_list,deleteN,stopwords_set=set()):
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_words_list),1):
        if n >1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n+=1
    return feature_words

def text_features(train_data,test_data_list,feature_words,flag='nltk'):
    def text_features(text,feature_words):
        text_words=set(text)
        
        if flag=='nltk':## nltk特征 dict
            features={word:1 if word in text_words else 0 for word in feature_words}
        elif flag=='sklearn':
            features=[1 if word in text_words else 0 for word in feature_words]
            
        else:
            features=[]
            
        return features
    train_feature_list=[text_features(text,feature_words) for text in train_data_list]
    test_feature_list=[text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list 

def text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag='nltk'):
    if flag=='nltk':
        train_flist=zip(train_feature_list,train_class_list)
        test_list=zip(test_feature_list,test_class_list)
        classifier=nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy=nltk.classify.accuracy(classifier,test_list)
        
    elif flag=="sklearn":
        classifier=MultinomialNB().fit(train_feature_list,train_class_list)
        test_accuracy=classifier.score(test_feature_list,test_class_list)
        
    else:
        test_accuracy=[]
    return test_accuracy

print("start")
folder_path="./Database/SogouC/Sample"
all_words_list,train_data_list,test_data_list,train_class_list,test_class_list=text_processing(folder_path,test_size=0.2)

stopwords_file="./stopwords_cn.txt"
stopwords_set=make_word_set(stopwords_file)

flag='sklearn'
deleteNs=range(0,1000,20)
test_accuracy_list=[]
for deleteN in deleteNs:
    feature_words=words_dict(all_words_list,deleteN,stopwords_set)
    train_feature_list,test_feature_list=text_features(train_data_list,test_data_list,feature_words,flag)
    test_accuracy=text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)
 
plt.plot(deleteNs, test_accuracy_list)
plt.title('Relationship of deleteNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()

#plt.savefig('result.png')

print("finished")      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    