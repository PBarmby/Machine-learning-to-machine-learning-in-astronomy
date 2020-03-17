#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:12:47 2019

@author: ryanchen1
"""
#print(bibCodeabstractArray["2017MNRAS.465.1959C"])
import ads
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
neuralCount=bayesCount=meanClusterCount=supportvectorClusterCount=aprioriClusterCount=linearRegreCount=logisticRegreCount=randomForestCount=decisionTreeCount=nearestneighCount=0
totalCount=0;
pst=PorterStemmer()
ads.config.token='umLarfQpDYvrHlGASUlcW5hyUSzTYMh6PJD8MUZ0'
#keywordsSearch=["redshift","galaxies","stars","cosmology","ISM","Galaxy","Sun","planets"]
#keywordsSearch=["collection:\"astronomy\" year:2000-2005 redshift","collection:\"astronomy\" year:2000-2008 redshift","collection:\"astronomy\" year:2000-2011 redshift","collection:\"astronomy\" year:2000-2014 redshift","collection:\"astronomy\" year:2000-2017 redshift","collection:\"astronomy\" year:2000-2019 redshift"]
#keywordsSearch=["collection:\"astronomy\" year:2000-2005 photometric","collection:\"astronomy\" year:2000-2008 photometric","collection:\"astronomy\" year:2000-2011 photometric","collection:\"astronomy\" year:2000-2014 photometric","collection:\"astronomy\" year:2000-2017 photometric","collection:\"astronomy\" year:2000-2020 photometric"]
keywordsSearch=["collection:\"astronomy\" year:2000-2005 machine learning","collection:\"astronomy\" year:2005-2010 machine learning","collection:\"astronomy\" year:2010-2015 machine learning","collection:\"astronomy\" year:2015-2020 machine learning"]
#keywordsSearch=["collection:\"astronomy\" year:2000-2020 photometric"]

#keywordsSearch=["collection:\"astronomy\" decision tree"]
sum_column_names=['neuralCount','bayesCount','meanClusterCount','supportvectorClusterCount','aprioriClusterCount','linearRegreCount','logisticRegreCount','randomForestCount','decisionTreeCount','nearestneighCount']
#sum_row_names=["redshift","galaxies","stars","cosmology","ISM","Galaxy","Sun","planets"]
#sum_row_names=["year:2000-2005 redshift","year:2000-2008 redshift","year:2000-2011 redshift","year:2000-2014 redshift","year:2000-2017 redshift","year:2000-2019 redshift"]
sum_row_names=["year:2000-2005 machine learning","year:2005-2010 machine learning","year:2010-2015 machine learning","year:2015-2020 machine learning"]
#sum_row_names=["year:2000-2020 photometric"]
#sum_row_names=["Decision Tree"]
#sum_row_names=["year:2000-2005 photometric","year:2000-2008 photometric","year:2000-2011 photometric","year:2000-2014 photometric","year:2000-2017 photometric","year:2000-2020 photometric"]

matrixResult=[]
for i in range(0,len(keywordsSearch)):
    neuralCount=bayesCount=meanClusterCount=supportvectorClusterCount=aprioriClusterCount=linearRegreCount=logisticRegreCount=randomForestCount=decisionTreeCount=nearestneighCount=0
    searchKeywords=keywordsSearch[i]
    abstractquery="machine learning"
    searchQuery=keywordsSearch[i]
    print("The Abstract Query is "+abstractquery)
    print('\n')
    print("The Search Keywords is "+str(searchQuery))
    papers=list(ads.SearchQuery(q='abs:\"machine learning\"'+keywordsSearch[i],max_pages=100000,fl=['abstract','bibcode','title','citation_count','keyword'],sort="citation_count"))
    print('\n')
    print("The number of paper found is "+str(len(papers)))
    print('\n')
    titlesArray=[]
    bibCodeabstractArray={}
    frequencyDict=FreqDist()
    doublefrequencyDict=FreqDist()
    triplefrequencyDict=FreqDist()
    citationTop3={}
    citationCt=0
    keywordsDictionList={}
    for paper in papers:
         if(paper.keyword!=None):
            keywordLists=paper.keyword
            for keywor in keywordLists:
                if(keywor in keywordsDictionList):
                    keywordsDictionList[keywor]+=1
                else:
                    keywordsDictionList[keywor]=1
                
         if(citationCt<3):
             citationTop3[paper.bibcode]=paper.title
             citationCt=citationCt+1
         #bibCodeabstractArray[paper.bibcode]=paper.abstract
         paperAbstract=str((paper.abstract)).lower()
         table=str.maketrans(",.?:;&()'><\"",12*" ")
         paperAbstract=paperAbstract.translate(table)
         abstractPaper=word_tokenize(paperAbstract)
         for wordssss in abstractPaper:
             wordssss=pst.stem(wordssss)
         bigramsData=list(nltk.bigrams(abstractPaper))
         trigramsData=list(nltk.trigrams(abstractPaper))
         for words in abstractPaper:
             frequencyDict[words.lower()]+=1
         for doubles in bigramsData:
             doublefrequencyDict[doubles]+=1
         for triples in trigramsData:
             triplefrequencyDict[triples]+=1
    frequencyDictTop2000=frequencyDict.most_common(2000)
    doublefrequencyDict1500=doublefrequencyDict.most_common(1500)
    triplefrequencyDict1500=triplefrequencyDict.most_common(1500)
    #print("Frenquency of Single Word in literature:")
#    print("\n")
   # print(frequencyDictTop2000)
#    print("\n")
#    print("\n")
#    print("\n")
    #print("Frenquency of Double Words in literature:")
#    print("\n")
    #print(doublefrequencyDict1500)
    
    for keyy in frequencyDictTop2000:
        if("bayes" in str(keyy)):
            _,bayesCount=keyy
        elif("apriori" in str(keyy)):
            _,aprioriClusterCount=keyy
        else:
            continue
        
    for key in doublefrequencyDict1500:
        if "neural" in str(key) and "network" in str(key):
            _,neuralCount=key
        elif ("mean" in str(key) and "cluster" in str(key)):
            _,meanClusterCount=key
        elif (("vector" in str(key) and "machine" in str(key))) or  (("svm") in str(key)):
            _,supportvectorClusterCount=key
        elif ("linear" in str(key) and "regression" in str(key)) :
            _,linearRegreCount=key
        elif ("logis" in str(key)and "regression"in str(key)):
            _,logisticRegreCount=key
        elif ("random"in str(key) and "forest"in str(key)) :
            _,randomForestCount=key
        elif ("decision" in str(key) and "tree" in str(key)) :
            _,decisionTreeCount=key
        elif ("nearest" and "neighbour") in str(key):
            _,nearestneighCount=key
        else:
            continue
    totalCount=neuralCount+bayesCount+meanClusterCount+supportvectorClusterCount+aprioriClusterCount+linearRegreCount+logisticRegreCount+randomForestCount+decisionTreeCount+nearestneighCount
    print("The total count of Machine Learning Terms being mentioned is "+str(totalCount))
    print("\n")
    label=['neuralCount','bayesCount','meanClusterCount','supportvectorClusterCount','aprioriClusterCount','linearRegreCount','logisticRegreCount','randomForestCount','decisionTreeCount','nearestneighCount']
    number=[neuralCount,bayesCount,meanClusterCount,supportvectorClusterCount,aprioriClusterCount,linearRegreCount,logisticRegreCount,randomForestCount,decisionTreeCount,nearestneighCount]
    labelAppend=[]
    numberAppend=[]
    for i in range(0,len(label)):
        if(number[i]!=0):
            labelAppend.append(label[i])
            numberAppend.append(number[i])
            matrixResult.append(number[i])
        else:
            matrixResult.append(0)
    index = np.arange(len(labelAppend))
    plt.bar(index, numberAppend,align='center',alpha=0.5)
    plt.xticks(index, labelAppend,rotation = 30)
    plt.xlabel('Methods Name', fontsize=5)
    plt.ylabel('Counts', fontsize=3)
    plt.title('Use of Machine Learning Techniques in '+searchQuery)
    print('\n')
    print('\n')
    plt.show()
    plt.draw()
    plt.savefig('distributinoTechniBar'+searchKeywords+'.png', dpi=1200)
    
    colors=['beige', 'yellowgreen', 'lightcoral', 'lightskyblue','red','purple','blue','yellow','brown','orange']
    colorAppend=[]
    for s in range(0,len(numberAppend)):
        colorAppend.append([colors[s]])
    plt.pie(numberAppend,labels=labelAppend,autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Use of Machine Learning Techniques in'+searchQuery)
    plt.axis('equal')
    plt.show()
    
    plt.savefig('distributinoTechniPie'+searchKeywords+'.png', dpi=1200)
#    print("\n")
#    print("\n")
#    print("\n")
    #print("Frenquency of Triple Words in literature:")
#    print("\n")
    #print(triplefrequencyDict1500)
#    print("\n")
    print("Top 3 Paper to Investigate:\n")
    for key in citationTop3:
        print("Paper Name",citationTop3[key],"BibCode:",key,'\n')
    data_sorted = {k: v for k, v in sorted(keywordsDictionList.items(), key=lambda x: x[1],reverse=True)}
    i=0
    for a,b in data_sorted.items():
        if(i<15):
            print(a,"Count:",b)
            i=i+1
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
finalResultmatrix=np.reshape(matrixResult,(len(keywordsSearch),len(sum_column_names)))
df=pd.DataFrame(finalResultmatrix,columns=sum_column_names, index=sum_row_names)
export_csv = df.to_csv ('astronomyKeyWordsSearchSummary.csv', index = True, header=True)
#df.to_hdf('astronomyKeyWordsSearchSummary.hdf', 'astronomyKeyWordsSearchSummary')
#print(pd.read_hdf('astronomyKeyWordsSearchSummary.hdf'))
