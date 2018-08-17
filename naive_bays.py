# -*- coding: utf-8 -*-
"""
Naive Bayes Document Classification

Dataset used: IMBD dataset movie review dataset
url: http://ai.stanford.edu/~amaas/data/sentiment/

Dataset description:
    Raw text data 

dataset directory hierarchy: 
    aclImdb
        -train
            -neg
            -pos
        -test
            -neg
            -pos

Created on Fri Mar 16 18:43:32 2018
@author: i_anu
"""
import os
from os import listdir
import string
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer

def getWords(stringWords,stemmer):
    '''
    Removes the punctuations, digits and stopwords to return the list of the words
    '''
    translator = str.maketrans(string.punctuation,' '*len(string.punctuation)) #translater to remove punctuations
    stringWords = stringWords.lower() #convert all to lowercase
    stringWords = re.sub(r'\d+', '', stringWords) #remove all digits 
    #sentences = list(map(lambda x: stemmer.stem(x),stringWords.split(".")))
    #wordList = [word for sentence in sentences for word in sentence.translate(translator).split()]
    stringList = stringWords.translate(translator).split() #removes the punctuations in the words and split
    wordList= list(map(lambda x: stemmer.stem(x),stringList))
    wordList = list(filter (lambda x: len(x)>=1,wordList)) #remove empty words
    return list(filter(lambda x: x not in stopwords.words('english'),wordList))

def getProbabilities(classesSummary,classKey,word,lengthOfVocab):
    '''
    Returns the Proabability of the given word in the Dataset on given class
    '''
    totalInClasses = 0
    try:
        numberOfOccurence = classesSummary[classKey][word]
    except:
        numberOfOccurence = 0
    for classes in classesSummary:
        try:
            occurence = classesSummary[classKey][word]
        except:
            occurence = 0
        totalInClasses += occurence
    #print(lengthOfVocab,totalInClass)
    return np.log10((float(numberOfOccurence) + 1) / (float(lengthOfVocab) + totalInClasses))

def getLogProbability(classesSummary,classKey,bagOfWords,lengthOfVocab,totalInClass,classProbability):
    '''
    Calualtes the log probability of for the given input vector
    '''
    probability = np.log10(classProbability)
    for word in bagOfWords:
        probability += getProbabilities(classesSummary,classKey,word,lengthOfVocab)
    return probability

def train(classValues,classDataValuesAndPaths,stemmer):
    '''
    Train the IMBD dataset
    '''
    counter = 0 
    classResults = {}
    classCounts = {}
    for classValue in classValues:
        classResults[classValue] = {}
        classCounts [classValue] = 0

    for classValue in classValues:
        for files in listdir(classDataValuesAndPaths[classValue]['train']):
            print(counter)
            classCounts[classValue] +=1
            readFile = open(os.path.join(classDataValuesAndPaths[classValue]['train'],files),
                            'r',encoding="utf8")
            textData = readFile.read()
            textData = BeautifulSoup(textData,'lxml')
            bagOfWords = getWords(textData.get_text(),stemmer)
            countDict = dict(Counter(bagOfWords))
            for word in countDict:
               if word in classResults[classValue]:
                   classResults[classValue] [word] += countDict[word]
               else:
                   classResults[classValue] [word] = countDict[word]
            counter+=1
    return classResults,classCounts

def classSummary(classValues,classResults,classCounts):
    '''
    Returns the summary of the class separation done
    '''
    summary = {}
    vocabulary = []
    total = 0
    for classes in classValues:
        classTotal = classCounts[classes]
        summary[classes] = {'Total':classTotal}
        total+= classTotal
        vocabulary += list(classResults[classes].keys())
    vocabulary = set(vocabulary)
    summary['Vocabulary'] = {'length':len(vocabulary),'Vocabulary':vocabulary}
    for classes in classValues:
        summary[classes]['Probability'] = float(summary[classes]['Total'])/float(total)
    return summary

def test(classValues,classDataValuesAndPaths,summary,classResults,stemmer):
    '''
    Tests the given model against the test dataset
    '''
    classOutputs = []
    counter = 0
    correct = 0
    for classValue in classValues:
        for files in listdir(classDataValuesAndPaths[classValue]['test']):
            readFile = open(os.path.join(classDataValuesAndPaths[classValue]['test'],files),
                            'r',encoding="utf8")
            textData = readFile.read()
            textData = BeautifulSoup(textData,'lxml')
            bagOfWords = getWords(textData.get_text(),stemmer)
            logProbability = -np.inf
            for classes in classValues:
                logClassProbability = getLogProbability(classResults,classes,bagOfWords,
                                                        summary['Vocabulary']['length'],
                                                        summary[classes]['Total'],
                                                        summary[classes]['Probability'])
                if(logClassProbability > logProbability):
                    logProbability = logClassProbability
                    outputClass = classes
            if classValue == outputClass:
                correct +=1        
            print(counter,classValue,outputClass)
            classOutputs.append({'File':files, 'Expected' : classValue, 'Output' : outputClass})
            counter +=1
            
    return classOutputs,float(correct)/counter
           

baseDir = os.getcwd()
datasetPath  = os.path.join(baseDir,'aclImdb')

#Training"
trainData = os.path.join(datasetPath,'train')
negatives = os.path.join(trainData,'neg')
positives = os.path.join(trainData,'pos')

testData = os.path.join(datasetPath,'test')
negTest = os.path.join(testData,'neg')
posTest = os.path.join(testData,'pos')

classValues = ['neg','pos']
classDataValuesAndPaths = {
        'neg':{'train':negatives,
               'test':negTest
               },
        'pos':{'train':positives,
               'test':posTest
               }
                           }
stemmer = PorterStemmer()
classSeparationAndCounts,classCounts = train(classValues,classDataValuesAndPaths,stemmer)
print("Training Complete") 
summary = classSummary(classValues,classSeparationAndCounts,classCounts)
testing,accuracy = test(classValues,classDataValuesAndPaths,summary,classSeparationAndCounts,stemmer)

'''
The misclassifcations
'''            
IncorrectOnes = list(filter(lambda x: x['Expected'] != x['Output'],testing))  
print(IncorrectOnes[:10])     

for classes in classValues:
    print(sorted(classSeparationAndCounts[classes],
                 key=classSeparationAndCounts[classes].get,
                 reverse=True)[:50])   

'''
Calcaulation of true Negatives and true Positives
'''
for classes in classValues:
    print('Expected class: ' + classes + ' Output Class: '+ classes )
    print(len(list(filter(lambda x: x['Output'] == classes == x['Expected'],testing))))

# Actual pos but result neg
len(list(filter(lambda x: x['Output'] == 'neg' != x['Expected'],testing)))

#Actual neg but result pos
len(list(filter(lambda x: x['Output'] == 'pos' != x['Expected'],testing)))
    
    



        
    
    


 


           
           