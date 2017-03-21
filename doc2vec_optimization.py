#import necessary packages
from __future__ import division
from os import listdir
from os.path import isfile, join
import multiprocessing
import gensim
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
import time
from itertools import product

#gather classified documents
classifiedDocLabels = []
classifiedDocLabels = [f for f in listdir("Documents/ClassifiedDocuments") if f.endswith('.txt')]

stopwords = stopwords.words("english") #stopwords list
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

#function to preprocess text
def process_text(openFile, numbers, stemming):
    #clean and tokenize document string
    raw = openFile.read().lower()
    raw = unicode(raw, errors='replace')
    cleanedText = ' '.join([word for word in raw.split() if word not in stopwords])
    
    if(numbers == True or stemming == True):
        tokens = tokenizer.tokenize(cleanedText)
        if(numbers == True):
            # remove numbers
            number_tokens = [re.sub(r'[\d]', ' ', i) for i in tokens]
            number_tokens = ' '.join(number_tokens).split()
        if(stemming == True):    
            if(numbers == True):
                #stem tokens
                stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
            else:
                stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
        if(stemming == True):
            cleanedText = ' '.join(stemmed_tokens)
        else:
            cleanedText = ' '.join(number_tokens)
    
    #return thet cleaned text 
    return cleanedText

#gather classified document text only 
labeledData = []
for doc in classifiedDocLabels:
    path = 'Documents/ClassifiedDocuments/' + doc
    f = open(path, 'r')
    cleanedText = process_text(f, False, False)
    labeledData.append(cleanedText)
    f.close()

#class needed for doc2vec model
class DocIterator(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])

#iterator object for the labeled data doc2vec model
labeledIt = DocIterator(labeledData, classifiedDocLabels)

#construct the set of hyperparameters to optimize
params = {"size": [10, 50, 100, 500, 1000, 5000],
	  "window": [5, 6, 7, 8, 9, 10],
          "min_count": [1, 2, 3, 4, 5, 10]}

#define my own scoring function for gridsearch
def scorer(estimator, X):
    #build vocab from sequence of sentence
    estimator.build_vocab(X)

    #training the model on the text corpus
    for epoch in range(10):
        estimator.train(X)
        estimator.alpha -= 0.002 # decrease the learning rate
        estimator.min_alpha = estimator.alpha # fix the learning rate, no deca
        estimator.train(X)
    
    #accuracy to return
    accuracy = 0

    for i in range(1, 11):
        string1 = 'True' + str(i) + '.txt'
        string2 = 'Misinformed' + str(i) + '.txt'
        if('True' in estimator.docvecs.most_similar(string1)[0][0]):
            accuracy = accuracy + 1
        if('Misinformed' in estimator.docvecs.most_similar(string2)[0][0]):
            accuracy = accuracy + 1
    
    return accuracy

#define function to do hyperparameter search
def hyperparameter_search(params, dociterator):
    combos = list(product(params['size'], params['window'], params['min_count']))
    print 'Testing ' + str((len(combos)))  + ' different combinations'
    maxAccuracyParams = []
    maxAccuracy = 0
    for combo in combos:
        model = gensim.models.Doc2Vec(size=combo[0], window=combo[1],
                                      min_count = combo[2],
                                      workers = multiprocessing.cpu_count(),
                                      alpha=0.025, min_alpha=0.025)
        
        accuracy = scorer(model, dociterator)/20
        
        if(maxAccuracy < accuracy):
            maxAccuracy = accuracy
            maxAccuracyParams = []
            maxAccuracyParams.append(combo)
        elif(maxAccuracy == accuracy):
            maxAccuracyParams.append(combo)
        else:
            pass

    return maxAccuracy, maxAccuracyParams
       
#tune the hyperparameters by scoring over all possible paramter combinations
print("[INFO] tuning hyperparameters")
start = time.time()
maxAcc, bestParams = hyperparameter_search(params, labeledIt)

#evaluate the results of the hyperparameter search
print("[INFO] parameter space search took {:.2f} seconds".format(
	time.time() - start))
print "[INFO] parameter search max accuracy: " + str(maxAcc)
print "[INFO] parameter search best parameters: " + str(bestParams)
print "[INFO] There are " + str(len(bestParams)) + " best parameter combos"
