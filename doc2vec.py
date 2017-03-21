from os import listdir
from os.path import isfile, join
import multiprocessing
import gensim
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

#gather classified documents
classifiedDocLabels = []
classifiedDocLabels = [f for f in listdir("Documents/ClassifiedDocuments") if f.endswith('.txt')]

#gather unlabeled documents
unlabeledDocLabels = []
unlabeledDocLabels = [f for f in listdir("Documents/UnlabeledDocumenets") if f.endswith('.txt')]

#array of combined doc labels
docLabels = []
for file in classifiedDocLabels:
    docLabels.append(file)
for file in unlabeledDocLabels: 
    docLabels.append(file)

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

i = 0 #conter to switch between two directories
data = [] #array to hold all the text documents
for doc in docLabels:
    if i < 20:
        path = 'Documents/ClassifiedDocuments/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, False, False)
        data.append(cleanedText)
        i = i + 1
        f.close()
    else:
        path = 'Documents/UnlabeledDocumenets/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, False, False)
        data.append(cleanedText)
        f.close()

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

#iterator object for the doc2vec model
it = DocIterator(data, docLabels)

#iterator object for the labeled data doc2vec model
labeledIt = DocIterator(labeledData, classifiedDocLabels)

#build the Doc2Vec model at a fixed learning rate
model = gensim.models.Doc2Vec(size=1000, window=8, min_count=4, 
                              workers = multiprocessing.cpu_count(),
                              alpha=0.025, min_alpha=0.025)

#build vocab from sequence of sentence
model.build_vocab(labeledIt)

#training the model on the text corpus
for epoch in range(10):
    model.train(labeledIt)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(labeledIt)

#save the model
model.save('Models/labeledDoc2Vec2.model')


#build the Doc2Vec model at a fixed learning rate
model = gensim.models.Doc2Vec(size=1000, window=8, min_count=4, 
                              workers = multiprocessing.cpu_count(),
                              alpha=0.025, min_alpha=0.025)

#build vocab from sequence of sentence
model.build_vocab(it)

#training the model on the text corpus
for epoch in range(10):
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)

#save the model
model.save('Models/Doc2Vec2.model')

