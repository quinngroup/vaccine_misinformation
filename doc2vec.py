from os import listdir
from os.path import isfile, join
import multiprocessing
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence

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

i = 0 #conter to switch between two directories
data = [] #array to hold all the text documents
for doc in docLabels:
    if i < 20:
        path = 'Documents/ClassifiedDocuments/' + doc
        f = open(path, 'r')
        data.append(f.read())
        i = i + 1
        f.close()
    else:
        path = 'Documents/UnlabeledDocumenets/' + doc
        f = open(path, 'r')
        data.append(f.read())
        f.close()

labeledData = []
for doc in classifiedDocLabels:
    path = 'Documents/ClassifiedDocuments/' + doc
    f = open(path, 'r')
    labeledData.append(f.read())
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

dims = [100, 200, 300]

for dim in dims:

    #build the Doc2Vec model at a fixed learning rate
    model = gensim.models.Doc2Vec(size=dim, window=8, min_count=3, 
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
    string = 'labeledDoc2Vec' + str(dim) + '.model'
    model.save(string)

for dim in dims:
    #build the Doc2Vec model at a fixed learning rate
    model = gensim.models.Doc2Vec(size=300, window=8, min_count=3, 
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
    string = 'Doc2Vec' + str(dim) + '.model'
    model.save(string)

