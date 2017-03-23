from __future__ import division
import gensim
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

'''
labeledModel1 = gensim.models.Doc2Vec.load('Models/labeledDoc2Vec.model')
labeledModel2 = gensim.models.Doc2Vec.load('Models/labeledDoc2Vec2.model')

models = [labeledModel1, labeledModel2]

for model in models:
    data = {'Cosine Similarity': [],
            'Prediction': []}
    correct_cosines = []
    incorrect_cosines = []
    for i in range(1, 11):
        string1 = 'True' + str(i) + '.txt'
        string2 = 'Misinformed' + str(i) + '.txt'
        data['Cosine Similarity'].append(model.docvecs.most_similar(string1)[0][1]) 
        data['Cosine Similarity'].append(model.docvecs.most_similar(string2)[0][1])
        if('True' in model.docvecs.most_similar(string1)[0][0]):
            data['Prediction'].append(1)
            correct_cosines.append(model.docvecs.most_similar(string1)[0][1])
        else:
            data['Prediction'].append(0)
            incorrect_cosines.append(model.docvecs.most_similar(string1)[0][1])
        if('Misinformed' in model.docvecs.most_similar(string2)[0][0]):
            data['Prediction'].append(1)
            correct_cosines.append(model.docvecs.most_similar(string2)[0][1])
        else:
            data['Prediction'].append(0)
            incorrect_cosines.append(model.docvecs.most_similar(string2)[0][1])

    stat, pval = stats.kruskal(correct_cosines, incorrect_cosines)
    print 'H-statistic of Kruskal-Wallis Test: ' + str(stat)
    print 'P-value of Kruskal-Wallis Test: ' + str(pval)
    
    num_bins = math.ceil(math.sqrt(len(correct_cosines) + len(incorrect_cosines)))
    plt.hist([correct_cosines,incorrect_cosines], bins=num_bins, stacked=True, normed = True, histtype='stepfilled')
    plt.xlabel('Pairwise Cosine Similarity')
    plt.ylabel('Density')
    low = min(data['Cosine Similarity'])
    high = max(data['Cosine Similarity'])
    plt.axis([low, high, 0, 5])
    plt.grid(True)
    plt.show()


total = 0

accuracy = 0

for i in range(1, 11):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    print string1 + ': ' + str(labeledModel1.docvecs.most_similar(string1)[0])
    print string2 + ':' + str(labeledModel1.docvecs.most_similar(string2)[0])
    total = total + labeledModel1.docvecs.most_similar(string1)[0][1] + labeledModel1.docvecs.most_similar(string2)[0][1]
    if('True' in labeledModel1.docvecs.most_similar(string1)[0][0]):
        accuracy = accuracy + 1
    if('Misinformed' in labeledModel1.docvecs.most_similar(string2)[0][0]):
         accuracy = accuracy + 1

print 'Average cosine similarity: ' + str(total / 20)
print 'Classification Accurary: ' + str(accuracy / 20)


'''

#Inference Tasks
model = gensim.models.Doc2Vec.load('Models/Doc2Vec2.model')

all_data = {'Feature Vector': [],
            'Classification': []}

labeled_data = {'Feature Vector': [],
                'Classification': []}

trueDocs = []
misDocs = []

unlabeled_data = {'Feature Vector': [],
                  'Classification': []}

docvecs = model.docvecs #get all the doc feature vectors

#iterate over first 20 docvecs which contain our labeled data
for i in range (0, 20):
    if i <= 9: #first ten are MISINFORMED, coded as 0 
        labeled_data['Feature Vector'].append(docvecs[i])
        labeled_data['Classification'].append(0)
        misDocs.append(docvecs.index_to_doctag(i))
        all_data['Feature Vector'].append(docvecs[i])
        all_data['Classification'].append(0)
    else: #last ten are TRUE, coded as 1
        labeled_data['Feature Vector'].append(docvecs[i])
        labeled_data['Classification'].append(1)
        trueDocs.append(docvecs.index_to_doctag(i))
        all_data['Feature Vector'].append(docvecs[i])
        all_data['Classification'].append(1)

for j in range(20, len(docvecs)):
    vec = docvecs[j]
    maxSimilarity = 0
    classification = 0
    #iterate over all classified documents to find most similar one
    for k in range (0,20):
        similarity = docvecs.similarity(j, k)
        if similarity > maxSimilarity:
            maxSimilarity = similarity
            classification = labeled_data['Classification'][k]
    unlabeled_data['Feature Vector'].append(vec)
    unlabeled_data['Classification'].append(classification)
    all_data['Feature Vector'].append(vec)
    all_data['Classification'].append(classification)
    if(classification == 1):
        trueDocs.append(docvecs.index_to_doctag(j))
    else:
        misDocs.append(docvecs.index_to_doctag(j))

countTrue = all_data['Classification'].count(1)
percentTrue = (countTrue) / len(all_data['Classification'])
print 'Count of true documents: ' + str(countTrue)
print 'Count of misinformed documents: ' + str(len(all_data['Classification']) - countTrue)
print 'Percentage of true documents: ' + str(percentTrue)
print 'Percentage of misinformed documents: ' + str(1 - percentTrue) 


stopwords = stopwords.words("english") #stopwords list
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

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

trueText = [] #array to hold text of true documents
i = 0 #conter to switch between two directories
data = [] #array to hold all the text documents
for doc in trueDocs:
    if i < 10:
        path = 'Documents/ClassifiedDocuments/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, True, False)
        trueText.append(cleanedText)
        i = i + 1
        f.close()
    else:
        path = 'Documents/UnlabeledDocumenets/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, True, False)
        trueText.append(cleanedText)
        f.close()

misText = [] #array to hold misinformed doc text
i = 0 #conter to switch between two directories
data = [] #array to hold all the text documents
for doc in misDocs:
    if i < 10:
        path = 'Documents/ClassifiedDocuments/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, True, False)
        misText.append(cleanedText)
        i = i + 1
        f.close()
    else:
        path = 'Documents/UnlabeledDocumenets/' + doc
        f = open(path, 'r')
        cleanedText = process_text(f, True, False)
        misText.append(cleanedText)
        f.close()

print len(trueText)
print len(misText)
