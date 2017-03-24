from __future__ import division
import gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
from wordcloud import WordCloud
from tsne import bh_sne
import numpy as np

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

trueText = ' '.join(trueText)
misText = ' '.join(misText)

wordcloud = WordCloud().generate(trueText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud for True Documents", fontsize=20)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(misText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("WordCloud for Misinformed Documents", fontsize=20)
plt.axis("off")
plt.show()


#TSNE
x_data = np.asarray(all_data['Feature Vector']).astype('float64')
y_data= np.asarray(all_data['Classification'])

'''
for label in all_data['Classification']:
    if label == 1:
        y_data.append('TRUE')
    else:
        y_data.append('MISINFORMED')
'''

# perform t-SNE embedding
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 2))
plt.colorbar(ticks=range(2))
plt.clim(-0.5, 1.5)
plt.grid()
plt.title('t-SNE of Document Vectors')
plt.show()
