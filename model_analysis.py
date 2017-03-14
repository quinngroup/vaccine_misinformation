from __future__ import division
import gensim

model100 = gensim.models.Doc2Vec.load('labeledDoc2Vec100.model')
model200 = gensim.models.Doc2Vec.load('labeledDoc2Vec200.model')
model300 = gensim.models.Doc2Vec.load('labeledDoc2Vec300.model')

total100 = 0
total200 = 0
total300 = 0

accuracy100 = 0
accuracy200 = 0
accuracy300 = 0

for i in range(1, 11):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    print string1 + ': ' + str(model100.docvecs.most_similar(string1)[0])
    print string2 + ':' + str(model100.docvecs.most_similar(string2)[0])
    total100 = total100 + model100.docvecs.most_similar(string1)[0][1] + model100.docvecs.most_similar(string2)[0][1]
    if('True' in model100.docvecs.most_similar(string1)[0][0]):
        accuracy100 = accuracy100 + 1
    if('Misinformed' in model100.docvecs.most_similar(string2)[0][0]):
         accuracy100 = accuracy100 + 1

for i in range(1, 11):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    print string1 + ': ' + str(model200.docvecs.most_similar(string1)[0])
    print string2 + ':' + str(model200.docvecs.most_similar(string2)[0])
    total200 = total200 + model200.docvecs.most_similar(string1)[0][1] + model200.docvecs.most_similar(string2)[0][1]
    if('True' in model200.docvecs.most_similar(string1)[0][0]):
        accuracy200 = accuracy200 + 1
    if('Misinformed' in model200.docvecs.most_similar(string2)[0][0]):
         accuracy200 = accuracy200 + 1

for i in range(1, 11):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    print string1 + ': ' + str(model300.docvecs.most_similar(string1)[0])
    print string2 + ':' + str(model300.docvecs.most_similar(string2)[0])
    total300 = total300 + model300.docvecs.most_similar(string1)[0][1] + model300.docvecs.most_similar(string2)[0][1]
    if('True' in model300.docvecs.most_similar(string1)[0][0]):
        accuracy300 = accuracy300 + 1
    if('Misinformed' in model300.docvecs.most_similar(string2)[0][0]):
         accuracy300 = accuracy300 + 1

print 'Average cosine similarity 100 dimensions: ' + str(total100 / 20)
print 'Classification Accurary 100 dimensions: ' + str(accuracy100 / 20)
print 'Average cosine similarity 200 dimensions: ' + str(total200 / 20)
print 'Classification Accurary 200 dimensions: ' + str(accuracy200 / 20)
print 'Average cosine similarity 300 dimensions: ' + str(total300 / 20)
print 'Classification Accurary 300 dimensions: ' + str(accuracy300 / 20)
