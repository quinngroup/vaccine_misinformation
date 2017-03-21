from __future__ import division
import gensim

labeledModel = gensim.models.Doc2Vec.load('Models/labeledDoc2Vec2.model')

total = 0

accuracy = 0

for i in range(1, 11):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    print string1 + ': ' + str(labeledModel.docvecs.most_similar(string1)[0])
    print string2 + ':' + str(labeledModel.docvecs.most_similar(string2)[0])
    total = total + labeledModel.docvecs.most_similar(string1)[0][1] + labeledModel.docvecs.most_similar(string2)[0][1]
    if('True' in labeledModel.docvecs.most_similar(string1)[0][0]):
        accuracy = accuracy + 1
    if('Misinformed' in labeledModel.docvecs.most_similar(string2)[0][0]):
         accuracy = accuracy + 1

print 'Average cosine similarity: ' + str(total / 20)
print 'Classification Accurary: ' + str(accuracy / 20)
