from __future__ import division
import gensim
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

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

'''
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
