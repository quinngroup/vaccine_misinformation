import gensim

model = gensim.models.Doc2Vec.load('doc2vec.model')
for i in range(1, 10):
    string1 = 'True' + str(i) + '.txt'
    string2 = 'Misinformed' + str(i) + '.txt'
    #print model.docvecs.most_similar(string1)
    #print model.docvecs.most_similar(string2)
    print model.docvecs.similarity(string1, string2)
