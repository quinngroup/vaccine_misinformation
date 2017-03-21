import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#NOTE: TRUE documents are coded as 1 and MISINFORMED as 0

#Inference Tasks
model = gensim.models.Doc2Vec.load('Models/Doc2Vec2.model')

all_data = {'Feature Vector': [],
            'Classification': []}

labeled_data = {'Feature Vector': [],
                'Classification': []}

unlabeled_data = {'Feature Vector': [],
                  'Classification': []}

docvecs = model.docvecs #get all the doc feature vectors

#iterate over first 20 docvecs which contain our labeled data
for i in range (0, 20):
    if i <= 9: #first ten are MISINFORMED, coded as 0 
        labeled_data['Feature Vector'].append(docvecs[i])
        labeled_data['Classification'].append(0)
        all_data['Feature Vector'].append(docvecs[i])
        all_data['Classification'].append(0)
    else: #last ten are TRUE, coded as 1
        labeled_data['Feature Vector'].append(docvecs[i])
        labeled_data['Classification'].append(1)
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

#Classification Tasks

#Classification Using Unlabeled Data to predict Labeled Data

print '======================================================'
print 'Results shown here are the accuracies of classificaiton models using unlabeled data to predict the labeled data'
print '======================================================' 

#Logistic Regression
logReg = LogisticRegression()
logReg.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
logRegAcc = logReg.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'Logistic Regression Accuracy: ' + str(logRegAcc)

#Naive Bayes
bayes = BernoulliNB()
bayes.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
bayesAcc = bayes.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'Naive Bayes Accuracy: ' + str(bayesAcc)

#SVM
svm = SVC()
svm.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
svmAcc = svm.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'SVM Accuracy: ' + str(svmAcc)

#Random Forest
rf = RandomForestClassifier()
rf.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
rfAcc = rf.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'Random Forest Accuracy: ' + str(rfAcc)
