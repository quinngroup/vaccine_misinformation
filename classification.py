from __future__ import division
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

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

countTrue = all_data['Classification'].count(1)
percentTrue = (countTrue) / len(all_data['Classification'])
print 'Count of true documents: ' + str(countTrue)
print 'Count of misinformed documents: ' + str(len(all_data['Classification']) - countTrue)
print 'Percentage of true documents: ' + str(percentTrue)
print 'Percentage of misinformed documents: ' + str(1 - percentTrue) 

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
svm = LinearSVC()
svm.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
svmAcc = svm.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'SVM Accuracy: ' + str(svmAcc)

#Random Forest
rf = RandomForestClassifier()
rf.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
rfAcc = rf.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'Random Forest Accuracy: ' + str(rfAcc)

#Knn 
kn = KNeighborsClassifier()
kn.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
knAcc = kn.score(labeled_data['Feature Vector'], labeled_data['Classification'])
print 'KNN Accuracy: ' + str(knAcc)

#Classification Using All data and k-fold cross validation
print '======================================================'
print 'Results shown here are the cross validation accuracies of classificaiton models using k-fold cross validation on all the data'
print '======================================================' 

X = all_data['Feature Vector']
Y = all_data['Classification']

k_fold = KFold(n_splits=10, shuffle=True)

#Logistic Regression
logCross = cross_val_score(logReg, X, Y, cv=k_fold, n_jobs=1)
avgLog = np.mean(logCross)
print 'Logistic Regression Cross-Validation Accuracy: ' + str(avgLog)

#Naive Bayes
bayesCross = cross_val_score(bayes, X, Y, cv=k_fold, n_jobs=1)
avgBayes = np.mean(bayesCross)
print 'Naive Bayes Cross-Validation Accuracy: ' + str(avgBayes)

#SVM
svmCross = cross_val_score(svm, X, Y, cv=k_fold, n_jobs=1)
avgSVM = np.mean(svmCross)
print 'SVM Cross-Validation Accuracy: ' + str(avgSVM)

#Random Forest
rfCross = cross_val_score(rf, X, Y, cv=k_fold, n_jobs=1)
avgRF = np.mean(rfCross)
print 'Random Forest Cross-Validation Accuracy: ' + str(avgRF)

#Knn 
knnCross = cross_val_score(kn, X, Y, cv=k_fold, n_jobs=1)
avgKNN = np.mean(knnCross)
print 'KNN Cross-Validation Accuracy: ' + str(avgKNN)

#This section will compare "full" classifier to cosine distance method
print '======================================================'
print 'Results shown here are the proportion of how well classification models built on the entire dataset compares to our infered labels from cosine distnce'
print '======================================================' 

#Logistic Regression
logReg.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = logReg.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
logProp = matches / len(predictions)
print 'Logistic Regression Proportion: ' + str(logProp)

#Naive Bayes 
bayes.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = bayes.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
bayesProp = matches / len(predictions)
print 'Naive Bayes Proportion: ' + str(bayesProp)

#SVM
svm.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = svm.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
svmProp = matches / len(predictions)
print 'SVM Proportion: ' + str(svmProp)

#Random Forest
rf.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = rf.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
rfProp = matches / len(predictions)
print 'Random Forest Proportion: ' + str(rfProp)

#KNN 
kn.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = kn.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
knnProp = matches / len(predictions)
print 'KNN Proportion: ' + str(knnProp)

print '======================================================'
print 'Results shown here are the proportion of how well classification models built on the ground truth dataset compares to our infered labels from cosine distnce'
print '======================================================' 

#Logistic Regression
logReg.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = logReg.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
logProp = matches / len(predictions)
print 'Logistic Regression Proportion: ' + str(logProp)

#Naive Bayes 
bayes.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = bayes.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
bayesProp = matches / len(predictions)
print 'Naive Bayes Proportion: ' + str(bayesProp)

#SVM
svm.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = svm.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
svmProp = matches / len(predictions)
print 'SVM Proportion: ' + str(svmProp)

#Random Forest
rf.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = rf.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
rfProp = matches / len(predictions)
print 'Random Forest Proportion: ' + str(rfProp)

#KNN 
kn.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = kn.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
knnProp = matches / len(predictions)
print 'KNN Proportion: ' + str(knnProp)
