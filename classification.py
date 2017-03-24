from __future__ import division
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

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

#dictionary to hold all graphing results
graphingData = {'Task 1 Acc' : [],
                'Task 1 Classifier': [],
                'Task 2 Acc': [],
                'Task 2 Classifier': [],
                'Task 3 Acc': [],
                'Task 3 Classifier': [],
                'Task 4 Acc': [],
                'Task 4 Classifier': []}

#Classification Using Unlabeled Data to predict Labeled Data

print '======================================================'
print 'Results shown here are the accuracies of classificaiton models using unlabeled data to predict the labeled data'
print '======================================================' 

#Logistic Regression
logReg = LogisticRegression()
logReg.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
logRegAcc = logReg.score(labeled_data['Feature Vector'], labeled_data['Classification'])
graphingData['Task 1 Acc'].append(logRegAcc)
graphingData['Task 1 Classifier'].append('Logistic Regression')
print 'Logistic Regression Accuracy: ' + str(logRegAcc)

#Naive Bayes
bayes = BernoulliNB()
bayes.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
bayesAcc = bayes.score(labeled_data['Feature Vector'], labeled_data['Classification'])
graphingData['Task 1 Acc'].append(bayesAcc)
graphingData['Task 1 Classifier'].append('Naive Bayes')
print 'Naive Bayes Accuracy: ' + str(bayesAcc)

#SVM
svm = LinearSVC()
svm.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
svmAcc = svm.score(labeled_data['Feature Vector'], labeled_data['Classification'])
graphingData['Task 1 Acc'].append(svmAcc)
graphingData['Task 1 Classifier'].append('SVM')
print 'SVM Accuracy: ' + str(svmAcc)

#Random Forest
rf = RandomForestClassifier()
rf.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
rfAcc = rf.score(labeled_data['Feature Vector'], labeled_data['Classification'])
graphingData['Task 1 Acc'].append(rfAcc)
graphingData['Task 1 Classifier'].append('Random Forest')
print 'Random Forest Accuracy: ' + str(rfAcc)

#Knn 
kn = KNeighborsClassifier()
kn.fit(unlabeled_data['Feature Vector'], unlabeled_data['Classification'])
knAcc = kn.score(labeled_data['Feature Vector'], labeled_data['Classification'])
graphingData['Task 1 Acc'].append(knAcc)
graphingData['Task 1 Classifier'].append('KNN')
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
logSD = np.std(logCross)
graphingData['Task 2 Acc'].append(avgLog)
graphingData['Task 2 Classifier'].append('Logistic Regression')
print 'Logistic Regression Cross-Validation Accuracy: ' + str(avgLog)
print 'Logistic Regression Cross-Validation Standard Deviation: ' + str(logSD)

#Naive Bayes
bayesCross = cross_val_score(bayes, X, Y, cv=k_fold, n_jobs=1)
avgBayes = np.mean(bayesCross)
bayesSD = np.std(bayesCross)
graphingData['Task 2 Acc'].append(avgBayes)
graphingData['Task 2 Classifier'].append('Naive Bayes')
print 'Naive Bayes Cross-Validation Accuracy: ' + str(avgBayes)
print 'Naive Bayes Cross-Validation Standard Deviation: ' + str(bayesSD)

#SVM
svmCross = cross_val_score(svm, X, Y, cv=k_fold, n_jobs=1)
avgSVM = np.mean(svmCross)
svmSD = np.std(svmCross)
graphingData['Task 2 Acc'].append(avgSVM)
graphingData['Task 2 Classifier'].append('SVM')
print 'SVM Cross-Validation Accuracy: ' + str(avgSVM)
print 'SVM Cross-Validation Standard Deviation: ' + str(svmSD)

#Random Forest
rfCross = cross_val_score(rf, X, Y, cv=k_fold, n_jobs=1)
avgRF = np.mean(rfCross)
sdRF = np.std(rfCross)
graphingData['Task 2 Acc'].append(avgRF)
graphingData['Task 2 Classifier'].append('Random Forest')
print 'Random Forest Cross-Validation Accuracy: ' + str(avgRF)
print 'Random Forest Cross-Validation Standard Deviation: ' + str(sdRF)

#Knn 
knnCross = cross_val_score(kn, X, Y, cv=k_fold, n_jobs=1)
avgKNN = np.mean(knnCross)
sdKNN = np.std(knnCross)
graphingData['Task 2 Acc'].append(avgKNN)
graphingData['Task 2 Classifier'].append('KNN')
print 'KNN Cross-Validation Accuracy: ' + str(avgKNN)
print 'KNN Cross-Validation Standard Deviation: ' + str(sdKNN)

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
graphingData['Task 3 Acc'].append(logProp)
graphingData['Task 3 Classifier'].append('Logistic Regression')
print 'Logistic Regression Proportion: ' + str(logProp)

#Naive Bayes 
bayes.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = bayes.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
bayesProp = matches / len(predictions)
graphingData['Task 3 Acc'].append(bayesProp)
graphingData['Task 3 Classifier'].append('Naive Bayes')
print 'Naive Bayes Proportion: ' + str(bayesProp)

#SVM
svm.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = svm.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
svmProp = matches / len(predictions)
graphingData['Task 3 Acc'].append(svmProp)
graphingData['Task 3 Classifier'].append('SVM')
print 'SVM Proportion: ' + str(svmProp)

#Random Forest
rf.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = rf.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
rfProp = matches / len(predictions)
graphingData['Task 3 Acc'].append(rfProp)
graphingData['Task 3 Classifier'].append('Random Forest')
print 'Random Forest Proportion: ' + str(rfProp)

#KNN 
kn.fit(all_data['Feature Vector'], all_data['Classification'])
predictions = kn.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
knnProp = matches / len(predictions)
graphingData['Task 3 Acc'].append(knnProp)
graphingData['Task 3 Classifier'].append('KNN')
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
graphingData['Task 4 Acc'].append(logProp)
graphingData['Task 4 Classifier'].append('Logistic Regression')
print 'Logistic Regression Proportion: ' + str(logProp)

#Naive Bayes 
bayes.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = bayes.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
bayesProp = matches / len(predictions)
graphingData['Task 4 Acc'].append(bayesProp)
graphingData['Task 4 Classifier'].append('Naive Bayes')
print 'Naive Bayes Proportion: ' + str(bayesProp)

#SVM
svm.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = svm.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
svmProp = matches / len(predictions)
graphingData['Task 4 Acc'].append(svmProp)
graphingData['Task 4 Classifier'].append('SVM')
print 'SVM Proportion: ' + str(svmProp)

#Random Forest
rf.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = rf.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
rfProp = matches / len(predictions)
graphingData['Task 4 Acc'].append(rfProp)
graphingData['Task 4 Classifier'].append('Random Forest')
print 'Random Forest Proportion: ' + str(rfProp)

#KNN 
kn.fit(labeled_data['Feature Vector'], labeled_data['Classification'])
predictions = kn.predict(unlabeled_data['Feature Vector'])
matches = 0
for i in range(len(predictions)):
    if(unlabeled_data['Classification'][i] == predictions[i]):
        matches = matches + 1
knnProp = matches / len(predictions)
graphingData['Task 4 Acc'].append(knnProp)
graphingData['Task 4 Classifier'].append('KNN')
print 'KNN Proportion: ' + str(knnProp)

#Plot some results
sns.set(font_scale = 1.6)
fig, (axes) = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(20, 15)
fig.subplots_adjust(hspace=0.3) 
sns.barplot(data=graphingData,x='Task 1 Classifier',y='Task 1 Acc',
            ax=axes[0][0])
axes[0][0].set(ylim=(0.4,1.0), xlabel='Algorithm', ylabel='Accuracy', title='Classification Task 1')
sns.barplot(data=graphingData,x='Task 2 Classifier',y='Task 2 Acc',
            ax=axes[0][1])
axes[0][1].set(ylim=(0.7,0.95), xlabel='Algorithm', ylabel='Accuracy', title='Classification Task 2')
sns.barplot(data=graphingData,x='Task 3 Classifier',y='Task 3 Acc',
            ax=axes[1][0])
axes[1][0].set(ylim=(0.8,1.0), xlabel='Algorithm', ylabel='Proportion', title='Classification Task 3')
sns.barplot(data=graphingData,x='Task 4 Classifier',y='Task 4 Acc',
            ax=axes[1][1])
axes[1][1].set(ylim=(0.6,1.0), xlabel='Algorithm', ylabel='Proportion', title='Classification Task 4')
plt.show()
