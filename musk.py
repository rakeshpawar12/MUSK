# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:13:34 2019

@author: rakesh
"""

import pandas as pd
from pandas import Series,DataFrame
import numpy as np

df=pd.read_csv("musk_csv.csv")
df=df.drop("ID",axis=1)
df=df.drop("molecule_name",axis=1)
df=df.drop("conformation_name",axis=1)
df.info()

from sklearn.model_selection import train_test_split
y=df["class"]
x=df.drop("class",axis=1)
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=123)

####################
#logistic regression
import statsmodels.api as sm
logit_model=sm.Logit(yTrain,xTrain)

result=logit_model.fit()

print(result.summary())

#for predication
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(xTrain,yTrain)

log_predT=logmodel.predict(xTrain)
log_pred

log_pred=logmodel.predict(xTest)
log_pred
###confusion matrix
from sklearn.metrics import confusion_matrix

#acc of train 
confusion_matrix(yTrain,log_predT)
cmlogT=confusion_matrix(yTest,log_pred)
acculog=(cmlog[0,0]+cmlog[1,1])/np.sum(cmlog)
acculog

#acc of test
confusion_matrix(yTest,log_pred)
cmlog=confusion_matrix(yTest,log_pred)
acculog=(cmlog[0,0]+cmlog[1,1])/np.sum(cmlog)
acculog

from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve

lg_roc_auc=roc_auc_score(yTest,logmodel.predict(xTest))
lg_roc_auc


################
#Random forest 
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,oob_score=True,random_state=123)
random_forest.fit(xTrain,yTrain)
Y_predication = random_forest.predict(xTest)
random_forest.score(xTrain,yTrain)
cmlog2=confusion_matrix(yTest,Y_predication)
print(cmlog2)

acculog2=(cmlog2[0,0]+cmlog2[1,1])/np.sum(cmlog2)
acculog2 

lg_roc_auc2=roc_auc_score(yTest,random_forest.predict(xTest))
lg_roc_auc2



#############
#decision Tree
from sklearn.tree import DecisionTreeClassifier 
decision_tree= DecisionTreeClassifier()
decision_tree.fit(xTrain,yTrain)
Y_pred_DT=decision_tree.predict(xTest)

#confusion matrix
cmlogDT=confusion_matrix(yTest,Y_pred_DT)
cmlogDT
acculogDT=(cmlogDT[0,0]+cmlogDT[1,1])/np.sum(cmlogDT)
acculogDT

lg_roc_aucDT=roc_auc_score(yTest,decision_tree.predict(xTest))
lg_roc_aucDT

###################
#SUPPORT VECTOR MACHINE (SVM)
from sklearn import svm,datasets
from sklearn.model_selection import GridSearchCV

#tuning parameters
Cs=[0.001,0.01,0.1,1,10]
gammas=[0.001,0.01,0.1,1]
param_grid ={'C':Cs,'gamma':gammas}
param_grid

grid_search = GridSearchCV(svm.SVC(kernel='rbf'),param_grid,cv=5)
grid_search.fit(xTrain,yTrain)
print(grid_search.best_params_)

#fitting svm with best parameter
from sklearn.svm import SVC, LinearSVC
svc=SVC(C=1,gamma=0.001,probability=True)
svc.fit(xTrain,yTrain)
Y_pred_SVM=svc.predict(xTest)

from sklearn.metrics import confusion_matrix
cmlog3=confusion_matrix(yTest,Y_pred_SVM)
cmlog3
acculog3=(cmlog3[0,0]+cmlog3[1,1])/np.sum(cmlog3)
acculog3

lg_roc_aucSVM=roc_auc_score(yTest,svc.predict(xTest))
lg_roc_aucSVM

##############################
# NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# create model
model = Sequential()
model.add(Dense(12, input_dim=166, activation='relu'))
model.add(Dense(166, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(xTrain, yTrain, validation_split=0.2, epochs=166, batch_size=100, verbose=0)

accuracy = model.evaluate(xTest, yTest, verbose=0)
accuracy


#########################
# accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



















