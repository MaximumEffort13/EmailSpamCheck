import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

dataFrame = pd.read_csv('emails.csv')
dataFrame
dataFrame.shape
dataFrame.isnull().any()
dataFrame.drop(columns='Email No.', inplace=True)
dataFrame
dataFrame.columns
dataFrame.Prediction.unique()
dataFrame['Prediction'] = dataFrame['Prediction'].replace({0:'Not spam',1:'Spam'})
dataFrame

X= dataFrame.drop(columns='Prediction', axis=1)
Y=dataFrame['Prediction']
X.columns
Y.head()
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=1)

KN = KNeighborsClassifier
knn = KN(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("Prediction: \n")
print(y_pred)

metrixAcc = metrics.accuracy_score(y_test, y_pred)
print("KNN accuracy: ", metrixAcc)

confMat = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", confMat)

finalMod = SVC(C = 1)
finalMod.fit(x_train, y_train)
y_pred =  finalMod.predict(x_test)
SVCacc =  metrics.accuracy_score(y_test, y_pred)
print("SVM accuracy: ", SVCacc)
SVCconf = metrics.confusion_matrix(y_test, y_pred)
print("SVM confusion matrix: \n", SVCconf)
pickle.dump(finalMod, open('221168925_best_model.pkl', 'wb'))
