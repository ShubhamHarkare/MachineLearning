import numpy as np
import pandas as pd

df=pd.read_csv('Train.csv')

y_train=df.pop('Class')

df_t=pd.read_csv('Test.csv')

dataset=pd.concat([df,df_t],ignore_index=True)

X=dataset.iloc[:,:].values
X_train=X[:1763,:]
X_test=X[1763:,:]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#FIRST CLASSIFIER#-
from sklearn.linear_model import RidgeClassifier
classifier = RidgeClassifier()
classifier.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
classifier1=GaussianNB()
classifier1.fit(X_train,y_train)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


y_pred=classifier.predict(X_test)

from pandas import DataFrame
submission_df=DataFrame(y_pred,columns=['Class'])
submission_df.to_csv('my_submission_file1.csv', index=False)
