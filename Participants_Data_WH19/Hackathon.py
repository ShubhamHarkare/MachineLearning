import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Train.csv')

y_train=df.pop('Sentiment')

df_t=pd.read_csv("Test.csv")

dataset=pd.concat([df,df_t],ignore_index=True)

print(len(dataset.Text_ID.unique()))

dataset=dataset.drop('Text_ID',axis=1)

Prod_Type=dataset.pop('Product_Type')

from wordcloud import WordCloud,STOPWORDS
text=' '.join(dataset.Product_Description)

wordcloud=WordCloud(width=2000,height=2000).generate(text)
plt.imshow(wordcloud)


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 9092):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Product_Description'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.extend(['link','sxsw','rt','mention'])
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(corpus).toarray()

from pandas import DataFrame
dataset=pd.DataFrame(X)

dataset['Product_Type']=Prod_Type

X=dataset.iloc[:,:].values

X_train=X[:6364,:]
X_test=X[6364:,:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=150)
classifier1.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(X_train,y_train)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from xgboost import XGBClassifier
classifier3=XGBClassifier()
classifier3.fit(X_train,y_train)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred=classifier3.predict_proba(X_test)

submission_df=pd.DataFrame(y_pred,columns=['Class_0','Class_1','Class_2','Class_3'])
submission_df.to_csv('my_submission_file.csv', index=False)