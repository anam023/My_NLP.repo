# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:51:56 2021

@author: Anam Fatima
"""


import nltk
import pandas as pd
import re


import os
os.getcwd()
os.chdir('C:\\Users\\Anam Fatima\\Downloads\\smsspamcollection')

messages= pd.read_csv('SMSSpamCollection', sep='\t', 
                      names=['label','message'])

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

corpus=[]
for i in range(len(messages)):
    review= re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review= review.lower()
    review= review.split()
    
    review= [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
    
# creating bag of words model using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 5000)
x= cv.fit_transform(corpus).toarray() 
#we should select columns which have frequently occured words

#we will convert our label column into dummy variable
y= pd.get_dummies(messages['label'])
y= y.iloc[:,1].values

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split( x, y, test_size=0.20, random_state=0)

#training model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()
mnb_model= mnb.fit(x_train, y_train)

y_pred= mnb_model.predict(x_test)

from sklearn.metrics import confusion_matrix 
cm= confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)

