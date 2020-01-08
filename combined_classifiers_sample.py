# -*- coding: utf-8 -*-
"""

Combined Classifiers for high level 
model selection. 

Modified. Not All Classifiers used.


SAMPLE ONLY

"""

from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.svm import SVC
import pandas


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train
        vectorize_text = vectorizer.fit_transform(train_data.Title)
        classifier.fit(vectorize_text, train_data.To_Predict)

        # score
        vectorize_text = vectorizer.transform(test_data.Title)
        score = classifier.score(vectorize_text, test_data.To_Predict)
        string += '. Has score: ' + str(score)
        print(string)

# open data-set and divide it
data = pandas.read_csv('for_domain_improved.csv')
learn = data[:5500] 
test = data[5500:] 

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        MultinomialNB(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
    ],
    learn,
    test
)
