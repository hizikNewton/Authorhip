import numpy as np
import string,re
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import pandas as pd


#from cf_model.processing.errors import InvalidModelInputError
from CF_Model.cf_model.processing.data_management import load_stopwords

stop_words = load_stopwords('stop_word.txt')


class DeNoise(BaseEstimator, TransformerMixin):


    def fit(self, X, y):
        return self

    def transform(self,X):
        #strip_html
        #X = BeautifulSoup(X[0], "html.parser")
        #remove_between_square_brackets
        if(isinstance(X,str)):
           X = re.sub('\[[^]]*\]', '', X)
        #elif(isinstance(X,numpy.ndarray)):
            #X.apply(lambda x:re.sub('\[[^]]*\]', '', str(x)) )
        else:
            X = [re.sub('\[[^]]*\]', '', i) for i in X]
        return X

class Tokenize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self,X):
        #X = self.text.copy()
        
        if(isinstance(X,str)):
            tokens= word_tokenize(X)
        else:
            tokens = [word_tokenize(i) for i in X ]
        return tokens

class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.table = str.maketrans('', '', string.punctuation)

    def fit(self, X, y=None ):
        return self

    def transform(self,X):
        # convert to lower case
        if(isinstance(X[0],str)):
            tokens = [w.lower() for w in X]
            stripped = [i.translate(self.table) for i in tokens ]
            x = [word for word in stripped if word.isalpha()]
        else:
            x=[]
            for i in X:
                tokens = [w.lower() for w in i]
                # remove punctuation from each word
                stripped = [i.translate(self.table) for i in tokens ]
                
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                x.append(words)
                
        return x

class FilterStopWords(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        if(isinstance(X[0],str)):
            words = [w for w in X if w not in stop_words]
            x = [' '.join(words)]
        else:
            x = []
            for i in X:
                words = [w for w in i if w not in stop_words]
                x.append(' '.join(words))
        return x
