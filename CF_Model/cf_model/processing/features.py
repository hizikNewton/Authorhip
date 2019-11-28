import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from CF_Model.cf_model.processing.data_management import load_stopwords


stopwords = load_stopwords(file_name='stop_word.txt')
'''
class Count_Vectorizer(BaseEstimator,TransformerMixin):
    
    def fit(self, X, y):
        return self

    def transform(self,X):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
        return X


class TFIDTransformer(BaseEstimator,TransformerMixin):
    
    
    def fit(self, X, y=None):
       
        return self

    def transform(self,X):
        transformer = TfidfTransformer()
        x = transformer.fit_transform(X)
        return x
'''
class TFIDVectorizer(BaseEstimator,TransformerMixin):
    def __init__(self, training_state=True):
        super().__init__()
        self.training_state = training_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Creating the Tf-Idf model directly
        if self.training_state:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2),stop_words = set(stopwords))
            X = self.vectorizer.fit_transform(X).toarray()
            
        else:            
            X = self.vectorizer.transform(X).toarray()
        
        return X
