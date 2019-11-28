from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from CF_Model.cf_model.processing import preprocessors as pp
from CF_Model.cf_model.processing.features import TFIDVectorizer
from CF_Model.cf_model.processing.data_management import load_dataset
from sklearn import svm

cf_pipe = Pipeline(
    [
        ('deNoise',
            pp.DeNoise()),
        
        ('tokenize',
            pp.Tokenize()),
        
        ('normalize',
            pp.Normalize()),
        
        ('filter_stopword',
            pp.FilterStopWords()),
        
        
        ('tfidVectorize',TFIDVectorizer()),

        ('svm_clf',svm.SVC(kernel = 'rbf', gamma="scale",probability=True)),
    ]
)
