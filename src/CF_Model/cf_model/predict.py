import numpy as np
import csv
from CF_Model.cf_model.processing.data_management import load_pipeline
from CF_Model.cf_model.config import config
import logging
import typing as t
import os
_logger = logging.getLogger(__name__)
dataset = config.DATASET_DIR
pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}.pkl'
_cf_pipe = load_pipeline(file_name=pipeline_file_name)
testfile = config.TEST_DATA_FILE



def make_prediction(f):
    if os.path.isdir(f):
        with open(f,'r',encoding='utf-8') as f:
            f  = f.read()
    pipe = _cf_pipe.set_params(tfidVectorize__training_state = False)
    prediction = pipe.predict(f).tolist()[0]
    score = pipe.predict_proba(f).tolist()[0]
    dec_fn = pipe.decision_function(f).tolist()[0]
    output = np.exp(prediction)
    
    with open(f'{dataset}/datacategory.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = {'predictions': rows[prediction]['category'],
                'acc':score[prediction],'dfv':dec_fn[prediction]}
    return results

