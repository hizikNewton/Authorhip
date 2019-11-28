from CF_Model.cf_model.predict import make_prediction
from pathlib import Path
import os
import numpy as np

pdir = Path(__file__).resolve().parent
filename=Path.joinpath(pdir,'test_dataset')

def test_prediction(rawtext):
    res = make_prediction(rawtext)
    return res
    '''
    for f in os.listdir(filename):
        f = Path(filename).joinpath(f)
        res = make_prediction(f)
        result[f] = res
    '''