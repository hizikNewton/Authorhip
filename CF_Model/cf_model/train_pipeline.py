import numpy as np
from sklearn.model_selection import train_test_split
import pipeline
from processing.data_management import (load_dataset, save_pipeline)
from sklearn.datasets import load_files
from config import config
import os,glob
from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

_logger = logging.getLogger(__name__)

books = os.path.join(config.DATASET_DIR,'txt_book')


class Train():
    def CreateData(self,category):
        setattr(self,field,category)
        #self.english_author = english_author
    
    def __init__(self):
        self.txtbook = []
        
        self.category = []
        self.filenames =[]
        self.content = []
        self.dirpath = []
        
        pathlist = []
        for dirpath,dirname,filename in os.walk(books):
            [self.txtbook.append(i) for i in [dirname,filename] if i!=[]]
            pathlist.append(dirpath)
        self.filenames = self.txtbook[1:]
        self.txtbook =[i for i in zip(self.txtbook,pathlist)]
        
        if len(self.txtbook[0][0])==len(self.txtbook[1:]):
            for i in self.txtbook[1:]:
                x,y = i
                self.dirpath.append([os.path.join(y,i) for i in x])

            for pathlist in self.dirpath:
                rawcontent = ""
                for path in pathlist:
                    with open(path,'r',encoding = 'utf-8') as f:
                        rawcontentarray = f.read()
                    rawcontent+=rawcontentarray
                self.content.append(rawcontent)
            self.category = self.txtbook[0][0]


    def run_training(self):
        dataSet = list(zip(self.category,self.filenames,self.content))
        df = pd.DataFrame(data = dataSet, columns=['category', 'filenames','content'])
        dftosave = df[['category', 'filenames']]
        dftosave.to_csv(f'{config.DATASET_DIR}/datacategory.csv',index=False,header=True)
        df['category_id'] = df['category'].factorize()[0]
        category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', 'category']].values)
        X = df.content
        y = df.category_id
        pipe = pipeline.cf_pipe.fit(X,y)
        save_pipeline(pipeline_to_persist=pipeline.cf_pipe)


if __name__ == '__main__':
        #get data
        train = Train()
        train.run_training()
