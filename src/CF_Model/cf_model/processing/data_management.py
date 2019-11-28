import joblib
from sklearn.pipeline import Pipeline
import os
from CF_Model.cf_model.config import config
from sklearn.datasets import load_files

import logging
import typing as t


_logger = logging.getLogger(__name__)


def load_dataset(file_name):
    if os.path.isdir(file_name):
        books = load_files(file_name)
    else:
        books = load_files(f'{config.DATASET_DIR}/{file_name}',encoding="utf-8")
    return books

def load_stopwords(file_name):
    if file_name:
        fn = os.path.join(config.STOPWORD_DIR,file_name)
        with open(fn,'r',encoding="utf-8") as f:
            stopwords=f.read()
    else:    
        stopwords = []
        for i in os.listdir(config.STOPWORD_DIR):
            fn = os.path.join(config.STOPWORD_DIR,i)
            with open(fn,'r',encoding="utf-8") as f:
                stopwords.append(f.read())
    return stopwords


def save_pipeline(*, pipeline_to_persist):
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name 

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')


def load_pipeline(*, file_name: str
                  ) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = os.path.join(config.TRAINED_MODEL_DIR,file_name)
    print(file_path)
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
