import os
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
STOPWORD_DIR =  DATASET_DIR/'StopWords'
# data
CF_model_path = pathlib.Path(__file__).resolve().parents[2]
TEST_DATA_FILE = CF_model_path/'tests/'
TARGET = ''


# variables
FEATURES = []

PIPELINE_NAME = 'svm_cf'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_pipe'

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
