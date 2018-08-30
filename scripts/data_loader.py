import pandas as pd
import numpy as np
import pickle
import os
import warnings
import logging
import time
from os import path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        logger.info('\nTimings: %r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def load_data(data_path, pickle_path):
    """Loads csv data
    
    Parameter
    ---------
    path : Path where csv is located
    
    Returns
    -------
    data : DataFrame containing the csv data
    """
    if path.exists(pickle_path):
        data = pd.read_pickle(pickle_path)
    else:
        data = pd.read_csv(data_path)
        data.to_pickle(pickle_path)
    return data


@timeit
def extract_X_y(data):
    """Extracts feature X and target y from the dataframe
    
    Parameter
    ---------
    data : Dataframe containing X and y
    
    Returns
    -------
    X : Feature variables
    y : Target variable
    """
    return data.drop(["Survived"], axis=1), data['Survived']


@timeit
def splitting(X, Y, val_split=0.2, random_state=42):
    """Splitting X and y into train-validation 
    
    Parameter
    ---------
    X : Feature
    y : Target
    val_split : 0.2
    
    Returns
    -------
    data : DataFrame containing the csv data
    """
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=val_split, random_state=random_state)
    return X_train, X_val, y_train, y_val