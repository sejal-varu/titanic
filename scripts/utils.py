import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pprint import pprint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
import logging
import time

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


def create_grid(classifier, param_grid):
    """ Creates a GridSearchCV grid to tune hyperparameters
        
        Parameters
        ----------
        classifier : model
        param_grid : dictionary containing different hyperparameters
        
        Returns
        ----------
        trans : pandas DataFrame
        
    """
    pipe = make_pipeline(Transformer(), StandardScaler(), classifier(random_state=42))
    return GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid,
                        scoring="roc_auc")


def pickle_save_model(model, name):
    """ Dumps machine learning model into a pickle file
        
        Parameters
        ----------
        model : machine learning model
        name  : model name to save as
        
    """
    with open('../models/' + name + '.pkl', 'wb') as md:
        pickle.dump(model, md)


def pickle_load_model(name):
    """ Loads machine learning model from a pickle file
        
        Parameters
        ----------
        name  : model name to load
        
        Returns
        ----------
        model : machine learning model
        
    """
    with open('../models/' + name + '.pkl', 'rb') as md:
        model = pickle.load(md)
    return model


@timeit
def train(X, y, classifier, param_grid, name):
    """ Train a machine learning model using GridSearchCV function and stores it 
        in a pickle file
        
        Parameters
        ----------
        X          : Feature variable
        y          : Target variable
        classifier : machine learning model 
        param_grid : dictionary of hyperparameters
        name       : name of the model to save as
        
    """
    grid = create_grid(classifier, param_grid)
    grid.fit(X, y)
    model = grid.best_estimator_
    # saving score while training
    roc_auc = grid.best_score_
    with open('../results/' + name + 'roc_auc_train.txt', 'w') as file:
        file.write(str(roc_auc))
    # saving model after training
    pickle_save_model(model, name)


@timeit
def test(name, X, y):
    """ Evaluate the performance metrics for test/validation data and store it in a
        pickle file
        
        Parameters
        ----------
        X          : Feature variable
        y          : Target variable
        name       : pickle model name to load from
        
    """
    model = pickle_load_model(name)
    y_pred = model.predict_proba(X)[:,1]
    roc = roc_auc_score(y, y_pred )
    conf_mat = confusion_matrix(y, model.predict(X))
    print(roc)
    with open('../results/'+ name + 'roc_auc_test.txt', 'w') as file:
        file.write(str(roc))
    with open('../results/'+ name + 'confusion_matrix_test.txt', 'w') as file:
        file.write(str(conf_mat))


@timeit
def predict(name, X):
    """ Loads a pickle model file and calculates label predictions on test/val data
        
        Parameters
        ----------
        X          : Feature variable
        name       : pickle model name to load

        Returns
        ----------
        y_pred : model class predictions
        
    """
    model = pickle_load_model(name)
    print(model)
    y_pred = model.predict(X)
    return y_pred

@timeit
def predict_probabilities(name, X):
    """ Loads a pickle model file and calculates probabilities predictions on test/val 
        data
        
        Parameters
        ----------
        X          : Feature variable
        name       : pickle model name to load

        Returns
        ----------
        y_pred : model class predictions containing probability values for both 0 and 1 class
        
    """
    model = pickle_load_model(name)
    y_pred = model.predict_proba(X)
    return y_pred


@timeit
def create_submission_csv(SK_ID_CURR, y_pred, path):
    submission_data_frame = pd.concat([SK_ID_CURR, pd.Series(y_pred, name='TARGET')], axis=1)
    submission_data_frame.to_csv(path, index=False)