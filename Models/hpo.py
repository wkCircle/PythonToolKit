from typing import Callable
import numpy as np 
import pandas as pd 
import functools as ft 

import hyperopt
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SequentialFeatureSelector as SFS, SelectFromModel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# self module
from Utils.misc import config_parser

class HPO:
    def __init__(self, space):
        self.space = space
    
    # TODO: should remove objective out of HPO class 
    # and modify get_model: it is bad to instantiate models in get_model
    # since we have to import a lot of model packages in the file. 
    # better way: instantiate models already in main.py and here get_model
    # should only modfiy hyperparameters of the already constructed estimator.
    @staticmethod
    def get_model(hps, estimator, multiout_wrapper=True):
        # parse dict 
        sel_hps = config_parser('sel', hps)
        est_hps = config_parser('est', hps)

        # parse selector 
        sel_estimator = sel_hps["estimator"]["type"]
        sel_estimator = eval(f"{sel_estimator}")
        del sel_hps["estimator"]
        
        # turn on/off MultiOutputRegressor 
        final_est = estimator(**est_hps)
        if multiout_wrapper: 
            final_est = MultiOutputRegressor(final_est)

        # biuld pipeline
        model = Pipeline([ 
            ('scal', StandardScaler()), 
            ('sel', RFE(sel_estimator, **sel_hps)), 
            ('est', final_est)
        ])

        return model 

    @staticmethod 
    def objective(hps, estimator: Callable, X, y, cv, scoring, multiout_wrapper=True):
        model = HPO.get_model(hps, estimator, multiout_wrapper)
        cv_res = cross_val_score(model, X, y, cv=cv, 
                                scoring=scoring, n_jobs=-1)
        return -cv_res.mean() # either min(), or mean()... #TODO: LSTM, wrap into class of all funcs here.
    
    def fmin(self, estimator, X, y, cv=None, multiout_wrapper=True, 
             scoring='neg_mean_squared_error', 
             max_evals=200, trials=None, seed=42):

        if cv is None: 
            cv = TimeSeriesSplit(n_splits=5)

        best_pt = hyperopt.fmin(
            fn=ft.partial(
                HPO.objective, estimator=estimator, 
                X=X, y=y, cv=cv, scoring=scoring, multiout_wrapper=multiout_wrapper
            ), 
            space=self.space, 
            algo=hyperopt.tpe.suggest, 
            max_evals=max_evals, 
            trials=trials, 
            rstate=np.random.RandomState(seed)
        )
        best_hyperparam = hyperopt.space_eval(best_pt, self.space)

        # note: trilas is inplacely changed if not None
        return best_hyperparam
