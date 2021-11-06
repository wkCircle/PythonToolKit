import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial


# self-defined modules
from Utils.misc import config_parser
# ref: https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization

class Tree: 

    def __init__(self, horizon, model_config={}, seed=42, tscv_splits=5):
        self.horizon = horizon 
        self.model_config = model_config
        self.seed = seed
        self.space = {
            'est__n_estimators': ho_scope.int(hp.quniform('n_estimators', low = 10, high = 100, q =10)),
            'est__max_depth': ho_scope.int(hp.quniform('max_depth', low = 5, high = 20, q = 1))
        }
        self.tscv = TimeSeriesSplit(n_splits=tscv_splits)

    @staticmethod
    def naivemodel(hps):
        sel_hps = config_parser('sel', hps)
        est_hps = config_parser('est', hps)

        model = Pipeline([
            ('scal', StandardScaler()),
            ('sel', RFE(**sel_hps)),
            ('est', RandomForestRegressor(**est_hps))
        ])
        return model

    @staticmethod
    def objective(hps, X, y, cv):
        model = Tree.naivemodel(hps)
        cv_res = cross_val_score(model, X, y, cv=cv,
                                 scoring='neg_mean_squared_error', n_jobs=-1)
        return -cv_res.mean()

    def fit(self, X, y, **kwargs):
        best = fmin(
            fn=partial(Tree.objective, X=X, y=y.squeeze(), cv=self.tscv),
            space=self.space,
            algo=tpe.suggest,
            max_evals=100,
            rstate=np.random.RandomState(42))

        self.final_model_fit = Tree.naivemodel(best).fit(X,y.squeeze())
        return self.final_model_fit

    def predict(self, X):
        #Need to assert model has already been fitted
        self.prediction = self.final_model_fit.predict(X)
        return self.prediction

    def __len__(self):
        """return the length of ``self.model``"""
        if not hasattr(self, 'model'): 
            raise AttributeError("attribute `model` has not been built. Please use the fit function firstly.")
        return len(self.model)
    
    @property
    def model(self): 
        if not hasattr(self, '_model'):
            raise AttributeError("attribute `model` has not been built. Please use the fit function firstly.")

        return self._model
    @model.setter
    def model(self, model):
        assert isinstance(model, dict)
        self._model = model
        
    @property
    def horizon(self): 
        return self._horizon
    
    @horizon.setter
    def horizon(self, horizon):
        assert horizon >= 1
        self._horizon = horizon
