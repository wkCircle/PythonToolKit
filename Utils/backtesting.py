# imports 
import numpy as np
import pandas as pd
import sys
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial

# self-defined modules
# sys.path.append('C:\\Users\\SchnM59\\PycharmProjects\\forecasty-lab\\Methanol')
from .misc import config_parser
from .metrics import MetricsCls

class BackTesting(): 
    """
    The class aims at generating backtesting prediction lines and incorporates metric overall metric performance by comparing to the ground truth. 
    """
    def __init__(self, init_period=.2, stride=1, horizon=1):
        self.init_period = init_period
        self.stride = stride
        self.horizon = horizon
    
    def backtesting_rolling_predict(self, model, X, y, model_config={}, record_to_pred='one_to_one'):
        """
        :param model:
        :param X:
        :param y:
        :param model_config:
        :param record_to_pred:
        :return:
        """
        # check
        if hasattr(model, 'horizon'):
            flag = model.horizon == self.horizon
            if not flag: warnings.warn('...')

        BackTesting._validate_model(model)
        assert record_to_pred in ['one_to_one', 'one_to_many'], 'One X-row needs to be associated with either predicting one or many y rows'
        assert len(X) == len(y)
        # initialization
        init_period = self.init_period
        stride = self.stride
        horizon = self.horizon
        N = len(X)
        assert init_period <= N - horizon, "arg 'init period' should be smaller than or equal to the N - horizon."
        fit_kwargs = config_parser(key='fit', space=model_config)
        predict_kwargs = config_parser(key='predict', space=model_config)

        # core algorithm: always use expanding window strategy to repeat the train and predict
        output, true_values = [], []
        init_period = int(init_period * N)
        for i in range(init_period, N-horizon, stride):
            model.fit(X[:i], y[:i], **fit_kwargs)
            if record_to_pred == 'one_to_many':
                output.append(
                    model.predict(X[:i], **predict_kwargs),
                true_values.append(y[:i]))                      #Need to check if this makes sense
            if record_to_pred == 'one_to_one':
                output.append(
                    model.predict(X[i:i+horizon], **predict_kwargs)),
                true_values.append(y[i:i+horizon])
        self.output = np.array(output)
        self.true_values = np.array(true_values)

    def calculate_performance(self, output_as_dataframe=False):
        assert len(self.output) == len(self.true_values), 'Number of predictions made must equal number of true y-values collected'
        assert len(self.output[0]) == len(self.true_values[0]), 'Forecasting horizon must equal length of output value'

        #Collect series of different forecasting horizons
        forecasts, actuals = {},{}
        horizons = len(self.output[0])
        for y in range(0, horizons):
            series_pred, series_actual = [], []
            for i in range(0, len(self.output)):
                series_pred.append(self.output[i][y])
                series_actual.append(self.true_values[i][y])
            forecasts[f"FC-Horizon= {y+1}"] = series_pred
            actuals[f"FC-Horizon= {y+1}"] = series_actual

        #Calculate the performance metrics
        results = {}
        evaluator = MetricsCls()
        for y in forecasts.keys():
            results[y] = evaluator.score(forecasts[y], actuals[y])
        if output_as_dataframe:
            results = pd.DataFrame(results)
        return results

    @staticmethod
    def _validate_model(model): 
        assert hasattr(model, 'fit'), "model doesn\'t have fit method."
        assert hasattr(model, 'predict'), "model doesn\'t have predict method."