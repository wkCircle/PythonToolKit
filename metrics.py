import sklearn 
import numpy as np 
import pandas as pd 
import re 

class MetricsCls:
    """
    Requirement: metric functions under the class should always have the y_true, and y_pred args.
    """
    
    def __init__(self, metrics=[], config={}):
        """
        metrics: list of strings of metric name which should be the same as the function name here.
        config: Eg, {'MAE__multioutput': 'uniform_average'}
        """
        if metrics == []: 
            metrics = ['MAE', 'RMSE', 'MAPE', 'DS']
        self.metrics_dict = dict(zip(metrics, [None]*len(metrics)))
        self.config_dict = config
    
    @property
    def metrics_dict(self):
        return self._metrics_dict
    
    @metrics_dict.setter
    def metrics_dict(self, value):
        """
        run all metrics stored in the metrics_dict
        """
        self._metrics_dict = value 
    
    def config_parser(self, metric): 
        """return the dict that contains only the keys with `metric`__ pattern. """
        # define pattern 
        prefix = f'{metric}__\w+'
        # find the keys matching the pattern required 
        keys = re.findall(prefix, '  '.join(self.config_dict.keys()))
        # modify the keys by deleting the prefix 
        keys_no_prefix = [k.replace(f'{metric}__', '') for k in keys]
        return dict(zip(keys_no_prefix, [self.config_dict[k] for k in keys] ))
    
    def score(self, y_true, y_pred, digit=4, inplace=False): 
        """ core function.
        run all metrics stored in the metrics_dict and return result as a dict.
        digit: number rounding
        inplace: bool - modify self.metrics_dict if True
        """
        res_dict = self.metrics_dict.copy()
        # loop over metrics
        for k in self.metrics_dict.keys(): 
            # fetch specific config
            config_dict = self.config_parser(k)
            # calculate the metric 
            res = eval(f'self.{k}(y_true, y_pred, **config_dict)')
            res = np.round(res, digit)
            # record the result
            res_dict[k] = res
        if inplace: 
            self.metrics_dict = res_dict
        else: 
            return res_dict
    
    @staticmethod
    def MAE(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average'):
        return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
        
    @staticmethod
    def RMSE(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True):
        return sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    
    @staticmethod
    def DS(y_true, y_pred, version='selfmade', **kwargs):
        version = version.lower()
        assert version in ['selfmade', 'bd']
        y_true = np.array(y_true).squeeze()
        y_pred = np.array(y_pred).squeeze()
        
        if version =='bd': 
            return MetricsCls._BD_DS(y_true, y_pred, **kwargs)
        
        assert version == 'selfmade'
        # matrices case is also valid
        d = np.diff(y_true, axis=0) * np.diff(y_pred, axis=0) > 0
        return 100.0 * d.sum() / len(d.reshape(-1))
    
    @staticmethod 
    def _BD_DS(y_true, y_pred, tolerance=0.01):
        """Business Desk version DS (version=='bd')."""
        # check 
        if tolerance < 0:
            raise ValueError('Tolerance cannot be less than zero!')
        # define common variables
        y_true = np.array(y_true).squeeze()
        y_pred = np.array(y_pred).squeeze()
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        # case not zero: modify true_diff and pred_diff
        if tolerance != 0:
            # get %change 
            tmp = pd.Series(y_true).pct_change().iloc[1:]
            true_diff[tmp.abs() < tolerance] = 0
            
            tmp = pd.Series(y_pred).pct_change().iloc[1:]
            pred_diff[tmp.abs() < tolerance] = 0

        # core formula
        d = (true_diff * pred_diff) > 0
        ## case of plateau for both y_true, y_pred
        d[(true_diff== 0) & (pred_diff == 0)] = 1
        dsymm = np.round(100 * d.sum() / len(d), 2)

        return dsymm
    
    @staticmethod
    def MAPE(y_true, y_pred, sample_weight=None, multioutput='uniform_average', version='sklearn'):
        version = version.lower()
        assert version in ['sklearn', 'selfmade']
        if version == 'sklearn':
            return sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred, sample_weight, multioutput)
        elif version == 'selfmade': 
            return MetricsCls._selfmade_MAPE(y_true, y_pred, sample_weight, multioutput)
        
    @staticmethod 
    def _selfmade_MAPE(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
        """doesn't deal with the zero divisor case"""
        assert multioutput in ['raw_values', 'uniform_average']
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape_array = np.abs( (y_true - y_pred)/y_true )
        mape = np.average(mape_array, weights=sample_weight, axis=0)
        if multioutput == 'raw_values': 
            return mape 
        elif multioutput == 'uniform_average': 
            return np.average(mape, weights=None)


# Testing the Class 
# obj=MetricsCls(config={'MAPE__version':'sklearn'})
# x, y=[0.025,0.5,0.5,0], [2,0.5,0, 5]
# print(obj.score(x,y))
# obj=MetricsCls(config={'MAPE__version':'selfmade'})
# print(obj.score(x,y))