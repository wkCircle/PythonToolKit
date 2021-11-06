from Utils.metrics import MetricsCls
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

class MovingAverage: 
    """
    Naive model that takes the mean value of the target feature from n step ago till -1 step ago (rolling window = n) as the prediction value for today. That is, $$ \hat(y_t) = frac{1}{n} * \Sum_{i=1}^{n} y_{t-i} $$

    :param horizon: how many periods the model needs to predict. 
    :type horizon: [int]
    :param look_back_period: how many periods to calculate the mean and then served as predictions. If None, which is the default value, then look_back_period will be set to equal ``horizon``. Notice that when this argument is equal to 1, the MovingAverage behavior is the same as the RandomWalk class.
    :type look_back_period: [int], optional
    """

    def __init__(self, horizon, look_back_period=None):
        self.horizon = horizon
        self.look_back_period = look_back_period
        if look_back_period is None: 
            self.look_back_period = self.horizon 
        
    
    def fit(self, df, y=None):
        """
        :param df: the input data for fitting. Actually the df here in the class should be the historic data of your target features.
        :type df: [pd.Series, pd.DataFrame]
        :param y: The ground truth of target value, defaults to None
        :type y: [pd.Series, pd.DataFrame], optional
        :return: self 
        :rtype: MovingAverage class
        """
        return self
        
    def predict(self, df):
        """
        :param df: the input data for prediction. Actually the df here in the class should be the historic data of your target features. 
        :type df: [pd.Series, pd.DataFrame]
        :return: prediction values of dataframe
        :rtype: [pd.Series, pd.DataFrame]
        """
        
        res = df.rolling(self.look_back_period, min_periods=1).mean()
        res.index = res.index.shift(self.horizon)
        return res
        
    def fit_predict(self, df):
        return self.fit(df).predict(df)
    
    def score(self, df, metrics=[], metrics_config={}):
        
        # cut off the unmatched timestamps for y_true (initial horizons) and for y_pred (last horizons)
        y_true = np.array(df)[self.horizon:]
        y_pred = self.predict(df)[:-self.horizon]
        
        # prepare output
        obj = MetricsCls(metrics=metrics, config = metrics_config)
        res = obj.score(y_true, y_pred)
        return res
    
    def plot(self, df, figsize=(15,15)): 
        """
        plot the prediction curve vs the actual curve
        :param df: the input data for prediction. Actually the df here in the class should be the historic data of your target features. 
        :type df: [pd.Series, pd.DataFrame]
        """
        df = pd.DataFrame(df)
        fig, ax = plt.subplots(len(df.columns), 1, sharex=True, figsize=figsize)
        for i, col in enumerate(df.columns): 
            ax[i].plot(self.predict(df[col]), c='b', label='predict')
            ax[i].plot(df[col], c='k', label='gt')
            ax[i].set_title(f"prediction for {col}")
            ax[i].legend()
        ax[i].tick_params(axis='x', labelbottom=True)
        plt.tight_layout()
        return ax 
