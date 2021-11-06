import matplotlib.pyplot as plt 
from Utils.metrics import MetricsCls
import numpy as np 
import pandas as pd 

class RandomWalk: 
    """
    Naive model that just uses the value of the target feature n (=horizon) step ago as the prediction value for today. 
    """
    def __init__(self, horizon):
        self.horizon = horizon
    
    def fit(self, df, y=None):
        """
        fit function that trains the model. Since RandomWalk is a simple algorithm, we actually won't do any weights tuning here, ie, the arguments ``df`` and ``y`` won't be used but just follow the convention of coding style that sklearn has.

        Args:
            df ([pd.Series, pd.DataFrame]): the input data for fitting. Actually the df here in the class should be the historic data of your target features.
            y ([pd.Series, pd.DataFrame], optional): The ground truth of target value. Defaults to None.

        Returns:
            object: self
        """
        return self
        
    def predict(self, df):
        """
        Args:
            df ([pd.Series, pd.DataFrame]): the input data for prediction. Actually the df here in the class should be the historic data of your target features. 
        Returns:
            prediction values of dataframe
        """
        res = df.copy()
        res.index = res.index.shift(self.horizon)
        return res
        
    def fit_predict(self, df):
        return self.fit(df).predict(df)
    
    def score(self, df, metrics=[], metrics_config={}):
        
        y_true = np.array(df)[self.horizon:]
        y_pred = self.predict(df)[:-self.horizon]
        
        # prepare output
        obj = MetricsCls(metrics=metrics, config = metrics_config)
        res = obj.score(y_true, y_pred)
        return res
    
    def plot(self, df, figsize=(15, 15)): 
        
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


