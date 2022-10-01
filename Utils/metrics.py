import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
import re


class MetricsCls:
    """
    This class helps calculate different metrics. User can calculate them all at once by creating an instance and calling ``score()``.
    User can also calculate only a certain metric by directly calling, eg, ``MAE()`` method via class or instance.
    Requirement:
    1. metric functions under the class should always have the y_true, and y_pred args.
    2. y_true, y_pred should always be (n_suamples,) or (n_samples, n_outputs) shape.

    Examples:
        >>> obj=MetricsCls(config={'MAPE__version':'sklearn'})
        >>> x, y=[0.025,0.5,0.5,0], [2,0.5,0, 5]
        >>> print(obj.score(x,y))
        {'MAE': 1.8688, 'RMSE': 2.6996, 'MAPE': 5629499534213140.0, 'DS': 0.0, 'RMSSE': 6.7799}
        >>> obj=MetricsCls(config={'MAPE__version':'selfmade'})
        >>> print(obj.score(x,y))
        {'MAE': 1.8688, 'RMSE': 2.6996, 'MAPE': inf, 'DS': 0.0, 'RMSSE': 6.7799}
    """

    def __init__(self, metrics: list = [], config: dict = {}):
        """
        Args:
            metrics: list of strings of metric name which should be the same as the function name here.
            config: Eg, {'MAE__multioutput': 'uniform_average'}
        """
        # set default metrics
        if metrics == []:
            metrics = ["MAE", "RMSE", "MAPE", "DS", "RMSSE"]
        self.metrics_dict = dict(zip(metrics, [None] * len(metrics)))
        self.config_dict = config

    @property
    def metrics_dict(self):
        return self._metrics_dict

    @metrics_dict.setter
    def metrics_dict(self, value):
        self._metrics_dict = value

    @staticmethod
    def config_parser(metric: str, space: dict):
        """
        return the dict that contains only the keys with ``metric__`` pattern.
        Then this pattern will be removed.
        """
        # define pattern
        prefix = f"{metric}__\w+"
        # find the keys matching the pattern required
        keys = re.findall(prefix, "  ".join(space.keys()))
        # modify the keys by deleting the prefix
        keys_no_prefix = [k.replace(f"{metric}__", "") for k in keys]
        return dict(zip(keys_no_prefix, [space[k] for k in keys]))

    def score(self, y_true, y_pred, digit: int = 4, inplace: bool = False):
        """
        Core function.
        Run all metrics stored in the metrics_dict and return result as a dict.

        Args:
            digit (int, optional): number rounding. Defaults to 4.
            inplace (bool, optional): modify self.metrics_dict if True. Defaults to False.

        Returns:
            dict: a dictionary that contains various metric name as keys and scores as values.

        """
        assert np.array(y_true).ndim <= 2
        assert np.array(y_pred).ndim <= 2
        res_dict = self.metrics_dict.copy()
        # loop over metrics
        for k in self.metrics_dict.keys():
            # fetch specific config
            config_dict = self.config_parser(k, self.config_dict)
            # calculate the metric
            res = eval(f"self.{k}(y_true, y_pred, **config_dict)")
            res = np.round(res, digit)
            # record the result
            res_dict[k] = res
        if inplace:
            self.metrics_dict = res_dict
        else:
            return res_dict

    @staticmethod
    def MAE(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
        return sklearn.metrics.mean_absolute_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )

    @staticmethod
    def RMSE(
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        squared=False,
    ):
        """
        squared (bool, optional): If True returns MSE value, if False (default) returns RMSE value.
        """
        return sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=squared,
        )

    @staticmethod
    def DS(y_true, y_pred, threshold=0.0, multioutput="uniform_average"):
        """
        directional symmetry that additionally considers the case of plateau
        (specified by threshold > 0). All percentage changes smaller than threshold
        are classified as plateau. Usually DS with threshold>0 has lower score than
        the normal version with threshold=0 due to the additional category to be
        compared.
        """
        # check & initialization
        assert (multioutput in ["uniform_average", "raw_values"]) or (
            isinstance(multioutput, (np.array, list))
        )
        if threshold < 0:
            raise ValueError("Tolerance cannot be less than zero!")

        # define common variables
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape((-1, 1))
        true_diff = np.diff(y_true, axis=0)
        pred_diff = np.diff(y_pred, axis=0)

        # case not zero: modify true_diff and pred_diff
        if threshold != 0:
            # get %change
            tmp = pd.DataFrame(y_true).pct_change(axis=0).iloc[1:, :]
            mask = np.asarray(tmp.abs() < threshold)
            true_diff[mask] = 0

            tmp = pd.DataFrame(y_pred).pct_change(axis=0).iloc[1:, :]
            mask = np.asarray(tmp.abs() < threshold)
            pred_diff[mask] = 0

        # core formula
        d = (true_diff * pred_diff) > 0
        d[(true_diff == 0) & (pred_diff == 0)] = True

        output = np.array([np.nan] * d.shape[-1])
        if d.size != 0:
            output = 100.0 * d.sum(axis=0) / d.shape[0]

        if multioutput == "raw_values":
            return output
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
        return np.average(output, weights=multioutput)

    @staticmethod
    def MAPE(
        y_true,
        y_pred,
        sample_weight=None,
        multioutput="uniform_average",
        version="sklearn",
    ):
        version = version.lower()
        assert version in ["sklearn", "selfmade"]
        if version == "sklearn":
            return sklearn.metrics.mean_absolute_percentage_error(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )
        elif version == "selfmade":
            return MetricsCls._MAPE_selfmade(y_true, y_pred, sample_weight, multioutput)

    @staticmethod
    def _MAPE_selfmade(
        y_true, y_pred, sample_weight=None, multioutput="uniform_average"
    ):
        """doesn't deal with the zero divisor case"""
        assert (multioutput in ["uniform_average", "raw_values"]) or (
            isinstance(multioutput, (np.array, list))
        )
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        mape_array = np.abs((y_true - y_pred) / y_true)
        mape = np.average(mape_array, weights=sample_weight, axis=0)
        if multioutput == "raw_values":
            return mape
        elif multioutput == "uniform_average":
            multioutput = None
        return np.average(mape, weights=multioutput)

    @staticmethod
    def RMSSE(
        y_true,
        y_pred,
        y_train=None,
        sp: int = 1,
        squared=False,
        multioutput="uniform_average",
    ):
        """
        Root Mean Squared Scaled Error (RMSSE) with formula from M5:

        .. math::
            \sqrt{
                \frac{
                    \frac{1}{h} \Sum_{t=n+1}^{n+h} (Y_t - \hat{Y_t})^2
                }{
                    \frac{1}{n-1} \Sum_{t=2}^{n} (Y_t - Y_{t-1})^2
                }
            }

        Hence, y_true and y_pred should have same length $h$ while y_train can be
        of length $n$ or None. When y_train is None, we take y_true instead to calculate the
        denominator part. Stonger version of MSSE can be found in sktime
        `here <https://www.sktime.org/en/v0.7.0/api_reference/modules/auto_generated
        /sktime.performance_metrics.forecasting.mean_squared_scaled_error.html>`_.
        Note that there is no standard definition of RMMSE, I use the def from M5 but there is
        alternative def, eg., [Theodosiou M. 2011]_.

        .. [Theodosiou M. 2011] Theodosiou, M. (2011). Forecasting monthly and quarterly time series
        using STL decomposition. International Journal of Forecasting, 27(4), 1178–1195.
        doi:10.1016/j.ijforecast.2010.11.002

        Args:
            y_train (np.array, pd.Series, pd.DataFrame, optional): training data of target feature
                of length $n$. If not provided, i.e. defaulted to None, takes y_true as y_train so $n=h$.
            sp (int, optional): seasonality periodicity to create naive forecast based on y_train.
            squared (bool, optional): If True returns MSSE value, if False (default) returns RMSSE value.

        Reference:
            M5 competition guidlines.
            https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf

        """
        # init
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if y_train is None:
            y_train = np.asarray(y_true.copy())

        # case no record to calculate
        if len(y_train) <= sp:
            if multioutput == "raw_values":
                return np.array([np.nan] * y_true.shape[-1])
            else:  # case uniform_average or np.array[float]
                return np.nan

        # naive forecasting
        naive_forecast = y_train[:-sp]
        EPS = np.finfo(np.float64).eps
        # core
        numerator = MetricsCls.RMSE(
            y_true, y_pred, squared=True, multioutput=multioutput
        )
        denominator = MetricsCls.RMSE(
            y_train[sp:], naive_forecast, squared=True, multioutput=multioutput
        )
        output = np.sqrt(numerator / np.maximum(denominator, EPS))
        if squared:
            output = output**2
        return output


