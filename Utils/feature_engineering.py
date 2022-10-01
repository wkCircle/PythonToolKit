#    Copyright 2022 Wen-Kai Chung
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import pandas as pd
from typing import Iterable, Union, List

# Feature Engineering functions
def create_lag(
    df: pd.DataFrame, 
    features: List[str], 
    lag: List[int] = [1], 
    droplagna: bool = True
) -> pd.DataFrame:
    """
    Create additional lagged features.

    .. TODO:: fix bug when lag=[0] case. Or forbid lag = [...,0,...]
    .. TODO:: add pytest for this function. 

    Args:
        df (pd.DataFrame): the input data.
        features (List[str]): the features in ``df.columns`` that you you 
            want to create lags.
        lag (List[int], optional): lags that you want to create. Defaults 
            to [1]. Eg, if lag=[1,3,9], then create feature of each features with lag 1,3, and 9.
        droplagna (bool, optional): drop the missing records/rows due to 
            lags. Notics that this doesn't drop all na records but only 
            those rows due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: New dataframe contains lags features as columns. This doesn't contain original df content.
    """
    # settup
    if isinstance(features, str):
        features = [features]
    if isinstance(lag, int):
        lag = [lag]
    # small helper 
    def get_suffix(lag: int): 
        if lag > 0: 
            return "lag"
        elif lag == 0: 
            return "" 
        elif lag < 0: 
            return "lead"
        else:
            raise ValueError(f"Weired value for lag={lag} received.")

    # make the shift
    resList = []
    for i in lag:
        res = df[features].shift(i)
        suffix = get_suffix(i)
        i = abs(i)
        res.columns = [str(s) + "_{}{}".format(suffix, i) for s in features]
        resList.append(res)
    res = pd.concat(resList, axis=1)

    if droplagna:
        lagmax, lagmin = max(lag), min(lag)
        if lagmax > 0 and lagmin > 0:
            res = res.iloc[lagmax:, :]
            # print('#lags={} is dropped'.format(lagmax))
        elif lagmax > 0 and lagmin < 0:
            res = res.iloc[lagmax:, :]
            res = res.iloc[:lagmin, :]
            # print('#lags={} and {} is dropped'.format(lagmax, lagmin))
        elif lagmax < 0 and lagmin < 0:
            res = res.iloc[:lagmin, :]
            # print('#lags={} is dropped'.format(lagmin))
        else:
            raise ValueError("Impossible case where lagmin > lagmax")
    return res


def create_lead(
    df: pd.DataFrame, 
    features: List[str], 
    lead: List[int] = [1], 
    dropleadna: bool = True
) -> pd.DataFrame:
    """
    Create additional leads features. This function is eqivalent to 
    ``create_lag(...,lag=[-x for x in lead],...)``

    Args:
        df (pd.DataFrame): the input data.
        features (Iterable, str): the features in ``df.columns`` that you 
            want to create leads.
        lead (List[int], optional): leads that you want to create. Defaults 
            to [1]. Eg, if lead=[1,3,9], then create feature of each 
            features with lead 1,3, and 9.
        droplagna (bool, optional): drop the missing records/rows due to lags. 
            Notics that this doesn't drop all na records but only those due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: New dataframe contains leads features as columns, which 
        doesn't contain original df content.
    """
    lag = -np.array(lead)
    return create_lag(df, features, lag=lag, droplagna=dropleadna)


def create_MA(
    df: pd.DataFrame, 
    features: List[str], 
    window: List[int], 
    **kwargs
) -> pd.DataFrame:
    """
    Create moving average featuers via ``w`` historical values where 
    ``w`` is each element of ``window``.

    Args:
        df (pd.DataFrame): intput dataframe. 
        features (List[str]): column names of ``df`` to generate new features.
        window (List[int]): lookback period to calculate the rolling mean.
        **kwargs (dict): any additional key-value pair seeting to 
            ``pd.DataFrame.rolling(w, min_periods=1, ...)`` function.

    Returns:
        pd.DataFrame: New dataframe collecting all generated features as 
        columns, which doesn't include features of original dataframe.
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]

    resList = []
    for w in window:
        res = df[features].rolling(w, min_periods=1, **kwargs).mean()
        res.columns = [str(s) + f"_ma{w}" for s in features]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return res


def create_EMA(
    df: pd.DataFrame, 
    features: List[str], 
    alpha: List[float], 
    adjust: bool = True, 
    **kwargs
) -> pd.DataFrame:
    """
    Create exponetial weighted moving average featuers via ``a`` 
    smoothing factor on historical values where ``a`` is each element 
    of ``alpha``.

    Args:
        df (pd.DataFrame): intput dataframe. 
        
        features (List[str]): column names of ``df`` to generate new features.
        
        alpha (List[float], optional): list of smoothing factors. Please refer 
            to `pd.DataFrame.ewm() <https://pandas.pydata.org/docs/reference/
            api/pandas.DataFrame.ewm.html>`_.
        
        **kwargs (dict): any additional key-value pair setting to 
            ``pd.DataFrame.ewm(alpha=a, min_periods=1, adjust=adjust, ...)`` 
            function.

    Returns:
        pd.DataFrame: New dataframe collecting all generated features as 
        columns, which doesn't include features of original dataframe.
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(alpha, int):
        alpha = [alpha]

    resList = []
    for a in alpha:
        res = df[features].ewm(alpha=a, min_periods=1, adjust=adjust, **kwargs).mean()
        res.columns = [str(s) + "_ema{:g}".format(a * 10) for s in features]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return res


def create_MS(
    df: pd.DataFrame, 
    features: List[str], 
    window: List[int], 
    **kwargs
) -> pd.DataFrame:
    """
    Create moving standard deviation featuers via ``w`` historical values 
    where ``w`` is each element of ``window``.

    Args:
        df (pd.DataFrame): intput dataframe. 
        features (List[str]): column names of ``df`` to generate new features.
        window (List[int]): lookback period to calculate the rolling std.
        **kwargs (dict): any additional key-value pair setting to 
            ``pd.DataFrame.rolling(w, min_periods=1, ...)`` function.

    Returns:
        pd.DataFrame: New dataframe collecting all generated features as 
        columns, which doesn't include features of original dataframe.
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]

    resList = []
    for w in window:
        res = df[features].rolling(w, min_periods=1, **kwargs).std()
        res.columns = [str(s) + f"_ms{w}" for s in features]
        # use feature mean to fill na  (shall be the first 3 records)
        res.fillna(res.mean(), inplace=True)
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return res


def create_skew(
    df: pd.DataFrame, 
    features: List[str], 
    window: List[int], 
    **kwargs
) -> pd.DataFrame: 
    """
    Create moving skew featuers via ``w`` historical values 
    where ``w`` is each element of ``window``.

    Args:
        df (pd.DataFrame): intput dataframe. 
        features (List[str]): column names of ``df`` to generate new features.
        window (List[int]): lookback period to calculate the rolling skew.
        **kwargs (dict): any additional key-value pair setting to 
            ``pd.DataFrame.rolling(w, min_periods=1, ...)`` function.

    Returns:
        pd.DataFrame: New dataframe collecting all generated features as 
        columns, which doesn't include features of original dataframe.
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]

    resList = []
    for w in window:
        res = df[features].rolling(w, min_periods=1, **kwargs).skew()
        res.columns = [str(s) + f"_skew{w}" for s in features]
        # use feature mean to fill na (shall be the first 3 records)
        res.fillna(res.mean(), inplace=True)
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return res


def create_fourier(
    series: pd.Series, 
    freq: float, 
    orders: List[int]
):
    """
    Generate fourier features from input series.

    Code idea comes from `kaggle seasonality tutorial <https://
    www.kaggle.com/ryanholbrook/seasonality>`_. 
    
    Args:
        series (pd.Series): target series to generate fourier features
        
        freq (int or float): frequency of the series which will be 
            regarded as denominator to parition the 2\pi periods, ie, 
            freq is the number of periods required to form one complete
            sin/cos cycle. 
        
        orders (List[int]): list of fourier order to create features. 
            Although not forbidden, it doesn't make sense to put order
            values larger than ``freq`` since sin/cos will repeat. 

    Returns:
        pd.DataFrame: New dataframe containing 2*len(orders) columns of
        various fourier features and same index as the input series.
    """
    name = series.name
    k = 2 * np.pi * series / freq
    features = {}
    for order in orders:
        features.update({
            f"{name}_ffsin_{freq}_{order}": np.sin(order * k),
            f"{name}_ffcos_{freq}_{order}": np.cos(order * k),
        })
    return pd.DataFrame(features, index=series.index)


def npshift(arr, num, fill_value=np.nan):
    """
    Helper function to do shift for ``np.array`` object like 
    ``pd.DataFrame.shift`` but supports multi-diemensions. 

    .. note: this is different from ``np.shift`` since our customized
       function fill values while ``np.shift`` return a matrix of smaller 
       size.

    .. TODO: check my own repo to recover this function to support multi-d.

    Args:
        arr (np.ndarray): Input array. 
        num (int): How much shift to be taken. 
        fill_value (any, optional): values to be filled after shift. 
            Defaults to np.nan.

    Returns:
        np.array: New array that is shifted and filled.
    """
    
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def percent_change_from_mean(data):
    """
    Calculate the %change between the last value and the mean of 
    previous values.
    
    Makes data more comparable if data absoulte scale changes a lot.
    Usage: df.rolling(window=20).aggregate(percent_change)
    """
    previous_values = data[:-1]
    last_value = data[-1]
    return (last_value - np.mean(previous_values)) / np.mean(previous_values)


def replace_outliers(df, window=7, k=3, method="mean"):
    """
    Args:
        df: pd.DataFrame/Series
        window (int, optional): rolling window size to decide rolling 
            mean/std/median.
        k (int, optional): multiplier of rolling std.
        method (str, optional): strategy to replace outliers. 
            Currently I only implment ['mean', 'median'] methods.

    Returns:
        pd.DataFrame of the same shape as input df.
    """
    # de-mean
    mean_series = df.rolling(window).mean()[window - 1 :]
    abs_meandiff = (df[window - 1 :] - mean_series).abs()
    std_series = df.rolling(window).std()[window - 1 :]
    median_series = df.rolling(window).median()[window - 1 :]
    # identify >k(=3 in default) standard deviations from zero
    this_mask = abs_meandiff > (std_series * k)
    tmp = df[: window - 1].astype(bool)
    tmp.values[:] = False
    this_mask = pd.concat([tmp, this_mask], axis=0)
    # Replace these values with the median accross the data
    if method == "median":
        to_use = median_series
    elif method == "mean":
        to_use = mean_series
    else:
        raise ValueError(f"method {method} not found.")
    return df.mask(this_mask, to_use)


def recover_from_diff(prediction_diff, y, dlist):
    """
    Transform difference-predicted series back to original
    prediction level via y. Only support first difference case.

    Args:
        prediction_diff: model.predict() result of training or testing data
        y: the original level of the feature that is the target of
        the prediction_diff.
        Usually, y should have same length as prediction_diff.
        dlist: list that stores 1st-difference record for each column of y.
        currently, the approach only deals with d=0,1 in dlist.

    Returns:
    prediction: recovered prediction series with tail entry truncated
       since the final prediction is out of bound. And usually the first
       prediction correspond to y[1] instead of y[0].
    """
    assert (np.array(dlist) <= 1).all(), (
        "Currently, the method doesn't deal with difference case"
        + " higher than 1 (in dlist)."
    )
    yarray = np.array(y, ndmin=1)
    parray = np.array(prediction_diff, ndmin=1)
    assert len(dlist) == yarray.shape[-1]

    res = parray.copy()
    for i in range(len(dlist)):
        # pass if no first difference on the current column
        if dlist[i] == 0:
            continue
        # concept: y_hat[i+1] = y[i] + delta_hat(y[i+1]-y[i])
        res[:, i] = yarray[:, i] + parray[:, i]
    return res.reshape(y.shape)

# # need testing this function, the advanced version of the original.
# def recover_from_diff(prediction_diff, y, dlist, step=1):
#      """
#     Transform difference-predicted series back to original
#     prediction level via y.
#     TODO: this function only supports recovery where the prediction is based on ground truth, ie,
#     the function doesn't support the case where the prediction is generated through the exogenous
#     feature prediction procedure
    
#     Parameters
#     ----------
#     prediction_diff: model.predict() result of training or testing data
#     y:  the original level of the feature that is the target of
#         the prediction_diff.
#         Usually, y should have same length as prediction_diff.
#     dlist: list - stores 1st-difference record for each column of y.
#         currently, the approach only deals with d=0,1 in dlist.
#     step: int - how many step ahead is the prediction compared to y.
    
#     Eg, if pred is 2-step-ahead on 1st-diff level, then step=2, dlist=[1]
#     Eg, if pred is 1-step-ahead on 2st-diff level, then step=1, dlist=[2]
    
#     Returns
#     -------
#     prediction: recovered prediction series with tail entry truncated
#        since the final prediction is out of bound. And usually the first
#        prediction correspond to y[1] instead of y[0].
#     """
#     import numpy as np
    
#     yarray = np.array(y, ndmin=1)
#     parray = np.array(prediction_diff, ndmin=1)
#     assert len(dlist) == yarray.shape[-1]
    
#     res = parray.copy()
#     # loop over each column
#     for i in range(len(dlist)):
#         # continue if no diff() at all on the current column
#         if dlist[i] == 0:
#             continue
#         # loop for recovering prediction back to 0th-hierarchy level
#         for hierarchy in range(dlist[i], 0, -1):
#             # get corresponding y_true series for 1 step diff() back transformation
#             helper = np.diff(yarray[:, i], n=hierarchy-1)
#             # locate the position: not sure and need double check
#             gap = helper.shape[0] - res.shape[0]
#             res[:, i] = helper[-res.shape[0]-gap-step:-gap-step] + res[:, i]
            
#     return res.reshape(y.shape)

#%% DataMelter 
class DataMelter(): 
    
    def __init__(self): 
        pass 
    
    @staticmethod 
    def _concat_back(frame: pd.DataFrame, frame2: pd.DataFrame, ):
        assert frame2.shape[0] % frame.shape[0] == 0, "frame is not divisible by frame2 wrt nrows."
        multiplier = int(frame2.shape[0] / frame.shape[0])
        augmented_frame = pd.concat([frame]*multiplier, axis=0)
        assert (augmented_frame.index == frame2.index).all(), "index of the two frames are not the same."
        return pd.concat([augmented_frame, frame2], axis=1)
    
    @staticmethod
    def melt_features(frame: pd.DataFrame, id_vars=None, inputcols=None, 
                  rename_inputcols: dict={}, var_name: str=None, value_name: str=None, ignore_index=False, concat_back=False): 
        """
        Unpivot a dataframe wide to long format. This function is an extension of ``pd.melt`` that provides additionally the mapper for input column names to dummy values.

        Args:
            frame (pd.DataFrame): target dataframe with some columns to be melted.
            id_vars (Iterable[str], optional): column(s) to use as identifier variables. Defaults to None.
            inputcols (Iterable[str], optional): Columns to melt. If not specified, used all columns that are not set as ``id_vars``. Defaults to None.
            rename_inputcols (dict, optional): Mapper that converts ``input_cols`` names to dummy values under the column ``var_name``. Defaults to {}.
            var_name (str, optional): Name for the melted variable column. If None, it uses ``frame.columns.name`` or `variable`.. Defaults to None.
            value_name (str, optional): Name for the melted value column.. Defaults to None.
            ignore_index (bool, optional): whether to ignore index when melting the data. Defaults to False.
            concat_back (bool, optional): whether to concat the melted features back to original dataframe. Defaults to False.

        Returns:
            pd.DataFrame: the melted dataframe
        """

        check = pd.Index(inputcols).isin(frame.columns)
        assert check.all(), f"Cannot find columns {inputcols[~check]} in frame."
        # prepare 
        value_vars = rename_inputcols.values()
        if len(value_vars) == 0: 
            value_vars = inputcols 
        # output 
        res = frame.rename(
            columns=rename_inputcols
        ).melt(
            id_vars=id_vars, value_vars=value_vars, 
            var_name=var_name, value_name=value_name, 
            ignore_index=ignore_index
        )

        # concat back if True 
        if concat_back: 
            res = DataMelter._concat_back(frame.drop(columns=inputcols), res, )
            
        return res 

    @staticmethod
    def melt_features_regex(frame: pd.DataFrame, features: Iterable[str], 
        rename_features: Iterable[str]=None, var_name: Iterable[str]=None, 
        value_name: Iterable[str]=None, ignore_index=False):
        """
        This method is similar to ``melt_features`` but can melt different group of features via regex.

        Args:
            frame (pd.DataFrame): the input dataframe that already contains the lead features.
            features (Iterable[str]): the target featues regex list for prediction task.
            dummy_value (Iterable[str], optional): . Defaults to None.
            var_name (Iterable[str], optional): Name for the melted variable column which serves as a dummy variable. Defaults to None.
            value_name (Iterable[str], optional): Name for melted value column. Defaults to None.
            ignore_index (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: the melted dataframe.
        """
        # initial check 
        assert not isinstance(features, str), "targets_stem should be Iterable, not str."
        
        if isinstance(var_name, str) or var_name is None: 
            var_name = [var_name] * len(features)
        else:
            assert len(var_name) == len(features)

        if rename_features is None: 
            rename_features = features * len(features)
        elif isinstance(rename_features, str):
            rename_features = [rename_features] * len(features)
        else: 
            assert len(rename_features) == len(features)

        if isinstance(value_name, str): 
            value_name = [value_name] * len(features)
        elif value_name is None: 
            value_name = features
        else:
            assert len(value_name) == len(features)
        

        # melt targets down to long format
        collector = []
        for i, feature_pattern in enumerate(features): 
            cols = frame.filter(regex=feature_pattern).columns
            tomap = cols.str.extract(f"({rename_features[i]})", expand=False)
            long_df = DataMelter.melt_features(
                frame, inputcols=cols, 
                rename_inputcols=dict(zip(cols, tomap)), 
                var_name=var_name[i], value_name=value_name[i], 
                ignore_index=ignore_index
            )
            collector.append(long_df)
        
        # prepare output 
        output = pd.concat(collector, axis=1)
        uniq_cols = output.columns.drop_duplicates()
        return output , uniq_cols
