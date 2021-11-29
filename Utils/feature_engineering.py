import numpy as np 
import pandas as pd 
import re 
import warnings 
from typing import Iterable

# Feature Engineering functions
def create_lag(df, features, lag=[1], droplagna=True):
    """create additional lagged features

    Args:
        df (pd.DataFrame): the input data. 
        features (Iterable, str): the features in ``df.columns`` that you you want to create lags
        lag (list, optional): lags that you want to create. Defaults to [1]. For example, if lag=[1,3,9], then create feature of each features with lag 1,3, and 9. 
        droplagna (bool, optional): drop the missing records/rows due to lags. Notics that this doesn't drop all na records but only those due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: new dataframe contains lags features as columns. This doesn't contain original df content.
    """
    # settup 
    if isinstance(features, str): 
        features = [features]
    if isinstance(lag, int): 
        lag = [lag]
        
    # make the shift
    resList = []
    for i in lag: 
        res = df[features].shift(i)
        suffix = 'lag' if i > 0 else 'lead'
        i = i if i > 0 else -i
        res.columns = [str(s)+'_{}{}'.format(suffix, i)  for s in features ]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    
    if droplagna: 
        lagmax, lagmin = max(lag), min(lag)
        if lagmax > 0 and lagmin > 0: 
            res = res.iloc[lagmax:, :]
            print('#lags={} is dropped'.format(lagmax))
        elif lagmax > 0 and lagmin < 0:
            res = res.iloc[lagmax:, :]
            res = res.iloc[:lagmin, :]
            print('#lags={} and {} is dropped'.format(lagmax, lagmin))
        elif lagmax < 0 and lagmin < 0: 
            res = res.iloc[:lagmin, :]
            print('#lags={} is dropped'.format(lagmin))
        else: 
            raise ValueError('Impossible case where lagmin > lagmax')

    return res

def create_lead(df, features: Iterable, lead=[1], dropleadna=True):
    """create additional leads features. This function is eqivalent to ``create_lag(...,lag=[-l for l in lead],...)``

    Args:
        df (pd.DataFrame): the input data. 
        features (Iterable, str): the features in ``df.columns`` that you you want to create lags
        lead (list, optional): leads that you want to create. Defaults to [1]. For example, if lead=[1,3,9], then create feature of each features with lead 1,3, and 9. 
        droplagna (bool, optional): drop the missing records/rows due to lags. Notics that this doesn't drop all na records but only those due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: new dataframe contains leads features as columns. This doesn't contain original df content.
    """
    lag = -np.array(lead)
    return create_lag(df, features, lag=lag, droplagna=dropleadna)

def add_MA(df, features, window, **kwargs):

    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]
    
    resList = []
    for w in window: 
        res = df[features].rolling(w, min_periods=1, **kwargs).mean()
        res.columns = [str(s)+f'_ma{w}' for s in features]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    
    return pd.concat([df, res], axis=1), res.columns

def add_EMA(df, features, alpha, adjust=True, **kwargs):

    if isinstance(features, str):
        features = [features]
    if isinstance(alpha, int):
        alpha = [alpha]
    
    resList = []
    for a in alpha: 
        res = df[features].ewm(alpha=a, min_periods=1, 
                             adjust=adjust, **kwargs).mean()
        res.columns = [str(s)+'_ema{:g}'.format(a*10) for s in features]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return pd.concat([df, res], axis=1), res.columns

def add_MS(df, features, window, **kwargs):

    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]
        
    resList = []
    for w in window: 
        res = df[features].rolling(w, min_periods=1, **kwargs).std()
        res.columns = [str(s)+f'_ms{w}' for s in features]
        # use feature mean to fill na  (shall be the first 3 records)
        res.fillna(res.mean(), inplace=True)
        resList.append(res)
    res = pd.concat(resList, axis=1)
    
    return pd.concat([df, res], axis=1), res.columns

def add_skew(df, features, window, **kwargs): 
    """
    Args:
        df ([type]): [description]
        features ([type]): [description]
        window ([type]): should be at least 3. Otherwise returned are all NaN.

    Returns:
        [type]: [description]
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(window, int):
        window = [window]

    resList = []
    for w in window: 
        res = df[features].rolling(w, min_periods=1, **kwargs).skew()
        res.columns = [str(s)+f'_skew{w}' for s in features]
        # use feature mean to fill na (shall be the first 3 records)
        res.fillna(res.mean(), inplace=True)
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return pd.concat([df, res], axis=1), res.columns

def npshift(arr, num, fill_value=np.nan):
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

def percent_change(data):
    """
    Calculate the %change between the last value and the mean of previous values.
    Makes data more comparable if the abs values of the data change a lot.
    Usage: df.rolling(window=20).aggregate(percent_change)
    """
    previous_values = data[:-1]
    last_value = data[-1]
    return (last_value - np.mean(previous_values)) / np.mean(previous_values)


def fourier_features(index, freq, order):
    """
    `Reference`_: https://www.kaggle.com/ryanholbrook/seasonality
    Example: 
    >>> # Compute Fourier features to the 4th order (8 new features) for a
    >>> # series y with daily observations and annual seasonality:
    >>> # fourier_features(y, freq=365.25, order=4)
    """
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)


def replace_outliers(df, window=7, k=3, method='mean'):
    """
    Inputs:
        df: pd.DataFrame/Series
        window: rolling window size to decide rolling mean/std/median.
        k: multiplier of rolling std.
    Return:
        pd.DataFrame of the same shape as input df.
    """
    # de-mean
    mean_series = df.rolling(window).mean()[window-1:]
    abs_meandiff = (df[window-1:] - mean_series).abs()
    std_series = df.rolling(window).std()[window-1:]
    median_series = df.rolling(window).median()[window-1:]
    # identify >k(=3 in default) standard deviations from zero
    this_mask = abs_meandiff > (std_series * k)
    tmp = df[:window-1].astype(bool)
    tmp.values[:] = False
    this_mask = pd.concat([tmp, this_mask], axis=0)
    # Replace these values with the median accross the data
    if method == 'median':
        to_use = median_series
    elif method == 'mean':
        to_use = mean_series
    else:
        raise ValueError(f'method {method} not found.')
    return df.mask( this_mask, to_use )


def recover_from_diff(prediction_diff, y, dlist):
    """
    Transform difference-predicted series back to original
    prediction level via y.
    Only support first difference case.
    
    Parameters
    ----------
    prediction_diff: model.predict() result of training or testing data
    y: the original level of the feature that is the target of
       the prediction_diff.
       Usually, y should have same length as prediction_diff.
    dlist: list that stores 1st-difference record for each column of y.
       currently, the approach only deals with d=0,1 in dlist.
    
    Returns
    -------
    prediction: recovered prediction series with tail entry truncated
       since the final prediction is out of bound. And usually the first
       prediction correspond to y[1] instead of y[0].
    """
    import numpy as np
    
    assert (np.array(dlist) <= 1).all(), "Currently, the method doesn't\
        deal with difference case higher than 1 (in dlist)."
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
    def melt_features(frame: pd.DataFrame, id_cols=None, input_cols=None, 
                  input2dummy_value={}, dummy_name: str=None, value_name: str=None, ignore_index=False, concat_back=False): 
        """
        Unpivot a dataframe wide to long format. This function is an extension of ``pd.melt`` that provides additionally the mapper for input column names to dummy values.

        :param frame: the input dataframe.
        :type frame: pd.DataFrame
        :param id_cols: column(s) to use as identifier variables, defaults to None
        :type id_cols: [tuple, list, or ndarray], optional
        :param input_cols: [Columns to unpivot. If not specified, used all columns that are not set as ``id_vars``], defaults to None
        :type input_cols: [tuple, list, or ndarray], optional
        :param input2dummy_value: [Mapper that convert ``input_cols`` names to dummy values under the column ``dummy_name``], defaults to {}
        :type input2dummy_value: dict, optional
        :param dummy_name: [Name to use for the `variable` column. If None it uses ``frame.columns.name`` or `variable`.], defaults to None
        :type dummy_name: str, optional
        :param value_name: [Name to use for the ‘value’ column.], defaults to None
        :type value_name: str, optional
        :param ignore_index: [whether to ignore index when melting the data], defaults to False.
        :type ignore_index: bool, optional
        :param concat_back: [whether to concat the melted features back to original dataframe], defaults to False.
        :type concat_back: bool, optional
        :return: [Unpivoted DataFrame.]
        :rtype: [pd.DataFrame]
        """

        check = pd.Index(input_cols).isin(frame.columns)
        assert check.all(), f"Cannot find columns {input_cols[~check]} in frame."
        # prepare 
        value_vars = input2dummy_value.values()
        if len(value_vars) == 0: 
            value_vars = input_cols 
        # output 
        res = frame.rename(
            columns=input2dummy_value
        ).melt(
            id_vars=id_cols, value_vars=value_vars, 
            var_name=dummy_name, value_name=value_name, 
            ignore_index=ignore_index
        )

        # concat back if True 
        if concat_back: 
            res = DataMelter._concat_back(frame.drop(columns=input_cols), res, )
            
        return res 

    @staticmethod
    def melt_features_regex(frame: pd.DataFrame, features: Iterable[str], 
        dummy_value: Iterable[str]=None, dummy_name: Iterable[str]=None, 
        value_name: Iterable[str]=None, ignore_index=False):
        """
        This method is similar to ``melt_features`` but can melt different group of features via regex.

        :param frame: [the input dataframe that already contains the lead features]
        :type frame: pd.DataFrame
        :param features: [the target featues regex list for prediction task]
        :type features: Iterable[str].

        :param targets_suffix_regex: [the suffix name regex pattern for targets_stem]
        :type targets_suffix_regex: str, optional

        :param dummy_name:
        :type dummy_name: Iterable[str], defaults to None. 

        :param value_name:
        :type value_name: Iterable[str], defaults to None. 

        :param ignore_index:
        :type ignore_index: bool, defaults to False. 

        :return: [description]
        :rtype: [type]
        """
        # initial check 
        assert not isinstance(features, str), "targets_stem should be Iterable, not str."
        
        if isinstance(dummy_name, str) or dummy_name is None: 
            dummy_name = [dummy_name] * len(features)
        else:
            assert len(dummy_name) == len(features)

        if dummy_value is None: 
            dummy_value = features * len(features)
        elif isinstance(dummy_value, str):
            dummy_value = [dummy_value] * len(features)
        else: 
            assert len(dummy_value) == len(features)

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
            tomap = cols.str.extract(f"({dummy_value[i]})", expand=False)
            long_df = DataMelter.melt_features(
                frame, input_cols=cols, 
                input2dummy_value=dict(zip(cols, tomap)), 
                dummy_name=dummy_name[i], value_name=value_name[i]
            )
            collector.append(long_df)
        
        # prepare output 
        output = pd.concat(collector, axis=1)
        uniq_cols = output.columns.drop_duplicates()
        return output , uniq_cols

# %%
