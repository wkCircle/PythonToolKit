import numpy as np 
import pandas as pd 
import re 
import warnings 

# warning class  
class DuplicatedValueWarning(UserWarning): 
    """use for category of warning type"""
    pass 


#%% Feature Engineering functions
def create_lag(df, target, lag=[1], droplagna=True):
    """create additional lagged features

    Args:
        df (pd.DataFrame): the input data. 
        target (Iterable, str): the features in ``df.columns`` that you you want to create lags
        lag (list, optional): lags that you want to create. Defaults to [1]. For example, if lag=[1,3,9], then create feature of each target with lag 1,3, and 9. 
        droplagna (bool, optional): drop the missing records/rows due to lags. Notics that this doesn't drop all na records but only those due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: new dataframe contains lags features as columns. This doesn't contain original df content.
    """
    # settup 
    if isinstance(target, str): 
        target = [target]
    if isinstance(lag, int): 
        lag = [lag]
        
    # make the shift
    resList = []
    for i in lag: 
        res = df[target].shift(i)
        suffix = 'lag' if i > 0 else 'lead'
        i = i if i > 0 else -i
        res.columns = [str(s)+'_{}{}'.format(suffix, i)  for s in target ]
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

def create_lead(df, target, lead=[1], dropleadna=True):
    """create additional leads features. This function is eqivalent to ``create_lag(...,lag=[-l for l in lead],...)``

    Args:
        df (pd.DataFrame): the input data. 
        target (Iterable, str): the features in ``df.columns`` that you you want to create lags
        lead (list, optional): leads that you want to create. Defaults to [1]. For example, if lead=[1,3,9], then create feature of each target with lead 1,3, and 9. 
        droplagna (bool, optional): drop the missing records/rows due to lags. Notics that this doesn't drop all na records but only those due to shift/lags. Defaults to True.

    Returns:
        pd.DataFrame: new dataframe contains leads features as columns. This doesn't contain original df content.
    """
    lag = -np.array(lead)
    return create_lag(df, target, lag=lag, droplagna=dropleadna)

def add_MA(df, target, window, **kwargs):

    if isinstance(target, str):
        target = [target]
    if isinstance(window, int):
        window = [window]
    
    resList = []
    for w in window: 
        res = df[target].rolling(w, min_periods=1, **kwargs).mean()
        res.columns = [str(s)+f'_ma{w}' for s in target]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    
    return pd.concat([df, res], axis=1), res.columns

def add_EMA(df, target, alpha, adjust=True, **kwargs):

    if isinstance(target, str):
        target = [target]
    if isinstance(alpha, int):
        alpha = [alpha]
    
    resList = []
    for a in alpha: 
        res = df[target].ewm(alpha=a, min_periods=1, 
                             adjust=adjust, **kwargs).mean()
        res.columns = [str(s)+'_ema{:g}'.format(a*10) for s in target]
        resList.append(res)
    res = pd.concat(resList, axis=1)
    return pd.concat([df, res], axis=1), res.columns

def add_MS(df, target, window, **kwargs):

    if isinstance(target, str):
        target = [target]
    if isinstance(window, int):
        window = [window]
        
    resList = []
    for w in window: 
        res = df[target].rolling(w, min_periods=1, **kwargs).std()
        res.columns = [str(s)+f'_ms{w}' for s in target]
        # use feature mean to fill na
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

def config_parser(key: str, space: dict): 
    """
    extract the setting from space dict based on key pattern. 
    Args:
        key (str): form the regex pattern to parse and extract from space dict.
        space (dict): the dict that will be searched. 

    Returns:
        dict: contains the setting related to key pattern, key pattern will be removed from the dict string keys.

    Example: 
        >>> space = {'RMSE__multiple_weight': 'uniform_average', 'RMSE__sample_weight': None, 
                     'MAE__multiple_weight': 'raw_values', 'MAE__sample_weight': None, }
        >>> config_parser('RMSE')
        {'multiple_weight': 'uniform_average', 'sample_weight':None}
    """
    # define pattern 
    prefix = f'{key}__\w+'
    # find the keys matching the pattern required 
    keys = re.findall(prefix, '  '.join(space.keys()))
    # modify the keys by deleting the prefix 
    keys_no_prefix = [k.replace(f'{key}__', '') for k in keys]
    return dict(zip(keys_no_prefix, [space[k] for k in keys] ))

def percent_change(data):
    """
    Calculate the %change between the last value and the mean of previous values.
    Makes data more comparable if the abs values of the data change a lot.
    Usage: df.rolling(window=20).aggregate(percent_change)
    """
    previous_values = data[:-1]
    last_value = values[-1]
    return (last_value - np.mean(previous_values)) / np.mean(previous_values)


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

