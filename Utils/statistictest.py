from typing import Any
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import seaborn as sns 
import numpy as np
import pandas as pd 


#%% testing functions
def adftest(series, maxlag: int=None, regression: str='c', autolag: str='AIC', verbose: bool=True, **kwargs):
    """
    ADFuller test with verbosity ability.
    
    Args:
        series (pd.Series, np.array): the time series of 1-D and of float type with NO missing data. 
        maxlag (int, optional): Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} is used when None.
        regression (str, optional): must be in ['c', 'ct', 'ctt', 'n']. Defaults to 'c'. 
            constant and trend order to include in regression. 
            * 'c': constant only (default).
            * 'ct': constant and trend. 
            * 'ctt': constant, linear, and quadratic trend. 
            * 'n': no constant, no trend. 
        autolag (str, optional): must be in ['AIC', 'BIC', 't-stat', None]
            Method to use when automatically determining the lag length among the values 0, 1, ..., maxlag.
        verbose (bool, optional): prints ADF statistics, bestlag, and p-value if True. Defaults to True.
        kwargs: any other key-value pairs for sm.tsa.stattools.adfuller().

    Returns:
        a tuple of results: adf, pvalue, usedlag, nobs, critical values, icbest, resstore. 
            For the details please refer to official documentation 
            `Ref`_: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    """
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
    res = sm.tsa.stattools.adfuller(np.array(series), maxlag, regression, autolag, **kwargs)
    if verbose:
        print('variable:{:20s} - ADF Statistic: {:13f} \tp-value: {:10f}'.\
              format(series.name, res[0], res[1]))
        if 'autolag' in kwargs.keys():
            print('IC: {:6s} \t\t\tbest_lag: {:9d}'.\
                  format(kwargs['autolag'], res[2]))
        print('Critical Values: ', end='')
        for key, value in res[4].items():
            print('{:2s}: {:>7.3f}\t'.\
                  format(key, value), end='')
        print()
    return res


def adfuller_table(df: pd.DataFrame, maxlag=None, regression='c', verbose=False, alpha=0.05, **kwargs):
    """
    Iterate over ``adftest()`` to generate a table. 

    Args:
        df (pd.DataFrame): a dataframe where each column is a timeseries of type float with NO missing data. 
        verbose (bool, optional): prints ADF statistics, bestlag, and p-value if True. 
            See ``adftest()``. Defaults to False.
        alpha (float, optional): significance level. Must be in [0.01, 0.05, 0.1] Defaults to 0.05.
        kwargs: any other key-value pairs for adftest(), or equivalently, sm.tsa.stattools.adfuller().

    Returns:
        pd.DataFrame: An ADFuller table where each row is the result of each time series in original df and 
            columns are AIC, BIC information.

    """
    # validation: 
    assert alpha in [0.01, 0.05, 0.1], "Value of alpha is not valid. Only [0.01,0.05,0.1] are possible."
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
    # TODO: add Hannan–Quinn column also 
    columns = [f'AIC_{int(alpha*100)}%level', 'AIC_bestlag', 
               f'BIC_{int(alpha*100)}%level', 'BIC_bestlag']
    table = pd.DataFrame(columns=columns)
    for col in df.columns: 
        # sig=True means test statistics > critical value 
        # => pass ADF test (reject H0:unit root)
        row = []
        for autolag in ['AIC', 'BIC']:    
            res = adftest(df[col],  maxlag, regression, autolag, verbose=verbose, **kwargs)
            sig = True if abs(res[0]) > abs(res[4][f'{int(alpha*100)}%']) else False
            row.extend([sig, res[2]])
        table = pd.concat([table, pd.DataFrame({col: row}, index=table.columns).T], axis=0)
    table.index.name = 'ADFuller Table alpha={}'.format(alpha)
    return table


def grangers_causation_table(data, xnames, ynames, maxlag, 
                             test='ssr_chi2test', alpha=None):
    """
    Check Granger Causality of all possible combinations of the Time series.
    The values in the table are the P-Values/boolean (reject H0 or not). 
    H0: X does not cause Y (iff coefs of X on Y is 0)
    
    Args:
        data (pd.DataFrame): containing the time series variables
        xnames (list[str]): of time series variable names to test granger causality on ynames.
        ynames (list[str]): of time sereis variable names to be granger predicted.
        maxlag (int): max lags.
        test (str): must be one of ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']. 
            Defaults to 'ssr_chi2test'.
        alpha (float, optional): significance level. Return boolean table if alpha is specified not 
            as None. Defaults to None. 
    
    Returns:
        pd.DataFrame: table showing Granger test result. 

    """
    res = pd.DataFrame(np.zeros((len(xnames), len(ynames))), 
                       columns=ynames, index=xnames)
    for c in res.columns:
        for r in res.index:
            test_result = sm.tsa.stattools.grangercausalitytests(data[[r, c]], 
                          maxlag=maxlag, verbose=False)
            p_values = [ round(test_result[i+1][0][test][1],4) 
                         for i in range(maxlag) ]
            min_p_value = np.min(p_values)
            res.loc[r, c] = min_p_value
    res.columns = res.columns + '_y'
    res.index =  res.index + '_x'
    if alpha is None: 
        res.index.name = 'Granger Causation Table'
        return res
    res.index.name = 'Granger Causation Table alpha={}'.format(alpha)
    return res < alpha


def durbin_watson_test(resids: Any, axis: int=0, verbose: bool=False):
    """
    Test for serial correlation of error terms.
    Documentation comes from `Ref`_: https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html

    Args: 
        resids (array_like): Data for which to compute the Durbin-Watson statistic. Usually regression/VAR model residuals. 
            resids should NOT contain missing values, otherwise the corresponding dw statistic will be output as 0. 
        axis (int, optional): Axis to use if data has more than 1 dimension. 
            axis=0 means treat each column of resids as time series and calculate dw statistics. 
            axis=1 means treat each row of resids as time series and calculate dw statistics.
            Defaults to 0. 
        verbose (bool, optional): prints formatted Durbin-Watson test statistic results for 
            each dim of resids if True. Defaults to False.

    Returns: 
        (float, array_like): the Durbin-Watson statistic. 

    """
    # res
    res = sm.stats.stattools.durbin_watson(resids, axis=axis)

    # verbose
    if verbose: 
        print( '\n'.join([
            'Durbin-Watson test-stat of autocorrelation of error terms:', 
            'd=0: perfect autocorrelation', 
            'd=2: no autocorrelation', 
            'd=4: perfect negative autocorrelation.' 
        ]))
        for value in res:
            print( '{:32s}: {:>6.2f}'.format(col, value))
    print('\n', end='')
    
    return res