import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import seaborn as sns 
import numpy as np
import pandas as pd 


#%% testing functions
def adftest(series, verbose=True, **kwargs):
    """adfuller + printing"""
    
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
    res = sm.tsa.stattools.adfuller(series.values, **kwargs)
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


def adfuller_table(df, verbose=False, alpha=0.05, **kwargs):
    """iterate over adftest() to generate a table"""
    # validation: 
    assert alpha in [0.01, 0.05, 0.1], "Value of alpha is not valid. Only [0.01,0.05,0.1] are possible."
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
    # TODO: add Hannan–Quinn column also 
    columns = [f'AIC_{int(alpha*100)}%level', 'AIC_bestlag', 
               f'BIC_{int(alpha*100)}%level', 'BIC_bestlag']
    table = pd.DataFrame(columns=columns)
    for col in df.columns: 
        row = []
        for autolag in ['AIC', 'BIC']:
            res = adftest(df[col], verbose=verbose, 
                          autolag=autolag, **kwargs)
            # sig=True means test statistics > critical value 
            # => pass ADF test (reject H0:unit root)
            sig = True if abs(res[0]) > \
                  abs(res[4][f'{int(alpha*100)}%']) else False
            row.extend([sig, res[2]])
        table = pd.concat([table, pd.Series(row, index=table.columns, name=col)], axis=0)
    table.index.name = 'ADFuller Table alpha={}'.format(alpha)
    return table


def grangers_causation_table(data, xnames, ynames, maxlag, 
                             test='ssr_chi2test', alpha=None):
    """
    Check Granger Causality of all possible combinations of the Time series.
    The values in the table are the P-Values/boolean (reject H0 or not). 
    H0: X does not cause Y (iff coefs of X on Y is 0)
    
    Inputs
    ------
    data: pd.DataFrame - containing the time series variables
    xnames: list of TS variable names to test granger causality on ynames.
    ynames: list of TS variable names to be granger predicted.
    maxlag: int - max lags.
    test  : str - 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
    alpha : float - significance level. 
            Return boolean table if alpha is specified != None.
    
    Returns 
    -------
    pd.DataFrame table showing Granger test result. 
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


def durbin_watson_test(model, verbose=False):
    """
    Test for serial correlation of error terms.
    model: statsmodel.VAR model
    """
    # verbose
    if verbose: 
        print( '\n'.join([
            'Durbin-Watson test-stat of autocorrelation of error terms:', 
            'd=0: perfect autocorrelation', 
            'd=2: no autocorrelation', 
            'd=4: perfect negative autocorrelation.' 
        ]))
    print('\n', end='')
    
    # res
    res = sm.stats.stattools.durbin_watson(model.resid)
    for col, value in zip(model.names, res):
        print( '{:32s}: {:>6.2f}'.format(col, value))
    
    return res