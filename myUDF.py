
#%% plotting functions
def adjust_bright(color, amount=1.2):
    """
    Adjust color brightness in plots for use.
    Inputs
    ------
    color: str | list, 
        color can be basic color string name or rgb list. 
    amount: float, 
        the level of brightness of the input color to be adjusted. 
        the higher the amount, the brighter the color is.
    
    Returns
    -------
    color with brightness level adjusted.
    
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(
        c[0], max(0, min(1, amount * c[1])), c[2])

def missingval_plot(df, figsize=(20,6), show=True):
    """
    Visualize index location of missin values of each feature.
    Doesn't work for 1-dim df.
    df: pd.DataFrame 
    """
    import seaborn as sns 
    
    # check all are bool
    if (df.dtypes != bool).any():
        df = df.reset_index().T.isna()
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    g = sns.heatmap(df, cmap='Blues', cbar=True, 
                    yticklabels=df.index.values)
    # customize colorbar
    colorbar = g.collections[0].colorbar
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['non-missing', 'missing'])
    # customize title
    ax.set_title('Distribution of Missing Values', fontsize=16)
    # customize font size in ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    if show: 
        plt.show()

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """
    Create a sample plot for indices of a cross-validation object.
    """
    import numpy as np 
    import matplotlib.pyplot as plt
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, 
                   cmap=plt.cm.coolwarm, vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits)+.5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax    
        
def corrtri_plot(df, figsize=(10,10)):
    """correlation plot of the dataframe"""
    # sns.set() #: will cause plt warning later in lag_plot
    import numpy as np 
    import seaborn as sns
    
    c = df.corr()
    mask = np.triu(c.corr(), k=1)
    plt.figure(figsize=figsize)
    plt.tick_params(axis='both', which='major', labelsize=10, 
                    bottom=False, labelbottom=False, 
                    top=False, labeltop=True)
    g = sns.heatmap(c, annot=True, fmt='.1f', cmap='coolwarm', 
                    square=True, mask=mask, linewidths=1, cbar=False)
    plt.show()
    
def acpac_plot(data, figsize=(10,5)):
    """Autocorrelation and Partial-aurocorrelation plots."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    for i, col in enumerate(data.columns):
        fig, ax = plt.subplots(1,2,figsize=figsize)
        plot_acf(data[col], lags=30, 
                 title='AC: ' + data[col].name, 
                 ax=ax[0])  # missing='drop'
        plot_pacf(data[col], lags=30, 
                  title='PAC: ' + data[col].name, 
                 ax=ax[1])
        plt.show()
        
def residac_plot(model, cols=None, figsize=(16, 8), ylim=(-.3, .3)):
    """
    model: var/vecm model (from statsmodels)
    cols: can be integer/str list.
    """
    import pandas as pd 
    import matplotlib.pyplot as plt
    
    # set up
    if cols is not None: 
        cols = list(cols)
        assert len(cols)==pd.DataFrame(model.resid).shape[1], \
            "cols length not matched with model.resid columns."
    else: 
        cols = list(model.names)

    # make sure DataFrame type
    resid = pd.DataFrame(model.resid)
    if isinstance(model.resid, np.ndarray): 
        resid = pd.DataFrame(resid, columns=cols)

    # plot 
    plt.figure(figsize=figsize)
    for count, (name, series) in enumerate(resid[cols].iteritems()):
        ax = plt.subplot( len(cols)//3 +1, 3, count+1)
        ax.set(ylim=ylim)
        pd.plotting.autocorrelation_plot(series, ax=ax)
        ax.set_ylabel('')
        plt.title(f'Residual Autocorrelation: {name}')
        ax.figure.tight_layout(pad=0.5)

    return ax


def periodogram(series, division=8, figsize=(15,10)):
    """
    plot periodogram of the series to find most important 
    frequency/periodicity
    """
    import numpy as np 
    import matplotlib.pyplot as plt
    dt = 1
    T = len(series.index)
    t = np.arange(0, T, dt)
    f = series.fillna(0)
    n = len(t)
    fhat = np.fft.fft(f, n, )
    PSD = np.conj(fhat) / n
    freq = np.arange(n) # * (1/(dt*n))
    L = np.arange(1, np.floor(n/division), dtype="int") 
    # n/4, otherwise too long of a stretch to see anything
    
    plt.figure(figsize=figsize)
    plt.plot(freq[L], np.abs(PSD[L]), linewidth=2, 
             label='Lag Importance')
    plt.xlim(freq[L[0]], freq[L[-1]])
    plt.legend()
    plt.hlines(0, xmin=0, xmax=n, colors='black')
    plt.title('Periodogram of ' + series.name)
    plt.show()
    
def rfft_plot(series, ylim=(0,400), figsize=(15,10)):
    """plot real valued fourier transform to find most important 
    frequency/periodicity"""
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np

    fft = tf.signal.rfft(series)
    f_per_dataset = np.arange(0, len(fft))

    n_samples_d = len(series)
    d_per_year = 365.2524
    years_per_dataset = n_samples_d/(d_per_year)
    f_per_year = f_per_dataset/years_per_dataset
    
    plt.figure(figsize=figsize)
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(*ylim)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 4, 12, 52, 365.2524], 
               labels=['1/Year', '1/quarter', 
                       '1/month', '1/week', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')
    plt.show()

    
def lag_plot(df, lag=1, redcols=None, figsize=(20,15), alpha=.3):
    """
    plot t+lag against t of each feature in df.
    df: pd.dataframe
    redcols: list/array of column names to be colored with red.
    """
    import matplotlib.pyplot as plt
    import pandas as pd 
    plt.figure(figsize=figsize)
    for i, col in enumerate(df.columns):
        ax = plt.subplot(len(df.columns)//5 +1 , 5, i+1)
        color = 'k'
        if redcols is not None and col in redcols:
            color = 'r'
        pd.plotting.lag_plot(df[col], lag=lag, alpha=alpha)
        plt.title(col, color=color)
    ax.figure.tight_layout(pad=0.5)
    

def pca_plot(data, n_comp=None, regex=None, figsize=(5,3)):
    """
    Plot n_comp pricipal components of data via PCA.
    data:   pd.DataFrame / np.ndarray
    regex:  string pattern to filter data. 
            Use all data if not specified.
    n_comp: number of components desired in PCA. 
            Default to data column numbers if not specified.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    x = data
    if regex is not None: 
        x = x.filter(regex=regex)
    xSfit = StandardScaler().fit_transform(x)
    if n_comp is None:
        n_comp = xSfit.shape[1]
    pca = PCA(n_components=n_comp)
    pca.fit(xSfit)
    v = pca.explained_variance_ratio_.round(2)
    xtick = range(1,n_comp+1)
    ax.bar(xtick,v) # range(1,n_comp+1)
    plt.xticks(xtick, x.columns, rotation='vertical')
    plt.xlabel("PCA components")
    plt.title("Variance Explained by each dimension")
    plt.show()












#%% testing functions
def adftest(series, verbose=True, **kwargs):
    """adfuller + printing"""
    
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
    from statsmodels.tsa.stattools import adfuller
    res = adfuller(series.values, **kwargs)
    if verbose:
        print('ADF Statistic: {:13f} \tp-value: {:10f}'.\
              format(res[0], res[1]))
        if 'autolag' in kwargs.keys():
            print('IC: {:6s} \t\t\tbest_lag: {:9d}'.\
                  format(kwargs['autolag'], res[2]))
        print('Critical Values: ', end='')
        for key, value in res[4].items():
            print('{:2s}: {:>7.3f}\t'.\
                  format(key, value), end='')
    return res


def adfuller_table(df, verbose=False, alpha=0.05, **kwargs):
    """iterate over adftest() to generate a table"""
    import pandas as pd 
    # kwargs: maxlag: default=12*(nobs/100)^{1/4}, regression, autolag
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
        table = table.append(
            pd.Series(row, index=table.columns, name=col)
        )
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
            test_result = grangercausalitytests(data[[r, c]], 
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
    res = statsmodels.stats.stattools.durbin_watson(model.resid)
    for col, value in zip(model.names, res):
        print( '{:32s}: {:>6.2f}'.format(col, value))
    
    return res
