import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import pandas as pd 
from statsmodels.graphics import tsaplots
from scipy.signal import periodogram
from .misc import get_equivalent_days
import re 


#%% plotting functions
def adjust_bright(color, amount=1.2):
    """
    Adjust color brightness in plots for use.
    Args: 
        color: str | list, 
            color can be basic color string name or rgb list. 
        amount: float, 
            the level of brightness of the input color to be adjusted. 
            the higher the amount, the brighter the color is.
    
    Returns:
        tuple: color with brightness level adjusted.
    
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

def missingval_plot(df, figsize=(20,6)):
    """
    Visualize index location of missin values of each feature. Doesn't work for 1-dim df.

    Args: 
        df (pd.DataFrame): the data to be plotted for missing value patterns. 

    """
    
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
    plt.show()

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """
    Create a sample plot for indices of a cross-validation object.
    """

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

    c = df.corr()
    mask = np.triu(c.corr(), k=1)
    plt.figure(figsize=figsize)
    plt.tick_params(axis='both', which='major', labelsize=10, 
                    bottom=False, labelbottom=False, 
                    top=False, labeltop=True)
    g = sns.heatmap(c, annot=True, fmt='.1f', cmap='coolwarm', 
                    square=True, mask=mask, linewidths=1, cbar=False)
    plt.show()
    
def acpac_plot(data, features: list=None, lags: int=None, figsize: tuple=(10,5), **kwargs):
    """
    Autocorrelation and Partial-aurocorrelation plots.
    See `Refenence`_: https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
    
    Args: 
        data ([pd.DataFrame, pd.Series]): time series data that contains ``features`` as columns for ac/pac plots. 
        features (list, optional): list of columns of ``data``. If None, features will be all ``data.columns``. Defaults to None. 
        lags (int, optional): An int or array of lag values, used on horizontal axis. Uses np.arange(lags) when lags is an int. 
            If not provided, lags=np.arange(len(corr)) is used. 
        kwargs (optional): other arguments specific to acf/pacf plots. Use nested dict with keys 'acf' and 'pacf' to specify configs respectively.
        figsize (tuple, optional): adjust figure size (width, height).

    Returns: 
        None: direcly plt.show() plots.
    """
    
    if features is None: 
        features = data.columns 

    for i, col in enumerate(features):
        fig, ax = plt.subplots(1,2,figsize=figsize)
        tsaplots.plot_acf(data[col], lags=lags, title='AC: ' + data[col].name, ax=ax[0], **kwargs.get("acf", {}))
        tsaplots.plot_pacf(data[col], lags=lags, title='PAC: ' + data[col].name, ax=ax[1], **kwargs.get("pacf", {}))
    
        
def residac_plot(model, cols=None, figsize=(16, 8), ylim=(-.3, .3)):
    """
    model: var/vecm model (from statsmodels)
    cols: can be integer/str list.
    """

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


# periodogram plots 
def rfft_plot(series, ylim=(0,400), figsize=(15,10)):
    """
    plot real valued fourier transform to find most important frequency/periodicity
    """
    import tensorflow as tf

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


def seasonal_plot(data, y, period, freq, ax=None, figsize=(20,10)):
    """
    Plot every ``period`` of ``y`` as a line on ``freq`` as x-axis.

    Example:
        >>> X = pd.read_csv(...)
        >>> X['day'] = X.index.dayofweek 
        >>> X['week'] = X.index.week
        >>> seasonal_plot(X, y='sales', period='week', freq='day')
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", n_colors=data[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=data,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    
    # loop over lines/period for "period" annotation
    for line, name in zip(ax.lines, data[period].unique()):
        # annotate besides the last pt of each curve
        y_ = line.get_ydata()
        if y_.size == 0: 
            continue 
        ax.annotate(
            name,
            xy=(1, y_[-1]), # on last data point
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax

def plot_periodogram(ts: pd.Series, ts_freq=None, detrend='linear', ax=None):
    # extract relevant freq
    assert isinstance(ts_freq, str) or ts_freq is None, "ts_freq type not valid."
    if ts_freq is None: 
        dates = ts.index 
        if isinstance(dates, pd.PeriodIndex):
            ts_freq = dates.freqstr     # may contain numbers
        else: 
            ts_freq = pd.infer_freq(dates)
    assert ts_freq is not None, (
        "Cannot decide the ts freq. Please either transform the date index to"+
        "period or set a value for ts_freq instead of the default value None."
    )
    # conver ts_freq to Timedelta
    value = re.search("^\d*", ts_freq)[0] # only match head string numbers
    if value == '': 
        value = 1.0
    denominator = get_equivalent_days(float(value), unit=ts_freq)
    
    # get power spectrum 
    # default is use 1Y as numerator
    fs = pd.Timedelta(365.2425, unit='D') / denominator
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    # plot 
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    # periods = 1/w = fs/w', where w' is the number for annotatioin in x-axis.
    ## Eg, if fs=365 and ts_freq is daily, then period =1Y if w' is 1, =Quarter if w' is 4. 
    ## Eg, if fs=365 and ts_freq is 2D, then period =Biannual if w' is 1, =Annual if w' is 2, =2/3Y if w' is 3, =Quarter if w' is 8.
    ## Eg, if fs=12 and ts_freq is monthly, then period =1Y if w' is 1, =Quarter if w' is 4. 
    # I intentionally define fs = 365 / ts_freq in the above so that second eg never happens.
    ax.set_xticks([ 1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=90,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def _lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    """
    helper function of plot_lag. 
    `Ref`_: https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = matplotlib.offsetbox.AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def lags_plot(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    """
    `Ref`_: https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', np.ceil(lags / nrows).astype(int))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = _lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig
    

def pca_plot(data, n_comp=None, regex=None, figsize=(5,3)):
    """
    Plot n_comp pricipal components of data via PCA.

    Args: 
        data:   pd.DataFrame / np.ndarray
        regex:  string pattern to filter data. 
                Use all data if not specified.
        n_comp: number of components desired in PCA. Default to data column numbers if not specified.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA 

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



def predict_gt_plot(at, y_true_tra, y_pred_tra, y_true_tes, y_pred_tes, 
                    figsize=(25,15), freq='MS'):
    """
    Plot the ground truth and prediction curves on both train and test set.
    y_tra and y_tes should have timestamp at index.

    Args: 
        at (int): specifies which prediction horizon it is which will be used to shift the timestamp of ground truth data, ie, ``y_tra`` and ``y_tes``.
        y_true_tra (pd.DataFrame, pd.Series): training set ground truth time series
        y_pred_tra (pd.DataFrame, pd.Series, np.array): training set prediction time series
        y_true_tes (pd.DataFrame, pd.Series): testing set ground truth time series
        y_pred_tes (pd.DataFrame, pd.Series, np.array): testing set prediction time series
        figsize (tuple of ints): 2-tuple integers (width, height) defining figure size, defaults to (25,15)
        freq (str, optional): freq of the input time series data, defaults to 'MS'

    Returns: 
        plt.Axes: axes that contains data content of the figure.

    """
    # initialization 
    y_true_tra, y_true_tes = pd.DataFrame(y_true_tra), pd.DataFrame(y_true_tes)
    y_pred_tra, y_pred_tes = np.array(y_pred_tra), np.array(y_pred_tes)
    if y_pred_tra.ndim == 1: 
        y_pred_tra = y_pred_tra.reshape(-1, 1)
    if y_pred_tes.ndim == 1: 
        y_pred_tes = y_pred_tes.reshape(-1, 1)
    # plot 
    num_targets = y_true_tra.shape[1]
    targets = y_true_tra.columns
    fig, ax = plt.subplots(num_targets, 1, figsize=figsize, sharex=True)
    for j, ta in enumerate(targets):
        ax[j].plot(y_true_tra.asfreq(freq).index.shift(at), y_pred_tra[:, j], c='b', label='train-predict')
        ax[j].plot(y_true_tra.asfreq(freq).index.shift(at), y_true_tra.values[:, j], c='k', label='gt-tra')
        ax[j].plot(y_true_tes.asfreq(freq).index.shift(at), y_pred_tes[:, j], c='orange', label='test-predict')
        ax[j].plot(y_true_tes.asfreq(freq).index.shift(at), y_true_tes.values[:, j], c='k', label='gt-tes')
        ax[j].set_title(f'prediction of {ta} at {at}th-horizon', fontdict = {'fontsize' : 20})
    plt.legend()
    plt.tight_layout()
    return ax