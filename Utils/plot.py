import matplotlib.pyplot as plt 
import statsmodels
import seaborn as sns 
import numpy as np
import pandas as pd 


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
    
def acpac_plot(data, features=[], figsize=(10,5)):
    """Autocorrelation and Partial-aurocorrelation plots."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    if features == []: 
        features = data.columns 
    for i, col in enumerate(features):
        fig, ax = plt.subplots(1,2,figsize=figsize)
        plot_acf(data[col], lags=30, 
                 title='AC: ' + data[col].name, 
                 ax=ax[0])  # missing='drop'
        plot_pacf(data[col], lags=30, 
                  title='PAC: ' + data[col].name, 
                 ax=ax[1])
    
        
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
def periodogram(series, division=8, figsize=(15,10), ax=None):
    """
    plot periodogram of the series to find most important 
    frequency/periodicity
    """

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
    if ax is None: 
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(freq[L], np.abs(PSD[L]), linewidth=2, 
             label='Lag Importance')
    ax.hlines(0, xmin=0, xmax=n, colors='black')
    ax.set_xlim(freq[L[0]], freq[L[-1]])
    ax.set_xlabel("Lag")
    ax.tick_params(axis='x', labelbottom=True)     # labelbottom: make xticks everywhere
    ax.set_title('Periodogram of ' + series.name)
    ax.legend()
    
    return ax 
    
def rfft_plot(series, ylim=(0,400), figsize=(15,10)):
    """plot real valued fourier transform to find most important 
    frequency/periodicity"""
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

    
def lag_plot(df, lag=1, redcols=None, figsize=(20,15), alpha=.3):
    """
    plot t+lag against t of each feature in df.
    df: pd.dataframe
    redcols: list/array of column names to be colored with red.
    """

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

    :param at: [specifies which prediction horizon it is which will be used to shift the timestamp of ground truth data, ie, ``y_tra`` and ``y_tes``.]
    :type at: [int]
    :param y_true_tra: training set ground truth time series
    :type y_true_tra: pd.DataFrame, pd.Series
    :param y_pred_tra: training set prediction time series
    :type y_pred_tra: pd.DataFrame, pd.Series, np.array
    :param y_true_tes: testing set ground truth time series
    :type y_true_tes: pd.DataFrame, pd.Series
    :param y_pred_tes: testing set prediction time series
    :type y_pred_tes: pd.DataFrame, pd.Series, np.array
    :param figsize: [description], defaults to (25,15)
    :type figsize: tuple, optional
    :param freq: freq of the input time series data, defaults to 'MS'
    :type freq: str, optional
    :return: axes that contains data content of the figure.
    :rtype: plt.Axes
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