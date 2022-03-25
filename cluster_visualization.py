'''
Available methods are the followings:
[1] cluster_pie
[2] cluster_hist
[3] cluster_scatter
[4] cluster_matrix
[5] cluster_radar
[6] create_cmap
[7] matplotlib_cmap
[8] adjust_label

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 22-07-2021

'''
import pandas as pd, numpy as np, math
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d
from scipy import (stats, linalg)

from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)
from mpl_toolkits.axes_grid1 import Grid
from itertools import product
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__  = ["matplotlib_cmap", "create_cmap",
            "cluster_pie", 
            "cluster_hist", 
            "cluster_scatter",
            "cluster_matrix",
            "cluster_radar", 
            "adjust_label"]

def matplotlib_cmap(name='viridis', n=10):

    '''
    Parameters
    ----------
    name : matplotlib Colormap str, default='viridis'
        Name of a colormap known to Matplotlib. 
    
    n : int, defualt=10
        Number of shades for defined color map.
    
    Returns
    -------
    colors : list of color-hex
        List of color-hex codes from defined Matplotlib Colormap. 
        Such list contains "n" shades.
        
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    colors = [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(n)]
    return colors

def create_cmap(colors=None):
    
    '''
    Creating matplotlib.colors.Colormap (Colormaps).
    
    Parameters
    ----------
    colors : list of hex-colors or RGBs, default=None
        The beginning color code [(255,10,8),(255,255,255)]
  
    Returns
    -------
    matplotlib ListedColormap
    
    '''
    # Convert to RGB
    to_RGB = lambda c: tuple(int(c[i:i+2],16) for i in (0,2,4))
    if colors is None: colors = [(255,10,8), (255,255,255)]
    colors = np.array([to_RGB(c.lstrip('#')) if isinstance(c, str) 
                       else c for c in colors])/256
    rgb = []
    for c1,c2 in zip(colors[:-1],colors[1:]):
        rgb.append(np.array([np.linspace(c1[i],c2[i],256) 
                             for i in range(3)]).T)
    rgb = np.vstack(rgb)
    rgb = rgb[np.arange(len(rgb)-1,-1,-1),:]
    return ListedColormap(rgb, name="Customized_cmap")

def cluster_pie(y, ax=None, labels=None, colors=None, 
                pie_kwds=None, tight_layout=True):
    
    '''
    Plot a cluster pie chart.
    
    Parameters
    ----------
    y : 1D-array or pd.Series
        Array of cluster labels (0 to n_clusters-1)
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.
    
    labels : list, default: None
        A sequence of strings providing the labels for each 
        class. If None, 'Cluster {n+1}' is assigned, where n 
        is the class in y.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to
        number of classes. If None, it uses default colors 
        from Matplotlib
    
    pie_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.pie".

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    References
    ----------
    .. [1] https://matplotlib.org/stable/api/_as_gen/
           matplotlib.pyplot.pie.html
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(5, 4))[1]
    
    unq, unq_count = np.unique(y, return_counts=True)
    labels = ([f"Cluster {n}" for n in unq+1] 
              if labels is None else labels)
    
    kwds = dict(explode = (unq_count==max(unq_count)).astype(int)*0.1,
                colors  = ([ax._get_lines.get_next_color() for _ in unq] 
                           if colors is None else colors), 
                labels  = ['{}\n({:,d})'.format(*v) for 
                           v in zip(labels, unq_count)], 
                autopct = "{:,.0f}%".format, 
                shadow  = True, 
                startangle = 90, labeldistance=1.3,
                wedgeprops = dict(edgecolor="#2f3640", lw=2), 
                textprops  = dict(fontsize=13, ha="center", 
                                  fontweight=500, color="#2f3640"))
    ax.pie(unq_count, **(kwds if pie_kwds is 
                         None else {**kwds,**pie_kwds}))
    ax.axis('equal')
    if tight_layout: plt.tight_layout()
    return ax

def cluster_hist(x, y=None, ax=None, labels=None, colors=None, 
                 fill_kwds=None, plot_kwds=None, tight_layout=True, 
                 bins="fd", sigma=None, whis=1.5, plot_order=None):
    
    '''
    Plot a cluster Kernal Density Estimation (KDE) chart.
    
    Parameters
    ----------
    x : 1D-array or pd.Series
        Sample input. All elements must be finite, i.e. no NaNs or 
        infs. 

    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None, an 
        array of single class is used instead.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.

    labels : list, default: None
        A sequence of strings providing the labels for each class. If 
        None, 'Cluster {n+1}' is assigned, where n is the class in y.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.
        This overrides `fill_kwds`, and `plot_kwds`.
    
    fill_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.fill_between".
    
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
        
    bins : int or str, default="fd"
        Number of bins (np.histogram_bin_edges) [1].
    
    sigma : float, default=None
        Standard deviation for Gaussian kernel. Sigma must be greater 
        than 0. The higher the sigma the smoother the probability 
        density curve (PDF). If None, it uses bin_width derived from
        `bins`.
        
    whis : float, default=1.5  
        It determines the reach of the whiskers to the beyond the 
        first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
        whis*IQR, respectively. This applies to both coordinates and 
        lower and upper bounds accordingly.
    
    plot_order : list of int, default=None
        List of class order to be plotted.
    
    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/numpy.
           histogram_bin_edges.html#numpy.histogram_bin_edges
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # ===============================================================
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(7,6))[1]
    # Default value of y
    y = (np.zeros(len(X)) if y is None else y).astype(int)
    # Default colors
    unique = np.unique(y)
    colors = ([ax._get_lines.get_next_color() for _ in unique] 
              if colors is None else colors)
    # ---------------------------------------------------------------
    # Default : plot_order and labels
    plot_order = np.array(unique if plot_order is None else plot_order)
    labels = ([f"Cluster {n}" for n in plot_order+1] 
              if labels is None else labels)
    # ---------------------------------------------------------------
    if fill_kwds is None: fill_kwds = {}
    if plot_kwds is None: plot_kwds = {}
    # ===============================================================
    
    # Plot histogram
    # ===============================================================
    # Determine bins.
    bins   = np.histogram(x, bins)[1]
    xticks = bins[1:] + np.diff(bins)
    if sigma is None: sigma = np.diff(bins)[0]
    # ---------------------------------------------------------------
    for n,c in enumerate(plot_order):
        
        # 1-D Gaussian filter (smoothen density curve)
        density = np.histogram(x[y==c], bins, density=True)[0]
        pdf = gaussian_filter1d(density, sigma)
    # ---------------------------------------------------------------
        # Density plot (ax.fill_between)
        kwds = {**dict(label=labels[n], alpha=0.3), **fill_kwds}
        kwds.update(dict(color=colors[n]))
        ax.fill_between(xticks, pdf, **kwds)
    # ---------------------------------------------------------------    
        # Density line (ax.plot)
        kwds = {**dict(lw=2.5), **plot_kwds}
        ax.plot(xticks, pdf, **{**kwds,**dict(color=colors[n])})
    # ===============================================================    

    # Set other attributes.
    # ===============================================================
    # Set limits of coordinates.
    ax.set_ylim(*np.array(ax.get_ylim())*[[1,1/0.8]])
    ax.set_xlim(__IQR__(x, whis, *ax.get_xlim()))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=12)
    # ---------------------------------------------------------------
    if len(plot_order)>1: 
        ax.legend(loc=0, edgecolor="grey", ncol=1, 
                  borderaxespad=0.1, markerscale=1, 
                  columnspacing=0.2, labelspacing=0.3, 
                  handletextpad=0.2, prop=dict(size=12))
    if tight_layout: plt.tight_layout()
    # ===============================================================
  
    return ax

def cluster_scatter(x1, x2, y=None, ax=None, labels=None, colors=None, 
                    scatter_kwds=None, tight_layout=True, whis=1.5, 
                    frac=1, random_state=0, use_kde=False, cmap=None, 
                    plot_order=None, decision=False, n_grids=(500,500), 
                    estimator=None, match_labels=False):
    
    '''
    Plot a cluster scatter chart.
    
    Parameters
    ----------
    x1, x2 : 1D-array or pd.Series
        The horizontal, and vertical coordinates of the data points, 
        respectively. All elements must be finite, i.e. no NaNs or 
        infs. 
    
    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None, an 
        array of single class is used instead.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    labels : list, default: None
        A sequence of strings providing the labels for each class. If 
        None, 'Cluster {n+1}' is assigned, where n is the class in `y`.

    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.
        This overrides `scatter_kwds`.
    
    scatter_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.scatter".

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
              
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond the 
        first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
        whis*IQR, respectively. This applies to both coordinates and 
        lower and upper bounds accordingly.
    
    frac : float, default=1
        Fraction of items to be plotted.
         
    random_state : int, default=0
        Controls the randomness of sampling of instances to be plotted.
        
    use_kde : bool, default=False
        If True, a kernel-density estimate using Gaussian kernels is 
        used, otherwise scatter plots [1].
    
    cmap : str or Colormap, default=None
        A Colormap instance e.g. cm.get_cmap('Reds',20) or registered 
        colormap name. This is relevant when `use_kde` is True [2]. If 
        None, it defaults to "Blues". This overrides "ax.scatter".
        
    plot_order : list of int, default=None
        List of class order to be plotted.
        
    decision : bool, default=False
        If True, decision boundaries of all classes in `y` are drawn 
        based on `x1` and `x2` (pairwise), and settings of `estimator`. 

    n_grids : (int,int), default=(500,500)
        Number of grids on x and y axes. This is relevant when 
        `decision` is True.
    
    estimator : estimator object, default=None
        Create decision boundary. This is assumed to implement the 
        scikit-learn estimator interface i.e. self.predict(X) to 
        predict class for X. If None, it uses scikit-learn
        "LogisticRegression" with following parameters i.e. 
        "class_weight"="balanced", and "penalty"="none".
    
    match_labels : bool, default=False
        If True, it performs a label matching. This is recommended 
        for unsupervised `estimator`, where labels are assigned based 
        on randomness (`random_state`). The matching uses majority
        criterion and will go down the priority list until all classes
        are uniquely matched. This is relevant when `decision` is True.
        
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.stats.gaussian_kde.html
    .. [2] https://matplotlib.org/stable/tutorials/colors/colormaps.html
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # ===============================================================
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(7,6))[1]
    # ---------------------------------------------------------------
    # Default value for `y` (single class)
    y = (np.zeros(len(x1)) if y is None else y).astype(int)
    # ---------------------------------------------------------------
    # Default colors
    unique = np.unique(y)
    colors = np.r_[[ax._get_lines.get_next_color() for _ in unique] 
                   if colors is None else colors]
    # ---------------------------------------------------------------
    # Default `scatter_kwds`
    if scatter_kwds is None: scatter_kwds = {}
    # ---------------------------------------------------------------
    # Default `plot_order`
    plot_order = np.r_[unique if plot_order is None else plot_order]
    # ---------------------------------------------------------------
    # Default : labels
    labels = np.r_[[f"Cluster {n}" for n in plot_order + 1] 
                   if labels is None else labels]
    # ---------------------------------------------------------------
    indices = np.arange(len(x1))
    # Keep x1, x2, and y given plot_order
    indices = indices[np.isin(y, plot_order)]
    # Randomize x1, x2, and y.
    indices = __RandomIndices__(indices, frac, random_state)
    y  = np.array(y).ravel()[indices]
    x1 = np.array(x1).ravel()[indices]
    x2 = np.array(x2).ravel()[indices]
    # ===============================================================
    
    # Plot
    # ===============================================================
    # Compute kernel-density estimate using Gaussian kernels
    if use_kde==True:
        
        # Colormap
        if cmap is None: cmap = 'Blues' 
        if isinstance(cmap, str): cmap = cm.get_cmap(cmap, 50)
        kwds = dict(marker='o', alpha=0.5, s=10)
        data = np.vstack((x1, x2))
        try: values = gaussian_kde(data).evaluate(data)
        except: values = gaussian_kde(x1).evaluate(x1)
        kwds.update(scatter_kwds)
        ax.scatter(x1, x2, **{**kwds,**dict(c=values, cmap=cmap)})
    # ---------------------------------------------------------------
    # Scatter plot
    elif use_kde==False:
        
        # Default settings for "ax.scatter".
        kwds = dict(marker='o', alpha=0.5, s=10)
        kwds.update(scatter_kwds)
        for n,c in enumerate(plot_order):
            kwds.update(dict(facecolor=colors[n], label=labels[n]))
            ax.scatter(x1[(y==c)], x2[(y==c)], **kwds)
    # ---------------------------------------------------------------
        if decision & (len(plot_order)>1):
        
            # Create the grid for background colors
            m1 = np.linspace(min(x1)-1, max(x1)+1, n_grids[0]+1)
            m2 = np.linspace(min(x2)-1, max(x2)+1, n_grids[1]+1)
            m1, m2 = np.meshgrid(m1, m2)
            m12 = np.c_[m1.ravel(), m2.ravel()]

            # Use `LogisticRegression` as a default `estimator`.
            if estimator is None:
                kwargs = {"class_weight": "balanced", 
                          "penalty": "none"}
                estimator = LogisticRegression(**kwargs)
           
            # plot decision boundaries (contour)
            C = np.c_[x1.ravel(), x2.ravel()]
            estimator.fit(C, y)
            
            if match_labels:
                args = (y, estimator.predict(C), 
                        estimator.predict(m12))
                Z = label_matching(*args).reshape(m1.shape)
            else: Z = estimator.predict(m12).reshape(m1.shape)
            
            # ax.contourf(m1, m2, Z, cmap)
            ax.pcolormesh(m1, m2,Z, shading="auto", zorder=-1,
                           cmap=create_cmap(colors[::-1]))
    # ===============================================================       
    
    # Set other attributes.
    # ===============================================================
    # Set limits of coordinates.
    ax.set_xlim(__IQR__(x1, whis, *ax.get_xlim()))
    ax.set_ylim(__IQR__(x2, whis, *ax.get_ylim()))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=12)
    # ---------------------------------------------------------------
    if (not use_kde) & (len(plot_order)>1): 
        ax.legend(loc=0, edgecolor="grey", ncol=1, 
                  borderaxespad=0.1, markerscale=1, 
                  columnspacing=0.2, labelspacing=0.3, 
                  handletextpad=0.2, prop=dict(size=12))
    if tight_layout: plt.tight_layout()
    # ===============================================================
        
    return ax

def __IQR__(a, whis, a_min, a_max):
    
    '''Private Function: Interquatile range'''
    Q1,Q3 = np.percentile(a, [25, 75])
    if Q1==Q3: return (a_min, a_max)
    else: return (np.fmax(Q1-whis*(Q3-Q1),a_min), 
                  np.fmin(Q3+whis*(Q3-Q1),a_max))

def __RandomIndices__(indices, frac=1, random_state=0):
    
    '''Private Function: Random indices'''
    # Select samples
    rng = np.random.RandomState(random_state)
    kwds = dict(size=np.fmax(int(frac*len(indices)),10), replace=False)
    if frac<1: select = rng.choice(indices, **kwds)
    else: select = indices.copy()
    return np.isin(indices, select)

def label_matching(y_true, y_pred, y_trans):
    
    '''Private Function: Matching labels'''
    pred_labels = []
    true_labels = np.unique(y_true)
    for k in true_labels:
        unq, cnt = np.unique(y_pred[y_true==k], 
                             return_counts=True)
        
        # Determine label with maximum count.
        n_labels = np.r_[[0]*len(true_labels)]
        n_labels[unq] = cnt
        
        # Sort label indices from max to min.
        labels = np.argsort(n_labels)[::-1] 
        # Select label, whose count is the maximum and 
        # not in `pred_labels`
        index  = np.isin(labels, pred_labels)
        pred_labels += [labels[~index][0]]
        
    condlist, choicelist = [], []
    for a,b in zip(true_labels, pred_labels):
        condlist   += [y_trans==b]
        choicelist += [a]
        
    return np.select(condlist, choicelist)

def cluster_matrix(X, y=None, colors=None, whis=1.5, plot_order=None, 
                   off_diagonal="scatter", hist_kwds=None, 
                   scatter_kwds=None, n_limit=1000, figsize=None, 
                   show_corr=True, label_kwds=None):

    '''
    Draw a matrix of scatter plots towards a pairwise comparison to 
    observe interaction of variables. NaN is ignored. Each numeric 
    variable in data will be shared across a single row and the 
    x-axes across a single column.

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input sample.
       
    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None, an 
        array of single class is used instead. This overrides 
        `hist_kwds` and `scatter_kwds`.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib. 
        This overrides `hist_kwds` and `scatter_kwds`.
        
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond the 
        first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
        whis*IQR, respectively. This applies to both coordinates and 
        lower and upper bounds accordingly. This overrides `hist_kwds` 
        and `scatter_kwds`.
        
    plot_order : list of int, default=None
        List of class order to be plotted. This overrides `hist_kwds` 
        and `scatter_kwds`.
         
    off_diagonal : {"scatter", "kde", "both"}, default="scatter"
        Pick between "kde", "scatter", and "both" for either Kernel 
        Density Estimation or scatter or both in the off-diagonal. 
        This overrides `scatter_kwds`.
   
    hist_kwds : keywords, default=None
        Keyword arguments to be passed to "cluster_hist".
    
    scatter_kwds : keywords, default=None
        Keyword arguments to be passed to "cluster_scatter".
        
    n_limit : int, default=1000
        Number of instances to be plotted in the off-diagonal plots. 
        n_limit should be less than len(X).
        
    figsize : (float,float), default=None
        A tuple (width, height) in inches. If None, a default figsize 
        is applied i.e. (n_rows*1.3, n_cols*1.3).
    
    show_corr : bool, default=True
        If True, it shows a Pearson correlation coefficient for every 
        pariwise plot (off-diagonal).
        
    label_kwds : keyword, default=None
        Keyword arguments to be passed to "adjust_label". If None, it
        defaults to {"max_lines":2, "factor":0.95, "suffix":"...."}.
    
    Returns
    -------
    grid : numpy.ndarray, of shape (n_rows, n_cols)
        A matrix of plots.

    '''
    # ===============================================================
    # A grid of Axes (mpl_toolkits.axes_grid1).
    n_rows, n_cols = (X.shape[1],)*2
    figsize = (n_rows*1.3, n_cols*1.3) if figsize is None else figsize
    fig  = plt.figure(figsize=figsize)
    grid = np.array(Grid(fig, rect=111, nrows_ncols=(n_rows,n_cols), 
                         share_x=False, share_y=False, axes_pad=0))
    # ---------------------------------------------------------------
    # Fraction, Pairs of variables, and axes.
    frac, columns = np.fmin(n_limit/len(X),1), X.columns
    pairs = zip(list(product(columns, columns)), grid, 
                product(range(n_rows), range(n_cols)))
    y = (np.zeros(len(X)) if y is None else y).astype(int)
    # ===============================================================
    
    # Plot scatter.
    # ===============================================================
    for (var1,var2), ax, (r,c) in pairs:
        notna = (X[[var1,var2]].notna().sum(axis=1)==2).values
        if notna.sum()>0:
            x1 = X.loc[notna, var1].copy()
            x2 = X.loc[notna, var2].copy()
            y0 = y[notna].copy()
    # ---------------------------------------------------------------        
            # Diagonal plot
            if var1==var2:
                default   = {"plot_kwds": {"lw": 1}}
                hist_kwds = (default if hist_kwds is None else 
                             {**default, **hist_kwds})
                kwds = dict(y=y, ax=ax, colors=colors, whis=whis, 
                            plot_order=plot_order, tight_layout=False)
                ax = cluster_hist(x1, **{**hist_kwds, **kwds})
    # ---------------------------------------------------------------
            # Off-diagonal plot
            elif var1!=var2: 
                default = {"scatter_kwds": dict(s=10, alpha=0.5)}
                scatter_kwds = (default if scatter_kwds is None 
                                else {**default, **scatter_kwds})
    # ---------------------------------------------------------------               
                use_kde = (True if off_diagonal=="kde" else False)
                if (off_diagonal=="both") & (c>r): use_kde = False
                elif (off_diagonal=="both") & (c<r): use_kde = True   
    # ---------------------------------------------------------------                
                kwds = dict(y=y0, ax=ax, frac=frac, whis=whis,
                            colors=colors, plot_order=plot_order, 
                            tight_layout=False, use_kde=use_kde)    
                ax = cluster_scatter(x2, x1, **{**scatter_kwds, 
                                                **kwds})
    # ---------------------------------------------------------------
                if show_corr:
                    corr, pvalue = stats.pearsonr(x1, x2)
                    bbox = dict(boxstyle='round' , alpha=1, pad=0.2,
                                facecolor='white', edgecolor='grey')
                    ax.text(0.95, 0.05, ('{:.2f}'.format(corr)), 
                            size=11, transform=ax.transAxes, 
                            ha="right", va="bottom", bbox=bbox)
    # ---------------------------------------------------------------                
            if ax.get_legend() is not None: 
                ax.legend().set_visible(False)      
            ax.set(xticks=[], yticks=[])  
            if c==0: ax.set_ylabel(var1, fontsize=12)
            if r==n_rows-1: ax.set_xlabel(var2, fontsize=12)   
    # ===============================================================
    
    # Adjust labels both x and y axis.
    # ===============================================================
    plt.tight_layout(pad=0)
    if label_kwds is None: 
        label_kwds = {"max_lines":2, "factor":0.95, "suffix":"...."}
    for ax in grid[0::n_cols]: adjust_label(ax, "y", **label_kwds)
    for ax in grid[-n_cols: ]: adjust_label(ax, "x", **label_kwds) 
    # ===============================================================

    return grid.reshape((n_rows, n_cols))

def adjust_label(ax, which="x", max_lines=2, factor=0.95, 
                 suffix="....", fig=None):

    '''
    Adjust label to stay within defined length.
    
    Parameters
    ----------
    ax : Matplotlib axis object
        Predefined Matplotlib axis.
        
    which : {"x", "y"}, default="x"
        Which axis to change label.
    
    max_lines : int, default=2
        Maximum number of lines.
        
    factor : positive float, default=0.95
        Positive number that is used to scale either width of height 
        of `ax` to determine the box boundary.
    
    suffix : str, default="...."
        String that is added to the end of last line.
    
    fig : Figure, default=None
        The Figure that `ax` is built in. If None, it defaults to 
        current figure (plt.gcf()).
        
    Returns
    -------
    ax : Matplotlib axis object
        Matplotlib axis.
        
    '''
    if fig is None: fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    Bbox_axis_ = ax.get_window_extent(renderer=renderer)
    Bbox_label = getattr(ax, f"{which}axis").get_label()
    Bbox_label = Bbox_label.get_window_extent(renderer=renderer)
    name = "width" if which=="x" else "height"
    denom = getattr(Bbox_axis_, name) * factor

    if denom > 0:
        ratio  = getattr(Bbox_label, name) / denom
        label_ = list(getattr(ax,f"get_{which}label")())
        length = int(np.floor(len(label_)/ratio))
        
        # Maximum line of texts.
        n_lines = int(np.ceil(ratio))
        max_lines = int(min(n_lines, max_lines))

        new_label = []
        for n in range(max_lines):
            t = "".join(label_[n*length:(n+1)*length])
            if (length==len(t)) & (n==max_lines-1) & (n_lines > max_lines):
                new_label += [t[:-len(suffix)] + suffix]
            else: new_label += [t]   
        getattr(ax, f"set_{which}label")("\n".join(new_label))
    return ax

def cluster_radar(X, y=None, ax=None, q=50, colors=None, labels=None, 
                  fill_kwds=None, plot_kwds=None, labels_format=None,
                  plot_order=None, tight_layout=True):

    '''
    Plot a cluster radar chart. X is normalized.
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input sample.
        
    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None,
        an array of single class is used instead.

    ax : Matplotlib axis object, default=None
        Predefined PolarAxesSubplot. If None, ax is created 
        with default figsize.

    q : float, default=50
        Percentile to compute, which must be between 0 and 100 
        e.g. If `q` is 70, that means values (normalized) from 
        all classes will be determined at 70th-percentile. 
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to
        number of classes. If None, it uses default colors 
        from Matplotlib. 

    labels : list, default: None
        A sequence of strings providing the labels for each 
        class. If None, 'Cluster {n+1}' is assigned, where n 
        is the class in y.
        
    fill_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.fill".
    
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    labels_format : string formatter, default=None
        String formatters (function) for ax.set_xticklabels 
        values. If None, it defaults to "{:,.2f}".format.

    plot_order : list of int, default=None
        List of class order to be plotted. 
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
        
    Returns
    -------
    ax : Matplotlib axis object

    '''
    # Check number of columns.
    if X.shape[1]<=2:
        raise ValueError(f"Number of columns must be greater " 
                         f"than 2. Got {X.shape[1]} instead.")
    
    # Create matplotlib.axes if ax is None.
    if ax is None: 
        fig = plt.figure(figsize=(6.5, 6))
        ax  = plt.subplot(polar=True)
    
    # Default value: y
    y = (np.zeros(len(X)) if y is None else y).astype(int)
    
    # Default : colors
    unique = np.unique(y)
    colors = ([ax._get_lines.get_next_color() for _ in unique] 
              if colors is None else colors)
    
    # Default : plot_order
    plot_order = np.array(unique if plot_order 
                          is None else plot_order)
    
    # Default : labels
    labels = ([f"Cluster {n}" for n in plot_order+1] 
              if labels is None else labels)
    
    # Default : xticklabels format
    labels_format = ("{:,.2f}".format if labels_format 
                     is None else labels_format)
    
    # To display all variables, particularly those that 
    # have difference in scale, we normalize `X`. This 
    # makes `X` stays within 0 and 1, which allows 
    # comparison possible across variables and classes.
    a_min, a_max = np.nanpercentile(X.values, q=[0,100], axis=0)
    norm_X = ((X.values - a_min) / 
              np.where((a_max-a_min)==0, 1, a_max-a_min))

    # Angle of plots.
    angles = [n/float(X.shape[1])*2*np.pi 
              for n in range(norm_X.shape[1])]
    angles+= angles[:1]
    angles = np.array(angles)

    # If you want the first axis to be on top.
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
     
    # Draw one axis per variable and add ticklabels. 
    ax.set_xticks(angles[:-1])
    xticklabels = [f"{f}\n(" + ", ".join((labels_format(v0), 
                                          labels_format(v1))) + ")"
                   for f,v0,v1 in zip(X.columns, a_min, a_max)]  
    ax.set_xticklabels(xticklabels, color='#3d3d3d', fontsize=12)
    
    # Set alignment of ticks.
    for n,t in enumerate(ax.get_xticklabels()):
        if (0<angles[n]<np.pi): t._horizontalalignment = 'left'
        elif (angles[n]>np.pi): t._horizontalalignment = 'right'
        else: t._horizontalalignment = 'center'

    for n,c in enumerate(plot_order):
        
        values = np.nanpercentile(norm_X[y==c], q, axis=0).tolist()
        values+= [values[0]]
        
        # ax.plot
        kwds = {'linewidth' : 2.5, 
                'color'     : colors[n], 
                "label"     : labels[n],
                "zorder"    : 2*(n + 1) - 1,
                "solid_capstyle":'round', 
                "solid_joinstyle":"round"}
        ax.plot(angles, values, **(kwds if plot_kwds is None 
                                   else {**kwds,**plot_kwds}))

        kwds = {'alpha':0.3, 'color':colors[n], "zorder":2*(n+1)}
        ax.fill(angles, values, **(kwds if fill_kwds is None 
                                   else {**kwds,**fill_kwds}))
        
    # Remove lines for radial axis (y)
    ax.set(yticks=[], yticklabels=[])
    ax.yaxis.grid(False)    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)
    
    # Keyword arguments for annotations.
    default = dict(textcoords='offset points', fontsize=12, color="grey", 
                   xytext=(10, 10), va="center", ha="left", zorder=-1, 
                   bbox=dict(facecolor="w", pad=0.2, 
                             edgecolor='none', boxstyle="round"),
                   arrowprops = dict(arrowstyle = "-", color="#cccccc"))
    
    # Draw minor grid along with annotations.
    for v in np.linspace(y_min, y_max, 5)[:-1]:
        kwds = dict(color="#cccccc", lw=0.8, zorder=-1)
        ax.plot(angles, np.full(len(angles),v), **kwds)
        ax.annotate("{:,.2f}".format(v) , (0, v), **default ) 
        
    # Draw perimeter.
    ax.plot(angles, np.full(len(angles), y_max), 
            color="grey", lw=2, zorder=-1)       
    ax.xaxis.grid(True, color="#cccccc", lw=0.8, zorder=-1)

    # Remove spines
    ax.spines["polar"].set_color("none")
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax = adjust_legend(ax, len(labels))
    
    if tight_layout: plt.tight_layout()
        
    return ax

def bounding_box(obj):
    
    '''Private function: Get bounding box'''
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    return (obj.get_window_extent(renderer=renderer)
            .transformed(plt.gca().transAxes))
    
def adjust_legend(ax, n_labels):

    '''Private function: Relocate legend to the bottom'''
    # Get (x0,y0), and (x1,y1) of all text boxes.
    ax_bbox = bounding_box(ax)
    x_, y_ = [], []
    for t in ax.get_xticklabels():
        t_bbox = bounding_box(t)
        x_.append([t_bbox.x0, t_bbox.x1])
        y_.append([t_bbox.y0, t_bbox.y1])

    # Determine maximum width.
    x_pos, y_pos = np.r_[x_], np.r_[y_]
    x_min, x_max = np.percentile(x_pos.ravel(), q=[0,100])
    max_width = (x_max - x_min)*0.9

    # Find height of textbox in axis coordinates.
    y0, y1 = y_pos[np.argmin(y_pos[:,0]), :]
    ax_height = ax_bbox.height
    t0_height = (y1 - y0)/ax_height
    new_y0    = (y0-ax_bbox.y0)/ax_height

    # Determine `ncol` that is best fit the legend.
    best_cols = 1
    kwds = dict(edgecolor="none" , borderaxespad=0.25, 
                markerscale =1.00, columnspacing=0.30, 
                labelspacing=0.70, handletextpad=0.50, 
                prop={"size":12} , loc='center')

    for c in range(1, n_labels+1):
        kwds.update({"ncol":c})
        legend = bounding_box(ax.legend(**kwds)) 
        gap = np.ceil(n_labels/c) * c - n_labels
        if (gap < 2) & (legend.width < max_width):
            best_cols = c

    kwds.update({"ncol": best_cols})
    # bbox_to_anchor=(1.05, 1)
    legend = ax.legend(**kwds) 
    legend_bbox = bounding_box(legend)
    dy = (legend_bbox.y1 - legend_bbox.y0)/ax_height
    dy = new_y0 - (0.5 * dy + t0_height)
    legend.set_bbox_to_anchor([0.5, dy], transform=ax.transAxes)
    
    return ax