'''
Available methods are the followings:
[1] cluster_pie
[2] cluster_hist
[3] cluster_scatter
[4] cluster_matrix
[5] cluster_radar
[6] create_cmap
[7] matplotlib_cmap

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 22-07-2021

'''
import pandas as pd, numpy as np, math
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d
from scipy import (stats, linalg)

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1 import Grid
from itertools import product

__all__  = ["matplotlib_cmap", "create_cmap",
            "cluster_pie", 
            "cluster_hist", 
            "cluster_scatter",
            "cluster_matrix",
            "cluster_radar"]

def matplotlib_cmap(name='viridis', n=10):

    '''
    Parameters
    ----------
    name : matplotlib Colormap or str, default='viridis'
        Name of a colormap known to Matplotlib. 
    
    n : int, defualt=10
        Number of shades for defined color map.
    
    Returns
    -------
    colors : list of color-hex
        List of color-hex codes from defined Matplotlib
        Colormap. Such list contains "n" shades.
        
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    colors = [c_hex % (c[i,0],c[i,1],c[i,2]) 
              for i in range(n)]
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
    labels = ([f"Cluster ({n})" for n in unq+1] 
              if labels is None else labels)
    
    kwds = dict(explode = (unq_count==max(unq_count)).astype(int)*0.1,
                colors  = ([ax._get_lines.get_next_color() for _ in unq] 
                           if colors is None else colors), 
                labels  = ['{}\n({:,d})'.format(*v) for 
                           v in zip(labels, unq_count)], 
                autopct = "{:,.1f}%".format, 
                shadow  = True, 
                startangle=90, labeldistance=1.3,
                wedgeprops = dict(edgecolor="#2f3640", lw=2), 
                textprops  = dict(fontsize=14, ha="center", 
                                  fontweight=500, color="#2f3640"))
    ax.pie(unq_count, **(kwds if pie_kwds is 
                         None else {**kwds,**pie_kwds}))
    ax.axis('equal')
    if tight_layout: plt.tight_layout()
    return ax

def cluster_hist(x, y=None, ax=None, labels=None, colors=None, 
                 fill_kwds=None, plot_kwds=None, 
                 tight_layout=True, bins="fd", sigma=0.5, 
                 whis=1.5, plot_order=None):
    
    '''
    Plot a cluster Kernal Density Estimation (KDE) chart.
    
    Parameters
    ----------
    x : 1D-array or pd.Series
        Sample input. All elements must be finite, i.e. no 
        NaNs or infs. 

    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None,
        an array of single class is used instead.
    
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
    
    fill_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.fill_between".
    
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
        
    bins : int or str, default="fd"
        Number of bins (np.histogram_bin_edges) [1].
    
    sigma : float, default=0.5
        Standard deviation for Gaussian kernel. Sigma must 
        be greater than 0. The higher the sigma the smoother
        the probability density curve (PDF)
        
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond 
        the first and third quartiles, which are Q1 - whis*IQR, 
        and Q3 + whis*IQR, respectively. This applies to both
        coordinates and lower and upper bounds accordingly.
    
    plot_order : list of int, default=None
        List of class order to be plotted.
    
    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/
           numpy.histogram_bin_edges.html#numpy.histogram_
           bin_edges
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(5, 4))[1]
    
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
    labels = ([f"Cluster ({n})" for n in plot_order+1] 
              if labels is None else labels)
    
    # Determine bins.
    bins   = np.histogram(x, bins)[1]
    xticks = bins[1:] + np.diff(bins)
    
    for n,c in enumerate(plot_order):
        
        # 1-D Gaussian filter (smoothen density curve)
        density = np.histogram(x[y==c], bins, density=True)[0]
        pdf = gaussian_filter1d(density, sigma)

        # Density plot (ax.fill_between)
        kwds = dict(color=colors[n], label=labels[n], alpha=0.3)
        if fill_kwds is not None: kwds.update(fill_kwds)
        ax.fill_between(xticks, pdf, **kwds)
        
        # Density line (ax.plot)
        kwds = dict(color=colors[n], lw=2.5)
        ax.plot(xticks, pdf, **({**kwds, **plot_kwds} 
                                if plot_kwds is not None else kwds))
    if len(plot_order)>1: ax.legend(loc=0, fontsize=11)
    ax.set_ylim(*np.array(ax.get_ylim())*[[1,1/0.85]])
    ax.set_xlim(__IQR__(x, whis, *ax.get_xlim()))
    if tight_layout: plt.tight_layout()
    return ax

def cluster_scatter(x1, x2, y=None, ax=None, labels=None, 
                    colors=None, scatter_kwds=None, 
                    tight_layout=True, whis=1.5, 
                    frac=1, random_state=0,
                    use_kde=False, cmap=None, plot_order=None):
    
    '''
    Plot a cluster scatter chart.
    
    Parameters
    ----------
    x1, x2 : 1D-array or pd.Series
        The horizontal, and vertical coordinates of the data 
        points, respectively. All elements must be finite, 
        i.e. no NaNs or infs. 
    
    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None,
        an array of single class is used instead.
    
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
    
    scatter_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.scatter".

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
              
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond 
        the first and third quartiles, which are Q1 - whis*IQR, 
        and Q3 + whis*IQR, respectively. This applies to both
        coordinates and lower and upper bounds accordingly.
    
    frac : float, default=1
        Fraction of items to be plotted.
        
    random_state : int, default=0
        Seed for random number generator to randomize samples
        to be plotted.
    
    use_kde : bool, default=False
        If True, a kernel-density estimate using Gaussian 
        kernels is used, otherwise scatter plots [1].
    
    cmap : str or Colormap, default=None
        A Colormap instance e.g. cm.get_cmap('Reds',20) or 
        registered colormap name. This is relevant when use_kde 
        is True [2]. If None, it defaults to "Blues".
        
    plot_order : list of int, default=None
        List of class order to be plotted.
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.stats.gaussian_kde.html
    .. [2] https://matplotlib.org/stable/tutorials/colors/
           colormaps.html
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(5, 4))[1]
    
    # Default value: y
    y = (np.zeros(len(x1)) if y is None else y).astype(int)
    
    # Default : colors
    unique = np.unique(y)
    colors = ([ax._get_lines.get_next_color() for _ in unique] 
              if colors is None else colors)
    
    # Default : plot_order
    plot_order = np.array(unique if plot_order 
                          is None else plot_order)
    
    # Default : labels
    labels = ([f"Cluster ({n})" for n in plot_order+1] 
              if labels is None else labels)
 
    # Keep x1, x1, and y given plot_order
    plot_index = np.isin(y, plot_order)
    x1 = np.array(x1)[plot_index]
    x2 = np.array(x2)[plot_index]
    y  = np.array(y)[plot_index]
    
    # Randomize x1, x2, and y (if not None)
    indices= __indices__(len(x1), frac, random_state)
    Select = lambda x, index: np.array(x).ravel()[indices]
    x1, x2 = Select(x1, indices), Select(x2, indices), 
    y = Select(y, indices)

    # Compute kernel-density estimate using Gaussian kernels
    if use_kde==True:
        
        if cmap is None: cmap = 'Blues' 
        if isinstance(cmap, str): cmap = cm.get_cmap(cmap,50)
    
        xx = np.vstack((x1, x2))
        kwds = dict(c=gaussian_kde(xx)(xx), cmap=cmap,
                    marker='s', alpha=0.8, s=10)
        ax.scatter(x1, x2, **({**kwds, **scatter_kwds} 
                              if scatter_kwds 
                              is not None else kwds))
    # Scatter plot
    elif use_kde==False:
        
        kwds = dict(fc="none", marker='s', alpha=0.8, s=10)
        for n,c in enumerate(plot_order):
            kwds.update(dict(ec=colors[n], label=labels[n]))
            ax.scatter(x1[(y==c)], x2[(y==c)], 
                       **({**kwds, **scatter_kwds} 
                          if scatter_kwds 
                          is not None else kwds))
    
    # Set limits of coordinates.
    ax.set_xlim(__IQR__(x1, whis, *ax.get_xlim()))
    ax.set_ylim(__IQR__(x2, whis, *ax.get_ylim()))
    if (not use_kde) & (len(plot_order)>1): 
        ax.legend(loc=0, fontsize=11)
    if tight_layout: plt.tight_layout()
        
    return ax
        
def __IQR__(a, whis, a_min, a_max):
    
    '''Interquatile range'''
    Q1,Q3 = np.percentile(a, [25, 75])
    if Q1==Q3: return (a_min, a_max)
    else: return (np.fmax(Q1-whis*(Q3-Q1),a_min), 
                  np.fmin(Q3+whis*(Q3-Q1),a_max))
    
def __indices__(N, frac=1, random_state=0):
    
    '''Random indices'''
    # Select samples
    np.random.seed(random_state)
    indices = np.arange(0,N)
    if frac<1:
        kwds = dict(size=np.fmax(int(frac*N), 10), 
                    replace=False)
        select = np.random.choice(indices, **kwds)
    else: select = indices.copy()
    return np.isin(indices, select)
    
def cluster_matrix(X, y=None, colors=None, whis=1.5, 
                   plot_order=None, off_diagonal="scatter",
                   hist_kwds=None, scatter_kwds=None, 
                   n_limit=1000, figsize=None, 
                   show_corr=True):

    '''
    Draw a matrix of scatter plots towards a pairwise 
    comparison to observe interaction of variables. NaN is 
    ignored. Each numeric variable in data will be shared 
    across a single row and the x-axes across a single column

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input sample.
        
    y : 1D-array or pd.Series of int, default=None
        Array of cluster labels (0 to n_clusters-1). If None,
        an array of single class is used instead. This 
        overrides hist_kwds and scatter_kwds.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to
        number of classes. If None, it uses default colors 
        from Matplotlib. This overrides hist_kwds and 
        scatter_kwds.
        
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond 
        the first and third quartiles, which are Q1 - whis*IQR, 
        and Q3 + whis*IQR, respectively. This applies to both
        coordinates and lower and upper bounds accordingly.
        This overrides hist_kwds and scatter_kwds.
        
    plot_order : list of int, default=None
        List of class order to be plotted. This overrides 
        hist_kwds and scatter_kwds.
         
    off_diagonal : {"scatter", "kde", "both"}, default="scatter"
        Pick between "kde", "scatter", and "both" for either 
        Kernel Density Estimation or scatter or both in the 
        off-diagonal. This overrides scatter_kwds.
   
    hist_kwds : keywords, default=None
        Keyword arguments to be passed to "cluster_hist".
    
    scatter_kwds : keywords, default=None
        Keyword arguments to be passed to "cluster_scatter".
        
    n_limit : int, default=1000
        Number of instances to be plotted in the off-diagonal 
        plots. n_limit should be less than len(X).
        
    figsize : (float,float), default=None
        A tuple (width, height) in inches. If None, a default
        figsize is applied i.e. (n_rows*1.3, n_cols*1.3).
    
    show_corr : bool, default=True
        If True, it shows a Pearson correlation coefficient for
        every pariwise plot (off-diagonal).
    
    Returns
    -------
    grid : numpy.ndarray, of shape (n_rows, n_cols)
        A matrix of plots.

    '''
    # A grid of Axes (mpl_toolkits.axes_grid1).
    n_rows, n_cols = (X.shape[1],)*2
    figsize = (n_rows*1.3, n_cols*1.3) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
    grid = np.array(Grid(fig, rect=111, nrows_ncols=(n_rows, n_cols), 
                         share_x=False, share_y=False, axes_pad=0))
    
    # Fraction, Pairs of variables, and axes.
    frac, columns = np.fmin(n_limit/len(X),1), X.columns
    pairs = zip(list(product(columns, columns)), grid, 
                product(range(n_rows), range(n_cols)))
    
    y = (np.zeros(len(X)) if y is None else y).astype(int)

    for (var1, var2), ax, (r,c) in pairs:
        
        notna = (X[[var1, var2]].notna().sum(axis=1)==2).values
        
        if notna.sum()>0:
            
            x1 = X.loc[notna, var1].copy()
            x2 = X.loc[notna, var2].copy()
            y0 = y[notna].copy()
            
            # Diagonal plot
            if var1==var2:
                
                default   = {"plot_kwds": dict(lw=1)}
                hist_kwds = (default if hist_kwds is None 
                             else {**default, **hist_kwds})
                
                kwds = dict(y=y, ax=ax, colors=colors, whis=whis, 
                            plot_order=plot_order, tight_layout=False)
                ax = cluster_hist(x1, **{**hist_kwds, **kwds})
                ax.legend().set_visible(False)

            # Off-diagonal plot
            elif var1!=var2: 
                
                default = {"scatter_kwds": dict(s=5, alpha=0.5)}
                scatter_kwds = (default if scatter_kwds is None 
                                else {**default, **scatter_kwds})
                    
                use_kde = (True if off_diagonal=="kde" else False)
                if (off_diagonal=="both") & (c>r): use_kde = False
                elif (off_diagonal=="both") & (c<r): use_kde = True   
                    
                kwds = dict(y=y0, ax=ax, frac=frac, whis=whis,
                            colors=colors, plot_order=plot_order, 
                            tight_layout=False, use_kde=use_kde)
                    
                ax = cluster_scatter(x2, x1, **{**scatter_kwds, **kwds})
                
                if ax.get_legend() is not None:
                    ax.legend().set_visible(False)
                
                if show_corr:
                    corr, pvalue = stats.pearsonr(x1, x2)
                    ax.text(0.95, 0.06, ('%.2f' % corr), size=11, 
                            transform=ax.transAxes, ha="right",
                            bbox=dict(boxstyle='square', alpha=0.8, 
                                      facecolor='white', edgecolor='none'))
                    
            ax.set(xticks=[], yticks=[])  
            if (c==0) & (r==n_rows-1):
                ax.set_ylabel(var1, fontsize=11)
                ax.set_xlabel(var2, fontsize=11)
            elif (c==0):
                ax.set_ylabel(var1, fontsize=11)
            elif r==n_rows-1:
                ax.set_xlabel(var2, fontsize=11)
            
    plt.tight_layout(pad=0)
    
    return grid.reshape((n_rows, n_cols))

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
    # Create matplotlib.axes if ax is None.
    if ax is None: 
        fig = plt.figure(figsize=(6,6))
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
    labels = ([f"Cluster ({n})" for n in plot_order+1] 
              if labels is None else labels)
    
    # Default : xticklabels format
    labels_format = ("{:,.2f}".format if labels_format 
                     is None else labels_format)
    
    # To display all variables, particularly those that 
    # have difference in scale, we normalize `X`. This 
    # makes `X` stays within 0 and 1, which allows 
    # comparison possible across variables and classes.
    a_min, a_max = np.nanpercentile(X.values, q=[0,100], axis=0)
    norm_X = (X.values - a_min)/ np.where((a_max-a_min)==0, 
                                          1, a_max-a_min)

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
    xticklabels = [f"{f}\n(" + " ".join((labels_format(v0), 
                                         labels_format(v1))) + ")"
                   for f,v0,v1 in zip(X.columns, a_min, a_max)]  
    ax.set_xticklabels(xticklabels, color='#3d3d3d', fontsize=11)

    # Set alignment of ticks.
    for n,t in enumerate(ax.get_xticklabels()):
        if (0<angles[n]<np.pi): t._horizontalalignment = 'left'
        elif (angles[n]>np.pi): t._horizontalalignment = 'right'
        else: t._horizontalalignment = 'center'

    for n,c in enumerate(plot_order):
        
        values = np.nanpercentile(norm_X[y==c], q, axis=0).tolist()
        values+= [values[0]]
        
        # ax.plot
        kwds = {'lw':2.5,'color':colors[n],"label":labels[n]}
        ax.plot(angles, values, **(kwds if plot_kwds is None 
                                   else {**kwds,**plot_kwds}))

        kwds = {'alpha':0.4, 'color':colors[n]}
        ax.fill(angles, values, **(kwds if plot_kwds is None 
                                   else {**kwds,**fill_kwds}))
        
    # Remove lines for radial axis (y)
    ax.set(yticks=[], yticklabels=[], ylim=(0,1))
    ax.yaxis.grid(False)
    
    for v in np.arange(0.25, 1.25, 0.25):
        kwds = dict(color='grey', lw=1, ls='--')
        if v==1: kwds.update(dict(lw=3, ls='-'))
        ax.plot(angles, np.full(len(angles),v), **kwds)
    ax.xaxis.grid(True, color='grey', lw=1, ls='--')

    # Remove spines
    ax.spines["polar"].set_color("none")
    
    ax.legend(fontsize=11, bbox_to_anchor=(0,0))
    if tight_layout: plt.tight_layout()
        
    return ax