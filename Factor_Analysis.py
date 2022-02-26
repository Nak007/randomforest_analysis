'''
Available methods are the followings:
[1] FactorRotation
[2] PrincipalComponents
[3] Cal_Bartlett
[4] Cal_KMOScore 
[5] Cal_PartialCorr 
[6] Cal_SMC

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 22-07-2021

'''
import pandas as pd, numpy as np, math
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d
from scipy import (stats, linalg)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.ticker import (FixedLocator, 
                               FixedFormatter, 
                               StrMethodFormatter,
                               FuncFormatter)
from itertools import product
import collections

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__  = ["FactorRotation", 
            "PrincipalComponents", 
            "Cal_Bartlett", 
            "Cal_KMOScore", "KMO_table",
            "Cal_PartialCorr", 
            "Cal_SMC"]

class FactorRotation():
    
    '''
    FactorRotation provides statistical information towards 
    interpretation of factor loadings, specific variances, and 
    communalities from "factor-analyzer" [1].
   
    Atrributes
    ----------
    loadings : pd.DataFrame
        The correlation coefficient between variables and factors.
    
    communalities : pd.DataFrame
        The elements are squared loadings, which represent the 
        variance of each item that can be explainded by the 
        corresponding factor. 

    variances : pd.DataFrame
        Total Variance Explained of initial eigenvalues (PCA), and 
        after extraction. 

    common_variances : pd.DataFrame
        Correlation or variance between variables that can be 
        explained by factors. Communalities are on the diagonal. 
        Value closer to 1 suggests that factors explain more of the 
        variance of variables [1].

    unique_variances : pd.DataFrame
        Unique variance is comprised of specific variances (diagonal) 
        and residual variances (off-diagonal). These variances can not 
        be explained by factors [1]. 

    rmsr : float
        According to the model assumptions stating that specific 
        factors are uncorrelated with one another, cov(ϵi,ϵj) = 0 for 
        i ≠ j, the off-diagonal elements from unique_variances should 
        be small towards zeros, which can be measured by Root-Mean-
        Square Residual (RMSR) [1].

    References
    ----------
    .. [1] https://factor-analyzer.readthedocs.io/en/latest/factor_
           analyzer.html#factor-analyzer-analyze-module
    .. [2] https://online.stat.psu.edu/stat505/book/export/html/691

    '''
    def __init__(self):
        pass
        
    def fit(self, estimator, features=None):
        
        '''
        Fit model.
        
        Parameters
        ----------
        estimator : estimator object.
            An estimator of "factor-analyzer" or other that implements 
            the similar interface [1].
        
        features : list of str, default=None
            A list of features. If None, features default to ["X1","X2", 
            ...,"Xn"] where n is the number of features according to 
            "loadings".
            
        Atrributes
        ----------
        loadings : pd.DataFrame
            The correlation coefficient between variables and factors.

        communalities : pd.DataFrame
            The elements are squared loadings, which represent the 
            variance of each item that can be explainded by the 
            corresponding factor. 

        variances : pd.DataFrame
            Total Variance Explained of initial eigenvalues (PCA), and 
            after extraction. 

        common_variances : pd.DataFrame
            Correlation or variance between variables that can be 
            explained by factors. Communalities are on the diagonal. 
            Value closer to 1 suggests that factors explain more of the 
            variance of variables [1].

        unique_variances : pd.DataFrame
            Unique variance is comprised of specific variances (diagonal) 
            and residual variances (off-diagonal). These variances can not 
            be explained by factors [1]. 

        rmsr : float
            According to the model assumptions stating that specific 
            factors are uncorrelated with one another, cov(ϵi,ϵj) = 0 for 
            i ≠ j, the off-diagonal elements from unique_variances should 
            be small towards zeros, which can be measured by Root-Mean-
            Square Residual (RMSR) [1].
        
        References
        ----------
        .. [1] https://factor-analyzer.readthedocs.io/en/latest/factor_
               analyzer.html#factor-analyzer-analyze-module
        .. [2] https://online.stat.psu.edu/stat505/book/export/html/691
        
        '''
        # Attributes from estimator
        n_factors = estimator.n_factors
        loadings  = estimator.loadings_
        corr_     = estimator.corr_
        n_features= len(corr_)
 
        factors = [f"F{n}" for n in range(1, n_factors+1)]
        if features is not None: features = list(features)
        else: features = [f"X{n+1}" for n in range(n_features)]
    
        # Initial and extracted variances
        columns = ["Total", "% Variance", "Cumulative %"]
        columns = list(product(["Initial","Extraction"], columns))
        columns = pd.MultiIndex.from_tuples(columns)
        index   = pd.Index([f"F{n+1}" for n in range(n_features)])
        
        initial = estimator.get_eigenvalues()[0]  
        initvar = initial / n_features
        initial = np.vstack((initial, initvar, np.cumsum(initvar))).T
        extract = np.vstack((np.vstack(estimator.get_factor_variance()).T,
                             np.full((n_features-n_factors,3), [np.nan]*3)))
        self.variances = pd.DataFrame(np.hstack((initial, extract)), 
                                      columns=columns, index=index)
    
        # Communality (common variance), and 
        # Uniqueness (specific variance + error).
        columns =(list(product(["Extraction"], factors + 
                               ["Communality","Uniqueness"])))
        columns = pd.MultiIndex.from_tuples(columns)
    
        commu = (loadings**2).sum(axis=1, keepdims=True)
        data  = np.hstack((loadings**2, commu, 1-commu))
        data  = np.vstack((data, data.sum(0, keepdims=True)))
        self.communalities = pd.DataFrame(data, columns=columns,
                                          index=features + ["Total"])
                                          
        # Correlation and Cov(e(i),e(j)) given n_factors
        kwds = dict(columns=features, index=features)
        corr = loadings.dot(loadings.T)
        self.common_variances = pd.DataFrame(corr, **kwds)
        self.unique_variances = corr_ - self.common_variances
            
        # Calculate Root Mean Square Residual (RMSR)
        # res(i,j) = Cov(e(i),e(j)) = 0
        off_diag = ~np.diag(np.full(n_features, True))
        errors = sum((corr_-corr)[off_diag]**2)
        denom  = n_features*(n_features-1)
        self.rmsr = np.sqrt(errors/denom)
        self.loadings = pd.DataFrame(loadings, 
                                     columns=factors, 
                                     index=features)
                                     
        return self
    
    def plotting(self, value=None, ax=None, cmap=None, pcolor_kwds=None, 
                 anno_kwds=None, anno_format=None, tight_layout=True):
        
        '''
        Plot chart of eigenvectors or correlations.

        Parameters
        ----------
        value : {"loading", "common", "unique"}
            Data input to be used in the functon. If None, it defaults to 
            "loading".
            
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with default 
            figsize.

        cmap : str or Colormap, default=None
            A Colormap instance e.g. cm.get_cmap('Reds',20) or registered 
            colormap name. If None, it defaults to "Blues". This overrides 
            `pcolor_kwds`.

        pcolor_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.pcolor".
            
        anno_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.annotate".
            
        anno_format : string formatter, default=None
            String formatters (function) for ax.annotate values. If None, 
            it defaults to "{:,.2f}".format.
            
        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        params = {"loading": ("loadings", 
                              r"Correlations, $\hat{L}=corr(X,PC)$"), 
                  "common" : ("common_variances", 
                              r"Correlations, $\hat{L}=corr(X,PC)$"), 
                  "unique" : ("unique_variances", 
                              r"Residual variances, $cov(e_{i},e_{j})$")}
        
        value = "loading" if value is None else value
        data, title = params[value]
        data = getattr(self, data)
        data = data.reindex(index=data.index[::-1])
        n_rows, n_cols = data.shape
 
        # Create matplotlib.axes if ax is None.
        if ax is None: 
            ax = plt.subplots(figsize=(n_cols, n_rows-.5))[1]
        
        # Matplotlib Colormap
        if cmap is None: cmap = 'Blues' 
        if isinstance(cmap, str): cmap = cm.get_cmap(cmap,50)
        
        abs_val = abs(data).values.ravel()
        kwds = dict(edgecolors='#4b4b4b', lw=1, alpha=0.8, 
                    vmin=min(abs_val), vmax=max(abs_val))
        kwds = (kwds if pcolor_kwds is None 
                else {**kwds, **pcolor_kwds})
        ax.pcolor(abs(data), **{**kwds,**{"cmap":cmap}})

        # Annotation.
        anno_xy = [[m+0.5,n+0.5] for n in range(n_rows) 
                   for m in range(n_cols)]
        anno_format = ("{:,.2f}".format if anno_format 
                       is None else anno_format)
        kwds = dict(xytext =(0,0), textcoords='offset points', 
                    va='center', ha='center', fontsize=12, 
                    fontweight='demibold', color="Black")
        kwds = (kwds if anno_kwds is None else {**kwds,**anno_kwds})
        
        for xy, v in zip(anno_xy, data.values.ravel()): 
            ax.annotate(anno_format(v), xy, **kwds)

        ax.tick_params(tick1On=False)
        ax.set_xticks(np.arange(0.5, n_cols))
        ax.set_xticklabels(data.columns)
        
        ax.set_yticks(np.arange(0.5, n_rows))
        ax.set_yticklabels(data.index)
        ax.set_title(title, fontsize=14)
        if tight_layout: plt.tight_layout()
            
        return ax

class PrincipalComponents:
    
    '''
    PrincipalComponents performs dimension reduction algorithm 
    so-called Principal Component Analysis (PCA) on the correlation 
    of X.
    
    Parameters
    ----------
    mineigval : float, default=1
        Minimum value of eigenvalues when choosing number of 
        prinicpal components. The algorithm chooses factor, whose 
        eigenvalue is more than mineigval. Only available when
        method is either "eigval" or None.

    minprop : float, default=0.8
        Minimum proportion of variation explained when choosing 
        number of prinicpal components. The algorithm select a
        group of factors (ranked from highest to lowest by 
        eigenvalues), whose sum of variation explained is greater 
        than or equal to minprop.
        
    method : {"eigval", "varprop"}, default=None
        If "eigval", mineigval is selected as a threshold, whereas 
        "varprop" uses minprop. If None, a maximum number of 
        factors is selected between two methods.
        
    Attributes
    ----------
    eigvals : ndarray of shape (n_components,)
        The variance that get explained by factors.

    eigvecs : pd.DataFrame of shape (n_features, n_components)
        Eigenvectors (or factors) are vectors whose direction 
        remain unchanged when a linear transformation is applied. 
        They represent the directions of maximum variance. The 
        factors are sorted by eigvals.

    variances : pd.DataFrame
        Variance that can be explained by a given factor. Starting 
        from the first factor, each subsequent factor is obtained 
        from partialling out the previous factor. Therefore the 
        first factor explains the most variance, and the last factor 
        explains the least [1].

    communalities : pd.DataFrame
        The communality is the sum of the squared component loadings 
        up to the number of components that gets extracted [1].

    References
    ----------
    .. [1] https://stats.idre.ucla.edu/spss/seminars/efa-spss/
   
    '''
    def __init__(self, minprop=0.8, mineigval=1.0, method=None):
    
        self.minprop = minprop
        self.mineigval = mineigval
        self.method = method
    
    def fit(self, X):
        
        '''
        Fit PCF model.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        Atrributes
        ----------
        eigvals : ndarray of shape (n_components,)
            The variance that get explained by factors
        
        eigvecs : pd.DataFrame of shape (n_features, n_components)
            Eigenvectors (or factors) are vectors whose direction 
            remain unchanged when a linear transformation is applied. 
            They represent the directions of maximum variance. The 
            factors are sorted by eigvals.
            
        variances : pd.DataFrame
            Variance that can be explained by a given factor. Starting 
            from the first factor, each subsequent factor is obtained 
            from partialling out the previous factor. Therefore the 
            first factor explains the most variance, and the last factor 
            explains the least [1].
            
        communalities : pd.DataFrame
            The communality is the sum of the squared component loadings 
            up to the number of components that gets extracted [1].
            
        References
        ----------
        .. [1] https://stats.idre.ucla.edu/spss/seminars/efa-spss/
 
        '''
        if isinstance(X, pd.DataFrame):
            if X.shape[1]<=2:
                raise ValueError(f'n_features must be greater than 2.' 
                                 f'Got {X.shape[1]} instead.')
        else: raise TypeError(f'Data must be pd.DataFrame.' 
                              f' Got {type(X)} instead.')
        
        # Create PC columns
        width = int(np.ceil(np.log(X.shape[1])/np.log(10)))
        self.components = ["PC{}".format(str(n).zfill(width)) 
                           for n in range(1,X.shape[1]+1)]
        
        # Standardize X
        self.mean = np.mean(X.values, axis=0)
        self.stdv = np.std(X.values, axis=0)
        std_X = ((X.values-self.mean)/self.stdv).copy()
        
        # Correlation matrix
        corr = pd.DataFrame(std_X).corr().values
        
        # Eigenvalues, Eigenvectors, and loadings
        eigvals, eigvecs = np.linalg.eigh(corr)
        indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[indices].real
        self.eigvecs = pd.DataFrame(eigvecs[:,indices].real, 
                                    columns=self.components, 
                                    index=X.columns)
        loadings = self.eigvecs*np.sqrt(self.eigvals)
        
        # Variance explained
        varprops = self.eigvals/self.eigvals.sum()
        cumprops = np.cumsum(varprops)
        
        # Factors
        data = np.vstack((self.components, self.eigvals, 
                          varprops, cumprops))
        columns = ["Factor", "Eigenvalues", "% Variance", "Cumulative %"]
        self.variances = (pd.DataFrame(data.T, columns=columns)
                          .set_index("Factor").astype("float64"))
    
        # Recommended number of factors
        n_minprop = np.argmax((cumprops>=self.minprop))+1
        n_maxeigval = (eigvals>=self.mineigval).sum()

        if self.method=="eigval": n_factors = n_maxeigval
        elif self.method=="varprop": n_factors = n_minprop
        else: n_factors = max([n_minprop, n_maxeigval, 1])
        self.n_factors = n_factors
        
        # Communalities
        columns = list(product(["Extraction"], 
                               self.components[:n_factors] +
                               ["Communality"]))
        columns = pd.MultiIndex.from_tuples(columns)
        variances = (loadings.values**2)[:,:n_factors]
        communalities = variances.sum(axis=1, keepdims=True)
        data = np.hstack((variances, communalities))
        data = np.vstack((data, data.sum(0, keepdims=True)))
        self.communalities = pd.DataFrame(data, columns=columns, 
                                          index=list(X)+["Total"])
        
        # Data for plots
        eigvecs_ = self.eigvecs.values[:,:n_factors]
        keys = ["eigvals", "eigvecs", "varprops", "princoms", "features"]
        results = collections.namedtuple("Results", keys)
        self.results_ = results(**{"eigvals" : self.eigvals[:n_factors], 
                                   "eigvecs" : eigvecs_, 
                                   "varprops": varprops, 
                                   "princoms": std_X.dot(eigvecs_),
                                   "features": list(X)})
            
        return self

    def transform(self, X, n_factors=None):

        '''
        Apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        n_factors : int, default=None
            Number of factors. If None, it is selected according 
            to "method".
            
        Returns
        -------
        PC : pd.DataFrame, of shape (n_samples, n_comps)
            Transformed X.
        
        '''
        std_X = (X-self.mean)/self.stdv
        n_factors = self.n_factors if n_factors is None else max(n_factors,1)
        PC = np.dot(std_X, self.eigvecs.values[:,:n_factors])
        return pd.DataFrame(PC, columns=self.components[:n_factors])
    
    def fit_transform(self, X, n_factors=None):
        
        '''
        Fit X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        n_factors : int, default=None
            Number of factors. If None, it is selected according 
            to "method".
            
        Returns
        -------
        PC : pd.DataFrame, of shape (n_samples, n_comps)
            Transformed X.
        '''
        self.fit(X)
        return self.transform(X, n_factors)
    
    def plot(self, ax=None, y=None, scatter_colors=None, arrow_colors=None, 
             pc_pair=(0,1), target=None, exclude=None, proportion=0.5, 
             max_display=10, show_corr=False, whis=1.5, scatter_kwds=None, 
             tight_layout=True):
    
        '''
        Plot results.

        Parameters
        ----------
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, it uses default size, 
            figsize=(9, 5).

        y : array-like of shape (n_samples,), default=None
            The target values (class labels) in integer. If None, a single
            class is used.

        scatter_colors : array-like of shape (n_classes,), default=None
            List of color-hex must be arranged in correspond to class label
            (integer). If None, it uses default colors from Matplotlib.

        arrow_colors : array-like of shape (n_features,), default=None
            List of color-hex must be arranged in correspond to n_features. 
            If None, it uses default colors from Matplotlib.

        pc_pair : (int, int), default=(0,1)
            Pair of principal component indices for x and y axis, 
            respectively.

        target : array-like of shape (n_samples,), default=None
            Binary target i.e. target=1, and non-target=0. If provided, it 
            draws red circle around targets.

        exclude : list of str, default=None
            List of features to be excluded from the plot.

        proportion : float, default=0.5
            Minimum proportion of variation explained by two principal 
            components when choosing variables to be displayed. It selects 
            variables whose explained variance is greater than or equal to
            `proportion`.

        max_display : int, greater than 1, default=10
            Maximum number of variables to be displayed. If None, it uses 
            all variables that satisfy `propotion`.

        show_corr : bool, default=False
            If True, it shows correlation coefficients between variable,
            and principal components on x and y axes, respectively in the 
            legend, otherwise explained variance.

        whis : float, default=1.5
            It determines the reach of the whiskers to the beyond the 
            first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
            whis*IQR, respectively. This applies to both coordinates and 
            lower and upper bounds accordingly.

        scatter_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.scatter".

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots i.e. 
            plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object
    
        '''
        args = (self.results_, ax, y, scatter_colors, arrow_colors, pc_pair, 
                target, exclude, proportion, max_display, show_corr, whis, 
                scatter_kwds, tight_layout)
        return plot_pcf_base(*args)

def plot_pcf_base(Results, ax=None, y=None, scatter_colors=None, 
                  arrow_colors=None, pc_pair=(0,1), target=None, 
                  exclude=None, proportion=0.5, max_display=10, 
                  show_corr=False, whis=1.5, scatter_kwds=None, 
                  tight_layout=True):
    
    '''
    Plot results from `PrincipalComponents`.

    Parameters
    ----------
    Results : namedtuple
        self.results_ (attribute) from `PrincipalComponents` class.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, it uses default size, 
        figsize=(9, 5).

    y : array-like of shape (n_samples,), default=None
        The target values (class labels) in integer. If None, a single
        class is used.

    scatter_colors : array-like of shape (n_classes,), default=None
        List of color-hex must be arranged in correspond to class label
        (integer). If None, it uses default colors from Matplotlib.
        
    arrow_colors : array-like of shape (n_features,), default=None
        List of color-hex must be arranged in correspond to `Results.
        features`. If None, it uses default colors from Matplotlib.
        
    pc_pair : (int, int), default=(0,1)
        Pair of principal component indices for x and y axis, 
        respectively.
 
    target : array-like of shape (n_samples,), default=None
        Binary target i.e. target=1, and non-target=0. If provided, it 
        draws red circle around targets.
        
    exclude : list of str, default=None
        List of features to be excluded from the plot.
   
    proportion : float, default=0.5
        Minimum proportion of variation explained by two principal 
        components when choosing variables to be displayed. It selects 
        variables whose explained variance is greater than or equal to
        `proportion`.
        
    max_display : int, greater than 1, default=10
        Maximum number of variables to be displayed. If None, it uses 
        all variables that satisfy `propotion`.
        
    show_corr : bool, default=False
        If True, it shows correlation coefficients between variable,
        and principal components on x and y axes, respectively in the 
        legend, otherwise explained variance.
 
    whis : float, default=1.5
        It determines the reach of the whiskers to the beyond the 
        first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
        whis*IQR, respectively. This applies to both coordinates and 
        lower and upper bounds accordingly.
    
    scatter_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.scatter".
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots i.e. 
        plt.tight_layout().

    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # ===============================================================
    # Default ax and colors.  
    if ax is None: ax = plt.subplots(figsize=(9,5))[1] 
    # Number of classes.
    n_samples = len(Results.princoms[:,0])
    if y is None: y = np.full(n_samples, 0)
    else: y = np.array(y).astype(int)
    n_classes = np.unique(y).shape[0]
    # ---------------------------------------------------------------
    # Default color palette.
    features = list(Results.features)
    n_features = len(features)
    colors = np.array([ax._get_lines.get_next_color() for _ 
                       in range(max(n_features, n_classes))])
    # --------------------------------------------------------------- 
    # Check items in color list (scatter_colors).
    if (scatter_colors is not None):
        if len(scatter_colors) < n_classes:
            raise ValueError(f'Numer of scatter colors must be '
                             f'greater than or equal to {n_classes}.' 
                             f' Got {len(scatter_colors)} instead.')
    elif n_classes > 1: scatter_colors = colors[:n_classes].tolist()
    else: scatter_colors = ["grey"]
    # ---------------------------------------------------------------
    # Check items in color list (arrow_colors).
    if (arrow_colors is not None):
        if len(arrow_colors) < n_features:
            raise ValueError(f'Numer of arrow colors must be '
                             f'greater than or equal to {n_features}.' 
                             f' Got {len(arrow_colors)} instead.')
    else: arrow_colors = colors[:n_features].tolist()
    # ===============================================================  
        
    # Scatter plots
    # ===============================================================
    patches, labels, (pc1, pc2) = [], [], pc_pair
    classes = np.unique(y)
    x1, x2 = Results.princoms[:,pc1], Results.princoms[:,pc2]
    if scatter_kwds is None: scatter_kwds = dict()
    # ---------------------------------------------------------------
    for n,c in enumerate(classes):
        kwds = {"s"         : scatter_kwds.get("s", 35), 
                "marker"    : scatter_kwds.get("marker", "o"), 
                "linewidth" : scatter_kwds.get("linewidth", 0.8), 
                "alpha"     : scatter_kwds.get("alpha", 0.25),
                "edgecolor" : "none", 
                "facecolor" : scatter_colors[n]}
        ax.scatter(x1[y==c], x2[y==c], **kwds)
        legend_kwds = dict(marker=kwds["marker"], markersize=8, 
                           markerfacecolor=kwds["facecolor"], 
                           markeredgecolor=kwds["edgecolor"],
                           alpha=kwds["alpha"], color='none')
        sc1 = mpl.lines.Line2D([0],[0], **legend_kwds)

        kwds.update(dict(facecolor="none", alpha=1, 
                         edgecolor=scatter_colors[n]))
        ax.scatter(x1[y==c], x2[y==c], **kwds)
        legend_kwds = dict(marker=kwds["marker"], 
                           markerfacecolor=kwds["facecolor"], 
                           markeredgecolor=kwds["edgecolor"],
                           color='none', markersize=8)
        sc2 = mpl.lines.Line2D([0],[0], **legend_kwds)
        patches += [(sc1, sc2)]
        if n_classes==1:
            labels += ["Samples (n={:,d})".format(len(x1))]
        else: labels += ["Class {} (n={:,d})".format(c,sum(y==c))]
    # ---------------------------------------------------------------                        
    ax.set_xlim(__IQR__(x1, whis, *ax.get_xlim()))
    ax.set_ylim(__IQR__(x2, whis, *ax.get_ylim()))
    # ---------------------------------------------------------------
    # Draw circle around targets.
    if target is not None:
        kwds = dict(edgecolor="#ff3f34", facecolor="none", 
                    s=scatter_kwds.get("s", 35) * 3.5)
        sc3 = ax.scatter(x1[target==1], x2[target==1], **kwds)
        kwds = dict(marker='o', color='none', markerfacecolor='none', 
                    markersize=8, markeredgecolor="r")
        patches += [mpl.lines.Line2D([0],[0], **kwds)]
        labels  += ["Targets (n={:,d})".format(target.sum())]
    # ===============================================================

    # ===============================================================
    # Calculate factor loadings.
    eigvecs  = Results.eigvecs * np.sqrt(Results.eigvals)
    varprops = Results.varprops
    scale = min(find_scale(eigvecs[:,pc_pair], ax)) 
    commu = pow(eigvecs[:,pc_pair], 2).sum(1)
    kwds  = dict(head_length=0.8, head_width=0.7, tail_width=0.2)
    arrowstyle = mpl.patches.ArrowStyle.Simple(**kwds)
    zorders = np.array(features)[np.argsort(commu)][::-1].tolist()
    # ---------------------------------------------------------------
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.
    # FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
    if exclude is None: exclude = []
    else:
        var = set(exclude).difference(features)
        if len(var)>0:
            raise ValueError(f'Exluded features must be {features}.' 
                             f'Got {var} instead.')
    # ---------------------------------------------------------------  
    proportion = min(proportion, max(commu))
    props = commu>=proportion
    if max_display is None: max_display = len(commu)
    order = [(f,c) for f,c in zip(features,commu)]
    order = sorted(order, key=lambda x: x[1], reverse=True)
    order = np.array(order)[:max_display,0].tolist()
    vectors = eigvecs[:,pc_pair]
    # ---------------------------------------------------------------
    for n,col in enumerate(features):
        if (col not in exclude) & props[n] & (col in order):
            size = tuple(eigvecs[n,pc_pair].real * 0.85 * scale)
            kwds = dict(mutation_scale=20, arrowstyle = arrowstyle, 
                        edgecolor="grey", facecolor=arrow_colors[n], 
                        alpha=1, linewidth=0.8, 
                        zorder=zorders.index(col)+n_classes+1)
            arrow = mpatches.FancyArrowPatch((0, 0), size, **kwds)
            ax.add_patch(arrow)
            patches += [arrow]
    # ---------------------------------------------------------------
            if show_corr==False: s = "({:.0%}) ".format(commu[n])
            else: s = "({:+.2f}, {:+.2f})".format(*vectors[n])
            labels += ["{} {}".format(col, s)]
    # ---------------------------------------------------------------
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=-1)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=-1)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    # ---------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    PC_str = "PC{} ({:.0%})".format
    ax.set_xlabel(PC_str(pc1+1, varprops[pc1]), fontsize=13)
    ax.set_ylabel(PC_str(pc2+1, varprops[pc2]), fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    # ---------------------------------------------------------------
    expvar = varprops[np.unique(pc_pair)].sum()
    text = "Explained Variance = {:.0%}".format(expvar)
    props = dict(boxstyle='square', facecolor='white', alpha=0.8, 
                 edgecolor="none", pad=0.1)
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=13,
            va='bottom', ha="left", bbox=props)
    # ---------------------------------------------------------------
    legend = ax.legend(patches, labels, edgecolor="none", ncol=1,
                       borderaxespad=0.25, markerscale=1.5, 
                       columnspacing=0.3, labelspacing=0.7, 
                       handletextpad=0.5, prop=dict(size=12), 
                       loc='upper left') 
    legend.set_bbox_to_anchor([1.01,1], transform = ax.transAxes)
    # ---------------------------------------------------------------
    ax.tick_params(axis='both', labelsize=11)
    if tight_layout: plt.tight_layout()
    # ===============================================================
    
    return ax

def find_scale(xy, ax):
    
    '''Calculate minimum scale for vectors'''
    x, y = xy[:,0], xy[:,1]
    ylim = np.absolute(ax.get_ylim())
    xlim = np.absolute(ax.get_xlim())
    lims = np.array(list(product(xlim, ylim)))
    shape = (len(x), 2)
    condlist = [(x < 0) & (y < 0), (x < 0) & (y >=0),
                (x > 0) & (y < 0), (x > 0) & (y >=0)]
    choicelist = [(np.full(shape, lims[n,:]) / abs(xy)).min(1) 
                  for n in range(4)]
    return np.select(condlist, choicelist)

def __IQR__(a, whis, a_min, a_max):
    
    '''Interquatile range'''
    Q1, Q3 = np.percentile(a, [25, 75])
    if Q1==Q3: return (a_min, a_max)
    else: return (np.fmax(Q1-whis*(Q3-Q1),a_min), 
                  np.fmin(Q3+whis*(Q3-Q1),a_max))

def Cal_Bartlett(X):
    
    '''
    Bartlett's Sphericity tests the hypothesis that the correlation 
    matrix is equal to the identity matrix, which would indicate that 
    your variables are unrelated and therefore unsuitable for 
    structure detection [1].
    
        H0: The correlation matrix is equal to I 
        H1: The correlation matrix is not equal to I
        
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/23.0.0?topic=
           detection-kmo-bartletts-test 
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    BartlettTest : collections.namedtuple
        Bartlett's Sphericity test result.
        
    '''
    corr = np.corrcoef(X.T)
    R = np.linalg.det(corr)
    p = corr.shape[0]
    df= p*(p-1)/2
    chisq = -((len(X)-1)-(2*p+5)/6)*np.log(R)
    p_value = 1-stats.chi2.cdf(chisq, df=df)
    critval= stats.chi2.ppf(1-0.05, df)
    
    keys = ["chisq", "df", "pvalue"]
    BTest = collections.namedtuple('BartlettTest', keys)
    BTest = BTest(*(chisq, df, p_value))

    return BTest

def Cal_KMOScore(X):
    
    '''
    The Kaiser-Meyer-Olkin Measure of Sampling Adequacy is a 
    statistic that indicates the proportion of variance in your 
    variables that might be caused by underlying factors. High values 
    (close to 1.0) generally indicate that a factor analysis may be 
    useful with your data. If the value is less than 0.50, the results 
    of the factor analysis probably won't be very useful.
    
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/23.0.0?topic=
           detection-kmo-bartletts-test
    .. [2] https://factor-analyzer.readthedocs.io/en/latest/_modules/
           factor_analyzer/factor_analyzer.html

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    MSATest : collections.namedtuple
        Measure of Sampling Adequacy (MSA).
        
    KMOTest : collections.namedtuple
        Kaiser-Meyer-Olkin (KMO).
        
    '''
    # Pair-wise correlations
    diag = np.identity(X.shape[1])
    corr = (X.corr().values**2) - diag
    
    # Partial correlations
    pcorr = (Cal_PartialCorr(X).values**2) - diag

    # Measure of Sampling Adequacy (MSA)
    pcorr_sum = np.sum(pcorr, axis=0)
    corr_sum  = np.sum(corr, axis=0)
    msa_score = corr_sum / (corr_sum + pcorr_sum)
    
    keys = ["Score", "Corr", "PartialCorr"]
    MSATest = collections.namedtuple('MSATest', keys)
    MSATest = MSATest(*(msa_score, corr_sum, pcorr_sum))

    # Kaiser-Meyer-Olkin (KMO)
    kmo_score = np.sum(corr) / (np.sum(corr) + np.sum(pcorr))
    KMOTest = collections.namedtuple('KMOTest', keys)
    KMOTest = KMOTest(*(kmo_score, np.sum(corr), np.sum(pcorr)))
    
    return MSATest, KMOTest

def Cal_PartialCorr(X):
    
    '''
    Partial correlation coefficients describes the linear 
    relationship between two variables while controlling for the 
    effects of one or more additional variables [1].
    
    If we want to find partial correlation of X, and Y while
    controlling Z. we regress variable X on variable Z, then subtract 
    X' from X, we have a residual eX. This eX will be uncorrelated 
    with Z, so any correlation X shares with another variable Y cannot 
    be due to Z. This method also applies to Y in order to compute eY. 
    The correlation between the two sets of residuals, corr(e(X), e(Y)) 
    is called a partial correlation [2]. 
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.
            
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/24.0.0?topic=
           option-partial-correlations
    .. [2] http://faculty.cas.usf.edu/mbrannick/regression/Partial.html
    
    Returns
    -------
    pcorr : pd.DataFrame of shape (n_features, n_features)
        Partial correlations.
    
    '''
    n_features = X.shape[1]
    X0 = np.array(X).copy()
    pcorr = np.zeros((n_features,)*2)
    
    for i,j in product(*((range(n_features),)*2)):
        if j-i > 0:
            resids = []
            # Controlled variables
            index = np.isin(range(n_features),[i,j])
            Xs = np.hstack((np.ones((X0.shape[0],1)), X0[:,~index]))
            
            for Z in (X0[:,[i]], X0[:,[j]]):
                
                # Determine betas (INV(X'X)X'Y) and residuals
                betas = np.linalg.inv(Xs.T.dot(Xs)).dot(Xs.T).dot(Z)
                resids.append(Z.ravel() - Xs.dot(betas).ravel())
                
            # Partial correlation between xi, and xj
            pr = np.corrcoef(np.array(resids))[0,1]
            pcorr[i,j] = pr
        
    pcorr = pcorr + pcorr.T 
    return pd.DataFrame(pcorr + np.identity(n_features), 
                        columns=X.columns, 
                        index=X.columns)

def Cal_SMC(X):
    
    '''
    Calculate the squared multiple correlations. This is equivalent 
    to regressing each variable on all others and calculating the r2 
    values.
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    smc : numpy array
        The squared multiple correlations matrix.
        
    '''  
    # smc = 1-1/np.diag(np.linalg.inv(np.corrcoef(X.T)))
    X0, smc = np.array(X), []
    for i in range(X0.shape[1]):
        
        # Controlled variables
        index = np.isin(range(X0.shape[1]),[i])
        Xs = np.hstack((np.ones((X0.shape[0],1)), X0[:,~index]))
        y  = X0[:,[i]]
        
        # Determine betas (INV(X'X)X'Y) and residuals
        betas = np.linalg.inv(Xs.T.dot(Xs)).dot(Xs.T).dot(y)
        smc.append(np.var(Xs.dot(betas))/np.var(y))
    
    return pd.DataFrame(smc, columns=["SMC"], 
                        index=X.columns)

def KMO_table():
    standard = [["0.0 $\geq$ KMO $>$ 0.5", "unacceptable"],
                ["0.5 $\geq$ KMO $>$ 0.6", "miserable"],
                ["0.6 $\geq$ KMO $>$ 0.7", "mediocre"],
                ["0.7 $\geq$ KMO $>$ 0.8", "middling"],
                ["0.8 $\geq$ KMO $>$ 0.9", "meritorious"],
                ["0.9 $\geq$ KMO $\geq$ 1.0", "marvelous"]]
    return pd.DataFrame(standard, columns=["KMO","Suitability"]) 