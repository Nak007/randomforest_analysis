'''
Available methods are the followings:
[1] Modified_RandomizedSearch
[2] ParameterVisualizer
[3] PlotGridSearch
[4] find_delta

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-01-2022

'''
import numpy as np, pandas as pd
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import matplotlib.tri as tri

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, make_scorer, 
                             confusion_matrix)
from sklearn.base import clone
from inspect import signature
import multiprocessing
from joblib import Parallel, delayed
from functools import partial
from itertools import product

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ["Modified_RandomizedSearch", 
           "ParameterVisualizer", 
           "PlotGridSearch", 
           "find_delta"]

def PlotGridSearch(gs, scoring=None, ax=None, colors=None, 
                   decimal=4, tight_layout=True):
    
    '''
    Plot results from fitted sklearn GridSearchCV.
    
    Parameters
    ----------
    gs : sklearn GridSearchCV object
    
    scoring : list of str, default=None
        List of scoring functions. If None, all specified scoring 
        functions are used.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number of 
        scorings. If None, it uses default colors from Matplotlib.
    
    decimal : int, default=4
        Decimal places for annotation of best value(s).
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.
           model_selection.GridSearchCV.html#sklearn.model_selection.
           GridSearchCV
           
    Returns
    -------
    ax : Matplotlib axis object
           
    '''
    
    # ============================================================
    # Attribute `cv_results_` from GridSearchCV.
    results = gs.cv_results_
    n_param = len(results['params'])
    num_format = ("{:,." + str(decimal) + "f}").format
    # -------------------------------------------------------------
    # Create matplotlib.axes if ax is None.
    if ax is None:
        width = np.fmax(n_param*0.3, 6.5)
        ax = plt.subplots(figsize=(width + 2, 4.5))[1]
    # -------------------------------------------------------------   
    # Get scoring metrics.
    if scoring is None: scoring = gs.scoring.keys()
    # Get default line color.      
    colors = ([ax._get_lines.get_next_color() for _ in 
               range(len(scoring))] if colors is None else colors)
    # ============================================================

    # =============================================================
    # Specified parameters.
    params = list(results["params"][0].keys())
    # Set x-axis according to number of parameters.
    x = np.arange(n_param)
    # -------------------------------------------------------------
    # Initialize parameters.
    kwargs = {"train": dict(lw=1, ls='--', alpha=0.8, color="k"),
              "test" : dict(lw=2, ls='-', color="k")}
    best_lines = []
    # Patches and labels for ax.leggend.
    patches = [mpl.lines.Line2D([0],[0], **kwargs["train"]), 
               mpl.lines.Line2D([0],[0], **kwargs["test"])] 
    patches+= [mpl.patches.Patch(fc=c, ec='none') for c in colors]
    labels  = ["Train", "Test"] + list(scoring)
    # -------------------------------------------------------------
    for (n,score) in enumerate(scoring):
        for sample in ('train','test'):
            
            # Mean and standard deviation of scores.
            score_mean = results[f"mean_{sample}_{score}"]
            score_std  = results[f"std_{sample}_{score}" ]
            lower = score_mean - score_std
            upper = score_mean + score_std

            # Plot standard deviation of test scores
            if sample=="test": 
                ax.fill_between(x, lower, upper, 
                                alpha=0.1, color=colors[n])
            
            # Plot mean score
            ax.plot(x, score_mean, **{**kwargs[sample], 
                                      **{"color":colors[n]}})
    # -------------------------------------------------------------        
        # Determine the best score
        best_index = np.argmax(results[f'rank_test_{score}']==1)
        best_score = results[f'mean_test_{score}'][best_index]
        best_rank  = x[best_index]
    # -------------------------------------------------------------
        # Annotate the best score.
        ax.scatter(best_rank, best_score, color=colors[n], 
                   marker='o', s=100, lw=2, facecolor='none')
        ax.annotate(num_format(best_score), (best_rank, best_score),
                    textcoords='offset points', xytext=(0,10), 
                    va='bottom', ha='center', fontsize=12, 
                    color=colors[n], fontweight='demibold', 
                    bbox=dict(boxstyle='square', fc='white', 
                              alpha=0.8, pad=0.2, lw=0))
    # -------------------------------------------------------------    
        # Positional and keyword arguments for `best_line`.
        best_lines.append((((best_rank,)*2, [0, best_score]), 
                           dict(ls='-.', lw=2, color=colors[n])))
    # -------------------------------------------------------------
    # Fix y-axis.
    y_min, y_max = ax.get_ylim()
    y_max = min(1.05, y_max)
    y_min = max(-0.5, y_min)
    ax.set_ylim(y_min, y_max)
    # -------------------------------------------------------------
    # Plot a dotted vertical line at the best score.
    for args, kwargs in best_lines: ax.plot(*args, **kwargs)
    # =============================================================   

    # Set other attributes.
    # =============================================================
    ax.set_xticks(x)
    ax.set_xlim(-0.5, n_param-0.5)
    xticklabels = results[f'param_{params[0]}'].data
    if ((isinstance(xticklabels[0], (dict,list,tuple)) | 
         len(params)>1)): 
        ax.set_xticklabels(np.arange(1, n_param+1))
        ax.set_xlabel(r"$n^{th}$ Set of parameters", fontsize=13)
    else:
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(params[0], fontsize=13)
    # -------------------------------------------------------------   
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=11)
    # -------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('Scores', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    text_y = ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', 
                     ha="center", transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    legend = ax.legend(patches, labels, edgecolor="none", ncol=1,
                       borderaxespad=0.25, markerscale=1.5, 
                       columnspacing=0.3, labelspacing=0.7, 
                       handletextpad=0.5, prop=dict(size=12), 
                       loc='upper left') 
    legend.set_bbox_to_anchor([1.01,1], transform = ax.transAxes)
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

class Modified_RandomizedSearch():
    
    '''
    Randomized search on hyper parameters.
    
    Parameters
    ----------
    estimator : estimator object
        A object of that type is instantiated for each grid point. 
        This is assumed to implement the scikit-learn estimator 
        interface. Either estimator needs to provide a score function, 
        or scoring must be passed.
    
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try. Distributions 
        must provide a ``rvs`` method for sampling (such as those 
        from scipy.stats.distributions). If a list is given, it is 
        sampled uniformly. If a list of dicts is given, first a dict 
        is sampled uniformly, and then a parameter is sampled using 
        that dict as above.
        
    n_iter : int, default=10
        Number of parameter settings that are produced.
        
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform 
        sampling from lists of possible values instead of scipy.stats 
        distributions. Pass an int for reproducible output across 
        multiple function calls.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a 
        joblib.parallel_backend context. -1 means using all processors.
        
    scoring : dict, default=None
        Dictionary with parameters names (`str`) as keys and list of 
        functions (e.g. sklearn.metrics) for classification that accept 
        parameters as follows: `y_true`, and `y_pred` or `y_score`. 
        If None, it defaults to `f1_score`, `accuracy_score`, and 
        `roc_auc_score`.
    
    Attributes
    ----------
    results : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that 
        can be imported into a pandas DataFrame e.g.

        {
         "train_score0" : [0.80, 0.84, 0.70],
         "train_score1" : [0.72, 0.91, 0.68],
         "test0_score0" : [0.81, 0.67, 0.70],
         "test0_score1" : [0.73, 0.63, 0.43],
         }

    params : list of dict
        It is used to store a list of parameter settings dicts for all 
        the parameter candidates.
        
    '''
    
    def __init__(self, estimator, param_distributions, n_iter=10, 
                 random_state=None, n_jobs=None, scoring=None):
        
        self.base_estimator = estimator
        
        # Number of processors required
        cpu_count = multiprocessing.cpu_count()
        if n_jobs == -1: n_jobs = cpu_count
        n_jobs = max(1, n_jobs) if isinstance(n_jobs, int) else 1
        self.n_jobs = min(n_jobs, cpu_count)
        
        # Uniformly randomize parameters sampled from given distributions.
        self.params = list(ParameterSampler(param_distributions, int(n_iter), 
                                            random_state=random_state))
        
        # Default scoring functions.
        if scoring is None:
            self.scoring = {'f1_score' : f1_score, 
                            'auc'      : roc_auc_score,
                            'accuracy' : accuracy_score}
        else: self.scoring = scoring
        
    def fit_estimator(self, estimator, params, X, y):

        '''Private function: Fit estimator given new parameters'''
        return clone(estimator).set_params(**params).fit(X, y) 
    
    def calulate_score(self, y_true, y_pred, y_score):
    
        '''Private function: Compute score(s)'''
        scores = dict()
        for name, scorer in self.scoring.items():
            if "y_pred" in signature(scorer).parameters:
                scores[name] = [scorer(y_true, y_pred)]
            else: scores[name] = [scorer(y_true, y_score)]
        return scores
    
    def fit(self, X, y, test_set=None):
        
        '''
        Run fit with all sets of parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and 
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target relative to X for binary classification.
        
        test_set : list, default=None
            A list of tuples, where first item is X and the seccond item 
            is y e.g. [(X_test0, y_test0), (X_test1, y_test1)]. 

        Attributes
        ----------
        results : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that 
            can be imported into a pandas DataFrame e.g.
            
            {
             "train_score0" : [0.80, 0.84, 0.70],
             "train_score1" : [0.72, 0.91, 0.68],
             "test0_score0" : [0.81, 0.67, 0.70],
             "test0_score1" : [0.73, 0.63, 0.43],
             }
             
             The order of the results corresponds to that in `params`, 
             which is a list of (randomized) parameter settings dicts for 
             all the parameter candidates.
        
        params : list of dict
            It is used to store a list of parameter settings dicts for all 
            the parameter candidates.
            
        '''
        # Fit `n_iter` randomized parameters.
        partial_ = partial(self.fit_estimator, X=X.copy(), y=y.copy(), 
                           estimator=self.base_estimator)
        parallel = Parallel(n_jobs=self.n_jobs)
        estimators = parallel(delayed(partial_)(params=params) 
                              for params in self.params)

        # Consolidate the validation sets.
        X_list = [("train", X, y)]
        if isinstance(test_set, list):
            for n, (X0, y0) in enumerate(test_set):
                X_list += [(f"test{n}", X0, y0)]

        # Compute score(s)
        self.results = dict()
        for estimator in estimators:
            for ds_name, Xi, yi in X_list:
                
                # Compute score(s) from self.scoring
                y_score = estimator.predict_proba(Xi)
                y_pred  = np.argmax(y_score, axis=1)
                scores_ = self.calulate_score(yi, y_pred, y_score[:,1])
                
                for sc_name, score in scores_.items():
                    key = f"{ds_name}_{sc_name}"
                    self.results[key] = self.results.get(key,[]) + score
        
        return self

def pcf_base(X, option="eigval", mineigval=0.8, proportion=0.75):
    
    '''
    Performs dimensionality reduction on obtained results using 
    principal component factoring.
    
    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and 
        n_features is the number of features.
        
    option : {"eigval", "varexp"}, default="eigval"
        The function to select variables

    mineigval : float, default=0.8
        Minimum value of eigenvalues when choosing number of 
        components. The algorithm selects component whose eigenvalue 
        is grater than or equal to `mineigval`. The minimum of
        component is 2. Only applicable if `option` is "eigval".
   
    proportion : float, default=0.75
        Minimum proportion of variation explained (cumulative) when 
        choosing components. The algorithm selects components when 
        all proportions of variation explained is greater than
        `proportion`. Only applicable if `option` is "varexp".
        
    Returns
    -------
    Results : collections.namedtuple
        A namedtuple class with following fields:
        - "eigvals"  : EigenValues of the selected components
        - "eigvecs"  : EigenVectors of the selected components
        - "varprops" : Variance explained by each of the components
        - "princoms" : Principal components
        - "features" : List of features
        
    '''
    # =============================================================
    if isinstance(X, pd.DataFrame):
        if X.shape[1]<=2:
            raise ValueError(f'n_features must be greater than 2.' 
                             f'Got {X.shape[1]} instead.')
    else: raise TypeError(f'Data must be pd.DataFrame.' 
                          f' Got {type(X)} instead.')
    # -------------------------------------------------------------   
    if option not in ["eigval", "varexp"]:
        raise ValueError(f'`option` must be either "eigval" or ' 
                         f'"varexp". Got {option} instead.')
    # -------------------------------------------------------------
    # Determine Eigenvalues and corresponding Eigenvector.
    std_X = ((X - np.mean(X,0)) / np.std(X,0)).values
    corr  = np.corrcoef(std_X.T)
    eigvals, eigvecs = np.linalg.eig(corr)
    if option=="varexp":
        cumvar = np.cumsum(eigvals/eigvals.sum())
        proportion = min(max(0, proportion),1)
        n_components = max(np.argmax(cumvar>=proportion)+1, 2)
    else: n_components = max((eigvals>=mineigval).sum(), 2)
    # -------------------------------------------------------------
    # Determine principal components.
    indices  = np.argsort(eigvals)[::-1]
    eigvals  = eigvals[indices][:n_components]
    eigvecs  = eigvecs[:,indices][:,:n_components]
    varprops = eigvals / X.shape[1]
    princoms = std_X.dot(eigvecs[:,:n_components])
    digit    = int(np.ceil(np.log(n_components)/np.log(10)))
    columns  = ["PC_{}".format(str(n).zfill(digit)) for n in 
                np.arange(n_components) + 1]
    eigvecs  = pd.DataFrame(eigvecs.real, columns=columns,
                            index=list(X))
    # -------------------------------------------------------------
    keys = ["eigvals", "eigvecs", "varprops", 
            "princoms", "features"]
    Results = collections.namedtuple("Results", keys)
    Results = Results(*(eigvals, eigvecs, varprops, 
                        princoms, list(X)))
    # =============================================================
    
    return Results

def plot_pcf_base(Results, ax=None, colors=None, pc_pair=(0,1), 
                  target=None, exclude=None, scatter_kwds=None, 
                  tight_layout=True):
    
    '''
    Plot results from `PCF_base`.

    Parameters
    ----------
    Results : namedtuple
        Result from `pcf_base` function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, it uses default figsize.
        
    colors : list of color-hex, default=None
        Color-hex must be arranged in same order as `Results.features`. 
        If None, it uses default colors from Matplotlib.
    
    pc_pair : (int, int), default=(0,1)
        Pair of principal component indices for x and y axis, 
        respectively.
        
    target : array-like of shape (n_samples,), default=None
        Binary target relative to X (parameters).
        
    exclude : list of str, default=None
        List of features to be excluded from the plot.
        
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
    if ax is None: ax = ax = plt.subplots(figsize=(9, 5))[1] 
    features = list(Results.features)
    n_features = len(features)
    colors = ([ax._get_lines.get_next_color() for _ in 
               range(n_features)] if colors is None else colors)
    # ===============================================================
    
    # Scatter plots
    # ===============================================================
    patches, labels, (pc1, pc2) = [], [], pc_pair
    x1, x2 = Results.princoms[:,pc1], Results.princoms[:,pc2]
    if scatter_kwds is None: scatter_kwds = dict()
    kwds = {"s"         : scatter_kwds.get("s", 35), 
            "marker"    : scatter_kwds.get("marker", "o"), 
            "linewidth" : scatter_kwds.get("linewidth", 0.8), 
            "alpha"     : scatter_kwds.get("alpha", 0.4),
            "edgecolor" : "none", 
            "facecolor" : scatter_kwds.get("facecolor", "grey")}
    ax.scatter(x1, x2, **kwds)
    legend_kwds = dict(marker=kwds["marker"], markersize=8, 
                       markerfacecolor=kwds["facecolor"], 
                       markeredgecolor=kwds["edgecolor"],
                       alpha=kwds["alpha"], color='none')
    sc1 = mpl.lines.Line2D([0],[0], **legend_kwds)
    # ---------------------------------------------------------------
    kwds.update(dict(facecolor="none", alpha=1, 
                     edgecolor=scatter_kwds.get("edgecolor", "grey")))
    ax.scatter(x1, x2, **kwds)
    legend_kwds = dict(marker=kwds["marker"], 
                       markerfacecolor=kwds["facecolor"], 
                       markeredgecolor=kwds["edgecolor"],
                       color='none', markersize=8)
    sc2 = mpl.lines.Line2D([0],[0], **legend_kwds)
    patches += [(sc1, sc2)]
    labels  += ["Parameters (n={:,d})".format(len(x1))]
    # ---------------------------------------------------------------  
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
    eigvecs  = Results.eigvecs.values.real * np.sqrt(Results.eigvals)
    varprops = Results.varprops
    scale = min(find_scale(eigvecs[:,pc_pair], ax)) 
    commu = pow(eigvecs[:,pc_pair], 2).sum(1)
    zorder= n_features + 2
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
    for n,col in enumerate(features):
        if col not in exclude:
            size = tuple(eigvecs[n,pc_pair].real * 0.85 * scale)
            kwds = dict(mutation_scale=20, arrowstyle = arrowstyle, 
                        edgecolor="grey", facecolor=colors[n], alpha=1,
                        linewidth=0.8, zorder=zorders.index(col)+2)
            arrow = mpatches.FancyArrowPatch((0, 0), size, **kwds)
            ax.add_patch(arrow)
            patches += [arrow]
            labels += ["{} ({:.1%})".format(col, commu[n])]
    # ---------------------------------------------------------------
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=-1)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=-1)
    # ---------------------------------------------------------------
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    # ---------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    PC_str = "PC{} ({:.1%})".format
    ax.set_xlabel(PC_str(pc1+1, varprops[pc1]), fontsize=13)
    ax.set_ylabel(PC_str(pc2+1, varprops[pc2]), fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    # ---------------------------------------------------------------
    expvar = varprops[np.unique(pc_pair)].sum()
    text = "Explained Variance = {:.1%}".format(expvar)
    props = dict(boxstyle='square', facecolor='white', alpha=0.8, 
                 edgecolor="none", pad=0.1)
    ax.text(1, 0.02, text, transform=ax.transAxes, fontsize=13,
            va='bottom', ha="right", bbox=props)
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

class ParameterVisualizer:
    
    '''
    Visualize parameters by using Principal Component Analysis.

    Parameters
    ----------
    option : {"eigval", "varexp"}, default="eigval"
        The function to select variables

    mineigval : float, default=0.8
        Minimum value of eigenvalues when choosing number of 
        components. The algorithm selects component whose eigenvalue 
        is grater than or equal to `mineigval`. The minimum of
        component is 2. Only applicable if `option` is "eigval".

    proportion : float, default=0.75
        Minimum proportion of variation explained (cumulative) when 
        choosing components. The algorithm selects components when 
        all proportions of variation explained is greater than
        `proportion`. Only applicable if `option` is "varexp".
        
    ''' 
    def __init__(self, option="eigval", mineigval=0.8, proportion=0.75):
        
        self.option = option
        self.mineigval = mineigval
        self.proportion = proportion
    
    def fit(self, X):
    
        '''
        Performs dimensionality reduction on obtained results using 
        principal component factoring.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and 
            n_features is the number of features.

        Returns
        -------
        Results : collections.namedtuple
            A namedtuple class with following fields:
            - "eigvals"  : EigenValues of the selected components
            - "eigvecs"  : EigenVectors of the selected components
            - "varprops" : Variance explained by each of the components
            - "princoms" : Principal components
            - "features" : List of features
            
        '''
        self.Results = pcf_base(X.copy(), option=self.option, 
                                mineigval=self.mineigval, 
                                proportion=self.proportion)
        return self
    
    def plot(self, ax=None, colors=None, pc_pair=(0,1), target=None, 
             exclude=None, scatter_kwds=None, tight_layout=True):
    
        '''
        Plot results.

        Parameters
        ----------
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, it uses default figsize.

        colors : list of color-hex, default=None
            Color-hex must be arranged in same order as `Results.features`. 
            If None, it uses default colors from Matplotlib.

        pc_pair : (int, int), default=(0,1)
            Pair of principal component indices for x and y axis, 
            respectively.

        target : array-like of shape (n_samples,), default=None
            Binary target relative to X (parameters).

        exclude : list of str, default=None
            List of features to be excluded from the plot.

        scatter_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.scatter".

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots i.e. 
            plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        return plot_pcf_base(self.Results, ax=ax, colors=colors, 
                             pc_pair=pc_pair, target=target, 
                             exclude=exclude, scatter_kwds=scatter_kwds, 
                             tight_layout=tight_layout)
    
    def get_param_grids(self, params):
    
        '''
        Create parameter grids from several sets of parameters.

        Parameters
        ----------
        params : list of dict
            List of dictionaries containing parameters of estimator.

        Returns
        -------
        param_grids : dict
            Dictionary with parameters names (str) as keys and lists of 
            parameters to try.

        '''
        param_grids = None
        for p in params:
            if param_grids is None:
                param_grids = dict([(k,[v]) for k,v in p.items()])
            else: 
                for k,v in p.items():
                    if v not in param_grids[k]:
                        param_grids[k] += [v]

        for k,v in param_grids.items():
            try: param_grids[k] = np.sort(v).tolist()
            except: pass

        return param_grids

def find_delta(rs, remove=True, percent=True):
    
    '''
    Determine differences of results from `Modified_RandomizedSearch` 
    between train and test sets i.e. train - test.
    
    Parameters
    ----------
    rs : `Modified_RandomizedSearch` object
        Fitted `Modified_RandomizedSearch` class.
    
    remove : bool, default=True
        If True, test results are excluded.
        
    percent : bool, default=True
        If True, difference is determined as a percentage change from
        trian, otherwise actual value.
        
    Returns
    -------
    results : dict
        A dict with keys as column headers and values as columns, 
        that can be imported into a pandas DataFrame.
    
    '''
    results = rs.results.copy()
    scoring = rs.scoring.keys()
    n_tests = int(len(results)/len(scoring) - 1)
    
    if n_tests > 0:
        for n in range(n_tests):
            for score in scoring:
                train = np.array(results[f"train_{score}"])
                test  = np.array(results[f"test{n}_{score}"])
                denom = np.where(train==0, 1, train) if percent else 1
                diff  = (train - test) / denom
                results[f"diff{n}_{score}"] = (train - test).tolist() 
                if remove: results.pop(f"test{n}_{score}")    
                    
    return results