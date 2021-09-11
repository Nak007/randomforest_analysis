'''
Available methods are the followings:
[1] PlotGridSearch
[2] FeatureImportance
[3] TreeInterpreter
[4] tts_randomstate
[5] cal_score
[6] Axes2grid
[7] get_classweights
[8] CalibatorEvaluation

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 27-06-2021

'''
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from treeinterpreter import treeinterpreter
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, 
                             accuracy_score,
                             f1_score, make_scorer, 
                             confusion_matrix)
from sklearn.base import clone
from sklearn.linear_model import(HuberRegressor, 
                                 LinearRegression,
                                 LogisticRegression) 
from inspect import signature
import collections
from itertools import product
from scipy.stats import pearsonr

__all__ = ["PlotGridSearch", 
           "FeatureImportance",
           "permutation_importance", 
           "dfc_importance", 
           "drop_column_importance", 
           "TreeInterpreter", 
           "tts_randomstate",
           "cal_score", 
           "Axes2grid",
           "get_classweights",
           "CalibatorEvaluation",
           "Calibrator_base"]

def PlotGridSearch(gs, scoring=None, ax=None, 
                   colors=None, decimal=4):
    
    '''
    Plot results from fitted sklearn GridSearchCV.
    
    Parameters
    ----------
    gs : sklearn GridSearchCV object
    
    scoring : list of str, default=None
        List of scoring functions. If None, all specified
        scoring functions are used instead.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to
        number of scorings. If None, it uses default 
        colors from Matplotlib.
    
    decimal : int, default=4
        Decimal places for annotation of best value(s).

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/
           sklearn.model_selection.GridSearchCV.html#sklearn.
           model_selection.GridSearchCV
           
    Returns
    -------
    ax : Matplotlib axis object
           
    '''
    # Attribute `cv_results_` from GridSearchCV.
    results = gs.cv_results_
    n_param = len(results['params'])
    num_format = ("{:,." + str(decimal) + "f}").format

    # Create matplotlib.axes if ax is None.
    if ax is None:
        width = np.fmax(n_param*0.24,6)
        ax = plt.subplots(figsize=(width,5))[1]
        
    # Get scoring metrics.
    if scoring is None: 
        scoring = gs.scoring.keys()
    
    # Specified parameters.
    params = list(results["params"][0].keys())
    
    # Set x-axis according to number of parameters.
    x = np.arange(n_param)
    ax.set_xticks(x)
    ax.set_xlim(-0.2, n_param-0.8)
    
    if len(params) > 1: 
        ax.set_xticklabels(np.arange(1, n_param+1))
        ax.set_xlabel(r"$n^{th}$ Set of parameters", fontsize=11.5)
        title ="GridSearchCV Results : " + ', '.join(params)
    else:
        ax.set_xticklabels(results[f'param_{params[0]}'].data)
        ax.set_xlabel(params[0], fontsize=11.5)
        title = "GridSearchCV Results"
    
    # Initialize parameters.
    kwargs = {"train": dict(lw=1, ls='--', alpha=0.8),
              "test" : dict(lw=2, ls='-')}
    best_lines = []
    
    for (n,score) in enumerate(scoring):

        # Get default line color.
        if colors is None: color = ax._get_lines.get_next_color()
        else: color = colors[n]

        for sample in ('train','test'):
            
            # Mean and standard deviation of scores.
            score_mean = results[f"mean_{sample}_{score}"]
            score_std  = results[f"std_{sample}_{score}" ]
            lower = score_mean - score_std
            upper = score_mean + score_std

            # Plot standard deviation of test scores
            if sample=="test": ax.fill_between(x, lower, upper, 
                                               alpha=0.1, color=color)
            
            # Plot mean score
            kwds = {"color":color, "label":f"{score} ({sample})"}
            ax.plot(x, score_mean, **{**kwargs[sample], **kwds})
             
        # Determine the best score
        best_index = np.argmax(results[f'rank_test_{score}']==1)
        best_score = results[f'mean_test_{score}'][best_index]
        best_rank  = x[best_index]

        # Annotate the best score.
        ax.scatter(best_rank, best_score, color=color, 
                   marker='o', s=100, lw=2, facecolor='none')
        ax.annotate(num_format(best_score), (best_rank, best_score),
                    textcoords='offset points', xytext=(10,10), 
                    va='center', ha='left', 
                    fontsize=12, color=color, fontweight='demibold', 
                    bbox=dict(boxstyle='square', fc='white', 
                              alpha=0.7, pad=0.2, lw=0))
        
        # Positional and keyword arguments for `best_line`.
        best_lines.append((((best_rank,)*2, [0, best_score]), 
                           dict(ls='-.', lw=2, color=color)))
    
    # Fix y-axis.
    ax.set_ylim(*ax.get_ylim())
    
    # Plot a dotted vertical line at the best score.
    for args, kwargs in best_lines: ax.plot(*args, **kwargs)
        
    ax.legend(loc="best", prop=dict(weight="ultralight", size=11.5))
    ax.set_ylabel("Score", fontsize=11.5)
    ax.set_title(title, fontweight='demibold', fontsize=14)
    ax.grid(False)
    plt.tight_layout()
    return ax
    
class FeatureImportance():
    
    '''
    This function determines the most predictive features 
    through use of "Feature importance", which is the most 
    useful interpretation tool. This function is built 
    specifically for "scikit-learn" RandomForestClassifier. 
    
    Parameters
    ----------
    methods : list of str, default=None
        If None, it defaults to all methods. The function 
        to measure the importance of variables. Supported 
        methods are
            "gain" : average infomation gain 
                     (estimator.feature_importances_).
            "dfc"  : directional feature constributions [3] 
            "perm" : permutation importance [4]
            "drop" : drop-column importance [5] 
                     
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification 
        that accept parameters as follows: `y_true`, and 
        `y_pred` or `y_score`. If None, it defaults to 
        `f1_score`, `accuracy_score`, and `roc_auc_score` [2].
        
    max_iter : int, default=10
        Maximum number of iterations of the algorithm for 
        a single predictor feature. This is relevant when
        "perm" is selected.

    random_state : int, default=None
        At every iteration, it controls the randomness of
        value permutation of a single predictor feature. This
        is relevant when "perm" is selected.

    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/
           model_evaluation.html
    .. [3] <function treeinterpreter>
    .. [4] <function permutation_importance>
    .. [5] <function drop_column_importance>
    
    Attributes
    ----------
    importances_ : collections.namedtuple
        A tuple subclasses with named fields as follows:
        - features      : list of features
        - gain_score    : Infomation gains
        - dfc_score     : Directional Feature Constributions
        - permute_score : Permutation importances
        - drop_score    : Drop-Column importances

    info : pd.DataFrame
        Result dataframe.

    result_ : Bunch
        Dictionary-like object, with the following attributes.
        - gain_score    : estimator.feature_importances_
        - dfc_score     : results from "dfc_importance"
        - permute_score : results from "permutation_importance"
        - drop_score    : results from "drop_column_importance"
        
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import pandas as pd
    
    Use the breast cancer wisconsin dataset 
    
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> cols = load_breast_cancer().feature_names
    >>> X = pd.DataFrame(X, columns=cols)
    
    Split train and test datasets
    >>> X_train, X_test, y_train, y_test = tts(X, y, test_size=0.15)

    Fit "RandomForestClassifier" model (with default settings)
    >>> clf = RandomForestClassifier().fit(X, y)
    
    Fit "FeatureImportance" model
    >>> model = FeatureImportance().fit(clf, X_train, y_train)
    
    Summary of results
    >>> model.info
    
    Plot results
    >>> model.plotting()
    
    '''
    def __init__(self, methods=None, scoring=None, 
                 max_iter=10, random_state=0):
        
        if methods is None: 
            self.methods = {"gain", "dfc", "permute", "drop"}
        else: self.methods = methods
        
        self.scoring = scoring 
        self.max_iter = max_iter
        self.random_state = random_state
        self.title = {"gain_score"   : "Infomation gain",
                      "dfc_score"    : "Feature Constribution", 
                      "permute_score": "Permutation", 
                      "drop_score"   : "Drop-Column",
                      "mean_score"   : "Mean of score(s)" }
       
    def fit(self, estimator, X_train, y_train):

        '''
        Fit model
        
        Parameters
        ----------
        estimator : estimator object
            Fitted RandomForestClassifier estimator.
            
        X_train : array-like of shape (n_samples, n_features) 
            The training input samples. 

        y_train : array-like of shape (n_samples,)
            The training target labels (binary).
        
        Attributes
        ----------
        importances_ : collections.namedtuple
            A tuple subclasses with named fields as follows:
            - features      : list of features
            - gain_score    : Infomation gains
            - dfc_score     : Directional Feature Constributions
            - permute_score : Permutation importances
            - drop_score    : Drop-Column importances
    
        info : pd.DataFrame
            Result dataframe.
        
        result_ : Bunch
            Dictionary-like object, with the following attributes.
            - gain_score    : estimator.feature_importances_
            - dfc_score     : results from "dfc_importance"
            - permute_score : results from "permutation_importance"
            - drop_score    : results from "drop_column_importance"
                
        '''
        # Convert `X_train` to pd.DataFrame
        X = _to_DataFrame(X_train)
        info = dict(features = list(X_train))
        self.result_ = dict(features = list(X_train))
        mean_score = np.zeros(X.shape[1])
        
        # Infomation Gain (sklearn)
        if "gain" in self.methods:
            
            importances = estimator.feature_importances_
            self.result_.update({"gain_score" : importances})
            info.update({"gain_score" : importances})
            mean_score += importances

        # Directional Feature Constributions.
        if "dfc" in self.methods:
            
            result = dfc_importance(estimator, X_train)
            self.result_.update({"dfc_score" : result})
            info.update({"dfc_score" : result["importances_mean"]})
            mean_score += result["importances_mean"]
            
        # Permutation importance
        if "permute" in self.methods:
            
            kwargs = {"scoring" : self.scoring, 
                      "max_iter" : self.max_iter, 
                      "random_state" : self.random_state}
            result = permutation_importance(estimator, X_train, 
                                            y_train, **kwargs)
            self.result_.update({"permute_score" : result})
            info.update({"permute_score" : result["importances_mean"]})
            mean_score += result["importances_mean"]
            
        # Drop-Columns importance
        if "drop" in self.methods:
    
            result = drop_column_importance(estimator, X_train, y_train,
                                            scoring = self.scoring)
            self.result_.update({"drop_score" : result})
            info.update({"drop_score" : result["importances_score"]})
            mean_score += result["importances_score"]
        
        # Model attributes.
        info.update({"mean_score" : mean_score/sum(mean_score)})
        Results = collections.namedtuple('Results', info.keys())
        self.importances_ = Results(**info)
        
        self.info = pd.DataFrame(info).set_index("features")\
        .sort_values(by="mean_score", ascending=False)
        
        return self
    
    def plotting(self, column="mean_score", ax=None, 
                 barh_kwds=None, sort_by=None, 
                 tight_layout=True):
        
        '''
        Plot values from attribute "info".
        
        Parameters
        ----------
        column : str, default="mean_score"
            Column name from attribute "self.info".
    
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis.
            
        barh_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.barh".
            
        sort_by : str, default=None
            Column name to sort by. If None, it defaults
            to `column`.
        
        tight_layout : bool, default=True
            If True, it adjusts the padding between and 
            around subplots i.e. plt.tight_layout().
        
        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        # Get data from self.info (attribute)
        by = column if sort_by is None else sort_by
        data = self.info.sort_values(by=by)[[column]]
        features = np.array(data.index)
        importances = data.values.ravel()
        
        # Default if ax is not provided.
        if ax is None:
            size = (6, np.fmax(0.35*len(features),5))
            ax = plt.subplots(figsize=size)[1]
     
        # Plot horizontal bar (importances).
        ax.barh(np.arange(len(features)), importances, 
                **self.__params__(importances, barh_kwds))
        
        # Set other plot parameters.
        amax = ax.get_xlim()[1]/0.85
        self.__setaxis__(ax, features, amax)
        if sort_by==column: self.__annotMean__(ax, importances, amax)
        self.__annotbarh__(ax, importances)
        ax.set_title(f'Feature Importance\n{self.title[column]}', 
                     fontsize=14, fontweight=600)
        if tight_layout: plt.tight_layout()
        return ax
        
    def __annotMean__(self, ax, importances, amax):
        
        '''Private: annotation of mean and median'''
        n_features, amin = len(importances), ax.get_xlim()[0]
        
        # Selection method i.e. mean, and median
        methods = dict([(fnc.__name__, 
                         sum(importances>=fnc(importances))) 
                        for fnc in [np.mean, np.median]])
        
        # Only select "mean" if results are identical.
        if methods["mean"]==methods["median"]: 
            methods.pop("median")
            
        for key in methods.keys():
            n = methods[key]
            ax.fill_between([amin, amax], n_features-n-0.5, 
                            n_features, alpha=0.1, color="grey")
            ax.annotate(f'{key} ({n})'.upper(), (amax, n_features-n-0.5),
                        textcoords='offset points', xytext=(-2,2), 
                        va='bottom', ha='right')
    
    def __annotbarh__(self, ax, importances):
        
        '''Private: annotation of horizontal bars'''
        bbox =  bbox=dict(boxstyle='square', fc='white', 
                          alpha=0.7, pad=0.1, lw=0)
        kwargs = dict(textcoords='offset points', xytext=(5,0), 
                      va='center', ha='left', bbox=bbox)
        for y,x in enumerate(importances): 
            ax.annotate('{:,.2%}'.format(x),(max(x,0),y),**kwargs)
            
    def __setaxis__(self, ax, features, amax):
        
        '''Private: set axis of x and y'''
        x = np.arange(len(features))
        ax.set_yticks(x)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_ylim(-0.5, len(features)-0.5)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlim(ax.get_xlim()[0], amax)
    
    def __params__(self, importances, barh_kwds):
        
        '''
        Private: keyword argument for horizontal bar
        
        References
        ----------
        .. [1] https://matplotlib.org/stable/api/
               colors_api.html#module-matplotlib.colors
        '''
        cmap = plt.get_cmap("winter")
        rescale = lambda y: 1-(y-min(y))/(max(y)-min(y))
        kwargs = dict(height=0.7, alpha=0.8, ec='none', 
                      color=cmap(rescale(importances)))
        if barh_kwds is not None: 
            return {**kwargs,**barh_kwds}
        else: return kwargs
        
def _to_DataFrame(X) -> pd.DataFrame:
    
    '''
    If `X` is not `pd.DataFrame`, column(s) will be
    automatically created with "Unnamed" format.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    
    '''
    if not (hasattr(X,'shape') or hasattr(X,'__array__')):
        raise TypeError(f'Data must be array-like. ' 
                        f'Got {type(X)} instead.')
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        try:
            z = int(np.log(X.shape[1])/np.log(10)+1)
            columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                       for n in range(1,X.shape[1]+1)]
        except: columns = ['Unnamed']
        return pd.DataFrame(X, columns=columns)
    return X

def permutation_importance(estimator, X_train, y_train, 
                           scoring=None, max_iter=10, random_state=0):
    
    '''
    Record baseline metric(s) by passing a validation set
    through the Random Forest (scikit-learn). Permute the 
    column values of a single predictor feature and then 
    pass all test samples back through the estimator and 
    recompute the metrics. 
    
    The importance of that feature is the difference between 
    the baseline and the drop in overall accuracy caused by 
    permuting the column.
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples. 

    y_train : array-like of shape (n_samples,)
        The training target labels (binary).
        
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification 
        that accept parameters as follows: `y_true`, and 
        `y_pred` or `y_score`. If None, it defaults to 
        `f1_score`, `accuracy_score`, and `roc_auc_score` [2].
        
    max_iter : int, default=10
        Maximum number of iterations of the algorithm for 
        a single predictor feature.

    random_state : int, default=0
        At every iteration, it controls the randomness of
        value permutation of a single predictor feature. 

    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/
           model_evaluation.html
    
    Returns
    -------   
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features,)
            Mean of feature importance over max_iter.

        importances_std : ndarray, shape (n_features,)
            Standard deviation over max_iter.

        importances : ndarray, shape (n_features, max_iter)
            Raw permutation importance scores.
  
    '''
    # Convert input into ndarray
    y_true = np.array(y_train).copy()
    X = np.array(X_train).copy()
    
    # Default metrics for `scoring`
    if scoring is None: 
        scoring = [f1_score, 
                   roc_auc_score, 
                   accuracy_score]
    
    # Calculate the baseline score.
    bs_score = estimator.predict_proba(X)
    bs_pred  = np.argmax(bs_score, axis=1)
    args = (y_true, bs_pred, bs_score[:,1], scoring)
    baseline = __CalScore__(*args)
    
    # Initialize parameters.
    importances = []
    np.random.seed(random_state)
    
    for n in np.arange(X.shape[1]):
        
        X_permute, m, mean_score = X.copy(), 0, []
        while m < max_iter:
            
            # Permute the column values (var) and then pass 
            # all permuted samples back through the model. 
            X_permute[:,n] = np.random.permutation(X[:,n])
            y_score = estimator.predict_proba(X_permute)
            y_pred  = np.argmax(y_score, axis=1)
            
            # Recompute the accuracy and measure against
            # baseline accuracy.
            args = (y_true, y_pred, y_score[:,1], scoring)
            mean_score.append(baseline - __CalScore__(*args))
            m = m + 1
            
        # Calculate mean score
        importances.append(mean_score)
    
    # Permutation-importance scores
    importances = np.array(importances)
    raw = importances/abs(importances).sum(axis=0)
    result = {"importances" : importances, 
              "importances_mean" : np.mean(raw, axis=1),
              "importances_std" : np.std(raw, axis=1)}
    return result

def __CalScore__(y_true, y_pred, y_score, scoring):
    
    '''Private function to compute mean score'''
    score_ = []
    for scorer in scoring:
        if "y_pred" in signature(scorer).parameters:
            score_.append(scorer(y_true, y_pred))
        else: score_.append(scorer(y_true, y_score))
    return float(np.nanmean(score_))

def dfc_importance(estimator, X_train):
    
    '''
    Directional Feature Contributions (DFCs)
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples.
        
    References
    ----------
    .. [1] https://pypi.org/project/treeinterpreter/
    .. [2] Palczewska et al, https://arxiv.org/pdf/1312.1121.pdf
    .. [3] Interpreting Random Forests, http://blog.datadive.net/
           interpreting-random-forests/)
    
    Returns
    -------   
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features,)
            Mean of feature importance over n_samples.

        importances_std : ndarray, shape (n_features,)
            Standard deviation of feature importance over n_samples.

        contributions : ndarray, shape (n_samples, n_features)
            Raw Directional Feature Contribution scores.
            
    '''
    pred, bias, cont = treeinterpreter.predict(estimator, X_train)
    abs_cont = abs(cont[:,:,1])
    importances = abs_cont/abs_cont.sum(axis=1).reshape(-1,1)
    result = {"contributions" : cont[:,:,1], 
              "importances_mean" : np.mean(importances, axis=0),
              "importances_std" : np.std(importances, axis=0)}
    return result

def drop_column_importance(estimator, X_train, y_train, scoring=None):
    
    '''
    Calculate a baseline performance score. Then drop a column 
    entirely, retrain the model, and recompute the performance 
    score. The importance value of a feature is the difference 
    between the baseline and the score from the model missing 
    that feature. 
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples. 

    y_train : array-like of shape (n_samples,)
        The training target labels (binary).
        
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification 
        that accept parameters as follows: `y_true`, and 
        `y_pred` or `y_score`. If None, it defaults to 
        `f1_score`, `accuracy_score`, and `roc_auc_score` [2].
    
    References
    ----------
    .. [1] https://explained.ai/rf-importance/index.html
    .. [2] https://scikit-learn.org/stable/modules/
           model_evaluation.html
           
    Retruns
    -------
    result : Bunch
        Dictionary-like object, with the following attributes.

        importances_score : ndarray, shape (n_features,)
            Feature importance scores.

        drop_score : ndarray, shape (n_features,)
            Raw mean scores.
            
    '''
    # Convert input into ndarray
    y_true = np.array(y_train).copy()
    X = np.array(X_train).copy()
    
    # Default metrics for `scoring`
    if scoring is None: 
        scoring = [f1_score, 
                   roc_auc_score, 
                   accuracy_score]
    
    # Calculate the baseline score.
    bs_score = estimator.predict_proba(X)
    bs_pred  = np.argmax(bs_score, axis=1)
    args = (y_true, bs_pred, bs_score[:,1], scoring)
    baseline = __CalScore__(*args)
    
    # Initialize parameters.
    drop_score  = np.zeros(X.shape[1])
    estimator_  = clone(estimator)
    col_indices = np.arange(X.shape[1])
    
    for i in col_indices:
        # Drop column and retrain model.
        X_drop  = X[:,col_indices[col_indices!=i]].copy()
        y_score = estimator_.fit(X_drop, y_true).predict_proba(X_drop)
        y_pred  = np.argmax(y_score, axis=1)

        # Recompute the accuracy and measure against
        # baseline accuracy.
        args = (y_true, y_pred, y_score[:,1], scoring)
        drop_score[i] = baseline - __CalScore__(*args)
    
    # Calculate `importances_score`
    importances_score = drop_score/abs(drop_score).sum()
    result = {"drop_score" : drop_score, 
              "importances_score" : importances_score}
    return result

class TreeInterpreter:
    
    '''
    Using `treeinterpreter` to determine directional feature
    contribution.
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X_train : array-like of shape (n_samples, n_features) 
        The training input samples. 

    y_train : array-like of shape (n_samples,)
        The training target labels (binary).
    
    Attributes
    ----------
    contributions : ndarray of shape (n_samples, n_features)
        Directional Feature Contributions.
        
    '''
    def __init__(self, estimator, y_train, X_train):
    
        _, bias, contributions = treeinterpreter.predict(estimator, X_train)
        self.X = X_train.values.copy()
        self.features = np.array(X_train.columns)
        self.bias = bias
        self.contributions = contributions[:,:,1]
        self.y_pred = estimator.predict(X_train)
        self.y_true = np.array(y_train).ravel().copy()
        self.n_samples = X_train.shape[0]
        
    def plotting(self, var, ax=None, correct_pred=False, 
                 no_outliers=True, whis=1.5, scatter_kwds=None,
                 tight_layout=False, colors=None, 
                 frac=1, random_state=0):
        
        '''
        Plot directional feature contributions.
        
        Parameters
        ----------
        var : str
            Column name in `X_train`.
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis.
        
        correct_pred : bool, default=False
            Select those instances from the training dataset that 
            are classified correctly i.e. True Positive and True 
            Negative.
            
        no_outliers : bool, default=True
            If True, any values that stay outside (Q1-whis*IQR, 
            Q3+whis*IQR) are ignored.
            
        whis : float, default=1.5
            It determines the reach of the whiskers to the beyond 
            the first and third quartiles, which are Q1 - whis*IQR, 
            and Q3 + whis*IQR, respectively. Only available when 
            `no_outliers` is True.
            
        scatter_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.scatter".
       
        tight_layout : bool, default=True
            If True, it adjusts the padding between and 
            around subplots i.e. plt.tight_layout().
            
        colors : list of color-hex, default=None
            Color-hex must be arranged in the follwoing order i.e.
            ["Class 0", "Class 1", "Trend line"]. For "Class 0", 
            and "Class 1", colors are only applied to "edgecolor". 
            If None, it uses default colors from Matplotlib.
            
        frac : float, default=1
            Fraction of axis items to be plotted.
        
        random_state : int, default=0
            Seed for random number generator to randomize samples
            to be plotted.
            
        Returns
        -------
        ax : Matplotlib axis object
            
        '''
        col_index = np.argmax(self.features==var)
        x = self.X[:, col_index].copy()
        if ax is None: ax = ax = plt.subplots()[1]
            
        # Default colors.
        if colors is None:
            colors = [ax._get_lines.get_next_color() 
                      for n in range(3)]    
        
        if no_outliers:
            lower, upper = self.__IQR__(x, whis)
            normal = (x>=lower) & (x<=upper)
        else: normal = np.full(self.n_samples, True)
            
        if correct_pred:
            correct = self.y_pred==self.y_true
        else: correct = np.full(self.n_samples, True)
        
        x = x[normal & correct] 
        y = self.y_true[normal & correct] 
        d = self.contributions[normal & correct, col_index]
        
        # Select samples
        np.random.seed(random_state)
        indices = np.arange(0,len(x))
        if frac<1:
            size = np.fmax(int(frac*len(x)),10)
            select = np.random.choice(indices, size=size, 
                                      replace=False)
        else: select = indices.copy()
        indices = np.isin(indices, select)
        
        # Scatter plot
        kwds = dict(fc="none", marker='s', alpha=0.8, s=10)
        if scatter_kwds is not None: kwds = {**kwds, **scatter_kwds}
        for n in range(2):
            kwds = {**kwds, **dict(ec=colors[n], label=f"Class ({n})")}
            ax.scatter(x[(y==n) & indices], d[(y==n) & indices], **kwds)
      
        # Robust regression (Huber)
        robust   = HuberRegressor(epsilon=1.3).fit(x.reshape(-1,1), d)
        robust_x = np.percentile(x,[0,100]).reshape(-1,1)
        ax.plot(robust_x, robust.predict(robust_x), lw=2, ls="-", 
                c=colors[2], label="Trend")
        
        # Standard error of regression
        robust_y = robust.predict(x.reshape(-1,1))
        error    = np.sqrt(np.mean((d-robust_y)**2))
        robust_x = np.unique(x)
        robust_y = robust.predict(robust_x.reshape(-1,1))
        ax.fill_between(robust_x, 
                        robust_y - error, 
                        robust_y + error, 
                        alpha=0.1, color=colors[2])
  
        corr  = "Correlation: {:,.2f} (p-value: {:.2%})".format(*pearsonr(x,d))
        pdata = "Data point: {:.2%}".format(len(x)/self.n_samples)       
                
        ax.set_title('\n'.join((corr, pdata)), fontweight='demibold', fontsize=12)
        ax.set_ylabel('Directional Feature\nContributions (DFC)', fontsize=10)
        ax.set_xlabel(f'{var}', fontsize=10)
        ax.axhline(0, ls='--', lw=1, c='k')
        ax.legend(loc='best', 
                  prop=dict(weight="ultralight", size=11))  
        if tight_layout: plt.tight_layout()
        return ax
    
    def __IQR__(self, a, whis):
        '''Interquatile range'''
        Q1,Q3 = np.percentile(a, [25, 75])
        return Q1-whis*(Q3-Q1), Q3+whis*(Q3-Q1)

def Axes2grid(n_axes=4, n_cols=2, figsize=(6,4.3), 
              locs=None, spans=None):
    
    '''
    Create axes at specific location inside specified 
    regular grid.
    
    Parameters
    ----------
    n_axes : int, default=4
        Number of axes required to fit inside grid.
        
    n_cols : int, default=2
        Number of grid columns in which to place axis.
        This will also be used to calculate number of 
        rows given number of axes (`n_axes`).
    
    figsize : (float, float), default=(6,4.3)
        Width, height in inches for an axis.
    
    locs : list of (int, int), default=None
        locations to place each of axes within grid i.e.
        (row, column). If None, locations are created, 
        where placement starts from left to right, and
        then top to bottom.

    spans : list of (int, int), default=None
        List of tuples for axis to span. First entry is 
        number of rows to span to the right while the
        second entry is number of columns to span 
        downwards. If None, every axis will default to
        (1,1).

    Returns
    -------
    fig : Matplotlib figure object
        The Figure instance.
    
    axes : list of Matplotlib axis object
        List of Matplotlib axes with length of `n_axes`.
    
    '''
    # Calculate number of rows needed.
    n_rows = np.ceil(n_axes/n_cols).astype(int)
    
    # Default values for `locs`, and `spans`.
    if locs is None: 
        locs = product(range(n_rows),range(n_cols))
    if spans is None: spans = list(((1,1),)*n_axes)

    # Set figure size
    width, height = figsize
    figsize=(n_cols*width, n_rows*height)
    fig = plt.figure(figsize=figsize)
    
    # Positional arguments for `subplot2grid`.
    args = [((n_rows,n_cols),) + (loc,) + span 
            for loc,span in zip(locs, spans)]
    return fig, [plt.subplot2grid(*arg) for arg in args]

class tts_randomstate():
    
    '''
    Observe estimator's performance through random selection
    of "random_state" that controls randomness in splitting 
    train and test data (train_test_split).
    
    Parameters
    ----------    
    estimator : estimator object
        Fitted scikit-learn RandomForestClassifier estimator.
    
    X : array-like of shape (n_samples, n_features) 
        The input samples. 

    y : array-like of shape (n_samples,)
        The target labels (binary).
    
    tts_kwds : keywords, default=None
        Keyword arguments to be passed to "train_test_split" [1].
            
    scoring : list of functions, default=None
        List of sklearn.metrics functions for classification 
        that accept parameters as follows: `y_true`, and 
        `y_pred` or `y_score`. If None, it defaults to 
        `f1_score`, `accuracy_score`, and `roc_auc_score` [2].
    
    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/
           sklearn.model_selection.train_test_split.html
    .. [2] https://scikit-learn.org/stable/modules/
           model_evaluation.html
    
    Attributes
    ----------
    info : pd.DataFrame object
        pd.DataFrame object, with the following columns.
        - "tp" : # of True-Positive samples
        - "tn" : # of True-Negative samples
        - "fp" : # of False-Positive samples
        - "fn" : # of False-Negative sampels
        - Other scorings.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    
    >>> tts = tts_randomstate(clf, X, y, tts_kwds={"test_size":0.3})
    >>> tts.fit([10,20,30])

    Result dataframe
    >>> tts.info
    
    Plot results
    >>> tts.plotting()
    
    '''
    def __init__(self, estimator, X, y, tts_kwds=None, scoring=None):
        
        self.params = estimator.get_params()
        self.X = np.array(X).copy()
        self.y = np.array(y).ravel().copy()
        self.tts_kwds = tts_kwds
        self.scoring = [f1_score, roc_auc_score, accuracy_score]
        if scoring is not None: self.scoring = scoring
        self.columns = ["random_state","train","tp","fp","fn","tn"] \
        + [fnc.__name__ for fnc in self.scoring]
    
    def fit(self, random_states):
        
        '''
        Fit model.
        
        Parameters
        ----------
        random_states : list of int
            List of random states (int) that control the shuffling 
            applied to the data before applying the split.
        
        Attributes
        ----------
        info : pd.DataFrame object
            pd.DataFrame object, with the following columns.
            - "tp" : # of True-Positive samples
            - "tn" : # of True-Negative samples
            - "fp" : # of False-Positive samples
            - "fn" : # of False-Negative sampels
            - Other metrics ("scoring").
        
        '''
        self.info = []
        for n,state in enumerate(random_states):
            
            # Split train and test datasets.
            tts_kwds = {**self.tts_kwds,**{"random_state":state}}
            X_train, X_test, y_train, y_test = tts(self.X, self.y, **tts_kwds)
            data = zip((X_test, X_train), (y_test, y_train))
            
            # Train model.
            # self.params.update({'class_weight': self.__weights__(y_train)})
            estimator = RandomForestClassifier(**self.params)
            estimator.fit(X_train, y_train)
            
            for n, (X, y_true) in enumerate(data):
                
                # Calculate score and prediction
                y_score = estimator.predict_proba(X)
                y_pred  = np.argmax(y_score, axis=1)
                
                # Confusion matrix
                tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
                
                info = {"train": bool(n), "random_state": state, 
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn}
                args = (y_true, y_pred, y_score[:,1])
                info.update(self.__CalScore__(*args))
                self.info.append(info)
        
        # Info dataframe
        self.info = pd.DataFrame(self.info)[self.columns]\
        .set_index(["random_state","train"])
        
        return self
    
    def __weights__(self, labels):
        
        '''Determine `class_weight` for imbalanced classes'''
        cls, cnt = np.unique(labels, return_counts=True)
        return dict([(c,np.round(sum(cnt)/n,4)) 
                     for c,n in zip(cls,cnt)])
    
    def __CalScore__(self, y_true, y_pred, y_score):
    
        '''Private function to compute mean score'''
        cal_score = []
        for scorer in self.scoring:
            params = signature(scorer).parameters
            y = y_pred if "y_pred" in params else y_score
            cal_score.append((scorer.__name__, scorer(y_true, y))) 
        return cal_score
 
    def plotting(self, scores, ax=None, colors=None, decimal=4, 
                 criterion="max", tight_layout=True):
        
        '''
        Plot outputs from `random_states`.
        
        Parameters
        ----------
        scores : list of str
            Columns in "info".
    
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is 
            created with default figsize.
            
        colors : list of color-hex, default=None
            Number of color-hex must be greater than or equal
            to number of scorings. If None, it uses default 
            colors from Matplotlib.

        decimal : int
            Decimal places for annotation of value(s) 
            according to `criterion`.
            
        criterion : {"min", "max"}, default="max"
            If "max", vertical line is drawn at maximum of 
            scores, whereas "min"
            
        tight_layout : bool, default=True
            If True, it adjusts the padding between and 
            around subplots i.e. plt.tight_layout().
        
        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        # results and random_state(s)
        results = self.info.reset_index().copy()
        num_format = ("{:,." + str(decimal) + "f}").format
        train_index = results["train"]==True
        random_states = results.loc[train_index,"random_state"].values
        n_random_states = random_states.shape[0]
        
        # Create matplotlib.axes if ax is None.
        if ax is None:
            width = np.fmax(n_random_states*0.24,6)
            ax = plt.subplots(figsize=(width,5))[1]
            
        # Set x-axis according to number of random_states.
        x = np.arange(n_random_states)
        ax.set_xticks(x)
        ax.set_xlim(-0.2, n_random_states-0.8)
        ax.set_xticklabels(random_states, rotation=45)
        ax.set_xlabel("random_state", fontsize=11.5)
  
        # Initialize parameters.
        kwargs = {"train": dict(lw=1, ls='--', alpha=0.8),
                  "test" : dict(lw=2, ls='-')}
        y_max, y_min, best_lines = -np.inf, np.inf, []
    
        for (n,score) in enumerate(scores):

            # Get default line color.
            if colors is None: color = ax._get_lines.get_next_color()
            else: color = colors[n]

            for n,sample in enumerate(('test', 'train')):

                # Mean and standard deviation of scores.
                score_mean = results.loc[train_index==bool(n), score]
                score_std  = np.std(score_mean)
                lower = score_mean - score_std
                upper = score_mean + score_std

                # Plot standard deviation of test scores
                if sample=="test": 
                    ax.fill_between(x, lower, upper, alpha=0.1, color=color)

                # Plot mean score
                kwds = {"color":color, "label":f"{score} ({sample})"}
                ax.plot(x, score_mean, **{**kwargs[sample], **kwds})
                
            # Determine the best score
            scr = results.loc[train_index==False, score].values
            best_index = getattr(np, f"arg{criterion}")(scr)
            best_score = scr[best_index]
            best_rank  = x[best_index]

            # Annotate the best score.
            ax.scatter(best_rank, best_score, color=color, 
                       marker='o', s=100, lw=2, facecolor='none')
            ax.annotate(num_format(best_score), (best_rank, best_score),
                        textcoords='offset points', xytext=(10,10), 
                        va='center', ha='left', fontsize=12, color=color, 
                        fontweight='demibold', 
                        bbox=dict(boxstyle='square', fc='white', 
                                  alpha=0.7, pad=0.2, lw=0))
        
            # Positional and keyword arguments for `best_line`.
            best_lines.append((((best_rank,)*2, [0, best_score]), 
                               dict(ls='-.', lw=2, color=color)))
    
        # Fix y-axis.
        ax.set_ylim(*ax.get_ylim())

        # Plot a dotted vertical line at the best score.
        for args, kwargs in best_lines: ax.plot(*args, **kwargs)
    
        ax.legend(loc="best", prop=dict(weight="ultralight", size=11.5))
        ax.set_ylabel("Score", fontsize=11.5)
        
        if self.tts_kwds is not None: 
            tts_kwds = ["{}: {}".format(*t) for t in self.tts_kwds.items()]
            tts_kwds = ' (' + ', '.join(tts_kwds) + ')'
        else: tts_kwds = ""
        ax.set_title("Train-Test split" + tts_kwds, 
                     fontweight='demibold', fontsize=14)
        ax.grid(False)
        if tight_layout: plt.tight_layout()
        return ax

class CalibatorEvaluation():
    
    '''
    Evaluate calibrated probailities against target, in order to
    obtain more accurate probabilities. This method requires 
    grouping and validation of bin edges.
    
    Parameters
    ----------
    bins : sequence of int, default=range(2,10)
        Sequence of monotonically increasing int. Each of items 
        is used as starting number of bins.
        
    equal_width : bool, default=True
        If True, it uses equal-width binning, otherwise equal-
        sample binning is used instead.

    monotonic : bool, default=True
        If True, bins will be collspsed until number of targets 
        is either monotonically increasing or decreasing pattern.
        
    Attributes
    ----------
    result : dict
        Dictionary of output from <Calibration_base>, where key
        is number of iterations.

    rmse : list of float 
        Root Mean Squared Error as per iteration.

    gini : list of float
        Gini Impurity as per iteration
        
    '''
    def __init__(self, bins=range(2,10), equal_width=True, 
                 monotonic=True):
        
        self.bins = bins
        self.equal_width = equal_width
        self.monotonic = monotonic
        
    def fit(self, y_train, y_test):
        
        '''
        Fit model.
        
        Parameters
        ----------
        y_train : tuple(y_true, y_proba) 
            Tuple of true labels of train dataset and corresponding
            calibrated probilities from estimator. 

        y_test : tuple(y_true, y_proba) 
            Tuple of true labels of test dataset and corresponding 
            calibrated probilities from estimator. 
        
        Attributes
        ----------
        result : dict
            Dictionary of output from `Calibrator_base`, where key
            is number of iterations.
            
        rmse : list of float 
            Root Mean Squared Error as per iteration.
            
        gini : list of float
            Gini Impurity as per iteration
   
        '''
        # Initialize parameters
        self.result = dict()
        kwds = {"equal_width": self.equal_width}
        self.rmse = {"train":[], "test":[]}
        self.gini = {"train":[], "test":[]}
        Result = collections.namedtuple("Calibr", ["train",'test'])
        
        for (n, bins) in enumerate(self.bins):
            
            kwds.update({"bins" : bins, 
                         "monotonic" : self.monotonic})
            res_train = Calibrator_base(*y_train, **kwds)
            kwds.update({"bins" : res_train.bin_edges, 
                         "monotonic" : False})
            res_test  = Calibrator_base(*y_test, **kwds)
            
            self.result[n] = Result(*(res_train, res_test))
            for m in ["rmse", "gini"]:
                getattr(self, m)["train"] += [getattr(res_train,m)]
                getattr(self, m)["test"]  += [getattr(res_test, m)]
            
        return self
    
    def __ax__(self, ax, figsize):
        
        '''Private: create axis if ax is None'''
        if ax is None: return plt.subplots(figsize=figsize)[1]
        else: return ax
        
    def __colors__(self, ax, colors, n=10):
        
        '''Private: get default line color'''
        if colors is not None: return colors
        else: return [ax._get_lines.get_next_color() for _ in range(n)]

    def plotting_hist(self, n_iter=0, use_train=True, ax=None, 
                      colors=None, tight_layout=True, decimal=2, 
                      actual_kwds=None, estimate_kwds=None):
        
        '''
        Histogram between nth iteration and actual values.

        Parameters
        ----------
        n_iter : int, default=0
            nth iteration.
            
        use_train : bool, default=True
            If True, it uses result from train dataset, otherwise
            test dataset.

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created 
            with default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than 1. If None, 
            it uses default colors from Matplotlib.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().
            
        decimal : int, default=2
            Decimal places for annotation of % targets.
            
        actual_kwds : keywords, default=None
            Keyword arguments of targets to be passed to `ax.bar`.
         
        estimate_kwds : keywords, default=None
            Keyword arguments of estimates to be passed to `ax.bar`.

        Returns
        -------
        ax : Matplotlib axis object

        '''
        # Extract data from `Result.info`
        if use_train: result = self.result[n_iter].train
        else: result = self.result[n_iter].test
            
        data ="Train" if use_train else "Test"
        info = pd.DataFrame(result.info).copy()
        p_targets = info["p_targets"].values
        p_samples = info["p_samples"].values
        mean_proba= info["mean_proba"].values
        x = np.arange(len(p_targets))
        
        # Create matplotlib.axes if ax is None.
        width = np.fmax(len(p_targets)*0.8, 6)
        ax = self.__ax__(ax, (width, 4.3))

        # Get default line color.
        colors = self.__colors__(ax, colors, 2)
        
        num_format = ("{:,." + str(decimal) + "f}").format
        anno_kwds  = dict(xytext =(0,4), textcoords='offset points', 
                          va='bottom', ha='center', fontsize=10, 
                          fontweight='demibold')
     
        # Vertical bar (actual).
        pct  = np.sum(p_samples * p_targets) 
        kwds = dict(width=0.4, ec='k', alpha=0.9, color=colors[0], 
                    label='Actual ({:.0%})'.format(pct))
        ax.bar(x-0.25, p_targets, **({**kwds, **actual_kwds} if 
                                     actual_kwds is not None else kwds))
        
        # Annotation (actual).
        kwds = {**anno_kwds, **dict(color=colors[0])}
        for xy in zip(x-0.25, p_targets): 
            ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
            
        # Vertical bar (estimate).
        pct  = np.sum(p_samples * mean_proba)   
        kwds = dict(width=0.4, ec='k', alpha=0.9, color=colors[1], 
                    label='Estimate ({:.0%})'.format(pct))    
        ax.bar(x+0.25, mean_proba, **({**kwds, **estimate_kwds} if 
                                      estimate_kwds is not None else kwds))
        
        # Annotation (actual).
        kwds = {**anno_kwds, **dict(color=colors[1])}
        for xy in zip(x+0.25, mean_proba): 
            ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
            
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)

        # Set title and labels.
        ax.set_xticks(x)
        ax.set_xticklabels(f"Group {n}" for n in x+1)
        ax.set_xlim(-0.5, len(p_targets)-0.5)
        ax.set_yticks([])
        ax.set_yticklabels('')
        ax.set_ylim(0, max(max(p_targets)/0.9, max(mean_proba))/0.9)
        ax.set_title(f'Actual vs Estimate ({data})', 
                     fontweight='demibold', fontsize=12)
        ax.legend(loc=2, fontsize=12)
        if tight_layout: plt.tight_layout()
        return ax
    
    def plotting_reliability(self, n_iter=0, use_train=True, 
                             ax=None, colors=None, 
                             tight_layout=True):
        
        '''
        Plot calibarated probabilities both train and test along 
        with actual percentage of targets (reliability curve).

        Parameters
        ----------
        n_iter : int, default=0
            nth iteration.
        
        use_train : bool, default=True
            If True, it uses result from train dataset, otherwise
            test dataset.

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created 
            with default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than 2. If None, 
            it uses default colors from Matplotlib.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object

        '''
        # Extract data from `Result.info`
        if use_train: result = self.result[n_iter].train
        else: result = self.result[n_iter].test
            
        data ="Train" if use_train else "Test"
        info = pd.DataFrame(result.info).copy()
        p_targets = info["p_targets"].values
        mean_pred = info["mean_proba"].values
        stdv_pred = info["std_proba"].values

        # Create matplotlib.axes if ax is None.   
        ax = self.__ax__(ax, (6, 4.3))

        # Get default line color.
        colors = self.__colors__(ax, colors, 2)
        
        # Diagonal line.
        ax.plot(p_targets, p_targets, lw=2, ls="-.", 
                color=colors[0], label='Actual Target')
        
        # Calibrated line.
        kwds = {"linewidth" : 1, 
                "marker"    : "s", 
                "color"     : colors[1], 
                "fillstyle" : "none", 
                "markersize": 5, 
                "label"     : "Calibrated"}
        ax.plot(p_targets, mean_pred, **kwds)
        lower, upper = mean_pred-stdv_pred, mean_pred+stdv_pred
        ax.fill_between(p_targets, lower, upper, alpha=0.1, 
                        color=colors[1], label="Standard Deviation")

        # Set title and labels.
        ax.set_xlabel('Actual Target (%)', fontsize=10)
        ax.set_ylabel('Average probability\nof estimates (%)', fontsize=10)
        title = ', '.join((f'binning: {result.binning}',
                           "bins: {:,.0f}".format(result.bins), f"({data})"))
        ax.set_title(title, fontweight='demibold', fontsize=12)
        ax.legend(loc=0, fontsize=12)
        
        error = "RMSE={:.2f}, Gini={:.2f}"
        bottom, top = ax.get_ylim()
        left, right = ax.get_xlim()
        ax.annotate(error.format(result.rmse, result.gini), 
                    (right, bottom), ha="right", va="bottom", 
                    xytext =(-5,5), textcoords='offset points', 
                    fontsize=12)
        if tight_layout: plt.tight_layout()
        return ax
    
    def plotting_metric(self, metric="gini", ax=None, 
                       colors=None, tight_layout=True):
        
        '''
        Plot evaluation metric.

        Parameters
        ----------
        metric: {"rmse", "gini"}, default="gini"
            Evaluation metric.
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created 
            with default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than 2. If None, 
            it uses default colors from Matplotlib.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object

        '''
        # Create matplotlib.axes if ax is None.   
        ax = self.__ax__(ax, (6, 4.3))

        # Get default line color.
        colors = self.__colors__(ax, colors, 2)
        
        train = getattr(self, metric)['train']
        test  = getattr(self, metric)['test']
     
        x  = np.arange(len(train))
        ax.plot(x, train, lw=2, color=colors[0], label="Train")
        ax.plot(x, test , lw=2, color=colors[1], label="Test", ls="--")
        
        y_label = ("Gini Impurity" if metric=="gini" 
                   else "Root Mean Square Error")
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlabel(r"$n^{th}$ iteration", fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xlim(min(x)-0.3, max(x)+0.3)
        ax.legend(loc="best", fontsize=12)
        ax.set_title("Calibration of probabilities", 
                     fontweight='demibold', fontsize=12)
        if tight_layout: plt.tight_layout()
        return ax

def Calibrator_base(y_true, y_proba, equal_width=True, 
                    bins=10, monotonic=True):
    
    '''
    Evaluate the performace of probabilities whether it follows 
    actual target percentage. 
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels or binary label indicators. 

    y_proba : array-like of shape (n_samples,)
        Target probability.
        
    equal_width : bool, default=True
        If True, it uses equal-width binning, otherwise equal-sample 
        binning is used instead. This is relevant when `bins` is int.
        
    bins : int or sequence of scalars or str, default=10
        If int, it defines the number of bins that algorithm starts 
        off with. If bins is a sequence, it defines a monotonically 
        increasing array of bin edges, including the rightmost edge 
        (similar to numpy.histogram). If bins is a string, it defines 
        the method used to calculate the optimal bin width, as defined 
        by `numpy.histogram_bin_edges`.
        
    monotonic : bool, default=True
        If True, bins will be collspsed until number of targets is
        either monotonically increasing or decreasing pattern.
        
    Returns
    -------
    Result : collections.namedtuple
        A namedtuple class with following fields:
        - bins     : Number of bins
        - binning  : Method of binning
        - bin_edges: Monotonic bin edges 
        - rmse     : Root Mean Squared Error
        - gini     : Gini Impurity
        - info     : Evaluation results (pd.DataFrame)

    '''
    # Select binning methods.
    if isinstance(bins, int):
        binning = "equal-width" if equal_width else "equal-sample"
        if equal_width: bin_edges = equalbins(y_proba, bins)
        else: bin_edges = equalsamp(y_proba, bins)
    elif isinstance(bins, str):
        bin_edges, binning = auto_bin(y_proba, bins=bins), bins
    else: bin_edges, binning = bins, "pre-determined"

    # Monotonise `bin_edges`.
    if monotonic:
        args = (y_true, y_proba, bin_edges)
        bin_edges = monotonize(*args)
    
    # Initialize parameters
    indices = np.digitize(y_proba, bin_edges)
    default = np.full(len(bin_edges)-1, np.nan)
    data = {"r_min"     : bin_edges[:-1], 
            "r_max"     : bin_edges[1 :],
            "p_samples" : default.copy(), 
            "min_proba" : default.copy(), 
            "max_proba" : default.copy(), 
            "mean_proba": default.copy(), 
            "std_proba" : default.copy(), 
            "p_targets" : default.copy()}
    
    # Evaluation table.
    for n in range(1, len(bin_edges)):
        proba = y_proba[indices==n]
        data["p_samples"][n-1] = len(proba)/len(y_true)
        data["min_proba"][n-1] = np.min(proba)
        data["max_proba"][n-1] = np.max(proba)
        data["mean_proba"][n-1]= np.mean(proba)
        data["std_proba"][n-1] = np.std(proba)
        data["p_targets"][n-1] = np.mean(y_true[indices==n])

    # Calculate errors
    gini = (1-(pow(data["p_targets"],2) + pow(1-data["p_targets"],2))) 
    gini = (gini * data["p_samples"]).sum()
    err  = data["mean_proba"] - data["p_targets"]
    rmse = np.sqrt(np.mean(err**2))

    keys = ["bins", "binning", "bin_edges", 
            "rmse", "gini", "info"]
    Result = collections.namedtuple("CalProb", keys)
    return Result(**{"bin_edges": bin_edges,
                     "bins"   : bins if isinstance(bins,int) else len(bins), 
                     "binning": binning, 
                     "rmse"   : rmse, 
                     "gini"   : gini,
                     "info"   : pd.DataFrame(data)})

def auto_bin(proba, bins="fd"):
    
    ''' Binning method '''
    kwds = {"bins" : bins, 
            "range": (0, 1+np.finfo(float).eps)}
    bins = np.histogram_bin_edges(proba, **kwds)
    return bins

def equalsamp(proba, bins=10):

    '''Equal sample'''
    q = np.linspace(0,100,np.clip(bins+1, 2, 20))
    bins = np.unique(np.percentile(proba, q))
    bins[[0,-1]] = [0., 1.+np.finfo(float).eps]
    return bins

def equalbins(proba, bins=10):
    
    '''Equal bin-width'''
    a_min = min(proba)
    a_max = max(proba)+np.finfo(float).eps
    bins  = np.linspace(a_min, a_max, bins+1)
    bins[[0,-1]] = [0., 1.+np.finfo(float).eps]
    return bins

def monotonize(y, x, bin_edges):
        
    '''
    This function collaspes bin (to the left) that does not follow 
    the trend. Collapsing repeats until either all values increase
    /decrease in the same direction (monotonic) or number of bins 
    equals to 2, whatever comes first. 

    Parameters
    ----------
    y : ndarray of int
        Array of binary classes or labels.

    X : ndarray of float
        Array of float to be binned.

    bins : list or array of float
        Bin edges.

    Returns
    -------
    bins : list of float
        Monotonic bin edges.
    
    '''
    # Loop until bins remain unchanged or number of bins is 
    # less than 4.
    bins, n_bins = np.array(bin_edges), 0
    while (len(bins)!=n_bins) and len(bins)>3:
        
        # Determine percent of actual targets in each bin.
        hist   = np.histogram(x[y==1],bins)[0]
        p_hist = hist/np.fmax(np.histogram(x, bins)[0],1)
        direction = np.sign(np.diff(p_hist))

        # If percent target of an adjacent bin equals to 0, 
        # such bin will be collasped to the left. Otherwise, 
        # the trend is computed and select only those 
        # considered majority.
        if (p_hist[1:]==0).sum()>0:
            index = np.full(len(direction),True)
            index[(p_hist[1:]==0)] = False
        elif (direction==0).sum()>0: 
            index = np.full(len(direction),True)
            index[np.argmax((direction==0))] = False
        else:   
            index = np.where(direction<0,0,1) 
            index = (index==np.argmax(np.bincount(index)))

        # Keep bin edges that follow the majority trend.
        n_bins = len(bins)
        bins = bins[np.hstack(([True],index,[True]))]
        
    return bins

def cal_score(proba, pdo=20, odd=1., point=200., decimal=0, 
              min_prob=0.0001, max_prob=0.9999):
    
    '''
    In general, the relationship between odds and scores 
    can be presented as a linear transformation
    
    score = offset + factor*ln(odds) --- Eq. (1)
    score + pdo = offset + factor*ln(2*Odds) --- Eq. (2)
    
    Solving (1) and (2), we obtain
    
    factor = pdo/ln(2) --- Eq. (3)
    offset = score - (factor*ln(odds)) --- Eq. (4)

    Parameters
    ----------
    proba : ndarray of scalar shape of (n_sample,)
        An array of probabilities (0 < proba < 1).
        
    pdo : int, default=20
        Point difference when odd is doubled.
    
    odds : float, optional, default: 1.0 
        This serves as a reference odd where "point" is 
        assigned to.
    
    point : int, optional, default: 200
        Point that assigned to a reference odd.
     
    decimal : int, default=0
        Decimal places for scores.
        
    min_prob : float, default=0.0001
        Minimum probability.
    
    max_prob : float, default=0.9999
        Maximum probability.
        
    Returns
    -------
    scores : ndarray of scalar
        An array of scores.
    '''
    proba_ = np.clip(proba, min_prob, max_prob)
    ln_odd = np.log([p/(1-p) for p in proba_])
    factor = pdo/np.log(2)
    offset = point - factor*np.log(odd)
    return np.round_(offset+factor*ln_odd, decimal)

def get_classweights(start, stop, num=20, backward=0, forward=0, decimal=4):
    
    '''
    Determine sequence of evenly spaced class weights 
    over a specified interval.
    
    Parameters
    ----------
    start : 1D-array
        The starting value of the sequence [1].
    
    stop : 1D-array
        The end value of the sequence [1].
    
    num : int, default=20
        Number of samples to generate. Must be 
        non-negative [1].
    
    forward : int, default=0
        Number of forward steps beyond the end value 
        wrt. step return from np.linspace.
    
    backward : int, default=0
        Number of backward steps beyond the starting 
        value wrt. step return from np.linspace.
        
    decimal : int, default=4
        Number of decimal places for class weights.
    
    References
    ----------
    ... [1] https://numpy.org/doc/stable/reference/
            generated/numpy.linspace.html
    ... [2] https://scikit-learn.org/stable/modules/
            generated/sklearn.ensemble.
            RandomForestClassifier.html
           
    Returns
    -------
    Weights : list of dicts
        List of weights associated with classes in the 
        form {class_label: weight}.
    
    Examples
    --------
    >>> get_classweights([1.,1.], [0.5, 2.], num=5)
    [{0: 1.000, 1: 1.000 },
     {0: 0.875, 1: 1.250 },
     {0: 0.750, 1: 1.500 },
     {0: 0.625, 1: 1.750 },
     {0: 0.500, 1: 2.000 }]
    
    '''
    start, stop = np.array(start),np.array(stop)
    steps = np.linspace(start, stop, num=num, retstep=True)[1]
    args  = (start-backward*steps, stop+forward*steps, 
             num+backward+forward)
    Weights = np.round(np.linspace(*args), decimal)
    Weights = [dict([wt for wt in enumerate(w)]) for w in Weights]
    return Weights