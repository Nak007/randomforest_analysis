'''
Available methods are the followings:
 [1] error_plot_base
 [2] f1score_plot_base
 [3] gini_plot_base
 [4] dist_plot_base
 [5] ks_plot_base
 [6] gains_plot_base
 [7] cumulift_plot_base
 [8] declift_plot_base
 [9] target_plot_base
[10] eval_classifier

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 05-10-2020

'''
import numpy as np, collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.ticker import(FixedLocator, FixedFormatter, 
                              StrMethodFormatter, FuncFormatter)
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score)
from sklearn.neighbors import KernelDensity

from scipy import stats
from cycler import cycler
from itertools import product

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ["error_plot_base", 
           "f1score_plot_base", 
           "gini_plot_base",
           "dist_plot_base", 
           "ks_plot_base",
           "gains_plot_base",
           "cumulift_plot_base",
           "declift_plot_base",
           "target_plot_base", 
           "eval_classifier"]

def relocate_text(fig, ax, text, ref_text=None):
    
    '''Private function: Relocate text'''
    x_min, x_max = ax.get_xlim()
    width  = (x_max - x_min)
    render = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=render) 
    tx_box = text.get_window_extent(renderer=render)
    
    if ref_text is not None:
        x1 = ref_text.get_window_extent(renderer=render).x1
    else: x1 = ax_box.x0
    offset = np.array([tx_box.x0 - max(ax_box.x0,x1), 
                       tx_box.x1 - ax_box.x1]) / ax_box.width

    if offset[0] < 0  : offset = (offset[0] - 0.01) * width
    elif offset[1] > 0: offset = (offset[1] + 0.01) * width
    else: offset = 0
    text.set_x(text._x - offset)

def __kde__(x, a_min=0, a_max=1, kernel_kwds=None):

    '''Private function: Kernel Density Estimator'''
    x = np.array(x).reshape(-1,1)
    kwds = {"bandwidth": 0.01, "kernel": 'gaussian'}
    if kernel_kwds is not None: kwds.update(kernel_kwds)
    kde = KernelDensity(**kwds).fit(x)
    z   = np.linspace(a_min, a_max, 101)
    pdf = np.exp(kde.score_samples(z.reshape(-1,1)))
    pdf = (pdf/sum(pdf)).ravel()
    return z, pdf

def error_plot_base(y_true, y_proba, ax=None, n_ticks=6, colors=None, 
                    plot_kwds=None, kernel_kwds=None, tight_layout=True):
    
    '''
    Type I error is false positive while a type II error is false 
    negative.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 4 i.e.
        ["Class 0", "Class 1", "False Positive", "False Negative"]. 
        If None, it uses default colors from Matplotlib.
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.
    
    kernel_kwds : keywords, default=None
        Keyword arguments to be passed to `KernelDensity`. If None, it 
        is assigned to default values i.e. bandwidth=0.01, kernel=
        'gaussian'.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
 
    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.
           neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.
           score_samples
    .. [2] https://en.wikipedia.org/wiki/Type_I_and_type_II_errors

    Returns
    -------
    ax : Matplotlib axis object  
            
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5,4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(4)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth' : 2.5}
    plot_kwds = (default if plot_kwds is None 
                 else {**default, **plot_kwds})
    # =============================================================
    
    # =============================================================
    cutoffs = np.linspace(min(y_proba), max(y_proba), 101)
    cutoffs = cutoffs[(0 < cutoffs) & (cutoffs < 1)]
    cfm = np.array([confusion_matrix(y_true, (y_proba>=c)).ravel() 
                    for c in cutoffs])/len(y_true)
    tn, fp, fn, tp = cfm[:,0], cfm[:,1], cfm[:,2], cfm[:,3]
    # -------------------------------------------------------------
    kwds = {1 : dict(color=colors[2], label="False Positive"),
            2 : dict(color=colors[3], label="False Negative")}
    for n in [1,2]:
        plot_kwds = {**plot_kwds, **kwds[n]}
        ax.plot(cutoffs, cfm[:,n], **plot_kwds)
        
    ax.set_xlim(*ax.get_xlim())
    all_falses = cfm[:,[1,2]].sum(1)
    threshold  = cutoffs[np.argmin(all_falses)]
    ax.axvline(threshold, ls="--", lw=1, color="grey")
    # =============================================================

    # =============================================================
    # Update `bandwidth`
    if kernel_kwds is not None:
        bandwidth = kernel_kwds.get("bandwidth", 0.01)           
    # Instantiate and fit the KDE model
    twinx_ax, max_pdf = ax.twinx(), 0
    a_min, a_max = np.percentile(y_proba, [0, 100])
    for n in np.unique(y_true):
        z, pdf = __kde__(y_proba[y_true==n], a_min, a_max, 
                         kernel_kwds)
        kwds = dict(color=colors[n], alpha=0.2)
        twinx_ax.fill_between(z, pdf, **kwds)
        max_pdf = np.fmax(max_pdf, max(pdf))
    # -------------------------------------------------------------
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_ylim(0, max_pdf/0.15)
    # =============================================================

    # Set other attributes.
    # =============================================================
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.tick_params(axis='both', labelsize=10.5)
    # -------------------------------------------------------------
    x_locator = FixedLocator([threshold])
    ax.xaxis.set_minor_locator(x_locator)
    ax.tick_params(axis="x", which="minor", length=4, color="k")
    # -------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('Scores', fontsize=13)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    # -------------------------------------------------------------
    y_min, y_max = ax.get_ylim()
    y_max = min(1.05, y_max)
    y_min = y_min - (y_max-y_min) / 0.85 * 0.1
    ax.set_ylim(y_min, min(1.05, y_max))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    text_y = ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', 
                     ha="center", transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    args = (ax.transData, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    props = dict(boxstyle='square', facecolor='white', alpha=0)
    text = ax.text(threshold, 1.01, 'x = {:,.2%}'.format(threshold), 
                   fontsize=13, va='bottom', ha="center", 
                   bbox=props, transform=transform)
    relocate_text(plt.gcf(), ax, text, text_y)
    # -------------------------------------------------------------
    ax.legend(edgecolor="none", borderaxespad=0., markerscale=1.5,
              columnspacing=0.3, handletextpad=0.5, loc="best",
              prop=dict(size=12)) 
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def f1score_plot_base(y_true, y_proba, ax=None, n_ticks=6, colors=None, 
                      plot_kwds=None, kernel_kwds=None, tight_layout=True):
    
    '''
    The F-score, also called the F1-score, is a measure of a model’s 
    performance on a dataset. It is used to evaluate binary 
    classification system, which classifies examples into "positive" 
    or "negative". The F-score is a way of combining the precision 
    and recall of the model, and it is defined as the harmonic mean 
    of the model’s precision and recall. 

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 5 i.e.
        ["Class 0", "Class 1", "Precision", "Recall", "F1 Score"]. 
        If None, it uses default colors from Matplotlib.
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.
    
    kernel_kwds : keywords, default=None
        Keyword arguments to be passed to `KernelDensity`. If None, it 
        is assigned to default values i.e. bandwidth=0.01, kernel=
        'gaussian'.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
 
    References
    ----------
    .. [1] https://deepai.org/machine-learning-glossary-and-terms/
           f-score
    .. [2] https://scikit-learn.org/stable/modules/generated/
           sklearn.neighbors.KernelDensity.html#sklearn.neighbors.
           KernelDensity.score_samples

    Returns
    -------
    ax : Matplotlib axis object  
            
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5,4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(5)] 
              + ["grey"] if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth' : 2.5}
    plot_kwds = (default if plot_kwds is None 
                 else {**default, **plot_kwds})
    # =============================================================
    
    # =============================================================
    # Determine Presion and Recall from all possible cutoffs. Order 
    # of outputs from `confusion_matrix` are as follow; tn, fp, fn, 
    # and tp.
    cutoffs = np.linspace(min(y_proba), max(y_proba), 101)
    cutoffs = cutoffs[(0 < cutoffs) & (cutoffs < 1)]
    cfm = np.array([confusion_matrix(y_true, (y_proba>=c)).ravel() 
                    for c in cutoffs])
    p = cfm[:,-1]/cfm[:,[1,3]].sum(axis=1)
    r = cfm[:,-1]/cfm[:,[2,3]].sum(axis=1)
    f1_score = 2*(p*r) / (p+r)
    # =============================================================
    
    # Plot F1-Score curve.
    # =============================================================
    kwds = {"label":"F1 Score", "color":colors[4]}
    ax.plot(cutoffs, f1_score, **{**plot_kwds, **kwds})
    max_score, proba = max(f1_score), cutoffs[np.argmax(f1_score)]
    ax.axvline(proba, ls="--", lw=1, color="grey")
    ax.axhline(max_score, ls="--", lw=1, color="grey")
    # -------------------------------------------------------------
    plot_kwds.update(dict(color=colors[2], label="Precision"))
    ax.plot(cutoffs, p, **plot_kwds)
    # -------------------------------------------------------------
    plot_kwds.update(dict(color=colors[3], label="Recall"))
    ax.plot(cutoffs, r, **plot_kwds)
    ax.set_xlim(*ax.get_xlim())
    # =============================================================

    # =============================================================
    # Update `bandwidth`
    if kernel_kwds is not None:
        bandwidth = kernel_kwds.get("bandwidth", 0.01)           
    twinx_ax, max_pdf = ax.twinx(), 0
    a_min, a_max = np.percentile(y_proba, [0, 100])
    for n in np.unique(y_true):
        z, pdf = __kde__(y_proba[y_true==n], a_min, a_max, 
                         kernel_kwds)
        kwds = dict(color=colors[n], alpha=0.2)
        twinx_ax.fill_between(z, pdf, **kwds)
        max_pdf = np.fmax(max_pdf, max(pdf))
    # -------------------------------------------------------------
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_ylim(0, max_pdf/0.15)
    # =============================================================

    # Set other attributes.
    # =============================================================
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.tick_params(axis='both', labelsize=10.5)
    # -------------------------------------------------------------
    x_locator = FixedLocator([proba])
    ax.xaxis.set_minor_locator(x_locator)
    ax.tick_params(axis="x", which="minor", length=4, color="k")
    y_locator = FixedLocator([max_score])
    ax.yaxis.set_minor_locator(y_locator)
    ax.tick_params(axis="y", which="minor", length=4, color="k")
    # -------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('F1-Score', fontsize=13)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    # -------------------------------------------------------------
    y_min, y_max = ax.get_ylim()
    y_max = min(1.05, y_max)
    y_min = y_min - (y_max-y_min) / 0.85 * 0.1
    ax.set_ylim(y_min, min(1.05, y_max))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transData)
    transform = transforms.blended_transform_factory(*args)
    s = "f(x)\n= {:,.2f}".format(max_score)
    ax.text(1.01, max_score, s, transform=transform, fontsize=13, 
            va="center", ha="left")
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    text_y = ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', 
                     ha="center", transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    args = (ax.transData, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    props = dict(boxstyle='square', facecolor='white', alpha=0)
    text = ax.text(proba, 1.01, 'x = {:,.2%}'.format(proba), 
                   fontsize=13, va='bottom', ha="center", 
                   bbox=props, transform=transform)
    relocate_text(plt.gcf(), ax, text, text_y)
    # -------------------------------------------------------------
    ax.legend(edgecolor="none", borderaxespad=0., markerscale=1.5,
              columnspacing=0.3, handletextpad=0.5, loc="best",
              prop=dict(size=12), bbox_to_anchor=(.72, 0.65)) 
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def gini_plot_base(y_true, y_proba, ax=None, n_ticks=6, colors=None, 
                   plot_kwds=None, fill_kwds=None, tight_layout=True):

    '''
    The ROC curve is the display of sensitivity and 1 - specificity 
    for different cutoff values for probability (If the probability 
    of positive response is above the cutoff, we predict a positive 
    outcome, if not we are predicting a negative one). Each cutoff 
    value defines one point on `ROC` curve, ranging cutoff from 0 to 
    1 will draw the whole `ROC` curve. 

    The Gini coefficient is the area or ROC curve above the random 
    classifier (diagonal line) that indicates the model’s 
    discriminatory power, namely, the effectiveness of the model in 
    differentiating between target, and non-target.
        
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 2. 
        If None, it uses default colors from Matplotlib.
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.
        
    fill_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.fill_between". If None, 
        it uses default settings.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    References
    ----------
    .. [1] ROC curve, https://en.wikipedia.org/wiki/Receiver_operating
           _characteristic
    .. [2] https://towardsdatascience.com/using-the-gini-coefficient-
           to-evaluate-the-performance-of-credit-score-models-
           59fe13ef420

    Returns
    -------
    ax : Matplotlib axis object  
            
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color()] * 2 
              if colors is None else colors)
    # -------------------------------------------------------------
    bins = np.percentile(y_proba, np.arange(101))
    bins = np.unique(np.r_[[0,1], bins])
    cfm = np.array([confusion_matrix(y_true, (y_proba>=c)).ravel() 
                    for c in bins])
    fpr = cfm[:,1] / cfm[:,[0,1]].sum(1)
    tpr = cfm[:,3] / cfm[:,[2,3]].sum(1)
    auc  = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1
    # =============================================================
    
    # =============================================================
    default = {'color' : colors[0], 'linewidth' : 2.5}
    plot_kwds = (default if plot_kwds is None 
                 else {**default, **plot_kwds})
    # -------------------------------------------------------------
    default = {'alpha':0.2 , 'color':colors[1], 'y1':tpr, 'y2':fpr}
    fill_kwds = (default if fill_kwds is None 
                 else {**default, **fill_kwds})
    # -------------------------------------------------------------
    label = 'ROC ({:,.2%})'.format(auc)
    ax.plot(fpr, tpr, **{**plot_kwds, **{'label' : label}})
    label = 'Gini ({:,.2%})'.format(gini)
    ax.fill_between(fpr, **{**fill_kwds, **{'label' : label}})
    ax.plot([0,1], [0,1], **dict(ls="--", lw=1, color="grey", 
                                 label = 'Random Classifier'))
    # =============================================================

    # Set other attributes.
    # =============================================================
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.tick_params(axis='both', labelsize=10.5)
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "g(x)", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    ax.legend(loc='best', edgecolor="none", borderaxespad=0., 
              columnspacing=0.3,handletextpad=0.5,markerscale=1.5,
              prop=dict(size=12), bbox_to_anchor=(1.0, 0.5)) 
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def dist_plot_base(y_true, y_proba, ax=None, use_kde=False, bins="fd", 
                   n_ticks=6, labels=None, colors=None, bar_kwds=None, 
                   fill_kwds=None, kernel_kwds=None, tight_layout=True):

    '''
    Distribution plot is intended to illustrate the separation of 
    two distributions from binary classification, according to 
    obtained probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    use_kde : bool, default=False
        If True, a kernel-density estimate using Gaussian kernels is 
        used, otherwise histogram plot.
        
    bins : int or str, default="fd"
        Number of bins (np.histogram_bin_edges).

    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.   
    
    labels : list, default: None
        A sequence of strings providing the labels for each class. 
        If None, 'Class {n+1}' is assigned, where n is the class in y.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.

    bar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.bar". If None, it uses 
        default settings. This is relevant when `use_kde` is False.
    
    fill_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.fill_between". If None, 
        it uses default settings. This is relevant when `use_kde` is
        True.
    
    kernel_kwds : keywords, default=None
        Keyword arguments to be passed to `KernelDensity`. If None, it 
        is assigned to default values i.e. bandwidth=0.01, kernel=
        'gaussian'. If `bandwidth` is "auto", it equals to half of the 
        bin width from np.histogram_bin_edges.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
 
    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/numpy.
           histogram_bin_edges.html#numpy.histogram_bin_edges
    .. [2] https://scikit-learn.org/stable/modules/generated/
           sklearn.neighbors.KernelDensity.html#sklearn.neighbors.
           KernelDensity.score_samples
    
    Returns
    -------
    ax : Matplotlib axis object
            
    '''  
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    if labels is None: labels = ['Class 0', 'Class 1']
    default =  {'alpha': 0.2, "width": 0.8}   
    if bar_kwds is None: bar_kwds = default
    else: bar_kwds = {**default, **bar_kwds}
    # =============================================================    
        
    # Create `bins` and `dist`.
    # =============================================================
    bins = np.unique(np.histogram_bin_edges(y_proba, bins=bins))
    eps  = np.finfo(float).eps
    bins[-1] = min(bins[-1] + eps , 1 + eps)
    x = bins[:-1] + np.diff(bins)
    # -------------------------------------------------------------
    # Distribution of samples.
    data = [y_proba[y_true==n] for n in [0,1]]
    kwds = dict(bins=bins, range=(0, 1+eps))
    dist = [np.histogram(data[n], **kwds)[0]
            /data[n].size for n in [0,1]]
    # =============================================================
  
    # Plot horizontal bar.
    # =============================================================
    text  = r' : {:.0%}$\leq$x$\leq${:.0%}'.format
    if use_kde==False:
        width = np.diff(bins)[0] * bar_kwds.get("width", 0.8)
        bar_kwds.update({"width": width})
        for n in [0,1]:
            label = labels[n] + text(min(data[n]), max(data[n]))
            bar_kwds.update({"color" : colors[n], "label" : label}) 
            ax.bar(x, dist[n], **bar_kwds)
            kwds = bar_kwds.copy()
            kwds.update({"facecolor" : "none", 
                         "edgecolor" : colors[n], 
                         "linewidth" : bar_kwds.get("linewidth",1.2),
                         "linestyle" : bar_kwds.get("linestyle","-"),
                         "label"     : None, 
                         "alpha"     : 1}) 
            ax.bar(x, dist[n], **kwds)
        ax.set_ylabel('% of Samples', fontsize=13)
    # -------------------------------------------------------------
    elif use_kde==True:
        # Update `bandwidth`
        if kernel_kwds is not None:
            bandwidth = kernel_kwds.get("bandwidth", 0.01)
            if bandwidth == "auto": 
                bandwidth = np.diff(bins)[0]/2
                kernel_kwds.update({"bandwidth": bandwidth}) 
                
        # Instantiate and fit the KDE model
        a_min, a_max = np.percentile(y_proba, [0, 100])
        for n in np.unique(y_true):
            label = labels[n] + text(min(data[n]), max(data[n]))
            z, pdf = __kde__(y_proba[y_true==n], a_min, a_max, 
                             kernel_kwds)
            kwds = dict(color=colors[n], alpha=0.2, label=label)
            if fill_kwds is not None: kwds.update(fill_kwds)
            ax.fill_between(z, pdf, **kwds)
            ax.plot(z, pdf, **dict(color=colors[n], lw=1.2))
        ax.set_ylabel('Density', fontsize=13)
    # ============================================================= 

    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=10.5)
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(t)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    ax.set_xlim(*ax.get_xlim())
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max/0.8)
    # -------------------------------------------------------------
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    ax.legend(loc='best', borderaxespad=0., columnspacing=0.3, 
              handletextpad=0.5, prop=dict(size=12))
    if tight_layout: plt.tight_layout()
    # =============================================================

    return ax

def ks_plot_base(y_true, y_proba, ax=None, bins=20, n_ticks=6, 
                 labels=None, colors=None, plot_kwds=None, 
                 kernel_kwds=None, tight_layout=True):

    '''
    In statistics, the Kolmogorov–Smirnov test is a nonparametric 
    test of the equality of continuous, one-dimensional probability 
    distributions that can be used to compare a sample with a 
    reference probability distribution (one-sample KS test), or to 
    compare two samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    bins : int or str, default=20
        Number of bins (np.histogram_bin_edges).

    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.   
    
    labels : list, default: None
        A sequence of strings providing the labels for each class. 
        If None, 'Class {n+1}' is assigned, where n is the class in y.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.
    
    kernel_kwds : keywords, default=None
        Keyword arguments to be passed to `KernelDensity`. If None, it 
        is assigned to default values i.e. bandwidth=0.01, kernel=
        'gaussian'. If `bandwidth` is "auto", it equals to half of the 
        bin width from np.histogram_bin_edges.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test
    .. [2] https://scikit-learn.org/stable/modules/generated/
           sklearn.neighbors.KernelDensity.html#sklearn.neighbors.
           KernelDensity.score_samples
        
    Returns
    -------
    ax : Matplotlib axis object
            
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(3)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    if labels is None: labels = ['Class 0','Class 1']
    default =  {'linewidth': 2.5} 
    if plot_kwds is None: plot_kwds = default
    else: plot_kwds = {**default, **plot_kwds}
    # ============================================================= 
    
    # ============================================================= 
    bins = np.unique(np.histogram_bin_edges(y_proba, bins=bins))
    eps  = np.finfo(float).eps
    bins[-1] = min(bins[-1] + eps , 1 + eps)
    # -------------------------------------------------------------
    # Cumulative (%) distribution.
    cumu_dist = []
    for n in range(2):
        dist = np.histogram(y_proba[y_true==n], bins)[0]
        cumu_dist += [np.hstack(([0], np.cumsum(dist)))/sum(dist)]
    # -------------------------------------------------------------   
    # Determine `ks_cutoff` from each threshold.
    ks = abs(cumu_dist[0] - cumu_dist[1])
    max_ks = np.argmax(ks)
    cutoff, ks = bins[max_ks], ks[max_ks]
    # =============================================================

    # =============================================================
    # Plot cumulative distribution.
    for n in [0,1]:
        plot_kwds.update({'color': colors[n], 'label': labels[n]})
        ax.plot(bins, cumu_dist[n], **plot_kwds)
    # Optimal KS
    ax.plot([cutoff]*2, [cumu_dist[0][max_ks], 
                         cumu_dist[1][max_ks]], lw = 1, ls="--", 
            label="KS = {:.2%}".format(ks), color="k")
    ax.set_xlim(*ax.get_xlim())
    # -------------------------------------------------------------
    # Update `bandwidth`
    if kernel_kwds is not None:
        bandwidth = kernel_kwds.get("bandwidth", 0.01)
        if bandwidth == "auto": 
            bandwidth = np.diff(bins)[0]/2
            kernel_kwds.update({"bandwidth": bandwidth}) 
    # -------------------------------------------------------------            
    # Instantiate and fit the KDE model
    twinx_ax, max_pdf = ax.twinx(), 0
    a_min, a_max = np.percentile(y_proba, [0, 100])
    for n in np.unique(y_true):
        z, pdf = __kde__(y_proba[y_true==n], a_min, a_max, None)
        kwds = dict(color=colors[n], alpha=0.2)
        twinx_ax.fill_between(z, pdf, **kwds)
        max_pdf = np.fmax(max_pdf, max(pdf))
    # -------------------------------------------------------------
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_ylim(0, max_pdf/0.15)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=10.5)
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(t)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    x_locator = FixedLocator([cutoff])
    ax.xaxis.set_minor_locator(x_locator)
    ax.tick_params(axis="x", which="minor", length=4, color="k")
    # -------------------------------------------------------------
    ax.set_ylabel('Cumulative Distribution', fontsize=13)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    y_min, y_max = ax.get_ylim()
    y_max = min(1.05, y_max)
    y_min = y_min - (y_max-y_min) / 0.85 * 0.1
    ax.set_ylim(y_min, min(1.05, y_max))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    text_y = ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', 
                     ha="center", transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    ax.legend(loc='best', borderaxespad=1, columnspacing=0.3, 
              handletextpad=0.5, framealpha=0, prop=dict(size=12),
              bbox_to_anchor=(1.05, 0.5))
    # -------------------------------------------------------------
    ax.axvline(cutoff, ls="--", lw=1, zorder=-1, color="#cccccc")
    args = (ax.transData, ax.transAxes)
    trans= transforms.blended_transform_factory(*args)
    # Relocate Text
    kwds = dict(fontsize=13,transform=trans,va='bottom',ha="center")
    text = ax.text(cutoff, 1.01, "x = {:.2%}".format(cutoff), **kwds)
    relocate_text(plt.gcf(), ax, text, text_y)
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def gains_plot(y_true, y_proba, ax=None, bins=20, n_ticks=6, 
               colors=None, plot_kwds=None, kernel_kwds=None, 
               tight_layout=True):

    '''
    Gain chart is used to evaluate performance of classification 
    model. It measures how much better one can expect to do with 
    the predictive model comparing without a model. It plots the 
    cumulative percentage of samples (x-axis) against the 
    cumulative percentage of targeted samples (y-axis).
        
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
  
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    bins : int, default=20
        Number of bins.

    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.   
 
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    References
    ----------
    .. [1] https://www.listendata.com/2014/08/excel-template-gain
           -and-lift-charts.html
    .. [2] https://community.tibco.com/wiki/gains-vs-roc-curves-do
           -you-understand-difference
        
    Returns
    -------
    ax : Matplotlib axis object
                  
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth' : 2.5, 'linestyle' : '-'}
    plot_kwds = (default if plot_kwds is None 
                 else {**default, **plot_kwds})
    # =============================================================
    
    # Data preparation
    # =============================================================
    q = np.linspace(0, 100, bins+1)
    bins = np.unique(np.percentile(1 - y_proba, q))
    bins[0], bins[-1] = 0, 1 + np.finfo(float).eps
    # -------------------------------------------------------------
    dist, cumdist = [], []
    for n in range(2):
        a = 1 - y_proba[y_true==n]
        hist  = np.hstack(([0], np.histogram(a, bins)[0])) 
        dist += [hist/sum(hist)]
        cumdist += [np.cumsum(dist[n])]
    # -------------------------------------------------------------
    distsum = np.hstack(([0], np.histogram(1 - y_proba, bins)[0]))
    cumsum  = np.cumsum(distsum)/sum(distsum)
    # =============================================================

    # Plot chart
    # =============================================================
    kwds = {**plot_kwds, **dict(color=colors[1],label="% Target")}
    ax.plot(cumsum, cumdist[1], **kwds)
    kwds = dict(color="grey", ls="-", lw=1, label="Random Classifier")
    ax.plot([0,1], [0,1], **kwds)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(-0.19, y_max)
    # -------------------------------------------------------------
    twinx_ax, width = ax.twinx(), np.diff(cumsum)[0]*0.8
    for n in range(2):
        a = dist[n]*(y_true==n).sum()
        a = np.where(a==0, np.nan, a)
        kwds = dict(width=width, ec="none", fc=colors[n], alpha=.2)
        twinx_ax.bar(cumsum, a, **kwds)
        kwds.update(dict(fc="none", lw=0.5, ec=colors[n], alpha=1.))
        twinx_ax.bar(cumsum, a, **kwds)
    # -------------------------------------------------------------
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_ylim(0, twinx_ax.get_ylim()[1]/0.15)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    args = (ax.transData, ax.transAxes)
    trans= transforms.blended_transform_factory(*args)
    y_pct = np.mean(y_true)
    ax.axvline(y_pct, ls="--", lw=1, zorder=-1, color="#cccccc")
    s = "x = {:.2%}".format(y_pct)
    # Relocate Text
    kwds = dict(fontsize=13, transform=trans, va='bottom', ha="center")
    relocate_text(plt.gcf(), ax, ax.text(y_pct, 1.01, s, **kwds))
    ax.xaxis.set_minor_locator(FixedLocator([y_pct]))
    ax.tick_params(axis="x", which="minor", length=3.5, color="k")
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transData)
    trans= transforms.blended_transform_factory(*args)
    estimate = np.interp(y_pct, cumsum, cumdist[1])
    ax.axhline(estimate, ls="--", lw=1, zorder=-1, color="#cccccc")
    kwds = dict(fontsize=13, transform=trans, va='center', ha="left")
    ax.text(1.01, estimate, "f(x)\n= {:.2%}".format(estimate), **kwds)
    ax.yaxis.set_minor_locator(FixedLocator([estimate]))
    ax.tick_params(axis="y", which="minor", length=3.5, color="k")
    # -------------------------------------------------------------
    ax.set_xlabel('% of Samples (Support)', fontsize=13)
    ax.set_ylabel('% of Targets', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    ax.tick_params(axis='both', labelsize=10.5)
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(t)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    ax.legend(loc='center right', borderaxespad=1, framealpha=0,  
              columnspacing=0.3, handletextpad=0.5, 
              prop=dict(size=12), bbox_to_anchor=(1.05, 0.4))
    if tight_layout: plt.tight_layout()
    # =============================================================
   
    return ax

def gains_plot_base(y_true, y_proba, ax=None, bins=20, n_ticks=6, 
                    colors=None, plot_kwds=None, kernel_kwds=None, 
                    tight_layout=True):

    '''
    Gain chart is used to evaluate performance of classification 
    model. It measures how much better one can expect to do with 
    the predictive model comparing without a model. It plots the 
    cumulative percentage of samples (x-axis) against the 
    cumulative percentage of targeted samples (y-axis).
        
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
  
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    bins : int, default=20
        Number of bins.

    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.   
 
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot". If None, it uses 
        default settings.
        
    kernel_kwds : keywords, default=None
        Keyword arguments to be passed to `KernelDensity`. If None, it 
        is assigned to default values i.e. bandwidth=0.01, kernel=
        'gaussian'. 
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    References
    ----------
    .. [1] https://www.listendata.com/2014/08/excel-template-gain
           -and-lift-charts.html
    .. [2] https://community.tibco.com/wiki/gains-vs-roc-curves-do
           -you-understand-difference
        
    Returns
    -------
    ax : Matplotlib axis object
                  
    '''
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth' : 2.5, 'linestyle' : '-'}
    plot_kwds = (default if plot_kwds is None 
                 else {**default, **plot_kwds})
    # =============================================================
    
    # Data preparation
    # =============================================================
    q = np.linspace(0, 100, bins+1)
    bins = np.unique(np.percentile(1 - y_proba, q))
    bins[0], bins[-1] = 0, 1 + np.finfo(float).eps
    # -------------------------------------------------------------
    dist, cumdist = [], []
    for n in range(2):
        a = 1 - y_proba[y_true==n]
        hist  = np.hstack(([0], np.histogram(a, bins)[0])) 
        dist += [hist/sum(hist)]
        cumdist += [np.cumsum(dist[n])]
    # -------------------------------------------------------------
    distsum = np.hstack(([0], np.histogram(1 - y_proba, bins)[0]))
    cumsum  = np.cumsum(distsum)/sum(distsum)
    # =============================================================

    # Plot chart
    # =============================================================
    kwds = {**plot_kwds, **dict(color=colors[1],label="% Target")}
    ax.plot(cumsum, cumdist[1], **kwds)
    kwds = dict(label="Random Classifier", ls="-", lw=1, 
                color="grey")
    ax.plot([0,1], [0,1], **kwds)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(-0.19, y_max)
    ax.set_xlim(*ax.get_xlim())
    # -------------------------------------------------------------
    # Update `bandwidth`
    if kernel_kwds is not None:
        bandwidth = kernel_kwds.get("bandwidth", 0.01)
        if bandwidth == "auto": 
            bandwidth = np.diff(bins)[0]/2
            kernel_kwds.update({"bandwidth": bandwidth}) 
    # -------------------------------------------------------------            
    # Instantiate and fit the KDE model
    twinx_ax, max_pdf = ax.twinx(), 0
    for n in np.unique(y_true):
        a = np.unique(1-y_proba[y_true==n])
        a = [stats.percentileofscore(1-y_proba,k) for k in a]
        z, pdf = __kde__(np.array(a)/100, 0, 1, None)
        kwds = dict(color=colors[n], alpha=0.2)
        #pdf = pdf * len(a) / len(y_true) # <-- weighted pdf
        twinx_ax.fill_between(z, pdf, **kwds)
        max_pdf = np.fmax(max_pdf, max(pdf))
    # -------------------------------------------------------------
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_ylim(0, max_pdf/0.15)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=10.5)
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(t)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    ax.set_xlabel('Cumulative % of Samples', fontsize=13)
    ax.set_ylabel('Cumulative % of Targets', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    y_pct= np.mean(y_true)
    args = (ax.transAxes, ax.transData)
    trans= transforms.blended_transform_factory(*args)
    estimate = np.interp(y_pct, cumsum, cumdist[1])
    ax.axhline(estimate, ls="--",lw=1, zorder=-1, color="#cccccc")
    kwds = dict(fontsize=13,transform=trans,va='center',ha="left")
    ax.text(1.01,estimate,"f(x)\n= {:.2%}".format(estimate),**kwds)
    ax.yaxis.set_minor_locator(FixedLocator([estimate]))
    ax.tick_params(axis="y", which="minor", length=3.5, color="k")
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    text_y = ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', 
                     ha="center", transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    args = (ax.transData, ax.transAxes)
    trans= transforms.blended_transform_factory(*args)
    ax.axvline(y_pct, ls="--", lw=1, zorder=-1, color="#cccccc")
    # Relocate Text
    text = ax.text(y_pct, 1.01, "x = {:.2%}".format(y_pct), 
                   fontsize=13, va='bottom', ha="center", 
                   transform=trans)
    relocate_text(plt.gcf(), ax, text, text_y)
    ax.xaxis.set_minor_locator(FixedLocator([y_pct]))
    ax.tick_params(axis="x", which="minor", length=3.5, color="k")
    # -------------------------------------------------------------
    ax.legend(loc='center right', borderaxespad=1, framealpha=0,  
              columnspacing=0.3, handletextpad=0.5, 
              prop=dict(size=12), bbox_to_anchor=(1.05, 0.4))
    if tight_layout: plt.tight_layout()
    # =============================================================
   
    return ax

def LiftTable(y_true, y_proba, step=10):
    
    '''
    ** Private Function **
    Function determines ith percentile based on defined incremental 
    change in percentile (`step`). Then, it collaspes bins that have 
    the same percentile values into one. Then, other parameters such 
    as cumulative percentage, lift, or target rate are calculated, 
    accordingly.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    step : int, default=10
        Incremental change of percentile and must be 0 < step < 1.
 
    Returns
    -------
    collections.namedtuple
    
    '''
    if isinstance(step, (int,float)):
        if (step <= 0.) | (step >=100.):
            raise ValueError(f'`step` must be 0 < step < 1. '
                             f'Got {step} instead.')
    else: raise ValueError(f'`step` must be either integer or float' 
                           f'Got {type(step)} instead.')
    
    # Compute sequence of percentiles.
    q = np.arange(100-step, -step, -step)
    q = np.unique(np.where(q<0, 0, q))
    bins = np.percentile(y_proba, q)[::-1]

    # Cumulative number of targets and samples.
    cum_n_target = [y_true[(y_proba>=n)].sum() for n in bins]
    cum_n_target = np.array(cum_n_target).astype(int)
    cum_n_sample = np.array([(y_proba>=n).sum() for n in bins])

    # Cumulative percentage of targets and samples.
    cum_p_target = cum_n_target / sum(y_true)
    cum_p_sample = cum_n_sample / len(y_true)
    cum_lift = cum_p_target / cum_p_sample
    
    # Number of samples in each bin.
    count = lambda a : np.hstack((a[0], np.diff(a)))
    n_target, n_sample = count(cum_n_target), count(cum_n_sample)
    
    # Percentage of targets, and lift (per bin).
    target_rate = n_target / n_sample
    dec_lift = (n_target/sum(y_true)) / (n_sample/len(y_true))
    
    keys = ['bins', 'cum_n_target', 'cum_n_sample', 'cum_p_target', 
            'cum_p_sample', 'n_target', 'n_sample', 'target_rate', 
            'cum_lift', 'dec_lift']
    Lift = collections.namedtuple('Lift', keys)  
    return Lift(bins = bins,
                cum_n_target = cum_n_target,
                cum_n_sample = cum_n_sample,
                cum_p_target = cum_p_target,
                cum_p_sample = cum_p_sample,
                n_target     = n_target,
                n_sample     = n_sample,
                target_rate  = target_rate,
                cum_lift     = cum_lift,
                dec_lift     = dec_lift)

def cumulift_plot_base(y_true, y_proba, ax=None, colors=None, bound=None, 
                       step=10, bar_kwds=None, font_kwds=None, 
                       anno_format=None, decimal=0, tight_layout=True):

    '''
    Cumulative lift measures how much better one can expect to do with 
    the predictive model comparing without a model (randomness). Lift 
    is the ratio of cumulative targets to cumulative samples. Moreover, 
    probability must be ordered in descending manner before grouping 
    into deciles.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.
       
    bound : (float, float), default=None
        The lower and upper percentile range of the bins. If None, 
        range is simply [0,100]. Values outside the range are ignored. 
        The first element of the range must be less than or equal to 
        the second. 
        
    step : int, default=10
        Incremental change of percentile and must be 0 < step < 1.
    
    bar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.bar". If None, it uses 
        default settings.
        
    font_kwds : keywords, default=None
        Keyword arguments to be passed to `ax.annotate`. If None, it 
        uses default settings.
        
    anno_format : function, default=None
        String formatting function method. If None, it uses default 
        settings i.e. '{:,.2f}'.format.   
    
    decimal : int, default=0
        Decimal places for displaying bad rate.
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    Returns
    -------
    ax : Matplotlib axis object
  
    '''
    # =============================================================
    if bound is None: bound = [0, 100] 
    if not isinstance(step, (int,float)): bins = 10
    if not callable(anno_format): anno_format = '{:,.2f}'.format 
    # -------------------------------------------------------------
    n_bins = np.ceil((bound[1] - bound[0])/step)
    width  = max(6.5, n_bins * 0.7)
    if ax is None: ax = plt.subplots(figsize=(width, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth': 1.2, "alpha": 1, "width": 0.7}
    bar_kwds = (default if bar_kwds is None 
                else {**default, **bar_kwds})
    # -------------------------------------------------------------
    default = dict(textcoords="offset points", fontsize=12, 
                   ha='center', va='bottom', xytext=(0,3))
    font_kwds = (default if font_kwds is None 
                 else {**default,**font_kwds})
    # =============================================================
    
    # Plot `Lift`.
    # =============================================================
    # `lift` is determined by interpolation.
    Lift = LiftTable(y_true, y_proba, step=step)
    decile = 100 - bound[1] + np.arange(1, n_bins + 1) * step
    decile = np.fmin(decile, 100 - bound[0]) / 100
    xp, fp = Lift.cum_p_sample, Lift.cum_lift
    lift = np.interp(decile, xp, fp)
    x = np.arange(len(decile))
    # -------------------------------------------------------------
    kwds = dict(color=colors[0], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none", bottom=1)
    bar0 = ax.bar(x, lift-1, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none", edgecolor=colors[0], alpha=1))
    ax.bar(x, lift-1, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    scale, y_scale = 0.65, 0.8
    y_min, y_max = ax.get_ylim()
    y_max = np.fmax(max(lift) / y_scale, 0.5)
    y_min = 1 - (y_max - 1)/scale * (1-scale)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.8, len(x)-0.2)
    ax.spines["left"].set_bounds((1, y_max))
    # -------------------------------------------------------------
    font_kwds.update(dict(color=colors[0]))
    for n,s in zip(x,lift): 
        ax.annotate(anno_format(s), (n,s), **font_kwds)
    # =============================================================
    
    # =============================================================
    y = np.interp(decile, Lift.cum_p_sample, Lift.n_target)
    y = y / sum(Lift.n_target)
    # -------------------------------------------------------------
    twinx_ax = ax.twinx()
    kwds = dict(color=colors[1], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none")
    bar1 = twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none",edgecolor=colors[1],alpha=1))
    twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    decimal = decimal if isinstance(decimal, int) else 0
    font_kwds.update(dict(color=colors[1]))
    for n,s in zip(x, y): 
        twinx_ax.annotate(("{:." + str(decimal) + "%}").format(s), 
                          (n,s), **font_kwds)
    # -------------------------------------------------------------
    tw_y_min, tw_y_max = twinx_ax.get_ylim()
    tw_y_max = max(y) / y_scale
    tw_y_max = tw_y_max + (tw_y_max / (1-scale))
    twinx_ax.set_ylim(0, tw_y_max if tw_y_max > 0 else 1)
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_frame_on(False)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    args = (ax.transAxes, ax.transData)
    transform = transforms.blended_transform_factory(*args)
    ax.text(1.01, 1, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    ax.axhline(1, ls="-", lw=1, zorder=1, color="k")
    # -------------------------------------------------------------
    ax.tick_params(axis='both', labelsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['{:.0%}'.format(n) for n in decile])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    # -------------------------------------------------------------
    ax.yaxis.set_minor_locator(FixedLocator([1]))
    ax.tick_params(axis="y", which="minor", length=3.5, color="k")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[(yticks>=1) & (yticks<y_max)])
    # -------------------------------------------------------------
    ax.set_xlabel('Cumulative % of Samples', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    ax.set_ylabel('Cumulative Lift', fontsize=13)
    y_min, y_max = ax.get_ylim()
    offset = (1 - y_min)/(y_max - y_min)
    incr = ((y_max - 1)/(y_max - y_min)) * 0.5
    text = ax.yaxis.get_label()
    left = text.get_position()[0]
    text.set_position((left, offset + incr))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    patches = [bar0[0], bar1[0]]
    labels  = ["Lift", "% Target"]
    ax.legend(patches, labels, loc='upper right', edgecolor="none",
              prop=dict(weight="ultralight", size=12), ncol=2, 
              borderaxespad=0., bbox_to_anchor=(1, 1.07), 
              columnspacing=0.5, handletextpad=0.5)
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def declift_plot_base(y_true, y_proba, ax=None, colors=None, bound=None, 
                      step=10, bar_kwds=None, font_kwds=None, 
                      anno_format=None, decimal=0, tight_layout=True):

    ''' 
    Unlike cumulative lift, decile lift uses target and support based 
    on the given decile (not cumulative). It measures how much gains 
    of additional samples would contribute to the prediction comparing 
    without a model (randomness). If `decile lift` is having value 
    larger than 1, that means such decile is better at predicting 
    target than random selection, and if it is less than 1, the model 
    is better at predicting non-target.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.
       
    bound : (float, float), default=None
        The lower and upper percentile range of the bins. If None, 
        range is simply [0,100]. Values outside the range are ignored. 
        The first element of the range must be less than or equal to 
        the second. 
        
    step : int, default=10
        Incremental change of percentile and must be 0 < step < 1.
    
    bar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.bar". If None, it uses 
        default settings.
        
    font_kwds : keywords, default=None
        Keyword arguments to be passed to `ax.annotate`. If None, it 
        uses default settings.
        
    anno_format : function, default=None
        String formatting function method. If None, it uses default 
        settings i.e. '{:,.2f}'.format.
        
    decimal : int, default=0
        Decimal places for displaying bad rate.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    Returns
    -------
    ax : Matplotlib axis object
  
    '''
    # =============================================================
    if bound is None: bound = [0, 100] 
    if not isinstance(step, (int,float)): bins = 10
    if not callable(anno_format): anno_format = '{:,.2f}'.format 
    # -------------------------------------------------------------
    n_bins = np.ceil((bound[1] - bound[0])/step)
    width  = max(6.5, n_bins * 0.7)
    if ax is None: ax = plt.subplots(figsize=(width, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth': 1.2, "alpha": 1, "width": 0.7}
    bar_kwds = (default if bar_kwds is None 
                else {**default, **bar_kwds})
    # -------------------------------------------------------------
    default = dict(textcoords="offset points", fontsize=12, 
                   ha='center', va='bottom', xytext=(0,5))
    font_kwds = (default if font_kwds is None 
                 else {**default,**font_kwds})
    # =============================================================
    
    # Plot `Lift`.
    # =============================================================
    # `lift` is determined by interpolation.
    Lift   = LiftTable(y_true, y_proba, step=step)
    decile = 100 - bound[1] + np.arange(1, n_bins + 1) * step
    decile = np.fmin(decile, 100 - bound[0]) / 100
    xp, fp = Lift.cum_p_sample, Lift.dec_lift
    lift   = np.interp(decile, xp, fp)
    x      = np.arange(len(decile))
    # -------------------------------------------------------------
    kwds = dict(color=colors[0], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none", bottom=0)
    bar0 = ax.bar(x, lift, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none", edgecolor=colors[0], alpha=1))
    ax.bar(x, lift, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    scale, y_scale = 0.65, 0.8
    y_min, y_max = ax.get_ylim()
    y_max = np.fmax(max(lift) / y_scale, 0.5)
    y_min = -y_max/scale * (1-scale)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.8, len(x)-0.2)
    ax.spines["left"].set_bounds((0, y_max))
    # -------------------------------------------------------------
    font_kwds.update(dict(color=colors[0]))
    for n,s in zip(x, lift): 
        ax.annotate(anno_format(s), (n,s), **font_kwds)
    ax.axhline(1, ls="--", lw=1, zorder=-1, color="#cccccc")
    # =============================================================
    
    # =============================================================
    y = np.interp(decile, Lift.cum_p_sample, Lift.n_target)
    y = y/sum(Lift.n_target)
    # -------------------------------------------------------------
    twinx_ax = ax.twinx()
    kwds = dict(color=colors[1], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none")
    bar1 = twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none",edgecolor=colors[1],alpha=1))
    twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    decimal = decimal if isinstance(decimal, int) else 0
    font_kwds.update(dict(color=colors[1]))
    for n,s in zip(x, y): 
        twinx_ax.annotate(("{:." + str(decimal) + "%}").format(s),
                          (n,s), **font_kwds)
    # -------------------------------------------------------------
    tw_y_min, tw_y_max = twinx_ax.get_ylim()
    tw_y_max = max(y) / y_scale
    tw_y_max = tw_y_max + (tw_y_max / (1-scale))
    twinx_ax.set_ylim(0, tw_y_max if tw_y_max > 0 else 1)
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_frame_on(False)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['{:.0%}'.format(n) for n in decile])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    # -------------------------------------------------------------
    ax.yaxis.set_minor_locator(FixedLocator([1]))
    ax.tick_params(axis="y", which="minor", length=3.5, color="k")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[(yticks>=0) & (yticks<y_max)])
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    ax.set_xlabel('% of Samples (Decile)', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    ax.set_ylabel('Decile Lift', fontsize=13)
    y_min, y_max = ax.get_ylim()
    offset = abs(y_min)/(y_max - y_min)
    incr = (y_max/(y_max - y_min)) * 0.5
    text = ax.yaxis.get_label()
    left = text.get_position()[0]
    text.set_position((left, offset+incr))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transData)
    transform = transforms.blended_transform_factory(*args)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    ax.axhline(0, ls="-", lw=1, zorder=1, color="k")
    # -------------------------------------------------------------
    patches = [bar0[0], bar1[0]]
    labels  = ["Lift", "% Target"]
    ax.legend(patches, labels, loc='upper right', edgecolor="none",
              prop=dict(weight="ultralight", size=12), ncol=2, 
              borderaxespad=0., bbox_to_anchor=(1, 1.07), 
              columnspacing=0.5, handletextpad=0.5)
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

def target_plot_base(y_true, y_proba, ax=None, colors=None, bound=None, 
                     step=10, bar_kwds=None, font_kwds=None, 
                     anno_format=None, decimal=0, tight_layout=True):

    '''
    The target ratio or target rate is the number of actual targets 
    over number of targeted samples in each decile. It shows how well 
    predictive model can assign positive response with high 
    probability and negative response with low probability. If the 
    model has a high discriminatory power, the target rate should 
    form a downward pattern.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.
       
    bound : (float, float), default=None
        The lower and upper percentile range of the bins. If None, 
        range is simply [0,100]. Values outside the range are ignored. 
        The first element of the range must be less than or equal to 
        the second. 
        
    step : int, default=10
        Incremental change of percentile and must be 0 < step < 1.
    
    bar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.bar". If None, it uses 
        default settings.
        
    font_kwds : keywords, default=None
        Keyword arguments to be passed to `ax.annotate`. If None, it 
        uses default settings.
        
    anno_format : function, default=None
        String formatting function method. If None, it uses default 
        settings i.e. '{:.0%}'.format.
        
    decimal : int, default=0
        Decimal places for displaying bad rate.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    Returns
    -------
    ax : Matplotlib axis object
  
    '''
    
    # =============================================================
    if bound is None: bound = [0, 100] 
    if not isinstance(step, (int,float)): bins = 10
    if not callable(anno_format): anno_format = '{:.0%}'.format 
    # -------------------------------------------------------------
    n_bins = np.ceil((bound[1] - bound[0])/step)
    width  = max(6.5, n_bins * 0.7)
    if ax is None: ax = plt.subplots(figsize=(width, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    default = {'linewidth': 1.2, "alpha": 1, "width": 0.7}
    bar_kwds = (default if bar_kwds is None 
                else {**default, **bar_kwds})
    # -------------------------------------------------------------
    default = dict(textcoords="offset points", fontsize=12, 
                   ha='center', va='bottom', xytext=(0,5))
    font_kwds = (default if font_kwds is None 
                 else {**default,**font_kwds})
    # =============================================================
   
    # Plot `Lift`.
    # =============================================================
    # `lift` is determined by interpolation.
    Lift   = LiftTable(y_true, y_proba, step=step)
    decile = 100 - bound[1] + np.arange(1, n_bins + 1) * step
    decile = np.fmin(decile, 100 - bound[0]) / 100
    xp, fp = Lift.cum_p_sample, Lift.target_rate
    lift   = np.interp(decile, xp, fp)
    x      = np.arange(len(decile))
    # -------------------------------------------------------------
    kwds = dict(color=colors[0], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none", bottom=0)
    bar0 = ax.bar(x, lift, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none", edgecolor=colors[0], alpha=1))
    ax.bar(x, lift, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    scale, y_scale = 0.65, 0.8
    y_min, y_max = ax.get_ylim()
    y_max = np.fmax(max(lift) / y_scale, 0.5)
    y_min = -y_max/scale * (1-scale)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.8, len(x)-0.2)
    ax.spines["left"].set_bounds((0, y_max))
    # -------------------------------------------------------------
    font_kwds.update(dict(color=colors[0]))
    for n,s in zip(x, lift): 
        ax.annotate(anno_format(s), (n,s), **font_kwds)
    # =============================================================
    
    # =============================================================
    y = np.interp(decile, Lift.cum_p_sample, Lift.n_target)
    y = y/sum(Lift.n_target)
    # -------------------------------------------------------------
    twinx_ax = ax.twinx()
    kwds = dict(color=colors[1], alpha=bar_kwds.get("alpha", 1), 
                edgecolor="none")
    bar1 = twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    kwds.update(dict(facecolor="none",edgecolor=colors[1],alpha=1))
    twinx_ax.bar(x, y, **{**bar_kwds, **kwds})
    # -------------------------------------------------------------
    decimal = decimal if isinstance(decimal, int) else 0
    font_kwds.update(dict(color=colors[1]))
    for n,s in zip(x, y): 
        twinx_ax.annotate(("{:." + str(decimal) + "%}").format(s), 
                          (n,s), **font_kwds)
    # -------------------------------------------------------------
    tw_y_min, tw_y_max = twinx_ax.get_ylim()
    tw_y_max = max(y) / y_scale
    tw_y_max = tw_y_max + (tw_y_max / (1-scale))
    twinx_ax.set_ylim(0, tw_y_max if tw_y_max > 0 else 1)
    twinx_ax.spines["right"].set_visible(False)
    twinx_ax.spines["top"].set_visible(False)
    twinx_ax.axes.yaxis.set_visible(False)
    twinx_ax.set_frame_on(False)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['{:.0%}'.format(n) for n in decile])
    t = ticker.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(t)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[(yticks>=0) & (yticks<=1)])
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    ax.set_xlabel('% of Samples (Decile)', fontsize=13)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    ax.set_ylabel('Target rate', fontsize=13)
    y_min, y_max = ax.get_ylim()
    offset = abs(y_min)/(y_max - y_min)
    incr = (y_max/(y_max - y_min)) * 0.5
    text = ax.yaxis.get_label()
    left = text.get_position()[0]
    text.set_position((left, offset+incr))
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transData)
    transform = transforms.blended_transform_factory(*args)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    ax.axhline(0, ls="-", lw=1, zorder=1, color="k")
    # -------------------------------------------------------------
    patches = [bar0[0], bar1[0]]
    labels  = ["Target rate", "% Target"]
    ax.legend(patches, labels, loc='upper right', edgecolor="none",
              prop=dict(weight="ultralight", size=12), ncol=2, 
              borderaxespad=0., bbox_to_anchor=(1, 1.07), 
              columnspacing=0.5, handletextpad=0.5)
    if tight_layout: plt.tight_layout()
    # =============================================================
  
    return ax

def create_cmap(c1=(23,10,8), c2=(255,255,255)):
    
    '''
    Creating `matplotlib.colors.Colormap` (Colormaps) with 
    two colors.

    Parameters
    ----------
    c1 : `hex code` or (r,g,b), default=(23,10,8)
        The beginning color code.
    
    c2 : `hex code` or (r,g,b), default=(255,255,255)
        The ending color code.
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
 
    '''
    to_rgb = lambda c : tuple(int(c.lstrip('#')[i:i+2],16) 
                           for i in (0,2,4))

    # Convert to RGB.
    if isinstance(c1,str): c1 = to_rgb(c1)
    if isinstance(c2,str): c2 = to_rgb(c2)
    colors = np.ones((256,4))
    for i in range(3):
        colors[:,i] = np.linspace(c1[i]/256,c2[i]/256,256)
    colors = colors[np.arange(255,-1,-1),:]
    return ListedColormap(colors)

def Axes2grid(n_axes=4, n_cols=2, figsize=(6.5,4), locs=None, spans=None):
    
    '''
    Create axes at specific location inside specified regular grid.
    
    Parameters
    ----------
    n_axes : int, default=4
        Number of axes required to fit inside grid.
    
    n_cols : int, default=2
        Number of grid columns in which to place axis. This will also 
        be used to calculate number of rows given number of axes 
        (`n_axes`).
    
    figsize : (float, float), default=(6.5,4)
        Width, height in inches for an axis.
    
    locs : list of (int, int), default=None
        locations to place each of axes within grid i.e.(row, column). 
        If None, locations are created, where placement starts from 
        left to right, and then top to bottom.

    spans : list of (int, int), default=None
        List of tuples for axis to span. First entry is number of rows 
        to span to the right while the second entry is number of 
        columns to span downwards. If None, every axis will default to
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
    figsize=(n_cols * width, n_rows * height)
    fig = plt.figure(figsize=figsize)
    
    # Positional arguments for `subplot2grid`.
    args = [((n_rows,n_cols),) + (loc,) + span 
            for loc,span in zip(locs, spans)]
    return fig, [plt.subplot2grid(*arg) for arg in args]

class eval_classifier():

    '''
    eval_classifier provides a quick access to all evaluation methods 
    under "model_validation.py". Moreover, it also allows adjustment 
    or modification to be made to any particular plot.
    
    Parameters
    ----------
    n_columns : int, default=3
        The number of columns. This is relevant when `axes` is not 
        provided (self.plot_all).
        
    plots : list of str, default=None
        List of validation metrics to be plotted. If None, all metrics 
        are selected. The available metrics are as follows:
        
            "error"  : Type I and type II errors
            "f1"     : Precision, Recall, and F1-Score
            "gini"   : Gini index
            "hist"   : Distribution of binary classes
            "ks"     : Kolmogorov–Smirnov test
            "gain"   : Gain chart
            "cumu"   : Cumulative lift
            "decile" : Decile lift
            "target" : Target rate
    
    ax_dict : dict of keywords, default=None
        Dictionary of keywords that are used to specify or override 
        properties in `plots`. "key" must match the specified metrics
        in `plot` e.g. {"error": {plot_kwds: {"linewidth": 3}}}.
        
    Attributes
    ----------
    funcs : collections.namedtuple
            
    '''
    
    def __init__(self, n_columns=3, plots=None, ax_dict=None):
       
        # =============================================================
        if not isinstance(n_columns, int) :
            raise ValueError(f"columns must be int. "
                             f"Got {type(n_columns)} instead.")
        else: self.n_columns = n_columns
        # -------------------------------------------------------------
        keys = ['error', 'f1', 'gini', 'hist', 'ks', 
                'gain', 'cumu', 'decile', 'target']
        funcs = collections.namedtuple('Functions', keys) 
        self.funcs = funcs(**{"error"  : error_plot_base,
                              "f1"     : f1score_plot_base,    
                              "gini"   : gini_plot_base,
                              "hist"   : dist_plot_base,    
                              "ks"     : ks_plot_base,
                              "gain"   : gains_plot_base,
                              "cumu"   : cumulift_plot_base,
                              "decile" : declift_plot_base, 
                              "target" : target_plot_base})
        # -------------------------------------------------------------
        metrics = list(self.funcs._fields)
        if plots is not None:
            diff = set(plots).difference(metrics)
            if len(diff)>0:
                raise ValueError(f"`plots` must be in {metrics}. "
                                 f"Got {diff} instead.")
            else: self.plots = plots
        else: self.plots = metrics
        # -------------------------------------------------------------   
        if ax_dict is not None:
            diff = set(ax_dict.keys()).difference(metrics)
            if len(diff)>0:
                raise ValueError(f"`ax_dict` key must be in {metrics}."
                                 f" Got {diff} instead.")
            else: self.ax_dict = ax_dict
        else: self.ax_dict = dict()
        # =============================================================
        
    def plot_all(self, y_true, y_proba, axes=None, tight_layout=True):
        
        '''
        Plots all evaluation metrics.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Target array (binary).

        y_proba : array-like of shape (n_samples,)
            Probability array.
            
        axes : list of Matplotlib axis object, optional
            List of `axes.Axes` objects.
        
        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().
        
        Returns
        -------
        axes : List of `axes.Axes` objects

        '''

        if axes==None:
            fig, axes = Axes2grid(n_axes=len(self.plots), 
                                  n_cols=self.n_columns)

        for name,ax in zip(self.plots, axes):
            args = (y_true, y_proba)
            kwds = self.ax_dict.get(name, dict())
            kwds.update({"ax": ax, "tight_layout": False})
            ax = getattr(self.funcs, name)(*args, **kwds)
        if tight_layout: plt.tight_layout()
            
        return axes
    
    def plot(self, y_true, y_proba, ax=None, plot="gini", tight_layout=True):
        
        '''
        Plots a single metric.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Target array (binary).

        y_proba : array-like of shape (n_samples,)
            Probability array.
       
        plot : str, default="gini"
            Validation metric to be plotted. The available metrics are as 
            follows:
        
                "error"  : Type I and type II errors
                "f1"     : Precision, Recall, and F1-Score
                "gini"   : Gini index
                "hist"   : Distribution of binary classes
                "ks"     : Kolmogorov–Smirnov test
                "gain"   : Gain chart
                "cumu"   : Cumulative lift
                "decile" : Decile lift
                "target" : Target rate
        
        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().
            
        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        args = (y_true, y_proba)
        kwds = self.ax_dict.get(plot, dict())
        kwds.update({"ax": ax, "tight_layout": tight_layout})
        return getattr(self.funcs, plot)(*args, **kwds)