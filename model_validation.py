'''
Available methods are the followings:
[1] eval_classifier 
[2] cfm_plot
[3] f1score_plot
[4] gini_plot
[5] dist_plot
[6] ks_plot
[7] gains_plot
[8] lift_plot
[9] create_cmap

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 05-10-2020

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from sklearn.metrics import (confusion_matrix, 
                             roc_curve, roc_auc_score,  
                             precision_recall_curve)

__all__ = ['eval_classifier',
           'cfm_plot',
           'f1score_plot',
           'gini_plot',
           'dist_plot',
           'ks_plot',
           'gains_plot',
           'lift_plot',
           'create_cmap']

def eval_classifier(y_true, y_proba, **params):

    '''
    `eval_classifier` provides a quick access to all evaluation 
    methods under `model_validation.py`. Moreover, it also allows 
    adjustment or modification to be made to any particular plot.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        params are used to specify or override properties of 
        
        axes : list of Matplotlib axis object, optional
            List of `axes.Axes` objects.
            
        columns : int, default=3
            The number of columns. This is relevant when `axes` is 
            not provided.
        
        plots : list of int, optional
            `plots` is a list that contains type of plots to be 
            displayed on `plt.figure`. If `plots` is not provided, 
            all plots are selected.
        
            The method indices are as follows:
            0 : 'Confusion matrix', 
            1 : 'F1-Score', 
            2 : 'Gini index', 
            3 : 'Distribution of binary classes', 
            4 : 'Kolmogorov–Smirnov test', 
            5 : 'Gains',
            6 : 'Lift'

        ax_dict : `list` of `dict`, optional
            List of dictionaries that is used to specify or override 
            properties of `plots`. The items must be arranged in the 
            same order as `plots`. `{}` is compulsory when adjustment 
            of properties is not required. If `ax_dict` is not defined, 
            `ax_dict` will default to list of `{}`.
            
    '''
    # ======================================================
    columns = update_(params, 'columns', 3)
    # ------------------------------------------------------
    default = [0,1,2,3,4,5,6,6,6]
    plots = update_(params, 'plots', default)
    # ------------------------------------------------------
    default = [dict()]*(len(plots)-2) + [{'plot':'decile'}, 
                                         {'plot':'rate'}]
    ax_dict = update_(params, 'ax_dict', default)
    # ------------------------------------------------------
    axes = update_(params, 'axes', None)
    # ------------------------------------------------------
    plts = [cfm_plot, f1score_plot, gini_plot, dist_plot, 
            ks_plot, gains_plot, lift_plot]
    # ======================================================

    # Check length of `ax_dict`.
    n_plot = len(plots)
    if len(ax_dict)!=n_plot:
        raise ValueError("Length of `ax_dict` must be {:,}. "
                         "Got {:,}".format(len(plots),
                                           len(ax_dict)))
    
    # Create `ax` if `axes` is not provided.
    if axes==None:
        have_axes = False
        # Determine number of rows and columns, and `loc` for 
        # each axis and `shape` for `plt.subplot2grid`.
        c, r = min(n_plot,columns), int(np.ceil(n_plot/columns))
        fig = plt.figure(figsize=(6*c,4*r))
        locs = [(m,n) for m in range(r) for n in range(c)]
        axes = [plt.subplot2grid((r,c),loc) for loc in 
                np.array(locs)[:n_plot]]
    else: have_axes = True
    
    # Plot `ax` according to `axes`.
    for (n, ax, params) in zip(plots, axes, ax_dict):
        plts[n](y_true, y_proba, **{**params, **{'ax' : ax}})
    
    if have_axes==False:
        fig.tight_layout()
        plt.show()

def cfm_plot(y_true, y_proba, **params):

    '''
    ** Confusion Matrix ** 
    
    A confusion matrix, also known as an error matrix, is a 
    specific table layout that allows visualization of the 
    performance of an algorithm, typically a supervised 
    learning one. This table is comprised of four elements,
    which are True-Positive, False-Positive, True-Negative, 
    and False-Negative.
    
    versionadded:: 05-10-2020
    
    References
    ----------
    .. [1] Confusion Matrix, https://endef eval_classifier(y_true, y_proba, **params):

    '''
    `eval_classifier` provides a quick access to all evaluation 
    methods under `model_validation.py`. Moreover, it also allows 
    adjustment or modification to be made to any particular plot.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        params are used to specify or override properties of 
        
        axes : list of Matplotlib axis object, optional
            List of `axes.Axes` objects.
            
        columns : int, default=3
            The number of columns. This is relevant when `axes` is 
            not provided.
        
        plots : list of int, optional
            `plots` is a list that contains type of plots to be 
            displayed on `plt.figure`. If `plots` is not provided, 
            all plots are selected.
        
            The method indices are as follows:
            0 : 'Confusion matrix', 
            1 : 'F1-Score', 
            2 : 'Gini index', 
            3 : 'Distribution of binary classes', 
            4 : 'Kolmogorov–Smirnov test', 
            5 : 'Gains',
            6 : 'Lift'

        ax_dict : `list` of `dict`, optional
            List of dictionaries that is used to specify or override 
            properties of `plots`. The items must be arranged in the 
            same order as `plots`. `{}` is compulsory when adjustment 
            of properties is not required. If `ax_dict` is not defined, 
            `ax_dict` will default to list of `{}`.
            
    '''
    # ======================================================
    columns = update_(params, 'columns', 3)
    # ------------------------------------------------------
    default = [0,1,2,3,4,5,6,6,6]
    plots = update_(params, 'plots', default)
    # ------------------------------------------------------
    default = [dict()]*(len(plots)-2) + [{'plot':'decile'}, 
                                         {'plot':'rate'}]
    ax_dict = update_(params, 'ax_dict', default)
    # ------------------------------------------------------
    axes = update_(params, 'axes', None)
    # ------------------------------------------------------
    plts = [cfm_plot, f1score_plot, gini_plot, dist_plot, 
            ks_plot, gains_plot, lift_plot]
    # ======================================================

    # Check length of `ax_dict`.
    n_plot = len(plots)
    if len(ax_dict)!=n_plot:
        raise ValueError("Length of `ax_dict` must be {:,}. "
                         "Got {:,}".format(len(plots),
                                           len(ax_dict)))
    
    # Create `ax` if `axes` is not provided.
    if axes==None:
        have_axes = False
        # Determine number of rows and columns, and `loc` for 
        # each axis and `shape` for `plt.subplot2grid`.
        c, r = min(n_plot,columns), int(np.ceil(n_plot/columns))
        fig = plt.figure(figsize=(6*c,4*r))
        locs = [(m,n) for m in range(r) for n in range(c)]
        axes = [plt.subplot2grid((r,c),loc) for loc in 
                np.array(locs)[:n_plot]]
    else: have_axes = True
    
    # Plot `ax` according to `axes`.
    for (n, ax, params) in zip(plots, axes, ax_dict):
        plts[n](y_true, y_proba, **{**params, **{'ax' : ax}})
    
    if have_axes==False:
        fig.tight_layout()
        plt.show().wikipedia.org/wiki/
           Confusion_matrix
    .. [2] https://www.geeksforgeeks.org/confusion-matrix-
           machine-learning/

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.

    **params : dictionary of properties, optional
        params are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.
        
        threshold : float, default=0.5
            Threshold of probabilities.
        
        mat_dict : dictionary
            A dictionary to override the default `ax.matshow` 
            properties. If None, it uses default settings. 
            
    '''
    # ======================================================
    default = {'cmap':create_cmap('#d1ccc0','#FF0000'), 
               'alpha':0.5}
    mat_dict = update_(params, 'mat_dict', default, True)
    # ------------------------------------------------------
    threshold = update_(params,'threshold', 0.5)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================

    # Compute confusion matrix elements i.e. 
    # True-Positive, False-Positive, 
    # True-Negative, and False-Negative.
    y_pred = (y_proba>=threshold).astype(int)
    cfm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cfm.ravel()
    
    # Plot the confusion matrix.
    ax.matshow(cfm, **mat_dict)
    
    # Set labels for `Confusion Matrix`.
    labels = [['True Positive' , tp, (0,0)], 
              ['False Positive', fp, (1,0)], 
              ['False Negative', fn, (0,1)], 
              ['True Negative' , tn, (1,1)]]
    
    # Set keyword arguments, and string formats.
    kwargs = dict(va='center', ha='center', fontsize=10)
    str_format1 = '{:}\n{:,.0f}\n({:.1%})'.format
    str_format2 = 'Predict\n(cutoff = {:.2%})'.format
    
    # Annotation for each element in cfm.
    for s,n,xy in labels:
        t = str_format1(s,n,n/len(y_true))
        ax.annotate(t, xy, **kwargs)
    
    # Set other attributes.
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_yticklabels(['True','False'], 
                       fontsize=10, rotation=90)
    ax.set_xticklabels(['True','False'], fontsize=10)
    ax.set_ylabel(str_format2(threshold))
    ax.set_xlabel('Actual')
    ax.set_facecolor('white')
    return ax

def f1score_plot(y_true, y_proba, **params):
    
    '''
    The F-score, also called the F1-score, is a measure of 
    a model’s accuracy on a dataset. It is used to evaluate 
    binary classification systems, which classify examples 
    into "positive" or "negative". The F-score is a way of 
    combining the precision and recall of the model, and it 
    is defined as the harmonic mean of the model’s precision 
    and recall. 
    
    versionadded:: 05-10-2020
    
    References
    ----------
    .. [1] https://deepai.org/machine-learning-glossary-and-
           terms/f-score

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        params are used to specify or override properties of 
          
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.

        plot_dict : dictionary
            A dictionary to override the default `ax.plot` 
            properties. If None, it uses default settings.
        
        fill_dict : dictionary
            A dictionary to override the default `ax.fill_between` 
            properties of "GINI". If None, it uses default 
            settings. 
            
    '''
    # ======================================================
    default = {'color' : '#eb3b5a' , 'linewidth' : 1.5, 
               'label' : 'F1-Score', 'linestyle' : '-'}
    plot_dict = update_(params, 'plot_dict', default, True)
    # ------------------------------------------------------
    default = {'alpha':0.2, 'color':'#d1ccc0'}
    fill_dict = update_(params, 'fill_dict', default, True)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================
    
    # Determine Presion and Recall from all possible cutoffs. 
    # Order of outputs from `confusion_matrix` are as follow; 
    # tn, fp, fn, and tp.
    cutoffs = np.unique(y_proba)
    cfm = np.array([confusion_matrix(y_true, (y_proba>=c)).ravel() 
                    for c in cutoffs])
    p = cfm[:,-1]/cfm[:,[1,3]].sum(axis=1)
    r = cfm[:,-1]/cfm[:,[2,3]].sum(axis=1)
    f1_score = 2*(p*r) / (p+r)

    # Plot F1-Score curve.
    label = plot_dict['label'] + ': max={:,.3g}'.format(max(f1_score))
    ax.plot(cutoffs, f1_score, **{**plot_dict,**{'label':label}})
    ax.fill_between(cutoffs, f1_score, **fill_dict)
    
    # Maximum F1-Score line.
    proba = cutoffs[np.argmax(f1_score)]
    label = 'Optimal point: P={:,.3f}'.format
    kwargs= {'color'     : '#3d3d3d', 
             'linewidth' : 1, 
             'linestyle' : '--', 
             'label'     : label(proba)}
    ax.axvline(proba, **kwargs)
    
    # Change format of ticklabels.
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)

    # Set other attributes.
    ax.set_xlabel('Probability (P)')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0,1.1)
    ax.set_title('F1-Score curves (Harmonic mean)')
    ax.set_facecolor('white')
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    ax.grid(False)
    return ax

def gini_plot(y_true, y_proba, **params):

    '''
    ** GINI index **
    
    The ROC curve is the display of sensitivity and 
    1 - specificity for different cut-off values for probability 
    (If the probability of positive response is above the 
    cut-off, we predict a positive outcome, if not we are 
    predicting a negative one). Each cut-off value defines one 
    point on `ROC` curve, ranging cut-off from 0 to 1 will draw 
    the whole `ROC` curve. 

    The `Gini` coefficient is the area or `ROC` curve above the 
    random classifier (diagonal line) that indicates the model’s 
    discriminatory power, namely, the effectiveness of the model 
    in differentiating between target, and non-target.
    
    versionadded:: 05-10-2020
    
    References
    ----------
    .. [1] ROC curve, https://en.wikipedia.org/wiki/Receiver_
           operating_characteristic
    .. [2] https://towardsdatascience.com/using-the-gini-
           coefficient-to-evaluate-the-performance-of-credit-
           score-models-59fe13ef420
        
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        `params` are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.

        plot_dict : dictionary
            A dictionary to override the default `ax.plot` 
            properties of "ROC". If None, it uses default settings.
        
        fill_dict : dictionary
            A dictionary to override the default `ax.fill_between` 
            properties of "GINI". If None, it uses default settings.
            
    '''
    # ======================================================
    # ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    # ------------------------------------------------------
    default = {'color' : 'red' , 'linewidth': 1.5, 
               'label' : 'ROC' , 'linestyle': '-'}
    plot_dict = update_(params, 'plot_dict', default, True)
    # ------------------------------------------------------
    default = {'alpha' : 0.2   , 'color':'#d1ccc0', 
               'label' : 'Gini', 'y1' : tpr, 'y2': fpr}
    fill_dict = update_(params, 'fill_dict', default, True)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================
    
    # Compute area under curve and gini coefficient. 
    auc = roc_auc_score(y_true, y_proba)
    gini = 2*auc - 1
    
    # Plot ROC curve.
    label = plot_dict['label'] + ' ({:,.2%})'.format(auc)
    ax.plot(fpr, tpr, **{**plot_dict, **{'label':label}})
    
    # Plot GINI curve.
    label = fill_dict['label'] + ' ({:,.2%})'.format(gini)
    ax.fill_between(fpr, **{**fill_dict, **{'label':label}})
    
    # Random classifier line.
    kwargs = {'color'    : '#3d3d3d', 
              'linewidth': 1, 
              'linestyle': '--',
              'label'    : 'Random classifier'}
    ax.plot([0,1], [0,1], **kwargs)
    
    # Change format of ticklabels.
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_formatter(t)
    
    # Set other attributes.
    ax.set_title('Receiver Operating Characteristic curve')
    ax.set_xlabel('False Positive Rate\n(1-Specificity)')
    ax.set_ylabel('True Positive Rate\n(Sensitivity)')
    ax.set_facecolor('white') 
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    ax.grid(False)
    return ax

def dist_plot(y_true, y_proba, **params):

    '''
    Distribution plot is intended to illustrate the separation 
    of two distributions from binary classification, according 
    to obtained probabilities.
    
    versionadded:: 05-10-2020

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.

    **params : dictionary of properties, optional
        `params` are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.
        
        bins : int, default=20
            Number of bins.

        n_tick : int, default: 5
            Number of ticks to be displayed on x-axis.
    
        colors : `list` of color-hex codes or (r,g,b)
            List must contain at the very least, 'n' of color-hex 
            codes or (r,g,b) that matches number of labels in 'y'. 
            If None, it defaults to `matplotlib_cmap`.

        labels : list of str
            List of labels whose items must be arranged in ascending 
            order. If None, algorithm will automatically assign 
            labels.
        
        bar_dict : dictionary
            A dictionary to override the default `ax.bar` properties. 
            If None, it uses default settings
        
        float_format : function
            String formatting function method e.g. '{:.2%}'.format.
            
    '''  
    # ======================================================
    bins = update_(params,'bins', 20)
    # ------------------------------------------------------
    n_tick = update_(params,'n_tick', 5)
    # ------------------------------------------------------
    colors = update_(params,'colors',['#d1ccc0','red'])
    # ------------------------------------------------------
    labels = update_(params, 'labels',['Class 0','Class 1'])
    labels = [(s + ' ({:,}, {:.0%})').format for s in labels]
    # ------------------------------------------------------
    default = {'edgecolor':'#353b48','alpha':0.5}
    bar_dict = update_(params, 'bar_dict', default, True)
    # ------------------------------------------------------
    default = '{:.0%}'.format
    float_format = update_(params,'float_format', default)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================    
    
    # Create `bins`.
    data = [y_proba[y_true==n] for n in [0,1]]
    bins = np.histogram(y_proba,bins)[1]
    bins[-1] = min(bins[-1]*1.0001,1.0001)

    # Distribution of samples.
    kwargs = dict(bins=bins, range=(0,1))
    dist = [np.histogram(data[n],**kwargs)[0]
            *100/data[n].size for n in [0,1]]
  
    # Plot horizontal bar.
    for n in [0,1]:
        N = (y_true==n).sum()
        kwargs = {'color' : colors[n], 
                  'label' : labels[n](N, N/y_true.size)}
        ax.bar(np.arange(bins.size-1), dist[n], 
               **{**bar_dict,**kwargs})

    # Set tick positions.
    step = (bins.size-2) / (n_tick-1)
    t = np.arange(0,bins.size-2,step)
    ax.set_xticks(np.concatenate((t,t[-1:]+step)))
    
    # Set tick labels.
    ax.set_xticklabels([float_format(n) for n in 
                        np.histogram(y_proba, n_tick-1)[1]])

    # Set other attributes.
    ax.set_ylim(0,min(np.max(dist)*1.2,110))
    s = 'P({}) = [{:.2%}, {:.2%}]'.format
    ax.set_title('Plot of two Distributions\n' + 
                 ', '.join((s(n,min(data[n]),max(data[n])) 
                            for n in [0,1])))
    ax.set_xlabel('Probability (P)')
    ax.set_ylabel('Percentage of\nsamples (%) by class')
    ax.set_facecolor('white')
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    ax.grid(False)
    return ax

def ks_plot(y_true, y_proba, **params):

    '''
    ** Kolmogorov–Smirnov **
    
    In statistics, the Kolmogorov–Smirnov test is a nonparametric 
    test of the equality of continuous, one-dimensional probability 
    distributions that can be used to compare a sample with a 
    reference probability distribution (one-sample KS test), or to 
    compare two samples.
    
    versionadded:: 05-10-2020

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        `params` are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.
        
        colors : list of color-hex codes or (r,g,b)
            List must contain at the very least, 'n' of color-hex 
            codes or (r,g,b) that matches number of labels in 'y'. 
            If None, it defaults to `matplotlib_cmap`.

        labels : list of str
            List of labels whose items must be arranged in ascending 
            order. If `None`, algorithm will automatically assign 
            labels.
        
        plot_dict : dictionary
            A dictionary to override the default `ax.plot` properties. 
            If None, it uses default settings.
        
        fill_dict : dictionary
            A dictionary to override the default `ax.fill_between` 
            properties. If None, it uses default settings.
            
    '''
    # ======================================================
    colors = update_(params,'colors',['#7f8fa6','red'])
    # ------------------------------------------------------
    labels = update_(params, 'labels',['Class 0','Class 1'])
    labels = [(s + ' ({:,}, {:.0%})').format for s in labels]
    # ------------------------------------------------------
    plot_dict = update_(params, 'plot_dict', 
                        {'linewidth': 2}, True)
    # ------------------------------------------------------
    default = {'alpha' : 0.2, 'color' : '#d1ccc0'}
    fill_dict = update_(params, 'fill_dict', default, True)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================
        
    # Get `bins` of probabilities.
    bins = np.histogram(y_proba,bins=100)[1]
    bins[-1] = min(bins[-1]*1.0001,1)
    bins = np.unique([0] + bins.tolist())
    
    # Cumulative (%) distribution.
    dist = [np.histogram(y_proba[y_true==n], bins)[0] for n in [0,1]]
    cumd = [np.array([0]+list(np.cumsum(d)))/sum(d) for d in dist]

    # Determine `ks_cutoff` from each threshold.
    ks = abs(cumd[0]-cumd[1])
    c  = np.argmax(ks)
    cutoff, ks = bins[c], ks[c]

    # Plot cumulative distribution.
    for n in [0,1]:
        N = (y_true==n).sum()
        kwargs = {'color': colors[n], 
                  'label': labels[n](N, N/y_true.size)}
        ax.plot(bins, cumd[n], **{**plot_dict,**kwargs})
    ax.fill_between(bins, cumd[0], cumd[1],**fill_dict)
    
    # Optimal `ks` line with `cutoff`.
    label  = 'KS = {:.2%}, P = {:.2%}'.format
    kwargs = {'color'    : 'k',
              'label'    : label(ks, cutoff),
              'linewidth': 1}
    ax.plot([cutoff]*2, [cumd[0][c],cumd[1][c]], **kwargs)
    
    # Change format of ticklabels.
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_formatter(t)
    
    # Set other attributes.
    ax.set_title('Kolmogorov–Smirnov')
    ax.set_xlabel('Probability (P)')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_facecolor('white')
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    ax.grid(False)
    return ax

def gains_plot(y_true, y_proba, **params):

    '''
    ** Gain chart **
    
    Gain chart is used to evaluate performance of classification 
    model. It measures how much better one can expect to do with 
    the predictive model comparing without a model. It plots the 
    cumulative percentage of positive samples (x-axis) against the 
    cumulative percentage of targeted samples (y-axis).
    
    versionadded:: 05-10-2020
    
    References
    ----------
    .. [1] https://www.listendata.com/2014/08/excel-template-gain
           -and-lift-charts.html
    .. [2] https://community.tibco.com/wiki/gains-vs-roc-curves-do
           -you-understand-difference
        
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        `params` are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.
        
        step : int, default=5
            Incremental change of percentile.

        decimal : int, default=4
            Decimal places that applied to all percentile values 
            before collapsing bins.
    
        plot_dict : dictionary
            A dictionary to override the default `ax.plot` properties. 
            If None, it uses default settings.
        
        fill_dict : dictionary
            A dictionary to override the default `ax.fill_between` 
            properties. If None, it uses default settings.
            
    '''
    # ======================================================
    default = {'color':'red'   , 'linewidth' : 1.5, 
               'label':'Model' , 'linestyle' : '-'}
    plot_dict = update_(params, 'plot_dict', default, True)
    # ------------------------------------------------------
    default = {'alpha':0.2, 'color':'#d1ccc0'}
    fill_dict = update_(params, 'fill_dict', default, True)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================
    
    # Compute cumulative percentage.
    bins = np.unique(y_proba); bins[-1] = np.inf
    dist = np.histogram(y_proba[y_true==1],bins)[0][::-1]
    dist = np.array([0] + dist.tolist())
    cum_event = np.cumsum(dist)/sum(dist)

    # Gain plot.
    x = (1 - bins[::-1]); x = np.where(x>=0,x,0)
    N = y_true.sum()
    p = ', Target = {:,.0f} ({:,.0%})'.format(N, N/y_true.size)
    label = {'label':plot_dict['label'] + p}
    ax.plot(x, cum_event, **{**plot_dict,**label})
    ax.fill_between(x, y1=cum_event, y2=x ,**fill_dict)
    
    # Random classifier (diagonal line).
    kwargs = {'color'    : '#3d3d3d', 
              'linestyle': '--', 
              'label'    : 'Random classifier'}
    ax.plot([0,1],[0,1], **kwargs)
    
    # Change format of ticklabels.
    t = ticker.PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(t)
    ax.yaxis.set_major_formatter(t)
    
    # Set other attributes.
    ax.set_title('Gains Chart')
    ax.set_xlabel('Percentage of targeted samples (Support)')
    ax.set_ylabel('Percentage of targets (Sensitivity)')
    ax.set_facecolor('white')
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    ax.grid(False)
    return ax

def lift_plot(y_true, y_proba, **params):

    ''' 
    ** Lift **
    
    Lift measures how much better one can expect to do with the 
    predictive model comparing without a model (randomness). Lift is 
    the ratio of targets to samples. Both numerator and denominator 
    can be either cumulative or decile percentage. Moreover, 
    probability must be ordered in descending manner before grouping 
    into deciles.
    
    versionadded:: 05-10-2020

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    **params : dictionary of properties, optional
        params are used to specify or override properties of 
        
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.
        
        plot : {'cumu', 'decile', 'rate'}, default='cumu'
            If the option chosen is 'cumu', the cumulative lift is 
            selected, while 'decile' is a decile lift, and 'rate' is 
            the target rate.
        
        bound : (`float`, `float`), optional
            The lower and upper range of the bins. If not provided, 
            range is simply [0,100]. Values outside the range are 
            ignored. The first element of the range must be less than 
            or equal to the second. 
        
        step : int, default=10
            Incremental change of percentile and must be between 0 and 
            100.

        decimal : int, default=4
            Decimal places that applied to all percentile values before 
            collapsing bins.
    
        plot_dict : dictionary
            A dictionary to override the default `ax.plot` properties. 
            If None, it uses default settings.
        
        fill_dict : dictionary
            A dictionary to override the default `ax.fill_between` 
            properties. If None, it uses default settings.
        
        font_dict : dictionary
            A dictionary to override the default `ax.annotate` 
            properties.
            
        anno_format : function, default=None
            String formatting function method e.g. '{:.2%}'.format.
            If None, it uses default settings.
        
    '''
    # ======================================================
    str_fmt = update_(params,'anno_format','{:,.2f}'.format)
    attrs = {'decile': ('dec_lift', -0.1, [1], str_fmt, 
                        'Decile Lift'), 
             'rate'  : ('p_target', -0.1, [0], str_fmt,
                        'Target Ratio (by decile)'), 
             'cumu'  : ('cum_lift',  0.9, [1], str_fmt, 
                        'Cumulative Lift')}
    attrs = attrs[update_(params,'plot', 'cumu')]
    # ------------------------------------------------------
    bound = update_(params, 'bound', [0,100])
    # ------------------------------------------------------
    step = update_(params,'step', 10)
    # ------------------------------------------------------
    decimal = update_(params,'decimal', 4) 
    # ------------------------------------------------------
    default = {'color':'red', 'linewidth':1.5, 
               'label':'Model' , 'linestyle':'-', 
               'marker':'s', 'markersize':5.5, 
               'fillstyle':'none'}
    plot_dict = update_(params, 'plot_dict', default, True)
    # ------------------------------------------------------
    default = {'alpha':0.2, 'color':'#d1ccc0'}
    fill_dict = update_(params, 'fill_dict', default, True)
    # ------------------------------------------------------
    default = dict(textcoords="offset points", 
                   color='red', fontsize=10, ha='center', 
                   va='bottom', xytext=(5,5))
    font_dict = update_(params, 'font_dict', default, True)
    # ------------------------------------------------------
    fig, ax = _ax_check(params)
    # ======================================================
    
    # Compute lift dictionary.
    kwargs = {'step':step, 'decimal':decimal}
    data = _lift_table_(y_true, y_proba, **kwargs)
    
    # Determine `decile`.
    decile = np.arange(bound[1]-step, bound[0]-step, -step)
    decile[-1] = bound[0]
    
    # `lift` is determined by interpolation.
    xp, fp = data['cum_p_sample'], data[attrs[0]]
    lift = np.interp(1-decile/100, xp, fp)
 
    # Add number of targets to `label`.
    N = y_true.sum()
    p =', Target = {:,.0f} ({:,.0%})'.format(N, N/y_true.size)
    label = {'label':plot_dict['label'] + p}
    
    # Plot lift and set its annotation.
    x = np.arange(len(decile))
    ax.plot(x, lift, **{**plot_dict,**label})
    ax.fill_between(x, y1=lift, y2=attrs[2]*len(lift), **fill_dict)
    for n,s in enumerate(lift): 
        ax.annotate(attrs[3](s), (n,s), **font_dict)
    
    # Random classifier (baseline).
    if attrs[0]!='p_target': 
        ax.axhline(1, **{'color'    : '#3d3d3d', 
                         'linestyle': '--',  
                         'label'    : 'Random classifier'})

    # Change format of ticklabels.
    ax.set_xticks(x)
    ax.set_xticklabels(['{:,.3g}'.format(s) 
                        for s in bound[1]-decile])
    
    # Set `ylim`.
    t = 2.5*np.diff(ax.get_yticks())[0]
    ax.set_ylim(max(min(lift)-t,attrs[1]),max(lift)+t)
    
    # Set other attributes.
    ax.set_title('Lift Chart, Bounds={}'.format(bound))
    ax.set_xlabel('Percentage of targeted samples (Support)')
    ax.set_ylabel(attrs[4])
    ax.set_facecolor('white')
    ax.grid(False)
    ax.legend(loc='best', framealpha=0, edgecolor='none')
    return ax

def update_(params, key, default, update=False):
    ''' 
    Update parameter.
    versionadded:: 05-10-2020
    '''
    if params.get(key) is not None:
        if update: return {**default,**params.get(key)}
        else: return params.get(key)
    else: return default
    
def _ax_check(params):
    ''' 
    Create `axes.Axes` object.
    versionadded:: 05-10-2020
    
    Returns
    -------
    (`plt.figure`, `axes.Axes`)
    '''
    if params.get('ax') is None:
        fig = plt.figure()
        return fig, fig.add_subplot()
    else: return None, params.get('ax')
    
def create_cmap(c1=None, c2=None):
    
    '''
    Creating `matplotlib.colors.Colormap` (Colormaps) with 
    two colors.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    c1 : `hex code` or (r,g,b), default=None
        The beginning color code. If None, `c1` is default 
        to (23,10,8).
    
    c2 : `hex code` or (r,g,b), default=None
        The ending color code. If `None`, `c2` is default to 
        (255,255,255).
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
    
    Examples
    --------
    >>> import numpy as np
    >>> create_cmap()
    <matplotlib.colors.ListedColormap at 0x12aa5aa58>
    '''
    to_rgb = lambda c : tuple(int(c.lstrip('#')[i:i+2],16) 
                           for i in (0,2,4))
    # Default values for `c1`, and `c2`.
    if c1 is None: c1 = (23,10,8)
    if c2 is None: c2 = (255,255,255)  
    # Convert to RGB.
    if isinstance(c1,str): c1 = to_rgb(c1)
    if isinstance(c2,str): c2 = to_rgb(c2)
    colors = np.ones((256,4))
    for i in range(3):
        colors[:,i] = np.linspace(c1[i]/256,c2[i]/256,256)
    colors = colors[np.arange(255,-1,-1),:]
    return ListedColormap(colors)

def _lift_table_(y_true, y_proba, step=10, decimal=4):
    
    '''
    
    Function determines `i` percentile based on defined `step`
    (incremental change in percentile). Then, it collaspes bins 
    that have the same percentile values into one. Then, other 
    parameters such as cumulative percentage, lift, or target 
    rate are calculated, accordingly.
    
    versionadded:: 05-10-2020
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    y_proba : array-like of shape (n_samples,)
        Probability array.
    
    step : int, default=5
        Incremental change of percentile.
    
    decimal : `int`, optional, default: 4
        Decimal places that applied to all percentile values 
        before collapsing bins.
    
    Returns
    -------
    dictionary
    
    '''
    # Compute sequence of percentiles.
    q = np.arange(100-step,-step,-step); q[-1] = 0
    bins = np.round(np.percentile(y_proba,q[q>=0]),decimal)
    bins = np.unique(bins)[::-1]

    # Cumulative number of targets and samples.
    cum_n_target = np.array([y_true[(y_proba>=n)].sum() 
                             for n in bins]).astype(int)
    cum_n_sample = np.array([(y_proba>=n).sum() for n in bins])

    # Cumulative percentage of targets and samples.
    cum_p_target = cum_n_target/y_true.sum()
    cum_p_sample = cum_n_sample/y_true.size
    cum_lift = cum_p_target/np.where(cum_p_sample==0,
                                     1, cum_p_sample) 
    
    # Number of samples in each bin.
    N = lambda a : (a[0:1],np.diff(a))
    n_target = np.concatenate(N(cum_n_target))
    n_sample = np.concatenate(N(cum_n_sample))
    
    # Percentage of targets, and lift (per bin).
    p_target = n_target/np.where(n_sample==0,1,n_sample)
    dec_lift = p_target*(y_true.size/y_true.sum())
    
    return {'cum_n_target' : cum_n_target,
            'cum_n_sample' : cum_n_sample,
            'cum_p_target' : cum_p_target,
            'cum_p_sample' : cum_p_sample,
            'n_target' : n_target,
            'n_sample' : cum_p_sample,
            'bins'     : bins, 
            'p_target' : p_target,
            'cum_lift' : cum_lift,
            'dec_lift' : dec_lift}