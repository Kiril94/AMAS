    # -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
import numpy as np
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions
from scipy import stats
from toolbox import fits
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# In[General plots]
def nice_plot(
    X,Y, SY = 0, errorbar = False,scatter = False, absolute_sigma = True, 
    show_plot = True, save_plot=False, figname = None, xlabel = 'x',ylabel='', 
    data_label=' ', figsize = (10,5), y_range= None, legend_loc = 0, 
    legend_fs =20, legend_ncol = 1, legend_color = 'white', label_fs = 25, ticksize = 20, axis= None, 
    figure = None, plot_legend = False, x_show_range = None, text_fs = 14, 
    dpi = 80, xlogscale = False, ylogscale = False, color = 'skyblue', 
    plot_style = 'ggplot', linestyle = 'solid', ecolor = 'deepskyblue', 
    capsize = 3, capthick = 0.3, err_markersize = 6,  elinewidth = .9, 
    alpha = 1, scr_markersize = 30, scr_markerstyle = 'o', linewidth = 3, fill_under_curve = False, 
    fill_color = 'skyblue'):
    r"""
    Simple x-y plot. 
    
    Parameters:
    ----------
    
    X, Y, SY: lists of array_like,
        input data, if produced from histogram, pass ONLY values where Y>0
    errorbar: bool, False by default, if True errorbars are plotted 
    Returns:
    -------
    ax: axis object
    fig: figure object
    """
    
    if not(axis==None):
        ax = axis
        fig = figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
    plt.style.use(plot_style)
    const_err = (type(SY)==float)
    if const_err:
        SY = np.ones(len(X))*SY
    if errorbar:
        ax.errorbar(X, Y, yerr=SY, marker = '.', mec=ecolor, color = ecolor, 
                    elinewidth=elinewidth, capsize = capsize, capthick=capthick, 
                    linestyle = 'none',markersize = err_markersize,
                    label =data_label, alpha = alpha)#plot data
    elif scatter:
        ax.scatter(X,Y, color = color,  label =data_label, 
                   linestyle = linestyle, s = scr_markersize,
        alpha = alpha, marker = scr_markerstyle)
    else:
        ax.plot(X,Y, color = color,  label =data_label, 
                linestyle = linestyle, alpha = alpha, linewidth = linewidth)
    
    if fill_under_curve:
        plt.fill_between(X,Y, color = fill_color)
    
    ax.set_ylabel(ylabel, fontsize = label_fs)
    ax.tick_params(axis = 'both',labelsize  = ticksize)
    if plot_legend:
        legend = ax.legend(loc = legend_loc, ncol = legend_ncol,
                           fontsize = legend_fs, shadow = True)
        legend.get_frame().set_facecolor(legend_color)
    
    ax.set_xlabel(xlabel, fontsize =label_fs)
    if xlogscale:
        ax.set_xscale('log')

    if ylogscale:
        ax.set_yscale('log')
    
    if y_range!=None:
        ax.set_ylim((y_range[0],y_range[1]))
    
    if not(x_show_range==None):
        ax.set_xlim(x_show_range)
    
    fig.tight_layout()

    if save_plot:
        fig.savefig(figname, dpi = dpi)
        
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
        
    return ax, fig


def nice_histogram(
    x_all, N_bins, poisson_error = False, show_plot = False, plot_hist = True, plot_errors = True, 
    plot_legend = False, save = False, figname = '', x_range = None, 
    data_label = 'Data, histogram', data_label_hist = '',figsize = (12,6), 
    histtype = 'step', color_hist = 'orange', xlabel = 'x', 
    ylabel = 'Frequency', label_fs = 20, legend_fs = 18, legend_loc = 0, legend_ncol = 1, legend_color = 'white',
    ticks_lsize = 20, xlog_scale = False, ylog_scale = False, axis = None, 
    figure = None, dpi = 80, ecolor = 'deepskyblue', capsize = 3, 
    capthick = 0.3, markersize = 6, elinewidth = .9, hist_alpha = .9,
    hist_linestyle = 'solid', hist_linewidth = 2, plot_style = 'ggplot'):
    """Produce a nice histogram.
    Returns: x, y, sy, binwidth, fig, ax."""
    if not(x_range==None):
        mask_x = (x_all>x_range[0]) & (x_all<x_range[1])
        x_all = x_all[mask_x]
      
    if poisson_error:
        x,y,sy, binwidth = fits.produce_hist_values(
            x_all,N_bins, x_range = x_range,
            log = xlog_scale, poisson_error = poisson_error)
    else:
        x,y, binwidth = fits.produce_hist_values(
            x_all,N_bins, x_range = x_range,
            log = xlog_scale, poisson_error = poisson_error)

        
    if not(axis==None):
        ax = axis
        fig = figure
    else:
        fig, ax = plt.subplots(figsize=figsize)  
    plt.style.use(plot_style)

    if plot_hist:
        ax.hist(x_all, bins=N_bins, range=(x_all.min(), x_all.max()), 
                histtype=histtype, linewidth=hist_linewidth, color=color_hist, 
                label=data_label_hist, alpha = hist_alpha, 
                linestyle = hist_linestyle)
    if poisson_error:
        if plot_errors:
            ax.errorbar(
                x, y, yerr=sy, xerr=0.0, label=data_label, marker = '.', 
                mec=ecolor, color = ecolor, elinewidth=elinewidth, 
                capsize = capsize, capthick=capthick, linestyle = 'none',
                markersize = markersize)
        else:
            print('no uncertainties were given, if you want to assume poisson errors set poisson_errors to True')
            
    ax.set_xlabel(xlabel, fontsize = label_fs)
    ax.set_ylabel(ylabel, fontsize = label_fs)
    ax.tick_params(axis ='both', labelsize = ticks_lsize)
    if xlog_scale:
        ax.set_xscale('log')
    if ylog_scale:
        ax.set_yscale('log')
    if plot_legend:
        legend = ax.legend(loc=legend_loc, fontsize = legend_fs, 
                           shadow = True, ncol = legend_ncol)
        legend.get_frame().set_facecolor(legend_color)
        
    if save:
        fig.tight_layout()
        fig.savefig(figname, dpi = dpi)
    
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
        
    if poisson_error:
        return x, y, sy, binwidth, fig, ax
    else:
        return x, y, binwidth, fig, ax


def scatter_hist(X0,X1,Y0,Y1, ax, ax_histx, ax_histy, N_bins_x, N_bins_y,
                 histlabel0, histlabel1):
    """Helper function to create additional axes with histograms."""
    
    #set tick parameters
    ax_histx.tick_params(axis="x", labelbottom=False,)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.tick_params(axis="y", labelsize = 18 )
    ax_histy.tick_params(axis ="x", labelsize = 18)
    
    # x ranges for histograms
    x_min = np.amin([X0,X1])
    x_max = np.amax([X0,X1])
    y_min = np.amin([Y0,Y1])
    y_max = np.amax([Y0,Y1])    
    
    # create actual histograms
    ax_histx.hist(X0, bins=N_bins_x,range=(x_min,x_max), 
                  alpha = .8, label = histlabel0)
    ax_histx.hist(X1, bins=N_bins_x,range=(x_min,x_max), 
                  alpha = .8, label = histlabel1)
    ax_histy.hist(Y0, bins=N_bins_y,range=(y_min,y_max),
                  orientation='horizontal', alpha = .8)
    ax_histy.hist(Y1, bins=N_bins_y, range=(y_min,y_max),
                  orientation='horizontal',alpha = .8)
    ax_histx.legend(fontsize = 12)


def plot_classification(X, y, classifier, N_bins_x = 40, N_bins_y = 40, 
                        label0='type I', label1 = 'type II', 
                        histlabel0 = 'type I', histlabel1 = 'type II',
                        xlabel='x', ylabel = 'y', 
                        figsize = (10,10), save = False, figname = '',
                        show_plot = False):

    r"""
    Create a scatter plot including separation according to classifier #
    with histograms of projections on x and y axis.
    
    Parameters:
    -----------
    X: array_like
        Input data with N rows and 2 columns, where N is the number of samples
        and 2 variables
    y: array_like
        Target data with N rows and 1 column, contains 0s and 1s
    classifier: class from sklearn
    label0, label1: str, optional,
        labels for first and second variables, default H0,H1
    xlabel, ylabel: str, optional, default x, y
    save: bool, default(False)
    figname: if save==True 
        saves figure under figname
    N_bins_x, N_bins_y: int 
        specifies number of bins for first and second variable
    label0, label1, histlabel0, histlabel1: str, optional
    xlabel, ylabel: str, optional
    figsize: (float,float), optional, default (10,10)
    show: bool, show plot if True
    
    Returns:
    -------
    classifier: object, fitted classifier that contains 
        fit parameters and can be used to make predictions
    ax_scatter, ax_histx, ax_histy: axis objects, can be used to add a fit
        or change parameters
    fig: fig object, can be used for saving when sth changed
    """
    
    
    classifier.fit(X, y)
    
    fig, ax = plt.subplots()#create figure
    scatter_kwargs = {'s': 100, 'edgecolor': 'k', 'alpha': 1,
                      'marker':'s'}
    contourf_kwargs = {'alpha': .3}
    #plot decision boundaries
    ax_scatter = plot_decision_regions(X=X, y=y, 
                                clf=classifier, legend=2,
                                scatter_kwargs=scatter_kwargs,
                                contourf_kwargs=contourf_kwargs,#
                                ax = ax)
    ax_scatter.set_xlim(X[:,0].min()*1.1, X[:,0].max()*1.1)
    ax_scatter.set_ylim(X[:,1].min()*1.1, X[:,1].max()*1.1)
    ax_scatter.tick_params(labelsize = 20)
    ax_scatter.set_xlabel(xlabel, fontsize = 20)
    ax_scatter.set_ylabel(ylabel, fontsize = 20)
    handles, labels = ax_scatter.get_legend_handles_labels()#get pos of axis  
    bbox = ax_scatter.get_position()
    left, bottom, width, height = bbox.bounds
    spacing = 0.04
    ax_scatter.legend(handles, [label0, label1], 
           framealpha=0.3, scatterpoints=1, fontsize = 20,
            bbox_to_anchor=(left+width, bottom+height, 0.5, 0.5))
    #set position for additional axes
    rect_histx = [left, bottom + height + spacing, width, 0.25]
    rect_histy = [left + width + spacing, bottom, 0.25, height]


    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    mask0 = y==0
    X0 = X[mask0,0]
    X1 = X[~mask0,0]
    Y0 = X[mask0,1]
    Y1 = X[~mask0,1]
    scatter_hist(X0,X1,Y0, Y1, ax_scatter, ax_histx, ax_histy,
                 N_bins_x=N_bins_x, N_bins_y=N_bins_y, 
                 histlabel0 = histlabel0 , histlabel1 = histlabel1)
    
    if save:
        fig.savefig(figname, bbox_inches = 'tight')
    
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return classifier, ax_scatter, ax_histx, ax_histy, fig

def add_zoom_inset(ax, zoom,loc, x,y, xlim, ylim , sy = None, 
                   xlabel = '', ylabel = '', label_fs = 18,
                   mark_inset_loc = (3,1), borderpad = 4):
    """Add inset axis that shows a region of the data.
    
    Parameters:
    -----------
    ax: axis object
    zoom: float, zoom factor
    loc: integer, location of inset axis
    x,y, sy: array_like, data to plot
    xlim, ylim: (float, float) limits for x and y axis
    xlabel, ylabel: str, label for x and y axis
    label_fs: float, fonstsize for x and y axis labels
    mark_inset_loc: (int, int), corners for connection to the new axis
    borderpad: float, distance from border
    """
    
    axins = zoomed_inset_axes(ax,zoom,loc = loc,
                              borderpad=borderpad) 
    if sy==None:
        axins.plot(x, y)
    else: 
        axins.errorbar(x, y, yerr=sy, fmt='.b',  ecolor='b', elinewidth=.5, 
             capsize=0, capthick=0.1)
        
    if not(ylim==None):
        axins.set_ylim(ylim)
    
    axins.set_xlabel(xlabel, fontsize = label_fs)
    axins.set_ylabel(ylabel, fontsize = label_fs)
    axins.set_xlim(xlim) # Limit the region for zoom
    mark_inset(ax, axins, loc1=mark_inset_loc[0], 
               loc2=mark_inset_loc[1], fc="none", ec="0.5")
   

# In[Random Numbers] 


def create_1d_hist(ax, values, bins, x_range, title):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    ax.hist(values, bins, x_range, histtype='step', density=False, lw=2)         
    ax.set(xlim=x_range, title=title)
    hist_data = np.histogram(values, bins, x_range)
    return hist_data

def get_chi2_ndf( hist, const):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    data = hist[0]
    const_unnormed = const * data.sum()
    chi2 = np.sum( (data - const_unnormed)**2 / data )
    ndof = data.size
    return chi2, ndof

def show_int_distribution(integers, save_plot = True, 
                          figname = '', show_plot= False):
    """Show histogram of integers, to see if random.(Author: Troels Petersen)
    modified by: Kiril Klein
    Parameters: 
        integers, array_like
    Returns: 
        dict_raw, dict_odd_even, dict_high_low: dictionaries
            contains chi2, ndf, p for the hypothesis of integers being random.
        AX: list of axes objects
        fig: figure
        """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax_number, ax_odd_even, ax_high_low = ax.flatten()

    # Fill 1d histograms and plot them:
    hist_numbers  = create_1d_hist(ax_number, integers, 10, 
                                  (-0.5, 9.5), 'Numbers posted') #Plot all digits
    hist_odd_even = create_1d_hist(ax_odd_even, integers % 2,  2, 
                                   (-0.5, 1.5), 'Even and odd numbers')#Is number even or odd
    hist_high_low = create_1d_hist(ax_high_low, integers // 5, 2, 
                                  (-0.5, 1.5), 'Above and equal to or below 5')#Is number >= or < 5
    fig.tight_layout()
    
    chi2_raw, ndf_raw  = get_chi2_ndf( hist_numbers,  1.0 / 10)
    chi2_odd_even, ndf_odd_even = get_chi2_ndf( hist_odd_even, 1.0 / 2 )
    chi2_high_low, ndf_high_low = get_chi2_ndf( hist_high_low, 1.0 / 2 )
    p_raw = stats.chi2.sf(chi2_raw, ndf_raw)
    p_odd_even = stats.chi2.sf(chi2_odd_even, ndf_odd_even)
    p_high_low = stats.chi2.sf(chi2_high_low, ndf_high_low)
    
    dict_raw = {'chi2':chi2_raw, 'ndf':ndf_raw ,'p': p_raw }
    dict_odd_even = {'chi2':chi2_odd_even, 'ndf':ndf_odd_even ,'p': p_odd_even }
    dict_high_low = {'chi2':chi2_high_low, 'ndf':ndf_high_low ,'p': p_high_low }
    AX = [ax_number, ax_odd_even, ax_high_low]
    if save_plot:
        fig.savefig(figname)
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return dict_raw, dict_odd_even, dict_high_low, fig, AX

# In[Helper functions]

def Color_palette(k):
    """Takes integer, Returns color scheme."""
    Color_schemes1 = [
        [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 
        u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
        ["#70d6ff","#ff70a6","#ff9770","#ffd670","#e9ff70"],
        ['#F61613', '#EECC86', '#34AE8E', '#636253', '#A26251'], 
        ["#8acdea","#746d75","#8c4843","#9e643c","#ede6f2"],
        ['#2CBDFE', '#47DBCD', '#9D2EC5',  '#F3A0F2', '#661D98', '#F5B14C'],
        ['#845EC2','#ffc75f','#f9f871','#ff5e78'],
        ['#fff3e6','#1a508b','#0d335d','#c1a1d3'],
        ["#f72585","#b5179e","#7209b7","#560bad","#480ca8",
        "#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"],
        ["#22223b","#4a4e69","#9a8c98","#c9ada7","#f2e9e4"],
        ["#d8f3dc","#b7e4c7","#95d5b2","#74c69d",
        "#52b788","#40916c","#2d6a4f","#1b4332","#081c15"],
        ["#ffbe0b","#fb5607","#ff006e","#8338ec","#3a86ff"],
        ["#7400b8","#6930c3","#5e60ce","#5390d9","#4ea8de",
        "#48bfe3","#56cfe1","#64dfdf","#72efdd","#80ffdb"],
        ["#cc8b86","#f9eae1","#7d4f50","#d1be9c","#aa998f"],
        ["#ff595e","#ffca3a","#8ac926","#1982c4","#6a4c93"],
        ["#8c1c13","#bf4342","#e7d7c1","#a78a7f","#735751"],
        ["#f72585","#b5179e","#7209b7","#560bad","#480ca8",
        "#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"],
        ["#006466","#065a60","#0b525b","#144552","#1b3a4b",
        "#212f45","#272640","#312244","#3e1f47","#4d194d"]]
    Color_schemes = Color_schemes1
    for i in range(len(Color_schemes1)):
        Color_schemes.append(reversed(Color_schemes1[i]))
    return Color_schemes[k]


# In[Example]
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from sklearn.metrics import plot_roc_curve

clf = LinearDiscriminantAnalysis(solver='svd')

rng = np.random.RandomState(0)
x1v1 = np.random.normal(-1,.2,200)
x2v1 = np.random.normal(0,.5,200)
x1v2 = np.random.normal(-1,.2,200)
x2v2 = np.random.normal(0,.5,200)

y1 = np.zeros(200, dtype = 'int')
y2 = np.ones(200, dtype = 'int')
X1 = np.concatenate((x1v1[:,np.newaxis],x1v2[:,np.newaxis]), axis = 1)
X2 = np.concatenate((x2v1[:,np.newaxis],x2v2[:,np.newaxis]), axis = 1)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,y2))
fitted_clf,_,ax_histx, ax_histy, fig = plot_classification(
        X,y, clf, save = True, figname = 'test2.png', show = False)
ax_histx.plot(np.linspace(-2,2,100), 10*stats.norm.pdf(np.linspace(-2,2,100), 
                                                      loc = 0,scale = .5))
display(fig)
plot_roc_curve(fitted_clf, X, y)
"""