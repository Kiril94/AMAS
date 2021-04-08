# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:23:38 2021

@author: klein

Basic functions for statistics.
"""
# In[]
import numpy as np
from scipy import stats
import urllib
import matplotlib.pyplot as plt
#import importlib
#import ExternalFunctions
#importlib.reload(ExternalFunctions)
from toolbox.ExternalFunctions import Chi2Regression
from toolbox.ExternalFunctions import nice_string_output
from toolbox.ExternalFunctions import compute_f
from iminuit.util import make_func_code
from iminuit import describe
from iminuit import Minuit
import sympy as sp
from scipy import special
import itertools
from decimal import Decimal
       
# In[]
def load_url_data(url): 
    """Load text data under url into numpy array. 
    Lines of text separate different data sets."""
    f = urllib.request.urlopen(url)
    Text_encoded = f.readlines()
    Text = [Text_encoded[i].decode("utf-8") for i in range(len(Text_encoded))]
    Start_Lines = []
    Stop_Lines = []
    if Text[0][0].isdigit() or Text[0][1].isdigit():
            Start_Lines.append(0)
            
    for i in range(len(Text)-1):
        col0, col1 = 0, 0
        if Text[i][col0]==' ':
            col0 = 1
            
        if Text[i+1][col1]==' ':
            col1 = 1
            
        if not(Text[i][col0].isdigit()) and (Text[i+1][col1].isdigit()):
            Start_Lines.append(i+1)
        elif not(Text[i+1][col1].isdigit()) and (Text[i][col0].isdigit()):
            Stop_Lines.append(i+1)
        else:
            pass

    if len(Stop_Lines)<len(Start_Lines):
        Stop_Lines.append(len(Text))
    
    Data = []
    for i in range(len(Start_Lines)):
        kwargs = {'skiprows':Start_Lines[i], 
                  'max_rows':(Stop_Lines[i]-Start_Lines[i])}
        f = urllib.request.urlopen(url)
        data = np.loadtxt(f, **kwargs)
        Data.append(data)
    if len(Data) ==1:
        return Data[0]
    else:
        return Data 
    
    
def weighted_avg(arr_mu, arr_sig):
    r"""
    Compute weighted average with uncertainty
    Parameters: 
    arr_mu, arr_sig: array_like
    Returns:
    mu: weighted average
    sig_mu: uncertainty on average
    """
    weights = 1/arr_sig**2
    mu = np.average(arr_mu, weights = weights)
    sig_mu = 1/np.sqrt(np.sum(1/arr_sig**2))
    return mu, sig_mu


def chi_sq(y_pred, y, sy):
    r"""Compute chi2. 
    Returns: sum((y_pred-y)**2/sy**2)"""
    return np.sum((y_pred-y)**2/sy**2)

def chi_sq_const(arr_mu, arr_sig):
    r"""Compute chi2 assuming y_pred=weighted_average
    Returns: sum((y_pred-y)**2/sy**2)"""
    y_pred, _ = weighted_avg(arr_mu, arr_sig)
    return chi_sq(y_pred, arr_mu, arr_sig)


def chi_sq_histtest(O1, O2): 
    r"""Compute chi2 between 2 histograms with same binning
    Parameters: O1, O2: array_like
        counts in histogram
    Returns: sum((O1-O2)**2/(O1+O2))"""
    return np.sum((O1-O2)**2/(O1+O2))    

def std_to_prob(std): 
    r"""Compute probability of having distance>std from the mean=0.
    """
    return (1-(stats.norm.cdf(abs(std))-stats.norm.cdf(-abs(std))))

def chauvenet_num(arr): 
    """Compute for each entry probability*len(arr) 
    assuming normal distribution."""
    z = stats.zscore(arr)
    ch_num = len(arr)*std_to_prob(z)
    return ch_num

def exclude_chauvenet(arr, threshold = .5, removed = True):
    """Based on Chauvenets criterion, removes outliers from an array. 
    Parameters:
        arr: array_like, input data
        threshold: float, (default 0.5) Threshold for chauvenet criterion 
        above which points are accepted
        removed: bool, 
            If True (default) return also array of removed points and prob.
    Returns: 
        Array of remaining points, array of removed points(optional)
    """
    removed, p_glob = [], []
    while np.min((chauvenet_num(arr)))<threshold:
        min_idx = np.argmin(chauvenet_num(arr))
        removed.append(arr[min_idx])
        p_glob.append(chauvenet_num(arr[min_idx]))
        arr_n = np.delete(arr, min_idx)
        arr = arr_n,
    if removed:
        return arr_n, removed, p_glob 
    else:
        return arr_n

def propagate_error(func, symb, cov=False):
    r"""
    Given a functional relationship y = f(x1,x2,x3,...) and corresponding 
    symbols symbols(y, x1, x2,...), computes the symbolic representation of
    function and uncertainty as well as latex code.

    Parameters:
    -----------------
    func: callable, 
        function that can be called by sympy (use e.g. sympy.function)
    symb: array_like, str
        contains symbols which give name to variables, e.g. ["y","x1","x2"...]
    cov: bool, (default = False)
        if True takes covariances between variables into account

    Returns:
    -----------
    eq_f, eq_sf: symbolic repr. of the function and uncertainty, latex code 
                 can be accessed via latex()
    sf_lamb: function to compute uncertainty, if result symbolic use .evalf(), 
        takes first the mean values then uncertainties in the order of symbols
     
    """

    mu_symb = []
    sig_symb = []
    Diff = []
    Contr = []
    N = len(symb)  # length of symbols array
    if cov:  # covariance
        V_symb = []
        num_cov = int(special.comb(N-1, 2))
        combis = list(itertools.combinations(symb[1:], 2))
        for i in range(num_cov):
            V_symb.append(sp.symbols("V_"+combis[i][0]+"_" + combis[i][1]))
    for i in range(N):  # create sympy symbols
        mu_symb.append(sp.symbols(symb[i]))
        sig_symb.append(sp.symbols("sigma_" + symb[i]))
    # create symbolic representation of the function
    mu_symb[0] = func(*mu_symb[1:])
    f = mu_symb[0]
    for i in range(N - 1):
        Diff.append(f.diff(mu_symb[1:][i]))
    for i in range(N - 1):
        Contr.append((Diff[i] * sig_symb[i + 1])**2)
    if cov:
        for i in range(num_cov):#contributions by covariance
            #differentiate again, redundant, might be improved if one can
            #find out in which order combinations are stored
            a = f.diff(sp.symbols("sigma_"+combis[i][0]))
            b = f.diff(sp.symbols("sigma_"+combis[i][1]))
            Contr.append(2*V_symb[i]*a*b)
    sig_symb[0] = sp.sqrt(sum(Contr))

    eq_f = sp.Eq(sp.symbols(symb[0]), mu_symb[0])
    eq_sf = sp.Eq(sp.symbols("sigma_" + symb[0]), sig_symb[0])
    if cov:
        sf_lamb = sp.lambdify((*mu_symb[1:], *sig_symb[1:], *V_symb),
                              sig_symb[0],
                              modules='sympy')
    else:
        sf_lamb = sp.lambdify((*mu_symb[1:], *sig_symb[1:]),
                              sig_symb[0],
                              modules='sympy')
        

    return eq_f, eq_sf, sf_lamb



def round_to_uncertainty(value, uncertainty):
    """Helper function for round_result."""
    # round the uncertainty to 1-2 significant digits
    u = Decimal(uncertainty).normalize()
    exponent = u.adjusted()  # find position of the most significant digit
    precision = (u.as_tuple().digits[0] == 1)  # is the first digit 1?
    u = u.scaleb(-exponent).quantize(Decimal(10)**-precision)
    # round the value to remove excess digits
    return round(Decimal(value).scaleb(-exponent).quantize(u)), u, exponent
    
def round_result(mean, err):
    """Print result with the right number of significant digits 
    determined by the uncertainty."""
    return print("{} ± {} (×10^{})".format(*round_to_uncertainty(mean, err)))


# In[]
def accept_reject(func, N, xmin, xmax, ymax, initial_factor = 2, random_state = 1):
    r"""Produce N random numbers distributed according to the function
        func using accept/reject method."""
    np.random.seed(random_state)
    L = xmax-xmin
    N_i = int(initial_factor*N)
    x_test = L*np.random.uniform(size=N_i)+xmin
    y_test = ymax*np.random.uniform(size = N_i)
    mask_func = y_test<func(x_test)
    if np.count_nonzero(mask_func)>N:
        x_func = x_test[mask_func]
        x_func = x_func[:N]
    else:
        x_func = accept_reject(
            func, N, xmin,xmax, ymax,
            initial_factor = int(initial_factor*2))
    return x_func
    
def transform_method(inv_func,xmin, xmax, N, initial_factor = 2): 
    r"""Produce N random numbers distributed according to f(x) 
    given the inverse inv_func of
    F(x) = \int_{-inf}^x f(x')dx' 
    """
    N_i = 2*N
    x = inv_func(np.random.uniform(size = N_i))
    x = x[(x>xmin) & (x<xmax)]
    if len(x)>N:
        x = x[:N]
    else:
        x = transform_method(
            inv_func,xmin, xmax, N, initial_factor = initial_factor*2 )
    return x

# In[]

def two_sample_test(mu1, mu2, sig_mu1, sig_mu2):
    """Compute p-value for mean values of two samples agreeing with each other.
    Assumes Gaussian distribution."""
    z = (mu1-mu2)/np.sqrt(sig_mu1**2+sig_mu2**2)
    return std_to_prob(np.abs(z)) 

def calc_separation(x, y):
    r"""Compute separation of two variables.
    Returns: 
        d: separation in terms of std 
        p: corr. p value"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    d = np.abs((mean_x - mean_y)) / np.sqrt(std_x**2 + std_y**2)
    
    return d, std_to_prob(d)

def runsTest(arr):
    """Runs test for randomness. 
    Parameters: arr array of digits
    Returns: p-value"""
    runs, np, nn = 0, 0, 0# Checking for start of new run
    median = np.median(arr)
    for i in range(len(arr)): 
        if (arr[i] >= median and arr[i-1] < median) or  (arr[i] < median and arr[i-1] >= median): 
                runs += 1  # no. of runs 
        if(arr[i]) >= median: 
            np += 1    # no. of positive values 
        else: 
            nn += 1    # no. of negative values 
    runs_exp = ((2*np*nn)/(np+nn))+1
    stan_dev = np.sqrt((2*np*nn*(2*np*nn-np-nn))/ 
                       (((np+nn)**2)*(np+nn-1))) 
  
    z = (runs-runs_exp)/stan_dev 
    p = std_to_prob(abs(z))
    return p 

def func_Poisson(x, N, lamb) :
    """Helper function for seq_freq_test. 
    Author: Troels Petersen"""
    if (x > -0.5) :
        return N * stats.poisson.pmf(x, lamb)
    else : 
        return 0.0

def seq_freq_test(
    integers, seq_l = 3, N_bins = 21, show_plot = True, figsize = (12,8)):
    r"""Compare sequence frequency with Poisson hypothesis.
    Poisson hypothesis is fitted and plotted.
    Author: Troels Petersen
    Parameters:
        integers: array_like, input data
        seq_l: int,(default 3) length of sequence to be tested
    Returns: chi2 p-value"""
    seq = []
    for i in range(-(seq_l-1), len(integers)-(seq_l-1) ) : 
        num = 0
        for j in range(seq_l):
            num+=integers[i+j]*int(10**(seq_l-1-j))
        seq.append(num)
    poisson_counts, _ = np.histogram(
        seq, int(10**seq_l+1), range=(-0.5, 10**seq_l+.5))
    xmin, xmax = -0.5, N_bins-.5
    
    fig, ax = plt.subplots(figsize=figsize)
    hist_poisson = ax.hist(poisson_counts, N_bins, 
                           range=(xmin, xmax), label = 'Sequence distribution')
    counts, x_edges, _ = hist_poisson
    
    x_centers = 0.5*(x_edges[1:] + x_edges[:-1])
    x = x_centers[counts>0]
    y = counts[counts>0]
    sy = np.sqrt(y)

    chi2_object = Chi2Regression(func_Poisson, x, y, sy)
    minuit = Minuit(chi2_object, pedantic=False, 
                    N = 10**seq_l, lamb = poisson_counts.mean())
    minuit.migrad()     # Launch the fit

    chi2_val = minuit.fval
    N_DOF = len(y) - len(minuit.args)
    chi2_prob = stats.chi2.sf(chi2_val, N_DOF)
    d = {'Entries'  : "{:d}".format(len(poisson_counts)),
     'Mean'     : "{:.5f}".format(poisson_counts.mean()),
     'STD Dev'  : "{:.5f}".format(poisson_counts.std(ddof=1)),
     'Chi2/ndf' : "{:.1f} / {:2d}".format(chi2_val, N_DOF),
     'Prob'     : "{:.6f}".format(chi2_prob),
     'N'        : "{:.1f} +/- {:.1f}".format(minuit.values['N'], minuit.errors['N']),
     'Lambda'   : "{:.2f} +/- {:.2f}".format(minuit.values['lamb'], minuit.errors['lamb'])
    }

    ax.text(0.62, 0.99, nice_string_output(d), family='monospace',
            transform=ax.transAxes, fontsize=14, verticalalignment='top')

    binwidth = (xmax-xmin) / N_bins 
    xaxis = np.linspace(xmin, xmax, 500)
    func_Poisson_vec = np.vectorize(func_Poisson)
    yaxis = binwidth*func_Poisson_vec(np.floor(xaxis+0.5), *minuit.args)
    ax.plot(xaxis, yaxis,label = 'Poisson distribution')
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    return chi2_prob, fig,ax


class NLLH: 
    """Class for computing the negative log likelihood. 
    The instance can be passed to a minimizer."""
    def __init__(self, f, data):
    
        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.func_code = make_func_code(describe(self.f)[1:])#Function signature for the minimizer
    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)    
        # compute the function value
        f = compute_f(self.f, self.data, *par)
        # compute the sum of the log values: the LLH
        logf = np.log(f)
        nllh = -np.sum(logf)
        
        return nllh
    
class NLLH_scan: 
    """Class for scanning the 1d/2d landscape of the NLLH."""
    def __init__(self, f, data):
    
        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.func_code = make_func_code(describe(self.f)[1:])#Function signature for the minimizer
    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)    
        # compute the function value
        arr_count = 0
        index, par_arr = [], []
        par = list(par)
        for i in range(len(par)):
            if not(type(par[i])==float or type(par[i])==int):#separate into float and array to create a meshgrid
                par_arr.append(par[i]) #stroe arrays into extra list
                index.append(i) #keep track of indices 
        Par_grid = np.meshgrid(*par_arr) #create n-dim meshgrid
        Par_grid_expanded = [None]*len(Par_grid)
        for i in range(len(Par_grid)):
            Par_grid_expanded[i] = np.expand_dims(Par_grid[i], len(index)) #expand dimensions to make function call in parallel
            #this allows to broadcast data and params
            Par_grid_expanded[i] = np.repeat(Par_grid_expanded[i], len(self.data), axis = len(index)) 
            par[index[i]] = Par_grid_expanded[i] #store back into par
        #now function can be called
        f = compute_f(self.f, self.data, *par)
        # compute the sum of the log values: the LLH
        
        logf = np.log(f)
        llh = -np.sum(logf, axis = len(index))#sum over the axis that represents the data points(last axis)
        
        return Par_grid, llh
    
    
def False_Negative_Rate(y_pred, y_true):
    """Compute the number of false positives. 
    Where positive is 1."""
    mask_pred = y_pred==0
    mask_true = y_true==1
    mask_FN = mask_pred&mask_true
    FN =  np.count_nonzero(mask_FN)
    P = np.count_nonzero(y_true==1)
    return FN/P

def False_Positive_Rate(y_pred, y_true):
    """Compute the number of false positives. 
    Where positive is 1."""
    mask_pred = y_pred==1
    mask_true = y_true==0
    mask_FP = mask_pred&mask_true
    FP = np.count_nonzero(mask_FP)
    N = np.count_nonzero(y_true==0)
    return FP/N

def True_Negative_Rate(y_pred, y_true):
    return 1 - False_Positive_Rate(y_pred, y_true)

def True_Positive_Rate(y_pred, y_true):
    return 1 - False_Negative_Rate(y_pred, y_true)

def Precision(y_pred, y_true):
    TPR = True_Positive_Rate(y_pred, y_true)
    FPR = False_Positive_Rate(y_pred, y_true)
    P = np.count_nonzero(y_true==1)
    N = np.count_nonzero(y_true==0)
    TP = TPR*P
    FP = FPR*N
    return TP/(TP+FP)

def False_Discovery_Rate(y_pred,y_true):
    return 1-Precision(y_pred, y_true)

def MCMH(post, prop, Theta0, num_iter = 1000, nwalkers = 10, burn_in = 500):
    """Performs the Monte_Carlo Metropolis Hasting algorithm for a 1d function.
        prop: proposal function, should take only size as argument
        Theta0: initial guess float or array"""
    if num_iter<burn_in:
        raise ValueError('num_iter should be larger than burn_in')
    if type(Theta0==float) or type(Theta0==int):
        Theta0 = Theta0*np.ones(nwalkers)
    Theta_arr = np.empty((num_iter,nwalkers))
    for i in range(num_iter):
        Theta_prop = Theta0+prop(size = nwalkers)
        rand = np.random.uniform(size = nwalkers)
        accept_mask = post(Theta_prop)/post(Theta0)>rand
        Theta0[accept_mask] = Theta_prop[accept_mask]
        Theta_arr[i, accept_mask] = Theta0[accept_mask]
        Theta_arr[i, ~accept_mask] = Theta0[~accept_mask]
    Theta_arr = Theta_arr[burn_in:,:].reshape(-1)
    return Theta_arr

def Create_uniform_angles(Npoints):
    """Given number of points, create angles corresponding to isotropic points on a unit sphere"""
    Phi = np.pi*2*np.random.uniform(size = Npoints)-np.pi
    Theta = np.arccos(2*np.random.uniform(size = Npoints)-1)-np.pi/2
    return Phi, Theta
def cumulative_autocorrelation(X, phi):
    """Compute cum. autocorrelation given X where rows of X are xi, yi, zi, 
    i.e. every column is a 3d vector."""
    Ntot = X.shape[1]
    C = []
    for i in range(0,Ntot):
        for j in range(0,i):
            cosphitemp = np.dot(X[:,i], X[:,j])-np.cos(phi)
            C.append(np.heaviside(cosphitemp,0))
    C = np.sum(np.array(C), axis = 0)*2/(Ntot* (Ntot-1))
    return C

def create_points3d(phi, theta, r = 1):
    """Create a point in spherical coordinates given phi and theta"""
    X = np.empty(( 3,len(phi)))
    X[0,:] = r*np.cos(phi)*np.sin(theta)
    X[1,:] = r*np.sin(phi)*np.sin(theta)
    X[2,:] = r*np.cos(theta)
    return X
