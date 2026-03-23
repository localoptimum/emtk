#!/Users/phillipbentley/anaconda3/bin/python

import h5py
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

RANDOM_SEED = 0
np.random.seed(seed=RANDOM_SEED)

from lmfit import Model 

from scipy.stats import cauchy
from scipy.stats import norm
from scipy.stats import uniform

import copy

import time

import emcee

# For parallel EMCEE
import os
os.environ["OMP_NUM_THREADS"] = "1"

#On linux setup Pool like this:
#from multiprocessing import Pool

#On mac, setup Pool like this:
import multiprocessing as mp
Pool = mp.get_context('fork').Pool


emcee_walkers = 64
emcee_burn = 10
emcee_iterations = 20

num_boots = 10

subsample_size = 45000

lsMinEvents = 50000
mcmcMinEvents = 1000000000 #10
lsMaxEvents = 10000000
mcmcMaxEvents = 500 #110000

parallel = True

testRun = False

verbose = False

def loadRawARCS(number):
    ldpath="/Users/phillipbentley/Code/python/mle/data/SNS/ARCS/ZrH2/IPTS-27751/nexus"
    stem="/ARCS_"
    tail=".nxs.h5"

    filename = ldpath + stem + str(number) + tail

    print(filename)

    f = h5py.File(filename, 'r')

    print( list(f.keys()) )

    entr = f['entry']
    b30 = entr['bank30_events']

    
    print( b30 )
    print( list(b30.keys()) )

    print(b30['event_id'])

    f.close()


def listkeys(obj):
    print( list( obj.keys()) )

def loadARCSmd(number):
    ldpath="/Users/phillipbentley/Code/python/mle/data/SNS/ARCS/ZrH2/IPTS-27751/nexus/"
    tail="-exported.nxs"

    filename = ldpath + str(number) + tail

    print(filename)

    f = h5py.File(filename, 'r')
    ws = f['MDEventWorkspace']
    cs = ws['coordinate_system']
    ed = ws['event_data']['event_data']
    bs = ws['box_structure']
    ex = ws['experiment0']
    pr = ws['process']
    vn = ws['visual_normalization']

    #for i in range(7):
    #    ed1 = ed[:,i]
    #    fig,ax = plt.subplots()
    #    plt.plot(ed1)

    dE = ed[:,6]
    wt = ed[:,0]

    #fig,ax = plt.subplots()
    #plt.plot(dE, wt)

    # Filter out zeros

    mask = wt > 0.0

    keepdE = dE[mask]
    keepwt = wt[mask]

    return keepdE, keepwt







""" We might need an integral function between two points (xmin, xmax) 
for every term in the fitting function, so that the relative likelihoods are 
normalised within the data bounds.  It will be good to check whether these
are actually needed or not in the final analysis, but in previous work
these proved to be necessary.
"""

def cauchy_integral(x1, x2, kappa):
    # Returns the integral of a cauchy distribution between two x values

    if x1 < x2:
        xmin = x1
        xmax = x2
    else:
        xmin = x2
        xmax = x1

    
    t1 = np.arctan(xmax/kappa)
    t2 = np.arctan(xmin/kappa)
    
    return (t1 - t2)/np.pi

    

def uniform_integral(x1, x2):
    # Returns the integral of a uniform distribution between two x values

    if(x1 == x2):
        return 0.0

    return np.absolute(x1-x2)



def gaussian_integral(x1, x2, mu=0.0, sigma=1.0):
    # Integral of a gaussian curve between two points
    t1 = norm.cdf(x1, loc=mu, scale=sigma)
    t2 = norm.cdf(x2, loc=mu, scale=sigma)
    intg = t1 - t2

    return np.absolute(intg)




evs, wts = loadARCSmd(201616)

# Shuffle them

perm = np.arange(evs.size)
np.random.shuffle(perm)

events = evs[perm]
weights= wts[perm]

sube = events[1:subsample_size-1]
subw = weights[1:subsample_size-1]


xmin = np.amin(sube)
xmax = np.amax(sube)





def optimal_n_bins(data) -> int:
    """Calculates the optimal number of bins from Freedman-Diaconis rule.
        See for example:
        https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
        https://en.wikipedia.org/wiki/Freedman–Diaconis_rule
    
        """
    
    # Protect against calling when there are no data points
    if data is None:
        raise ValueError(
            f"attempt to find optimal number of data points with no data defined."
        )

    #  Get the range of values for the events
    xmin=np.amin(data)
    xmax=np.amax(data)
    n_events = data.size
    
    # Apply the Freedman-Diaconis calculation
    # First calculate the interquartile range of the data
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    
    # If all the data points are equal (or maybe there is only one data point)
    # then the IQR is zero and that makes no sense for anything that comes after
    if iqr == 0.0:
        print("WARNING: interquartile range is zero.")
        return 0
    
    # If we get to this point it's probably OK, return the Freedman-Diaconis value
    return int((xmax - xmin)*n_events**(1.0/3.0)/(2.0*iqr))





def calculate_histogram(data, weights=None):
    """Calculates a histogram of the weighted events using
        numpy.histogram.  Just prepares the data, does not plot.  The
        actual plotting is done by plot_histogram().
    
        """

    # Protect against no data points
    if data is None:
        raise ValueError(
            f"attempt to compute histogram with no data defined."
        )
    
    # If we get here, we have events
    
    #  Get the range of values for the events
    xmin=np.amin(data)
    xmax=np.amax(data)
    
    # Calculate the optimum number of histogram bins
    opt_n_bin = optimal_n_bins(data)
    
    # Create that number of bins spanning the range of event values
    slic=(xmax - xmin)/(opt_n_bin+1)
    hbins = np.arange(xmin, xmax, slic)
    
    # Maybe the events are weighted, maybe they aren't.  Handle both scenarios.
    if weights is None:
        hst = np.histogram(data, bins=hbins, density=True)
    else:
        hst = np.histogram(data, bins=hbins, density=True, weights=weights)
        
    # The way that numpy makes histograms is not x-y pairs but x bins
    # We'll remove the last point and plot the histogram as a matplotlib step
    # later
    x_hist = hst[1]
    x_hist = x_hist[:-1]
    
    #Apply Thomas' shift.  Now we need to plot at the mid point of the x value
    #rather than the "pre" point with matplotlib.step
    x_hist = x_hist + 0.5*(x_hist[1] - x_hist[0])
    
    # Grab the y values
    y_hist = hst[0]
    
    # Error values are square root of y values (poisson statistics)
    e_hist = np.sqrt(y_hist)
    
    return x_hist, y_hist, e_hist





# Build MCMC Models with API

def simplex_weights(Qraw: np.ndarray) -> np.ndarray:
    # Raw Q values can run between 0-1 to keep things simple
    # The sum is assumed to be 1, preventing out of gamut values
    # That condition is enforced already in the last 4 terms of 
    # the log_prior above.
    # Note that Qraw has one dimension fewer
    # than the number of parameters, like this:
    # https://en.m.wikipedia.org/wiki/Ternary_plot
    
    Qraw = np.asarray(Qraw)
    
    Qsum = np.sum(Qraw)    
    Qlast = 1.0 - Qsum
    
    Qvals = np.append(Qraw, Qlast)
    return Qvals

def log_prior_function(theta):
    # The main role of this function is to set 
    # parameter bounds of the bayesian search space
    elmu, mu1, mu2, mu3, mu4, mubg1, mubg2, elsigma, s1, s2, s3, s4, sbg1, sbg2, me, m1, m2, m3, m4, mbg1 = theta
    
    if -100.0 < elmu < 100.0 and \
        100.0 < mu1  < 200.0 and \
        200.0 < mu2  < 350.0 and \
        350.0 < mu3  < 450.0 and \
        500.0 < mu4  < 620.0 and \
        600.0 < mubg1< 700.0 and \
        0.0   < mubg2< 200.0 and \
        10.0 < elsigma < 100.0 and \
        10.0 < s1 < 100.0 and \
        10.0 < s2 < 100.0 and \
        10.0 < s3 < 100.0 and \
        10.0 < s4 < 100.0 and \
        100.0 < sbg1 < 250.0 and \
        100.0  < sbg2 < 250.0 and \
        0.01 < me < 1.0 and\
        0.01 < m1 < 1.0 and\
        0.01 < m2 < 1.0 and\
        0.01 < m3 < 1.0 and\
        0.01 < m4 < 1.0 and\
        0.01 < mbg1 < 1.0 and\
        me + m1 + m2 + m3 + m4 + mbg1 < 1.0:
        return 0.0
    
    return -np.inf



def probability_mass_function(theta, xx, xmin, xmax, pweights, verbose=False):
    
    elmu, mu1, mu2, mu3, mu4, mubg1, mubg2, elsigma, s1, s2, s3, s4, sbg1, sbg2, me, m1, m2, m3, m4, mbg1 = theta

    pweights = np.asarray(pweights)
    
    if (pweights==None).any():
        use_weights = np.ones_like(xx)
    else:
        use_weights = pweights

    spscale = xmax-xmin


    mvals = simplex_weights(np.array([me, m1, m2, m3, m4, mbg1]))

    el = mvals[0] * norm.pdf(xx, scale=elsigma, loc=elmu) / gaussian_integral(xmin, xmax, elmu, elsigma)
    l1 = mvals[1] * norm.pdf(xx, scale=s1, loc=mu1) / gaussian_integral(xmin, xmax, mu1, s1)
    l2 = mvals[2] * norm.pdf(xx, scale=s2, loc=mu2) / gaussian_integral(xmin, xmax, mu2, s2)
    l3 = mvals[3] * norm.pdf(xx, scale=s3, loc=mu3) / gaussian_integral(xmin, xmax, mu3, s3)
    l4 = mvals[4] * norm.pdf(xx, scale=s4, loc=mu4) / gaussian_integral(xmin, xmax, mu4, s4)
    bg1= mvals[5] * norm.pdf(xx, scale=sbg1, loc=mubg1) / gaussian_integral(xmin, xmax, mubg1, sbg1)
    bg2= mvals[6] * norm.pdf(xx, scale=sbg2, loc=mubg2) / gaussian_integral(xmin, xmax, mubg2, sbg2)

    
    sol = (el + l1 + l2 + l3 + l4 + bg1 + bg2)**use_weights
    
    return sol



def log_likelihood_function(theta, data, xmin, xmax, pweights, mylpf, verbose=False):

    raiseError = False
    result = 0.0

    elmu, mu1, mu2, mu3, mu4, mubg1, mubg2, elsigma, s1, s2, s3, s4, sbg1, sbg2, me, m1, m2, m3, m4, mbg1 = theta

    pweights = np.asarray(pweights)
    
    if (pweights==None).any():
        use_weights = np.ones_like(data)
    else:
        use_weights = pweights
        
    
    lp = mylpf(theta)

    
    if np.isinf(lp):
        return -np.inf

    mvals = simplex_weights(np.array([me, m1, m2, m3, m4, mbg1]))

    if verbose:
        print("mvals:")
        print(mvals)

    el = mvals[0] * norm.pdf(data, scale=elsigma, loc=elmu) / gaussian_integral(xmin, xmax, elmu, elsigma)
    l1 = mvals[1] * norm.pdf(data, scale=s1, loc=mu1) / gaussian_integral(xmin, xmax, mu1, s1)
    l2 = mvals[2] * norm.pdf(data, scale=s2, loc=mu2) / gaussian_integral(xmin, xmax, mu2, s2)
    l3 = mvals[3] * norm.pdf(data, scale=s3, loc=mu3) / gaussian_integral(xmin, xmax, mu3, s3)
    l4 = mvals[4] * norm.pdf(data, scale=s4, loc=mu4) / gaussian_integral(xmin, xmax, mu4, s4)
    bg1= mvals[5] * norm.pdf(data, scale=sbg1, loc=mubg1) / gaussian_integral(xmin, xmax, mubg1, sbg1)
    bg2= mvals[6] * norm.pdf(data, scale=sbg2, loc=mubg2) / gaussian_integral(xmin, xmax, mubg2, sbg2)


    # Guard against zero values in each term
    minval = 1.0E-300

    msk = el < minval
    el[msk] = minval

    msk = l1 < minval
    l1[msk] = minval

    msk = l2 < minval
    l2[msk] = minval

    msk = l3 < minval
    l3[msk] = minval

    msk = l4 < minval
    l4[msk] = minval

    msk = bg1 < minval
    bg1[msk] = minval

    msk = bg2 < minval
    bg2[msk] = minval

    
 #   try:
    lel = np.log(el)
    ll1 = np.log(l1)
    ll2 = np.log(l2)
    ll3 = np.log(l3)
    ll4 = np.log(l4)
    lbg1 = np.log(bg1)
    lbg2 = np.log(bg2)

    lt1 = np.logaddexp(lel, ll1)
    lt2 = np.logaddexp(lt1, ll2)
    lt3 = np.logaddexp(lt2, ll3)
    lt4 = np.logaddexp(lt3, ll4)
    lt5 = np.logaddexp(lt4, lbg1)
    lt6 = np.logaddexp(lt5, lbg2)

    lll = np.sum(lt6 * use_weights)
    
    result = lp + lll

    if np.isnan(result):
        print("NaN in log_likelihood at", theta)
        verbose=True
                    
    if verbose:
        print("lp", lp)
        print("lll", lll)
        print("l1", l1)
        print("l2", l2)
        print("l3", l3)
        print("l4", l4)
        print("bg1", bg1)
        print("bg2", bg2)
        print("")
        print("m0", mvals[0])
        print("m1", mvals[1])
        print("m2", mvals[2])
        print("m3", mvals[3])
        print("m4", mvals[4])
        print("m5", mvals[5])
        print("m6", mvals[6])

    return result


def big_lse_pdf(x, amplitude, elmu, mu1, mu2, mu3, mu4, mubg1, mubg2, elsigma, s1, s2, s3, s4, sbg1, sbg2, me, m1, m2, m3, m4, mbg1):
    
    mvals = simplex_weights(np.array([me, m1, m2, m3, m4, mbg1]))

    el = mvals[0] * norm.pdf(x, scale=elsigma, loc=elmu) / gaussian_integral(xmin, xmax, elsigma)

    l1 = mvals[1] * norm.pdf(x, scale=s1, loc=mu1) / gaussian_integral(xmin, xmax, mu1, s1)
    l2 = mvals[2] * norm.pdf(x, scale=s2, loc=mu2) / gaussian_integral(xmin, xmax, mu2, s2)
    l3 = mvals[3] * norm.pdf(x, scale=s3, loc=mu3) / gaussian_integral(xmin, xmax, mu3, s3)
    l4 = mvals[4] * norm.pdf(x, scale=s4, loc=mu4) / gaussian_integral(xmin, xmax, mu4, s4)
    bg1= mvals[5] * norm.pdf(x, scale=sbg1, loc=mubg1) / gaussian_integral(xmin, xmax, mubg1, sbg1)
    bg2= mvals[6] * cauchy.pdf(x, scale=sbg2, loc=mubg2) / cauchy_integral(xmin, xmax, sbg2)
    
    sol = amplitude * (el + l1 + l2 + l3 + l4 + bg1 + bg2)
    
    return sol

def big_background_pdf(x, amplitude, elmu, mu1, mu2, mu3, mu4, mubg1, mubg2, elsigma, s1, s2, s3, s4, sbg1, sbg2, me, m1, m2, m3, m4, mbg1):

    mvals = simplex_weights(np.array([me, m1, m2, m3, m4, mbg1]))

    bg1= mvals[5] * norm.pdf(x, scale=sbg1, loc=mubg1) / gaussian_integral(xmin, xmax, mubg1, sbg1)
    bg2= mvals[6] * cauchy.pdf(x, scale=sbg2, loc=mubg2) / cauchy_integral(xmin, xmax, sbg2)
    
    sol = amplitude * (bg1 + bg2)
    
    return sol




least_squares_model = Model(big_lse_pdf)
least_squares_parameters = least_squares_model.make_params(amplitude=dict(value=1.0, min=0.0),\
    elmu = dict(value=2.0, min=-100.0, max=100.0),\
    mu1 = dict(value=145.0, min=100.0, max=200.0),\
    mu2 = dict(value=280.0, min=200.0, max=350.0),\
    mu3 = dict(value=420.0, min=350.0, max=450.0),\
    mu4 = dict(value=560.0, min=500.0, max=620.0),\
    mubg1=dict(value=620.0, min=600.0, max=700.0),\
    mubg2=dict(value=140.0, min=0.0, max=200.0),\

    elsigma = dict(value=20.0, min=10.0, max=100.0),\
    s1 = dict(value=30.0, min=10.0, max=100.0),\
    s2 = dict(value=30.0, min=10.0, max=100.0),\
    s3 = dict(value=30.0, min=10.0, max=100.0),\
    s4 = dict(value=30.0, min=10.0, max=100.0),\
    sbg1=dict(value=105.0, min=100.0, max=250.0),\
    sbg2=dict(value=105.0, min=100.0, max=250.0),\
    me=dict(value=0.11, min=0.01, max = 1.0),\
    m1=dict(value=0.11, min=0.01, max = 1.0),\
    m2=dict(value=0.11, min=0.01, max = 1.0),\
    m3=dict(value=0.11, min=0.01, max = 1.0),\
    m4=dict(value=0.11, min=0.01, max = 1.0),\
    mbg1=dict(value=0.11, min=0.01, max=1.0))


same_as_least_squares = np.array([2.0, 145.0, 280.0, 420.0, 560.0,\
                                 620.0, 140.0, 20.0, 30.0, 30.0,\
                                 30.0, 30.0, 105.0, 105.0,\
                                 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])


p0 = np.asarray(same_as_least_squares)
p0 = [p0 + (10.0 ** -2) * np.random.randn(same_as_least_squares.size) for k in range(emcee_walkers)]

if testRun == True:

    print("Running a single iteration test run.")
    
    histx, histy, histe = calculate_histogram(sube, subw)

    lse_result = least_squares_model.fit(histy, least_squares_parameters, weights=histe, x=histx)

    valdict = lse_result.best_values
    pvals = np.zeros(len(valdict))
    
    i = 0
    for key in valdict:
        pvals[i] = valdict[key]
        i=i+1

    print(pvals)
        
    quit()
    
    if parallel == False:
        start = time.time()
        print("Time started", start)
        sampler = emcee.EnsembleSampler(emcee_walkers, (same_as_least_squares.size), log_likelihood_function, threads=4, args=[sube, xmin, xmax, subw, log_prior_function])
        print("Burn in:")
        sampler_state = sampler.run_mcmc(p0, emcee_burn, progress=True)
        sampler.reset()
        print("Sampling:")
        sampler_state = sampler.run_mcmc(sampler_state, emcee_iterations, progress=True)
        print("MCMC sampling complete.")
        end = time.time()
        print("Time finished", end)
        single_time = end - start
        print("Single EMCEE run took {0:.1f} seconds".format(single_time))
    else:
        with Pool() as pool:
            start = time.time()
            print("Time started", start)
            sampler = emcee.EnsembleSampler(emcee_walkers, (same_as_least_squares.size), log_likelihood_function, pool=pool, args=[sube, xmin, xmax, subw, log_prior_function])
            print("Burn in:")
            sampler_state = sampler.run_mcmc(p0, emcee_burn, progress=True)
            sampler.reset()
            print("Sampling:")
            sampler_state = sampler.run_mcmc(sampler_state, emcee_iterations, progress=True)
            print("MCMC sampling complete.")
            end = time.time()
            print("Time finished", end)
            single_time = end - start
            print("Single EMCEE run took {0:.1f} seconds".format(single_time))

    print("End of test run.  Quitting.")
    quit()



def get_MCMC_params(sampler):
    samples = sampler.get_chain(flat=True)
    ndim = same_as_least_squares.size
    rge = range(ndim)

    nn = samples.shape[0]
    rootn = np.sqrt(nn)

    mcmc_parameter_values = np.zeros(ndim)
    mcmc_parameter_errors = np.zeros(ndim)

    for i in rge:
        mcmc_parameter_values[i] = np.mean(samples[:,i])
        stdd = np.std(samples[:,i])
        mcmc_parameter_errors[i] = stdd/rootn        

    return mcmc_parameter_values, mcmc_parameter_errors


# Bootstrap loop to measure convergence with MC sample of variances for each step
maxsiz = np.log10(evs.size)-1 # We do 10 trials on each step
print("Total events:", evs.size)
span0 = np.linspace(0, evs.size/num_boots, num_boots)
span0 = np.round(span0).astype(int)
print("span0:", span0)


print(maxsiz)
evreps=10
n_evs = np.logspace(2, maxsiz, evreps).astype(int)
print(n_evs)
n_evs[-1]=evs.size
print(n_evs)

overall_lse_results = np.zeros((evreps, same_as_least_squares.size+1))
overall_lse_errors  = np.zeros((evreps, same_as_least_squares.size+1))
overall_mcmc_results = np.zeros((evreps, same_as_least_squares.size))
overall_mcmc_errors = np.zeros((evreps, same_as_least_squares.size))





rep = 0

start = time.time()
print("Time started", start)

lse_result = None

for nn in n_evs:
    
    lse_results = np.zeros((num_boots, same_as_least_squares.size+1))
    lse_errors  = np.zeros((num_boots, same_as_least_squares.size+1))
    mcmc_results = np.zeros((num_boots, same_as_least_squares.size))
    mcmc_errors = np.zeros((num_boots, same_as_least_squares.size))
    
    for boot in range(num_boots):
        print("Rep", rep+1, "/", evreps, "| Size ", nn, "/", evs.size, "| boot", boot, "/", num_boots)

        slicestart = span0[boot]
        sliceend = span0[boot]+nn
        print("Subsampling [", slicestart, ":", sliceend, "]")
        subsample_events = evs[slicestart:sliceend]
        subsample_weights= wts[slicestart:sliceend]

        p0 = np.asarray(same_as_least_squares)
        p0 = [p0 + (10.0 ** -2) * np.random.randn(same_as_least_squares.size) for k in range(emcee_walkers)]
        
        if nn >= lsMinEvents and nn <= lsMaxEvents:
            print("Least Squares Analysis...")

            histx, histy, histe = calculate_histogram(subsample_events, subsample_weights)

            lse_result = least_squares_model.fit(histy, least_squares_parameters, weights=histe, x=histx, method='leastsq')

            valdict = lse_result.best_values
            pvals = np.zeros(len(valdict))
            
            i = 0
            for key in valdict:
                pvals[i] = valdict[key]
                i=i+1
                        
            lse_results[boot,:] = pvals

        else:
            print("   --- Skipping Least Squares Analysis")
            
        if nn >= mcmcMinEvents and nn <= mcmcMaxEvents:
            print("MCMC Analysis...")

            if parallel == False:
                sampler = emcee.EnsembleSampler(emcee_walkers, (same_as_least_squares.size), log_likelihood_function, threads=4, args=[subsample_events, xmin, xmax, subsample_weights, log_prior_function])
                print("Burn in:")
                sampler_state = sampler.run_mcmc(p0, emcee_burn, progress=False)
                sampler.reset()
                print("Sampling:")
                sampler_state = sampler.run_mcmc(sampler_state, emcee_iterations, progress=False)
                print("MCMC sampling complete.")
            else:
                with Pool() as pool:
                    sampler = emcee.EnsembleSampler(emcee_walkers, (same_as_least_squares.size), log_likelihood_function, pool=pool, args=[subsample_events, xmin, xmax, subsample_weights, log_prior_function])
                    print("Burn in:")
                    sampler_state = sampler.run_mcmc(p0, emcee_burn, progress=False)
                    sampler.reset()
                    print("Sampling:")
                    sampler_state = sampler.run_mcmc(sampler_state, emcee_iterations, progress=False)
                    print("MCMC sampling complete.")
                        
            mcmc_results[boot,:], mcmc_errors[boot,:] = get_MCMC_params(sampler)
            
        else:
            print("   --- Skipping MCMC Analysis")

        if verbose:
            print("Current bootstrap LSE grid:")
            print(lse_results)

            print("Current bootstrap MCMC grid:")
            print(mcmc_results)

    
    # All 10 boots have now been done, copy over the mean and stddev to the overall results arrays
    overall_mcmc_results[rep,:] = np.mean(mcmc_results, axis=0)
    overall_mcmc_errors[rep,:] = np.std(mcmc_errors, axis=0)
    overall_lse_results[rep,:] = np.mean(lse_results, axis=0)
    overall_lse_errors[rep,:] = np.std(lse_results, axis=0)

    if verbose:
        print("Current overall LSE grid:")
        print(overall_lse_results)
        print("Current overall LSE errors:")
        print(overall_lse_errors)
    
    rep = rep + 1
                
print("Convergence bootstrap loop complete.")

if verbose:
    print("Overall lse grid:")
    print(overall_lse_results)

end = time.time()
print("Time finished", end)
single_time = end - start
print("Bootstrap cycle took {0:.1f} seconds".format(single_time))


# Get the parameter names in the dictionary from the last LSE analysis run
valdict = lse_result.best_values
pnams = [None]*len(valdict)

i = 0
for key in valdict:
    pnams[i] = key
    i=i+1



with open('arcs_parameters.npy', 'wb') as f:
    np.save(f, n_evs)
    np.save(f, pnams)
    np.save(f, overall_lse_results)
    np.save(f, overall_lse_errors)
    np.save(f, overall_mcmc_results)
    np.save(f, overall_mcmc_errors)
    


