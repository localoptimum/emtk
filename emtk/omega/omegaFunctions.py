"""omega.py

This is a standard set of statistical functions that one normally
enounters in neutron scattering, such as background subtraction,
detector efficiency correction, etc.  The difference being, these
functions are traditionally written for histogram operations.  In a
histogram, we group events into bins and operate on each bin in turn,
taking into account the mean wavelength, time, weight etc of the bin.
In contrast, event mode requires that we treat each individual event
uniquely.
"""

from scipy.stats import gaussian_kde

import numpy as np

def kernel_density(data, xvals, normalise=False):

    
    '''Performs kernel density estimation (KDE) using scipy.

    Kernel density estimation is a smoother way of sampling the
    intensity of a distribution of points compared to a histogram.

    Note that if the array xvals is a grid of equally-spaced points
    and a top-hat function were used as the kernel function, then that
    would be mathematically identical to a traditional histogram.


    Arguments:
        data :
            numpy array (float) of events (x or Q values) for each neutron
            event.
        xvals :
            numpy array (float) of points at which to sample the KDE.
        normalise :
            boolean (optional).  If true, then the output spectrum
            is normalised so that its sum equals unity.

    '''
        
    kdeobject = gaussian_kde(data, bw_method='silverman')

    kdevals = kdeobject.evaluate(xvals)
        
    if normalise:
        datasum = data.size
        kdesum = np.sum(kdevals)
        fact = datasum / kdesum
        
        print("data sum:", datasum)
        print("kde sum:", kdesum)
        print(fact)
        
        kdevals = kdevals * fact

    return kdevals 



def kde_background_subtract(spectrum, background, ratio=None, verbose=False):
    ''' Uses kernel density estimation to establish the intensity of
        an arbitrary measured background at each point in the data set.  Then
        probabilistically removes data points that are correlated with this
        background.

        Arguments:
            spectrum : 
                numpy array of Q values of each neutron event
            background :
                numpy array of Q values of each neutron event in the background
                measurement
            ratio :
                (float, optional) the signal to noise ratio to assume, e.g. the 
                ratio of monitor counts in the spectra.  If none is given, then 
                the total of the spectrum and background will be calculated.
            verbose : 
                (boolean, optional) whether or not to output diagnostic
                information.
        
        Returns:
            Nothing.  The data set is overwritten.
        '''
    
    if ratio is None :
        bgsize = background.size
        dtsize = spectrum.size
        norm = dtsize/(bgsize+dtsize)
        ratio = bgsize/dtsize
    else:
        norm = ratio

    if verbose :
        print("Ratio:", ratio)
        print("Norm:", norm)
        print("1-norm:", 1.0-norm)

    # Now for each data point, we identify the probability of rejecting it.
    # Because of norm, we can just loop over all data points, estimate the
    # intensity at the data point based on the KDE of the background spectrum
    # and roll a random number to see if it survives.  If the random number is
    # below norm * KDE then the event in question dies.
    
    dice = np.random.uniform(0.0, 1.0, dtsize)
    
    kdebg = kernel_density(background, spectrum)

    kdesp = kernel_density(spectrum, spectrum)

    rejectProbability = kdebg / kdesp
    
    keepmask = dice <= rejectProbability*ratio
    rejectmask = np.invert(keepmask)

    subtracted = spectrum[rejectmask]

    if verbose:
        print("dice", dice)
        print("x   ", spectrum)
        print("spec", kdesp)
        print("back", kdebg)
        print("rej ", rejectProbability)
        print("masj", rejectmask)
    
    return subtracted

