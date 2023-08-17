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

from sklearn.neighbors import KernelDensity

import numpy as np

def kernel_density(data, xvals, normalise=False):
    return kernel_density_sklearn(data, xvals, normalise=False)


def kernel_density_sklearn(data, xvals, normalise=False, bandwidth="silverman", kernel='gaussian', rtol=1E-09):
    '''Performs kernel density estimation (KDE) using sci-kit learn.  This
was added 16-Aug-2023 when seeing that the scipy method was very slow.
This website explains why skl is advantageous in many scenarios:
https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

Also, the sci-kit learn implementation returns the log-likelihood, so
we have an extra step at the end to take the exponent.  The scipy API
is the cleanest ;)

    Arguments:
        data :
            numpy array (float) of events (x or Q values) for each neutron
            event.
        xvals :
            numpy array (float) of points at which to sample the KDE.
        normalise :
            boolean (optional).  If true, then the output spectrum
            is normalised so that its sum equals unity.
        bandwidth :
            float value of width to use, or either "scott" or "silverman" 
            as a string.
        kernel :
            supply the name of the kernel function to use.  There are 
            several - see the scikit-learn website for more details.
        rtol :
            float, the total relative error, essentially speed vs accuracy 
            parameter.  Default is 1 part in 10^9.

    '''

    kde_skl = KernelDensity(bandwidth=bandwidth, kernel=kernel, rtol=rtol)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xvals[:, np.newaxis])
    return np.exp(log_pdf)
    
    


def kernel_density_scipy(data, xvals, normalise=False):

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

    kdevals = kdevals
        
    if normalise:
        datasum = data.size
        kdesum = np.sum(kdevals)
        fact = datasum / kdesum
        
        print("data sum:", datasum)
        print("kde sum:", kdesum)
        print(fact)
        
        kdevals = kdevals * fact

    return kdevals 



def kde_background_subtract(spectrum, background, spectrum_weight=1.0, background_weight=1.0, ratio=None, verbose=False):
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
            spectrum_weight :
                (optional, float or numpy array of floats) the statistical weight
                of the spectrum (e.g. monitor counts, relative strength of signal)
            background_weight :
                (optional, float or numpy array of floats) the statistical weight
                of the background (e.g. monitor counts, relative strength of 
                background)
            verbose : 
                (boolean, optional) whether or not to output diagnostic
                information.
        
        Returns:
            Nothing.  The data set is overwritten.
        '''

    # Ensure data can be manipulated as an arrays, whether
    # float or array is passed
    spectrum = np.asarray(spectrum)
    background = np.asarray(background)
    spectrum_weight = np.asarray(spectrum_weight)
    background_weight = np.asarray(background_weight)

    
    # Check that the dimension of the supplied weights is either unity
    # or equals the length of the data array
    spectrum_weight_size = spectrum_weight.size
    spectrum_size = spectrum.size
    background_weight_size = background_weight.size
    background_size = background.size

    
    if spectrum_weight_size != spectrum_size:
        #print("Size unequal")
        if spectrum_weight_size != 1:
            print("Error: dimension of spectrum weight array (", \
                  spectrum_weight_size, \
                  ") does not equal that of spectrum (", spectrum_size, ")")
            raise SystemExit("Unequal array size")
    if background_weight_size != background_size:
        #print("Size unequal")
        if background_weight_size != 1:
            print("Error: dimension of background weight array (", \
                  background_weight_size, \
                  ") does not equal that of data (", background_size, ")")
            raise SystemExit("Unequal array size")


    # Warn for computation on large array sizes
    estd_time = 7.1E-08*spectrum.size*background.size

    if estd_time > 20.0:
        print("WARNING: kde_background_subtract: estimated run time:", int(estd_time), "seconds", flush=True)
    else:
        print("kde_background_subtract: estimated run time:", int(estd_time), "seconds", flush=True)
 

    # If statistical weights are provided, then we should scale these
    # so that the largest of them is unity

    if spectrum_weight_size == background_weight_size == 1:
        # single float values given for each weight
        if spectrum_weight > background_weight:
            background_weight = background_weight / spectrum_weight
            spectrum_weight = np.asarray(1.0)
        else:
            spectrum_weight = spectrum_weight / background_weight
            background_weight = np.asarray(1.0)

    if verbose:
        print("Spectrum weight:", spectrum_weight)
        print("Background weight:", background_weight)
            
        
    # The number of data points also needs to be taken into account,
    # not just the statistical weights
    if background_weight_size != 1:
        background_strength = np.sum(background_weight)
    else:
        background_strength = background.size * background_weight

    if spectrum_weight_size != 1:
        spectrum_strength = np.sum(spectrum_weight)
    else:
        spectrum_strength = spectrum.size * spectrum_weight

        #bgsize = background.size
        #dtsize = spectrum.size
        #norm = spectrum_strength/(background_strength + spectrum_strength)
        ratio = background_strength/spectrum_strength
    #else:
    #    norm = ratio

    if verbose :
        print("Ratio:", ratio)
        #print("Norm:", norm)
        #print("1-norm:", 1.0-norm)

    # Now for each data point, we identify the probability of
    # rejecting it.  Because of norm, we can just loop over all data
    # points, estimate the intensity at the data point based on the
    # KDE of the background spectrum and roll a random number to see
    # if it survives.  If the random number is below norm * KDE then
    # the event in question dies.

    # Random coin tosses matching size of data array
    dice = np.random.uniform(0.0, 1.0, spectrum_size)

    # Evaluate background intensity at data point locations
    kdebg = kernel_density(background, spectrum)
    #kdebg = kdebg #* background_weight

    # Perform identical evaluation of data points at data point
    # locations The reason to do this is in case there are any
    # artefacts associated with the kernel shape function.  In theory,
    # this step could be skipped entirely but it's likely the kernel
    # function is normalised, so then the relative intensity
    # at the centre point is constant and below 1.0
    kdesp = kernel_density(spectrum, spectrum)
    #kdesp = kdesp #* spectrum_weight

    # We probabilistically reject a point where the KDE of the
    # signal is below the KDE of the background
    point_reject_likelihood = kdebg / kdesp

    # that fraction is the *likelihood* ratio, the probability is
    #normalised, e.g.

    #point_reject_prob = kdebg / (kdebg + kdesp)
    #point_keep_prob = kdesp / (kdebg + kdesp)

    # Why does the likelihood work better for background subtraction?
    # Because in the tails the probability is too high


    # The following is working when weight=1.0.  When the weights are
    # not 1.0 the rejected fraction is OK if the weight of the signal
    # is larger than that of the background, but if the background
    # weight is larger than that of the signal then the number of
    # rejected events is much too high.  The problem, I think, is that
    # the likelihood ratio is correctly bounded to zero on one side
    # but goes to infinity on the other side, which is why the
    # calculation breaks down.

    # Having said that, the behaviour of the method is absolutely
    # correct when the weights are unity, whether the background is
    # large or small.
    
    reject_mask = dice <= point_reject_likelihood * ratio
    keep_mask = np.invert(reject_mask)

    subtracted = spectrum[keep_mask]

    subtracted_size = subtracted.size

    if verbose:
        print("dice", dice)
        print("x   ", spectrum)
        print("spec", kdesp)
        print("back", kdebg)
        print("rej ", point_reject_likelihood)
        print("masj", reject_mask)

        print("# in ", spectrum_size)
        print("# out", subtracted_size)
        print("survive fraction", subtracted_size/spectrum_size)
    
    return subtracted

