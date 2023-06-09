"""A maximum likelihood toolkit for use in neutron scattering.

Maximum likelihood estimation (MLE) is a reliable way to estimate
parameters of probability distriubtions.  Unlike least-squares
regression, for example, maximum likelihood does not require the data
to be binned into histograms.  Another advantage over least squares is
that it tends to better fit the long tails of distributions and resist
being overpowered by strong peaks.  This is because of the log-sum
(instead of the linear difference squared) at the core of the
routines.  For neutron scattering, typically fitting Lorentzian or
Voigtian lineshapes to curves on a log plot, this is ideal.  This can
make MLE a *lot* more accurate in many scenarios that occur commonly
in neutron scattering, muon spin relaxation etc.  It is also quite
tolerant of statistical noise.

The main drawback is that MLE can be quite time consuming to set up
for new functions.  Instead of a single probability distribution, in
an ideal world you define an analytical solution to the log-likelihood
estimates.  If no analytical form exists, the next best thing is to
provide analytical expressions for both the first and second
derivatives of the logarithm of the probability distribution.  These
are numerically generated in the base class if they don't exist.
Finally, for generation of synthetic test data, or for some
quality-of-fit metrics, the cumulative distribution function (CDF),
the quantile function (the inverse of the CDF) should also be
supplied.  The quantile function often does not exist in analytical
form, so a numerical inverse is performed on the CDF if the base
quantile function is not overloaded.

Typical usage example (for simple SANS):

# Prepare the data:
import emtk
import numpy as np
clength = 90.0 # correlation length (angstroms)
kappa = 1.0 / clength
curv = emtk.lorentzianCurve()
pvalues = np.array([kappa])
xrange = np.array([0.001, 0.1]) # minimum and maximum Q values (AA-1)
curv.generateTestSamples(pvalues, xrange, 2000) # generate data points
curv.generatebackground(xrange, ratio=0.1) # generate a flat 10% bg

# Fit the data:
curv.mle() # perform the maximum likelihood estimation 
curv.report() # summarise the results
curv.plotFit(logarithmic=True) # display a graph of the fitted data

"""

import numpy as np

import powerlaw

import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.special import erfinv
from scipy.optimize import minimize
from scipy.special import sici

from scipy.stats import gaussian_kde

def sinintegral(theta):
    si, ci = sici(theta)
    return si


def kernel_density(data, xvals, normalise=False):
    '''Performs kernel density estimation using scipy
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


def kdeBackgroundSubtract(spectrum, background, ratio=None):
    ''' Uses kernel density estimation to establish the intensity of
        an arbitrary measured background at each point in the data set.  Then
        probabilistically removes data points that are correlated with this
        background.

        Parameters:
            bg : 
                numpy array of Q values of background events
            ratio : 
                (float, optional) the signal to noise ratio to assume, e.g. the 
                ratio of monitor counts in the spectra.  If none is given, then 
                the total counts in each spectrum will be used 
        
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
    
    print("dice", dice)
    print("x   ", spectrum)
    print("spec", kdesp)
    print("back", kdebg)
    print("rej ", rejectProbability)
    print("masj", rejectmask)
    
    return subtracted
    


class MLECurve:
    """Base class for maximum likelihood estimation.  

    Attributes: 
        data : a numpy array (float) of Q values of detected neutrons 
        estimates : a numpy array (float) of estimated parameter values 
        guesses : the initial parameter estimates to use for iterating 
                  the estimates (only needed for non-analytic methods)

    Methods: 
        mle : 
            performs maximum likelihood estimation
        generatebackground : 
            create a flat, random background on the data
        llcurve : 
            the log-likelihood function of the distribution/
        dllcurve : 
            the first derivative of the log-likelihood function with
            respect to each parameter
        ddllcurve : 
            the second derivative of the log-likelihood function 
            with respect to each parameter
        generateTestSamples : 
            creates synthetic data that follows the probability 
            distribution
        Quantile : the quantile function of the distribution
        CDF : the cumulative distribution function
        sortData : sorts the data in numerical order.  This is needed 
                   in some statistical test metrics

    """

    def __init__(self):
        self.nparams=0
        self.estimates = np.array([None])
        self.guesses = np.array([None])
        self.variances = np.array([None])
        self.data = np.empty(0)
        self.hessian = np.empty(0)
        self.method = "only from guesses"
        self.forcenumeric = False
        self.issorted = False
        self.kdeobject = None

    def kde(self, xvals):
        yvals = kernel_density(self.data, xvals)
        return yvals

    def mle(self, verbose = False):
        """Numerical maximum likelihood estimation from the base class.  Uses
        Newton's method.  Derived classes can (should) override this
        method if there exists closed analytical forms for estimators,
        which will make things slightly more accurate and probably
        more stable under more extreme conditions (e.g. small data
        sets).

        Results are stored in the base class estimates attribute.

        Args:
            verbose:
                (optional) whether to print diagnostic information.

        Returns:
            Nothing.  May need a status return code eventually.

        """

        self.estimates = self.guesses
        run = True
        nsteps=0

        #print("Numerical MLE")
        #print(self.estimates, "as starting estimates")

        self.method = "numerically"

        while run:
            yzero = self.dllcurve(self.estimates) # the derivative (to solve)
            yprime = self.ddllcurve(self.estimates) # the second differential used to find the roots
            newvals = self.estimates - yzero / yprime # simple Newton iteration
            fracchange = newvals - self.estimates #self.estimates
            self.estimates = newvals
            nsteps = nsteps + 1
            if verbose:
                print(self.estimates, yzero, yprime, fracchange)
            if(nsteps > 50 or fracchange.all() < 1.0E-06):
                run=False

    # Can use Scipy.optimize.minimize if you want
    # This requires a NEGATIVE log likelihood to minimise
    def negll(self, params):
        """Negative log-likelihood function, useful for experimenting with
        scipy.optimize.minimize (experimental)

        Args:
            params: a numpy array of parameter values (float)

        Returns:
            The negative sum of the log-likelihood (float)
        """

        return -self.llcurve(params)

    # Then just call minimize on that function using the initial guesses
    # a solid gradient descent mode is probably fine for what we need
    def mlescipy(self):
        """Uses scipy minimize to find a numerical root of the negative log-likelihood.

        """

        result = minimize(self.negll, self.guesses, method='BFGS')
        return result


    def generatebackground(self, xrange, ratio=1.0):
        """Adds a flat background to the data
        
        Args:
            xrange:
                numpy array (float, 2 values) of minimum and maximum Q
                values to use in the background generation
            ratio: 
                The background-to-signal ratio
        """

        xmin = xrange[0]
        xmax = xrange[1]

        npts = float(self.data.size)

        nnew = npts * ratio

        print("Adding flat background of", nnew, "points")

        bgdata = np.random.uniform(xmin, xmax, int(nnew))

        self.data = np.append(self.data, bgdata)


    def setupGuesses(self):
        print("ERROR: base class setupGuesses called.")
        raise NotImplementedError()

    def llcurve(self, _params):
        """Returns the sum of log likelihood curve of the function - this
        must be overridden.  It is the maximum (or minimum of the
        negative) of this curve that we are trying to find.  This
        function requires parameter values to be passed so that we can
        calculate numerical differentials.
        
        Args:
            _params:
                numpy array of parameter values (float)
        
        Returns:
            the sum of the log-likelihood of the distribution (float)

        """

        print("ERROR: base class log likelihood function called.")
        raise NotImplementedError()
        return 0.0

    def dllcurve(self, pars=np.array([None, None])):
        """The first differential of the log-likelihood curve with respect to
        each parameter.  We are trying to find the roots of this
        curve.  This base class function performs a numerical
        derivative.  You can (should) override it with an analytical
        form if it exists.

        Args:
            pars:
                numpy array (float) of parameter values

        Returns:
            numpy array (float) with the same dimensions as pars, containing
            the partial derivatives in respective order

        """

        if np.any(pars == None):
            pars = self.estimates
        dydp = np.zeros_like(pars)

        for ii in range(len(dydp)):
            p0 = pars[ii]
            if p0 == 0.0:
                p1 = 0.0001
                p2 = -0.0001
            else:
                p1 = p0 + p0/1000.0
                p2 = p0 - p0/1000.0

            dp1 = p1 - p0
            dp2 = p0 - p2

            parray1 = np.array(pars) # looks like python is a bit like ruby
            parray2 = np.array(pars) # you have to actually duplicate the objects
            parray1[ii] = p1
            parray2[ii] = p2

            yzero = self.llcurve(pars)
            yone = self.llcurve(parray1)
            ytwo = self.llcurve(parray2)

            dy1 = yone-yzero
            dy2 = yzero-ytwo

            dy = 0.5*(dy1 + dy2)
            dp = 0.5*(dp1 + dp2) # probably it must always be that dp1 = dp2, but best be careful

            dydp[ii] = dy/dp

        return dydp


    def ddllcurve(self, pars=np.array([None, None])):
        """The second differential of the log-likelihood curve with respect to
        each parameter.  This is used to direct a newton iteration to
        the root of dllcurve.  This base class function performs a
        numerical derivative, you can (should) override it with an
        analytical expression if it exists.

        Args:
            pars:
                numpy array (float) of parameter values
        
        Returns:
            numpy array (float) with the same dimensions as pars, containing
            the second partial derivatives in respective order

        """

        if np.any(pars == None):
            pars = self.estimates
        dydp = np.zeros_like(pars)

        for ii in range(len(dydp)):
            p0 = pars[ii]
            if p0 == 0.0:
                p1 = 0.0001
                p2 = -0.0001
            else:
                p1 = p0 + p0/1000.0
                p2 = p0 - p0/1000.0

            dp1 = p1 - p0
            dp2 = p0 - p2

            parray1 = np.array(pars) # looks like python is a bit like ruby
            parray2 = np.array(pars) # you have to actually duplicate the objects
            parray1[ii] = p1
            parray2[ii] = p2

            yzero = self.dllcurve(pars)[ii]
            yone = self.dllcurve(parray1)[ii]
            ytwo = self.dllcurve(parray2)[ii]

            dy1 = yone-yzero
            dy2 = yzero-ytwo

            dy = 0.5*(dy1 + dy2)
            dp = 0.5*(dp1 + dp2) # probably dp1 = dp2 always, but best be careful

            dydp[ii] = dy/dp

        return dydp



    def generateTestSamples(self, params, xrng, nsamples, verbose=True):
        """Uses a quantile function to generate Monte-Carlo samples from the
        distribution.  This synthetic data can be used for testing and
        also for statistical quality-of-fit tests such as those
        comparing two distributions without assuming a functional
        form.

        Args:
            params:
                numpy array (float) of the parameter values
            xrng: 
                numpy array (float) of the min and max Q values
            nsamples:
                number of data points to generate
            verbose:
                optional boolean whether to describe activity

        """

        xmin = xrng[0]
        xmax = xrng[1]

        # Check that these minima and maxima are correct, swap them if not
        if xmin > xmax:
            tmp = xmin
            xmin = xmax
            xmax = tmp

        
        pmin = self.cdf(params, xmin)
        pmax = self.cdf(params, xmax)

        uniform = np.random.uniform(pmin, pmax, nsamples)
        self.data = self.Quantile(params, uniform)

        # Some functions (e.g. hard spheres) have unusual features
        # Remove nan and inf values from the array
        self.data = self.data[~np.isnan(self.data)]
        self.data = self.data[~np.isinf(self.data)]

        # Remove spurious points that exceed the x-range
        mask = self.data > xmax
        self.data = self.data[~mask]
        mask = self.data < xmin
        self.data = self.data[~mask]

        
        # We might now have fewer points than were asked for.  Create
        # additional points until we are done
        npass = np.size(self.data)

        ntries = 0

        while npass < nsamples:
            nneeded = nsamples - npass
            uniform = np.random.uniform(pmin, pmax, nneeded)
            extra = self.Quantile(params, uniform)
            extra = extra[~np.isnan(extra)]
            extra = extra[~np.isinf(extra)]
            mask = extra > xmax
            extra = extra[~mask]
            mask = extra < xmin
            extra = extra[~mask]

            nnew = np.size(extra)
            ntries = ntries + 1
            self.data = np.append(self.data, extra)
            npass = np.size(self.data)
            
        
        self.setupGuesses()
        if verbose:
            print("Generated", npass, "samples using parameters", params)


        

    def Quantile(self, params, p):
        """Returns the quantile function.  This base class performs a
        numerical inverse of the cumulative distribution function
        (CDF).  If possible, it should be overridden with an
        analytical expression, but these do not always exist.
        Basically, for a given P value, this finds the root value of x
        by solving: CDF(x) - P = 0, which is done using Newton
        iteration.  The quantile function is used to generate test
        samples, since a uniform distribution of random numbers in the
        range 0-1 can then be mapped to a range of random x values
        following whatever distribution is of interest.

        Args:
            params:
                numpy array (float) containing the parameters to use 
                for the quantile calculation
            p:
                numpy array (float) of probability values (y-values) to
                use to determine the corresponding x-values

        Returns:
            numpy array (float) with the same dimensions as p,
            containing the x, or Q, values associated with each p.

        """

        p = np.array(p, ndmin=1) # probs might be a single value or an array of probabilities

        nprobs= np.size(p)

        dx = 0.0001

        nmax = 15

        # Since this is SANS, we will assume x is pretty close to zero!
        guesses = 1.0E-3

        xx = np.zeros(nprobs)

        for pp in range(nprobs):

            xx[pp] = guesses

            fx = 1.0

            ii = 0

            while(ii < nmax and np.abs(fx) > 1.0E-13): # by trial and error

                #print("x", xx)

                fx = self.cdf(params, xx[pp])-p[pp]
                fxp= self.cdf(params, xx[pp]+dx)-p[pp]

                #print("f", fx)
                #print("ff", fxp)

                dy = fxp-fx
                #dyr= abs(dy / fx)

                #print("dyr", dyr)

                fprime = dy/dx

                # If the gradient is zero, then we reach a flat point and cannot continue
                if fprime == 0.0:
                    break
                
                
                xx[pp] = xx[pp] - fx/fprime

                ii = ii + 1

                # Validation that an inverse was found

            if fx > 1.0E-2:
                print("WARNING: numerical quantile function possibly failed to converge")
                print("   cdf(x)-p = ", fx)

        return xx


    def cdf(self, params, x):
        """Returns the cumulative distribution function of f(x) from -inf to
        x.  This function must be overridden.

        """

        print("WARNING: base class cdf function called.")
        raise NotImplementedError()
        return 0.0

    def sortData(self):
        """Performs a sort of the data, which is needed for some
        quality-of-fit test statistics.

        """

        self.data = np.sort(self.data)
        self.issorted = True

    def adtest(self):
        """Computes the Anderson Darling test statistic for the distribution
        This is work in progress - the cutoff values need to be
        figured out specifically for each distribution and maybe this
        is something that will actually appear in overloaded functions
        instead for that reason.

        """

        nn = float(len(self.data))
        if not self.issorted:
            self.sortData()

        rev = np.flip(self.data)

        # uses python equivalent of ECDF()
        # see
#https://stackoverflow.com/questions/15792552/numpy-scipy-equivalent-of-r-ecdfxx-function
        frac = (2.0*np.arange(1, nn+1) - 1) / nn
        t1 = np.log(self.cdf(self.estimates, self.data))
        t2 = np.log(1.0 - self.cdf(self.estimates, rev))

        SS = np.sum( frac * (t1 + t2) )

        AA2 = -nn - SS
        # Returns A-squared
        return AA2


    def kstest(self):
        """Computes the Kolmogorov-Smirnov test statistic for the curve.  This
        is here for completeness, but there are better ones to use for
        sure.  The CDF appropriate for the test statistic might be
        different from the usual CDF for a particular curve type, so
        this should be overloaded in that case.  For example, SANS
        data does not run from -inf to +inf, rather only positive Q
        starting at Q>0, so the CDF in that case is only the positive
        quadrant.

        """

        nn = float(len(self.data))
        if not self.issorted:
            self.sortData()

        cdfy = self.cdf(self.estimates, self.data)
        ecdfy = np.arange(1, nn+1) / nn

        fig = plt.plot(self.data, cdfy)
        plt.plot(self.data, ecdfy)
        plt.show()
        print("plotted")

        dif = np.absolute(cdfy - ecdfy)
        ks = np.amax(dif)
        return ks


    def verifyMaximum(self):
        """Verifies that the second derivative of the log-likelihood is
        negative, implying that we are at a maximum point.

        Returns:
            boolean value, representing whether this is a maximum point

        """

        grad2 = self.ddllcurve()
        #print(grad2)
        result = True
        if np.any(grad2 >= 0.0):
            result = False
        return result

#    def uncertainty(self):
        """This will calculate the Cramer-Rao bound from the Fisher
        Information.  It has not been fully implemented yet.

        """

        # We have much of this information already, as part of the
        # maximum likelihood estimation process
        #diag = self.ddllcurve() # these are the diagonal elements of the hessian matrix
        #rt = np.sqrt(fisher2)
        #return 1.0/rt


    def plotFit(self, logarithmic=True, nbins=50):
        """Uses matplotlib to plot a fit of the data

        A histogram is constructed of the data.  The integral of the
        data and the probability curve are matched.  Each are then
        plotted together.

        Args:
            logarithmic:
                (boolean, optional) selects between linear and 
                logarithmic axes
            nbins:
                int, number of bins for the histogram of the data

        Returns:
            no value is returned, but the figure itself will appear 
            inline in a jupyter notebook.

        """

        lw = np.amin(self.data)
        hi = np.amax(self.data)

        if np.any(self.estimates == None):
            linecolor = 'red'
            params = self.guesses
        else:
            linecolor = 'black'
            params = self.estimates

        slic = (hi-lw)/(nbins+1)

        #print(self.data)
        #print(hi, lw, slic)
        
        #if(logarithmic ==False):
        hbins = np.arange(lw, hi, slic)
        #else:
        #    hbins=np.logspace(np.log10(lw),np.log10(hi), 50)

        hst = np.histogram(self.data, bins=hbins)
      

        # The histogram is shifted by half a bin width of course, when
        # we want to plot the centre values on the linear error bar
        # plot
        if logarithmic is False:
            rl = np.roll(hbins,-1)
            xvals = 0.5*(hbins + rl)
            xvals = np.delete(xvals,-1)
        else:
            xvals = np.delete(hbins,-1)

        errors=np.sqrt(hst[0])
        yvals=hst[0]

        hnorm = np.sum(yvals)
        if not np.any(params == None):
            fitvals = self.curve(params, xvals)
            fnorm = np.sum(fitvals)
            fitvals = fitvals * hnorm/fnorm

        print(fitvals)
        print(hst[0])
        
            
        with open('temp-histo.npy', 'wb') as fl:
            np.save(fl, xvals)
            np.save(fl, yvals)
            
            
        fig,ax = plt.subplots()
        ax.errorbar(xvals, yvals, errors, fmt='o', mfc='none')

        if not np.any(params == None):
            ax.plot(xvals, fitvals, color=linecolor)
        if logarithmic is True:
            plt.yscale('log')
            plt.xscale('log')
        ax.set_xlabel('Q (Å-1)')
        ax.set_ylabel('Intensity')
        plt.show()








class gaussianCurve(MLECurve):
    """Implements an MLE analysis for a Gaussian (normal) curve.  This is
    provided as a thorough test case and validation of the underlying
    methodology of the toolkit.  It provides full examples of how to
    overload the functions with well known results that can be tested
    and checked.  It is expected to be rather limited in terms of neutron
    scattering analysis.

    For many distribution curve types, analytical expressions might not 
    exist for various parts of the toolkit.
    All three methods are therefore implemented here:
        (1) A purely analytical MLE estimation of parameters
        (2) A semi-numerical solution, where the differential and second 
            differential are both supplied analytically, but the search 
            is done numerically by the base class methods
        (3) A fully numerical solution, where the base class methods are 
            called 

    """

    def init(self, data):
        self.nparams=2

        if len(data) > 0:
            self.setupGuesses()

    def mleAnalytic(self):
        """Analytical MLE of gaussian distribution parameters.

        """

        self.method = "analytically"
        N = len(self.data)
        mu = (1.0/N) * np.sum(self.data)
        sigma2 = (1.0/N) * np.sum((self.data - mu)**2.0)
        sigma = np.sqrt(sigma2)
        self.estimates = np.array([mu, sigma])

    def mle(self):
        """Passes through to numerical MLE in the base class.

        """

        #self.mleAnalytic()
        MLECurve.mle(self)

    def curve(self, params, dat=np.array(None)):
        """The likelihood curve of a gaussian distriubtion.

        Args:
            params:
                numpy array (float) containing mu (float) as the mean of the 
                distribution, and sigma, the std. deviation of the 
                distribution, respectively
            dat:
                (float, optional) numpy array containing the Q points of 
                the detected neutrons.  If this array is not passed, 

        """

        mu = params[0]
        sigma = params[1]
        if np.any(dat == None):
            xx = self.data
        else:
            xx = dat
        normf = (1.0 / (sigma * np.sqrt(2.0*np.pi))) * \
            np.exp( -( ((xx-mu)**2.0)/ (2.0 * sigma**2.0)))
        return normf

    def llcurve(self, params):
        """The log-likelihood of a gaussian distribution.

        Args:
            params:
                A numpy array (float) containing the two parameters mu 
                and sigma

        Returns:
            The sum of the log likelihood (float)

        """

        normf = self.curve(params)
        # this stops warnings trying to do log(0.0):
        lg = np.log(normf, out=np.full_like(normf, 1.0E-30), where= normf!=0)
        return np.sum(lg)

    def dllcurveAnalytic(self, params=np.array([None, None])):
        """The analytical first derivative of the log-likelihood function.
        
        Args:
            params:
                A numpy array (float, optional) that contains the two
                parameters mu and sigma.  If no parameters are given,
                then the maximum likelihood estimates are used instead.

        Returns:
            A numpy array containing the partial derivatives with respect 
            to mu and sigma

        """

        if np.any(params == None):
            params = self.estimates

        mu = params[0]
        sigma = params[1]
        data = self.data

        d1 = np.sum( -2.0 * (mu - data)/sigma**2.0 )
        d2 = np.sum( -1.0/sigma + ((self.data - mu)**2.0) / sigma**3.0 )

        return np.array([d1, d2])

    def ddllcurveAnalytic(self, params=np.array([None, None])):
        """Analytical second derivative of the log likelihood function.

        Args:
            params:
                A numpy array (float, optional) that contains the two
                parameters mu and sigma.  If no parameters are given,
                then the maximum likelihood estimates are used instead.

        Returns:
            A numpy array containing the second partial derivatives with 
            respect to mu and sigma

        """

        if np.any(params == None):
            params = self.estimates

        mu = params[0]
        sigma = params[1]
        data = self.data

        dd1 = np.sum( -len(data)/sigma*2.0 )
        dd2 = np.sum( (1.0/sigma**2.0) - 3.0*((data-mu)**2.0) /sigma**4.0 )


        return np.array([dd1, dd2])

    def setupGuesses(self):
        """Creates initial parameter guesses to begin the numerical solution.

        """
        # This is actually the analytic solution for mu:
        self.guesses = np.array([np.mean(self.data), 0.0])

        # now generate a random sample over sigma and pick the best one as the initial estimate
        mindata = np.amin(self.data)
        maxdata = np.amax(self.data)
        nguess = 100
        sigmas = np.random.uniform(low=mindata/1000.0, high=maxdata, size=nguess)
        guessVals = np.zeros_like(sigmas)

        for ii in range(nguess):
            pars = np.array([self.guesses[0], sigmas[ii]])
            guessVals[ii] = self.llcurve(pars)

        bestEstimate = np.argmax(guessVals)
        self.guesses[1] = sigmas[bestEstimate]

    def cdf(self, params, x):
        """Analytical computation of the cumulative distribution function
        (CDF) of a gaussian distribution.

        Args:
            params:
                A numpy array (float) that contains the two
                parameters mu and sigma.  If no parameters are given,
                then the maximum likelihood estimates are used instead.
            x:
                A numpy array (float) of x values, i.e. Q values 

        Returns:
            A numpy array (float, same dimensions as x) containing a
            CDF value for each passed x value

        """

        mu = params[0]
        sigma= params[1]
        root2 = np.sqrt(2.0)
        cdf = 0.5 * ( 1.0 + erf((x - mu)/(sigma*root2)) )
        return cdf

    def Quantile(self, params, p):
        """Analytical computation of the quantile function (inverse CDF) 
        for a gaussian distribution.
        
        Args:
            params:
                A numpy array (float) that contains the two
                parameters mu and sigma.  If no parameters are given,
                then the maximum likelihood estimates are used instead.
            p:
                A numpy array (float) of p values, i.e. y-values on the 
                CDF curve, for which we are trying to find corresponding
                x-values through computation of the inverse function.

        Returns:
            A numpy array (float) of x values, one for each p,
            containing the result of the inverse mapping invCDF(p) -> x

        """

        mu = params[0]
        sigma = params[1]
        root2 = np.sqrt(2.0)
        qn = mu + sigma * root2 * erfinv(2.0 * p - 1.0)
        return qn


    def report(self):
        """Prints a brief report of the MLE fitting results.

        """

        print("Gaussian curve maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guesses (mu, sigma)")
        print(self.estimates, "solution obtained", self.method)

        if self.verifyMaximum():
            derivStr = "a maximum"
        else:
            derivStr = "not a maximum"

        print("The second derivative indicates that this is", derivStr)
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")









class lorentzianCurve(MLECurve):
    """Immplements an MLE analysis for a Cauchy distribution (Lorentzian).

    The Cauchy distribution has no closed analytical form for the
    estimates, so we call the base class numerical methods for this.

    The first and second derivatives of the log likelihood do exist,
    as do the CDF and Quantile, which makes life easy.

    In addition to overloading various base class methods, the
    following are exemplified:
    
    Methods:
        setupGuesses:
            Intialises the guess values to be used for iteration
        gridECDF:
            Histograms and computes an empirical cumulative 
            distribution function for statistical test metrics

    """

    def init(self, data):
        self.nparams=1

        if len(data) > 0:
            self.setupGuesses()

    def setupGuesses(self):
        """Creates initial guesses of the parameters before numerical MLE
        method is applied.

        """

        self.guesses = np.array([np.mean(self.data)])

        # now generate a random sample over sigma and pick the best one as the initial estimate
        mindata = np.amin(self.data)
        maxdata = np.amax(self.data)
        nguess = 50
        kappas = np.logspace(-4,-1, nguess)
        guessVals = np.zeros_like(kappas)

        for ii in range(nguess):
            pars = np.array([self.guesses[0], kappas[ii]])
            guessVals[ii] = self.llcurve(pars)

        bestEstimate = np.argmax(guessVals)
        self.guesses[0] = kappas[bestEstimate]

    def mle(self):
        """Passes through to the base class to perform numerical MLE.

        """

        MLECurve.mle(self)
        self.calcVariances()

    def curve(self, params, dat=np.array([None])):
        """Returns the basic likelihood curve for a Lorentzian function.

        """

        xx = dat
        
        kappa = params[0]
        if np.any(dat == None):
            xx = self.data

        lorf = (kappa / np.pi) * 1.0 / (xx**2.0 + kappa**2.0)
        return lorf

    def llcurve(self, params):
        """Returns the log-likelihood for a Lorentzian distribution.

        """

        lorf = self.curve(params)
        # this stops warnings trying to do log(0.0)
        lg = np.log(lorf, out=np.full_like(lorf, 1.0E-30), where= lorf!=0 )
        return np.sum(lg)

    def dllcurve(self, params=np.array([None])):
        """The analytical first derivative of the log-likelihood of a
        Lorentzian distribution.

        """

        if np.any(params == None):
            params = self.estimates

        kappa = params[0]
        data2 = self.data**2.0

        grad = np.sum( (1.0/kappa) - 2.0 * kappa / (kappa**2.0 + data2))
        return np.array([grad])

    def ddllcurve(self, params=np.array([None])):
        """The analytical second derivative of the log-likelihood of a
        Lorentzian distribution.

        """

        if np.any(params == None):
            params = self.estimates

        kappa = params[0]
        data2 = self.data**2.0

        grad = np.sum( -(1.0 / kappa**2.0) +\
                       (4.0*kappa**2.0 / (kappa**2.0 + data2)**2.0) - 2.0/(kappa**2.0 + data2))

        return np.array([grad])

    def cdf(self, params, x):

        """ksdlk sdlkds ds

        """

        kappa = params[0]
        cdf = (1.0/np.pi) * np.arctan( x / kappa ) + 0.5
        return cdf

    def halfcdf(self, params, x):
        # This is the CDF for half curve assuming the centre is at x=0
        kappa = params[0]
        cdf = 0.5 + np.arctan( x / kappa ) / np.pi
        return cdf


    def Quantile(self, params, p):
        kappa = params[0]
        qn = kappa * np.tan(np.pi * (p - 0.5))
        return qn

    def nQuantile(self, params, p):
        cdf = super().Quantile(params, p)
        return cdf

    def gridEcdf(self, dat):
        # Calculates the empirical CDF on a grid of 100 points
        xmin = np.amin(dat)
        xmax = np.amax(dat)

        ns = 100

        stp = (xmax - xmin)/float(ns)


        nn = float(len(dat))
        xs = np.arange(xmin, xmax, stp)

        print ("siz", xs.size, xmin, xmax, stp, (xmax-xmin)/stp)

        ecdfy = np.zeros_like(xs)

        for i in range(ns):
            ecdfy[i] = np.sum( dat < xs[i] ) / nn

        return (xs, ecdfy)





    def kstest(self):
        # Overloading the base class ks-test and doing a numerical KS test with two eCDFs
        nn = float(len(self.data))
        nni = int(nn)
        if self.issorted is False:
            self.sortData()

        minx = np.amin(self.data)
        maxx = np.amax(self.data)

        pmin = self.cdf(self.estimates, minx)
        pmax = self.cdf(self.estimates, maxx)

        uniform = np.random.uniform(pmin, pmax, nni)
        synth = self.Quantile(self.estimates, uniform)

        synth = np.sort(synth)

        ecdfx, ecdfy = self.gridEcdf(self.data) # self.CDF(self.estimates, self.data)
        #ecdfy = np.arange(1, nn+1) / nn
        scdfx, scdfy = self.gridEcdf(synth)

        dif = np.absolute(scdfy - ecdfy)
        ks = np.amax(dif)
        return ks

    def report(self):
        print("Lorentzian curve maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guess (kappa)")

        
        if not np.any(self.variances) == None :
            errstr = " +/- " + np.array2string(self.variances[0]) + "?"
        else:
            errstr = ""
            
        print(self.estimates, errstr, "solution obtained", self.method)

        print(self.estimates, "solution obtained", self.method)
        print("That a maximum was found is", self.verifyMaximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")


    def calcVariances(self, crit = 1.0):
        '''
        Calculates the variance of the kappa parameter.
        For a single parameter curve, as is the case for SANS, this is
        a simple computation of the inverse of the fisher information.
        This is the Cramer-Rao bound, the lower bound on the variance of 
        any unbiased estimator of theta.  It is a 'happy conincidence' that
        the inverse of the fisher information is the same size as the 
        sigma of a gaussian distribution of errors, but this comes from the central
        limit theorem basically, so it all makes sense if you accept the
        central limit theorem.

        The fisher information here is the "observed fisher information"
        which I have shamelessly stolen from equation 2.10 in University of
        Minnesota "Stat 5102 Notes: Fisher Information and Confidence Intervals
        Using Maximum Likelihood" by Charles J. Geyer, March 2007.

        This function is called automatically at the end of mle(), you don't need to call it.

        TODO:
            There are not yet any checks as to whether mle() was able to find a 
            sensible result.  This means that calcVariances() might be unstable 
            and some sanity checks are probably needed before deployment.

        Parameters:
            crit:
                (float, optional) value to scale the sigma by.  1-sigma
                is the usual error bar size for gaussians.  crit=1.96 would
                correspond to 2-sigma, 95% confidence range, etc...
        
        Returns:
            Nothing.  But it does store the results in self.variances np.array 
            for use elsewhere.  If not np.any(self.variances) == None, etc etc
        
        '''
        
        #crit = 1.96 (95% or roughly 2 sigma, is also a common option)
        
        secondDiff = self.ddllcurve(self.estimates)
        fisherI = -secondDiff/self.data.size
        
        variance = 1.0 / fisherI
        self.variances = np.array([variance])
        
        

    def infer(self, plot=False):

        # Bayesian inference

        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.data), np.amax(self.data) ])
        
        logmean = np.mean(np.log10( xrange)) 
        loghw = 0.5*np.std(np.log10(xrange))
        rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/rrange
        rrange = np.round(rrange)

        #print(rrange)

        rvals = np.arange(rrange[1], rrange[0], 1.0)

        revr = np.flip(rvals)

        #print(rvals)

        kappas = 1.0 / revr

        # Now setup prior

        prior = np.full_like(kappas, 1.0/kappas.size)#0.01)
        posterior = np.full_like(kappas, 1.0/kappas.size)#0.01)
        
        #print(kappas)

        kappas2 = kappas**2.0


        ##### Below, the following procedure is done in logspace to simplify, but here it is in
        # linear space so it's easier to see the mapping to classic bayesian inference
        # as described on wikipedia

        #for neutron in np.arange(0, self.data.size, 1):
        #    likelihoods = ((kappas / np.pi) / (self.data[neutron]**2.0 + kappas2))
        #    sml = np.sum(likelihoods * prior)
        #    posterior = prior * likelihoods / sml
        #    prior = np.copy(posterior)
        


        # To simplify, we will flip into logspace, so that combining probabilities is then a sum
        # Instead of normalising every step, we'll simply normalise everything at the very end
        prior = np.log10(prior)
        posterior = np.log10(posterior)
        
        for neutron in np.arange(0, self.data.size, 1):
            posterior = prior + np.log10((kappas / np.pi) / (self.data[neutron]**2.0 + kappas2))
            prior = np.copy(posterior) # this is copied over for the next iteration

        
        # Shift the weight distribution down to sensible values around unity, and normalise
        posterior = posterior - np.amax(posterior)
        posterior = 10.0**posterior

        #print(posterior)
        # normalise the curve (assumes the whole curve has been sampled)
        total = np.sum(posterior)
        posterior = posterior / total
            
        centre =  np.sum( posterior * kappas) # mean is the weighted sum, gaussian according to central limits

        # Likewise for the standard deviation
        diffs = (kappas - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(kappas, posterior)
            ax.set_xlabel('Parameter value')
            ax.set_ylabel('Inferred probability')

        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates





        

class lorentzianSquaredCurve(MLECurve):

    def cdf(self, params, x):
        kappa = params[0]
        ss    = params[1]

        x2 = x**2.0

        pi = np.pi
        pi2= pi**2.0
        kappa2 = kappa*2.0

        t1 = 0.5
        t2 = ss * x * kappa / (pi * (2.0 + ss) * (x2 + kappa2))
        t3 = np.arctan(x/kappa)/pi

        cdf = t1 + t2 + t3
        return cdf

    def curve(self, params, dat=np.array(None)):
        kappa = np.abs(params[0])
        ss = np.abs(params[1])

        k2 = kappa**2.0
        k3 = kappa**3.0

        pi = np.pi


        if np.any(dat == None):
            xx = self.data
        else:
            xx = dat

        x2 = xx**2.0

        t1 = 2.0 * ss * k3 / (pi * (2.0 + ss) * (x2 + k2)**2.0)
        t2 = 2.0 * kappa / (pi * (2.0 + ss) * (x2 + k2))
        return t1 + t2

    def llcurve(self, params):
        crv = self.curve(params)
        lg = np.log(crv, out=np.full_like(crv, 1.0E-30), where= crv!=0 )
        return np.sum(lg)

    def setupGuesses(self):
        self.guesses = np.array([0.0, 0.0])
        self.guesses[0] = 1.0/70.0
        self.guesses[1] = 1.0


    def report(self):
        print("Lorentzian Squared maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guesses (kappa, S)")
        print(self.estimates, "solution obtained", self.method)
        print("R =", 1.0/self.estimates[0])
        print("That a maximum was found is", self.verifyMaximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")









class hardSphereCurve(MLECurve):
    def init(self, data):
        self.nparams=1

        if len(data) > 0:
            self.setupGuesses()
    
    def curve(self, params, dat=np.array(None)):
        # Return the basic likelihood curve
        R = params[0]

        if np.any(dat == None):
            xx = self.data
        else:
            xx = dat

        xr = xx * R
        xr6= xr ** 6.0
            
        hrd = 15.0 * R * ( np.sin(xr) - xr * np.cos(xr))**2.0  / (2.0 * np.pi * xr6)
        return hrd

    def llcurve(self, params):
        # Return the sum of the log likelihood for the curve shape
        hrd = self.curve(params)
        # this stops warnings trying to do log(0.0)
        lg = np.log(hrd, out=np.full_like(hrd, 1.0E-30), where= hrd!=0)
        return np.sum(lg)

    def cdf(self, params, x):
        R = params[0]

        xr = x * R
        xr2 = xr ** 2.0
        xr4 = xr2 ** 2.0
        xr5 = xr ** 5.0
        
        cdft = (-3.0 - 5.0 * xr2 + 
            2.0 * np.pi * xr5 + (3.0 - xr2 + 2.0 * xr4) * np.cos(2.0 * xr) + 
            xr * (6.0 + xr2) *  np.sin(2.0 * xr) + 
            4.0 * xr5 * sinintegral(2.0 * xr))
        cdfb = (4.0 * np.pi * xr5)
        return cdft / cdfb

    
    def setupGuesses(self):
        self.guesses = np.array([90.0])



    
    def infer(self, plot=False):

        # Bayesian inference

        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.data), np.amax(self.data) ])
        
        logmean = np.mean(np.log10( xrange)) 
        loghw = 0.5*np.std(np.log10(xrange))
        rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/rrange
        rrange = np.round(rrange)

        #print(rrange)

        rvals = np.arange(rrange[1], rrange[0], 1.0)

        revr = np.flip(rvals)

        # Now setup prior

        prior = np.full_like(rvals, 1.0/rvals.size)#0.01)
        posterior = np.full_like(rvals, 1.0/rvals.size)#0.01)
        
        # flip into logspace - combining probabilities is then a sum
        prior = np.log10(prior)
        posterior = np.log10(posterior)
        
        for neutron in np.arange(0, self.data.size, 1):
            xr = self.data[neutron] * rvals
            xr6= self.data[neutron] ** 6.0
            
            hrd = 15.0 * rvals * ( np.sin(xr) - xr * np.cos(xr))**2.0  / (2.0 * np.pi * xr6)
            posterior = prior + np.log10(hrd)
            prior = np.copy(posterior)

        # Shift the weight distribution down to sensible values and normalise
        posterior = posterior - np.amax(posterior)
        posterior = 10.0**posterior

        # normalise the curve (assumes the whole curve has been sampled)
        total = np.sum(posterior)
        posterior = posterior / total
            
        centre =  np.sum( posterior * kappas) # mean is the weighted sum, gaussian according to central limits

        # Likewise for the standard deviation
        diffs = (kappas - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(kappas, posterior)
            ax.set_xlabel('Parameter value')
            ax.set_ylabel('Inferred probability')

        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates




class guinierCurve(MLECurve):
    def init(self, data):
        self.nparams=1

        if len(data) > 0:
            self.setupGuesses()
    
    
    def report(self):
        """Prints a brief report of the MLE fitting results.

        """

        print("Guinier model bayesian inference")
        print(np.size(self.data), "data points")


    
    def infer(self, plot=False):

        # Bayesian inference

        # First figure out range of R values
        xrange = np.array( [ np.amin(self.data), np.amax(self.data) ])
        
        logmean = np.mean(np.log10( xrange)) 
        loghw = 0.5*np.std(np.log10(xrange))
        rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/rrange
        rrange = np.round(rrange)

        #print(rrange)

        rvals = np.arange(rrange[1], rrange[0], 1.0)

        revr = np.flip(rvals)

        # Now setup prior

        prior = np.full_like(rvals, 1.0/rvals.size)#0.01)
        posterior = np.full_like(rvals, 1.0/rvals.size)#0.01)
        
        # flip into logspace - combining probabilities is then a sum
        prior = np.log10(prior)
        posterior = np.log10(posterior)
        
        for neutron in np.arange(0, self.data.size, 1):
            xr = self.data[neutron] * rvals
            xr2= self.data[neutron] ** 2.0
            gnr= np.exp(-xr2 / 3.0)
            posterior = prior + np.log10(gnr)
            prior = np.copy(posterior)

        # Shift the weight distribution down to sensible values and normalise
        posterior = posterior - np.amax(posterior)
        posterior = 10.0**posterior

        # normalise the curve (assumes the whole curve has been sampled)
        total = np.sum(posterior)
        posterior = posterior / total
            
        centre =  np.sum( posterior * rvals) # mean is the weighted sum, gaussian according to central limits

        # Likewise for the standard deviation
        diffs = (rvals - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(rvals, posterior)
            ax.set_xlabel('Radius of Gyration (R, Angstroms)')
            ax.set_ylabel('P(R)')

        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates

  


        



class porodCurve(MLECurve):
    def init(self, data):
        self.nparams=2

        if len(data) > 0:
            self.setupGuesses()

    def mle(self):
        """Analytical MLE of gaussian distribution parameters, using 
           powerlaw package.

           The exponent is calculated analytically, whilst the Qmin
           parameter must be optimised numerically.
        
           See https://arxiv.org/pdf/0706.1062.pdf
           and https://pypi.org/project/powerlaw/

        """

        self.method = "analytically"

        results = powerlaw.Fit(self.data)
        alpha = results.power_law.alpha
        xmin = results.power_law.xmin
        
        self.estimates = np.array([alpha, xmin])
    
    def curve(self, params, dat=np.array(None)):
        # Return the basic likelihood curve
        alpha = params[0]
        qmin = params[1]

        if np.any(dat == None):
            xx = self.data
        else:
            xx = dat
            
        #prd = (xx/qmin)**(-alpha) * (alpha - 1.0) / qmin
        prd = (xx/qmin)**(-alpha) * (alpha - 1.0) / qmin
        
        return prd

    def llcurve(self, params):
        # Return the sum of the log likelihood for the curve shape
        prd = self.curve(params)
        # this stops warnings trying to do log(0.0)
        lg = np.log(hrd, out=np.full_like(hrd, 1.0E-30), where= hrd!=0)
        return np.sum(lg)

    def cdf(self, params, xx):
        alpha = params[0]
        qmin = params[1]
        
        bkt = qmin / xx

        pcdf = 1.0 - bkt**(alpha-1.0)
        
        return pcdf

    def quantile(self, params, p):
        alpha = params[0]
        qmin = params[1]

        qtl = qmin * (1.0 - p)**(-1.0 / (alpha-1.0))
        
        return qtl

    
    def setupGuesses(self):
        self.guesses = np.array([4.0, 0.01])

    
    def report(self):
        """Prints a brief report of the MLE fitting results.

        """

        print("Generalised Porod curve maximum likelihood estimation")
        print(np.size(self.data), "data points")
        print(self.guesses, "as initial guesses (z, qmin)")
        print(self.estimates, "solution obtained", self.method)

        #if self.verifyMaximum():
        #    derivStr = "a maximum"
        #else:
        #    derivStr = "not a maximum"

        #print("The second derivative indicates that this is", derivStr)
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")









# TO DO:

#  P value tests
#  Goodness of fit metrics - "Is the fit good, or valid?"
#  Bayesian information criterion - "Which is the best model that fits my data"
