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

The typical usage example below generates synthetic data for SANS with
a 10% background level (which is huge) and obtains a maximum
likelihood estimate from that data.


import emtk.curves.lorentzian as lor
import numpy as np
import matplotlib.pyplot as plt

clength = 90.0 # correlation length in system

kappa = 1.0 / clength
curv = lor.LorentzianCurve()
pvalues = np.array([kappa])
xrange = np.array([0.001, 0.1])
curv.generateTestSamples(pvalues, xrange, 2000)
curv.generatebackground(xrange, ratio=0.1) # method is resistant to a 10% background, which is pretty big
curv.mle()
curv.report()
curv.plotFit(logarithmic=True)

"""

import numpy as np

from emtk.omega.omegaFunctions import kernel_density

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



class Curve:
    """Base class for maximum likelihood estimation.  

    Attributes: 
        data : a numpy array (float) of Q values of detected neutrons 
        estimates : a numpy array (float) of estimated parameter values 
        guesses : the initial parameter estimates to use for iterating 
                  the estimates (only needed for non-analytic methods)

    Methods: 
        mle : 
            performs maximum likelihood estimation
        generate_background : 
            create a flat, random background on the data
        llcurve : 
            the log-likelihood function of the distribution/
        d_llcurve : 
            the first derivative of the log-likelihood function with
            respect to each parameter
        dd_llcurve : 
            the second derivative of the log-likelihood function 
            with respect to each parameter
        generate_test_samples : 
            creates synthetic data that follows the probability 
            distribution
        quantile : the quantile function of the distribution
        cdf : the cumulative distribution function
        sort_data : sorts the data in numerical order.  This is needed 
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
            yzero = self.d_llcurve(self.estimates) # the derivative (to solve)
            yprime = self.dd_llcurve(self.estimates) # the second differential used to find the roots
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


    def generate_background(self, xrange, ratio=1.0):
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


    def setup_guesses(self):
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

    def d_llcurve(self, pars=np.array([None, None])):
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


    def dd_llcurve(self, pars=np.array([None, None])):
        """The second differential of the log-likelihood curve with respect to
        each parameter.  This is used to direct a newton iteration to
        the root of d_llcurve.  This base class function performs a
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

            yzero = self.d_llcurve(pars)[ii]
            yone = self.d_llcurve(parray1)[ii]
            ytwo = self.d_llcurve(parray2)[ii]

            dy1 = yone-yzero
            dy2 = yzero-ytwo

            dy = 0.5*(dy1 + dy2)
            dp = 0.5*(dp1 + dp2) # probably dp1 = dp2 always, but best be careful

            dydp[ii] = dy/dp

        return dydp



    def generate_test_samples(self, params, xrng, nsamples, verbose=True):
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
        self.data = self.quantile(params, uniform)

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
            extra = self.quantile(params, uniform)
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
            
        
        self.setup_guesses()
        if verbose:
            print("Generated", npass, "samples using parameters", params)


        

    def quantile(self, params, p):
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

    def sort_data(self):
        """Performs a sort of the data, which is needed for some
        quality-of-fit test statistics.

        """

        self.data = np.sort(self.data)
        self.issorted = True

    def ad_test(self):
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


    def ks_test(self):
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


    def verify_maximum(self):
        """Verifies that the second derivative of the log-likelihood is
        negative, implying that we are at a maximum point.

        Returns:
            boolean value, representing whether this is a maximum point

        """

        grad2 = self.dd_llcurve()
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
        #diag = self.dd_llcurve() # these are the diagonal elements of the hessian matrix
        #rt = np.sqrt(fisher2)
        #return 1.0/rt


    def plot_fit(self, logarithmic=True, nbins=50):
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

        #print(fitvals)
        #print(hst[0])
        
            
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



