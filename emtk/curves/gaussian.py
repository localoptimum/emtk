from . import curve as base

#import emtk.omega as omega

import matplotlib.pyplot as plt

import numpy as np

#import powerlaw

#import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.special import erfinv
#from scipy.optimize import minimize
#from scipy.special import sici

#from scipy.stats import gaussian_kde

#def sinintegral(theta):
#    si, ci = sici(theta)
#    return si











class GaussianCurve(base.Curve):
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
            self.setup_guesses()

    def mleAnalytic(self):
        """Analytical MLE of gaussian distribution parameters.

        """

        self.method = "analytically"
        N = len(self.data)
        mu = (1.0/N) * np.sum(self.data)
        sigma2 = (1.0/N) * np.sum((self.data - mu)**2.0)
        sigma = np.sqrt(sigma2)
        self.estimates = np.array([mu, sigma])

    def mle(self, verbose=False):
        """Passes through to numerical MLE in the base class.

        """

        #self.mleAnalytic()
        base.Curve.mle(self, verbose)

    def curve(self, params, dat=None):
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

        dat = np.asarray(dat)
        
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

    
    # To activate the following function for testing it,
    # rename it to d_llcurve() and it will be called by the
    # base class instead of the numerical method
    def d_llcurve_analytic(self, params=None):
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

        params = np.asarray(params)

        if np.any(params == None):
            params = self.estimates

        mu = params[0]
        sigma = params[1]
        data = self.data

        d1 = np.sum( -2.0 * (mu - data)/sigma**2.0 )
        d2 = np.sum( -1.0/sigma + ((self.data - mu)**2.0) / sigma**3.0 )

        return np.array([d1, d2])

    # To activate the following function for testing it,
    # rename it to dd_llcurve() and it will be called by the
    # base class instead of the numerical method
    def dd_llcurve_analytic(self, params=None):
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

        params = np.asarray(params)

        if np.any(params == None):
            params = self.estimates

        mu = params[0]
        sigma = params[1]
        data = self.data

        dd1 = np.sum( -len(data)/sigma*2.0 )
        dd2 = np.sum( (1.0/sigma**2.0) - 3.0*((data-mu)**2.0) /sigma**4.0 )


        return np.array([dd1, dd2])


    
    def setup_guesses(self):
        """Creates initial parameter guesses to begin the numerical solution.

        """
        # This is actually the analytic solution for mu:
        #self.guesses = np.array([np.mean(self.data), 0.0])

        # Lets not do that, instead lets just take something in the middle
        muguess = 0.5 * (np.amax(self.data) + np.amin(self.data))

        self.guesses = np.array([muguess, 0.25 * muguess])

        # now generate a random sample over sigma and pick the best one as the initial estimate
        #mindata = np.amin(self.data)
        #maxdata = np.amax(self.data)
        #nguess = 100
        #sigmas = np.random.uniform(low=mindata/1000.0, high=maxdata, size=nguess)
        #guess_vals = np.zeros_like(sigmas)

        #for ii in range(nguess):
        #    pars = np.array([self.guesses[0], sigmas[ii]])
        #    guess_vals[ii] = self.llcurve(pars)

        #best_estimate = np.argmax(guess_vals)
        #self.guesses[1] = sigmas[best_estimate]


    def infer(self, mu_range=None, sigma_range=None, plot=True):

        # Bayesian inference
        mu_range = np.asarray(mu_range)
        sigma_range = np.asarray(sigma_range)

        # First, figure out range of mu values
        if (mu_range == None).any():
            dmax = np.amax(self.data)
            dmin = np.amin(self.data)
        else:
            dmax = np.amax(mu_range)
            dmin = np.amin(mu_range)
        
        span = dmax - dmin
        mu_step = span/100.0
        mus = np.arange(dmin, dmax, mu_step)


        # And now sigma values
        if (sigma_range == None).any():
            sigma_min = span/20.0
            sigma_max = span/2.0
        else:
            sigma_min = np.amin(sigma_range)
            sigma_max = np.amax(sigma_range)
            
        sigma_step = (sigma_max - sigma_min)/100.0

        sigmas = np.arange(sigma_min, sigma_max, sigma_step)        

        # numpy meshspace creates two parallel array spaces
        # whilst mathematica would interleave this and you'd have to
        # index them separately.
        # The first indexed value increases horizontally, the other
        # increases vertically
        mu, sigma = np.meshgrid(mus, sigmas)

        
        prior_value = 1.0E-04
        
        prior = np.full_like(mu, prior_value)
        posterior = np.full_like(mu, prior_value)

        # Flip into logspace

        prior = np.log10(prior)
        posterior = np.log10(posterior)


        sigma_squared = sigma ** 2.0
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        # Infer
        
        for neutron in np.arange(0, self.data.size, 1):
            posterior = prior + np.log10( \
                norm * np.exp( -( ((self.data[neutron]-mu)**2.0) / (2.0 * sigma_squared))) \
                                         )
            prior = np.copy(posterior)

        # Shift the log likelihood to sensible values around zero
        maxp = np.amax(posterior)
        posterior = posterior - maxp
        
        # Flip out of logspace and normalise
        posterior = 10.0**posterior

        #total = np.sum(posterior)
        #posterior = posterior / total
        
        # 2D likelihood plot maybe
        
        if plot:
            fig, ax = plt.subplots()
            plt.contourf(mu, sigma, posterior)
            plt.axis('scaled')
            cbar = plt.colorbar()
            cbar.set_label('Relative likelihood')
            ax.set_xlabel('mu')
            ax.set_ylabel('sigma')
            plt.show



            
        
        
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

    def quantile(self, params, p):
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

        if self.verify_maximum():
            derivStr = "a maximum"
        else:
            derivStr = "not a maximum"

        print("The second derivative indicates that this is", derivStr)
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")






