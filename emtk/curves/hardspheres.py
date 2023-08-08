"""The hard sphere curve is a classic neutron scattering curve that
has been used countless times for both research on proteins and nano
technology substances, as well as calibration of instrumentation.

The curve follows the function:


        15 * (sin(Q*R) - Q*R*cos(Q*R))^2
I(Q) = ----------------------------------
                 2 * pi * (Q*R)^6

We could safely ignore the 15 / 2 pi normalisation factor when dealing with
the log likelihoods but it's left in here.



"""

from . import curve as base

import numpy as np

from scipy.special import sici




def sin_integral(theta):
    si, ci = sici(theta)
    return si




class HardSpheres(base.Curve):
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
            4.0 * xr5 * sin_integral(2.0 * xr))
        cdfb = (4.0 * np.pi * xr5)
        return cdft / cdfb

    
    def setup_guesses(self):
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


    def report(self):
        """Prints a brief report of the MLE fitting results.

        """

        print("Hard sphere curve maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guess")
        print(self.estimates, "solution obtained", self.method)

        if self.verify_maximum():
            deriv_str = "a maximum"
        else:
            deriv_str = "not a maximum"

        print("The second derivative indicates that this is", deriv_str)
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")
    
