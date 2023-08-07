
from . import curve as base

import numpy as np

import powerlaw

class PorodCurve(base.Curve):
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
