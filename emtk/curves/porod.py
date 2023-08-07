"""Porod scattering arises from flat surfaces, either planes or
fairly regular particle surfaces.  This gives rise to a scattering of the type

         A
I(Q) = -----
        Q^4

This code here includes the possibility of fractal Porod behaviour,
where the surface exhibits self-similarity over the probed length
scales and this modifies the dimensionality so that the curve now
follows

         A
I(Q) = -----
        Q^z

where non-integer values of z are related to the fractal
dimensionality.

Power laws usually don't actually exist in practice, only within an
asymptotic region of parameter space.  The "true" curve is usually
something like a log-normal or gamma distribution, but if you are in
the power law regime you cannot know what that might be.  The second
problem is that least squares fitting is notoriously bad at
determining the parameters of a power law.  The third problem is that
a power law diverges to infinity at Q=0.  The solution to all three of
these problems is to use maximum likelihood estimation to determine
qmin and z.  There is an iterative process of choosing a qmin,
estimating z, and then updating our qmin estimate as appropriate.

There are rigorous studies of this mathematical problem, a good one to
read is Clauset, Shalizi and Newman in SIAM REVIEW 2009 (doi:
10.1137/070710111 )

The wheel is not reinvented here.  The code below uses the existing
power law package, which is a python implementation very similar ot
the R package of the same name.  These determine (iteratively) qmin
and z.  Because it iterates over qmin and z it can be a bit slow, but
it's the best way to estimate the parameters.

"""


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








