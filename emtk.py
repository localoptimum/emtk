import numpy as np

import matplotlib.pyplot as plt 

from scipy.special import erf
from scipy.special import erfinv
from scipy.optimize import minimize
from scipy.special import sici

class MLECurve:
    def __init__(self):
        self.nparams=0
        self.estimates = np.empty(0)
        self.guesses = np.empty(0)
        self.data = np.empty(0)
        self.hessian = np.empty(0)
        self.method = "only from guesses"
        self.forceNumeric = False
        self.isSorted = False
        
    def mle(self):
        # The base class does a simple numerical approach with Newton iteration
        # Derived classes can overload the numerical functions with analytical ones
        # for greater accuracy and stability under more extreme conditions (e.g. small data sets)
        
        self.estimates = self.guesses
        run = True
        nsteps=0
        
        #print("Numerical MLE")
        #print(self.estimates, "as starting estimates")
        
        self.method = "numerically"

        while(run):
            y0 = self.dllcurve(self.estimates) # the curve for which we're trying to find the roots (the differential)
            yprime = self.ddllcurve(self.estimates) # the second differential used to find the roots
            newvals = self.estimates - y0 / yprime # simple Newton iteration
            fracChange = (newvals - self.estimates)#/self.estimates
            self.estimates = newvals
            nsteps = nsteps + 1
            #print(self.estimates, y0, yprime)
            if(nsteps > 50 or fracChange.all() < 1.0E-06):
                run=False


    # Can use Scipy.optimize.minimize if you want
    # This requires a NEGATIVE log likelihood to minimise
    def negLL(self, params):
        return( -self.llcurve(params))

    def generateBackground(self, xrange, ratio=1.0):
        xmin = xrange[0]
        xmax = xrange[1]

        nn = float(self.data.size)

        nnew = nn * ratio
        
        print("Adding flat background of", nnew, "points")

        bgdata = np.random.uniform(xmin, xmax, int(nnew))

        self.data = np.append(self.data, bgdata)
        
    
    # Then just call minimize on that function using the initial guesses
    # a solid gradient descent mode is probably fine for what we need
    def mleScipy(self):
        result = minimize(self.negLL, self.guesses, method='BFGS')
        return(result)
        
    def llcurve(self, params):
        # Returns the sum of log likelihood curve of the function - this must be overridden
        # It is the maximum (or minimum of the negative) of this curve that we are trying to find
        # This function requires parameter values to be passed so that we can calculate
        # numerical differentials
        print("ERROR: base class log likelihood function called.")
        raise NotImplementedError()
        return(None)
    
    def dllcurve(self, pars=np.array([None, None])):
        # Returns the differential of the llcurve with respect to each parameter
        # Only uses the current estimate values so no parameters to pass
        # We are trying to find the root of this curve where it equals zero
        if(np.any(pars == None)):
            pars = self.estimates 
        dydp = np.zeros_like(pars)
        
        for ii in range(len(dydp)):
            p0 = pars[ii]
            if(p0 == 0.0):
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
            
            y0 = self.llcurve(pars)
            y1 = self.llcurve(parray1)
            y2 = self.llcurve(parray2)
                        
            dy1 = y1-y0
            dy2 = y0-y2
            
            dy = 0.5*(dy1 + dy2)
            dp = 0.5*(dp1 + dp2) # probably it must always be that dp1 = dp2, but best be careful
            
            dydp[ii] = dy/dp            
        
        return(dydp)
            
        
    def ddllcurve(self, pars=np.array([None, None])):
        # Returns the differential of dllcurve with respect to each parameter
        # This is used to direct a newton iteration to the root of dllcurve
        if(np.any(pars == None)):
            pars = self.estimates 
        dydp = np.zeros_like(pars)
        
        for ii in range(len(dydp)):
            p0 = pars[ii]
            if(p0 == 0.0):
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
            
            y0 = self.dllcurve(pars)[ii]
            y1 = self.dllcurve(parray1)[ii]
            y2 = self.dllcurve(parray2)[ii]
            
            dy1 = y1-y0
            dy2 = y0-y2
            
            dy = 0.5*(dy1 + dy2)
            dp = 0.5*(dp1 + dp2) # probably dp1 = dp2 always, but best be careful
            
            dydp[ii] = dy/dp            
        
        return(dydp)
    
    def testSamples(self, params, xrange, nsamples):
        # Returns a numpy array of random samples from the curve.  
        # Must be overridden
        print("ERROR: attempt to generate samples from base class.")
        raise NotImplementedError()
        return(None)

    def Quantile(self, params, p):
        # If not overridden, this function will numerically solve for x: CDF(x) - P = 0
        # so as to determine the value x that corresponds to CDF(x) = P
        # using newton iteration
        # this could obviously be a bit slower than an analytic expression
        # so it's always best to provide a quantile function if you can

        ii = 0

        nmax = 20

        dx = 0.0001

        dyr = 1.0

        # Since this is SANS, we will assume x is close to zero!
        guesses = np.random.uniform(1.0E-4, 0.1, size=20)

        fs = self.CDF(params, guesses)-p

        opt = np.argmin(fs)

        xx = guesses[opt]

        fx = 1.0
        

        while(ii < nmax and fx > 1.0E-13): # This cutoff was just found by trial and error

            #print("x", xx)
            
            fx = self.CDF(params, xx)-p
            fxp= self.CDF(params, xx+dx)-p

            #print("f", fx)
            #print("ff", fxp)

            dy = fxp-fx
            dyr= abs(dy / fx)

            #print("dyr", dyr)
            
            fprime = dy/dx

            xx = xx - fx/fprime

            ii = ii + 1

        # Validation that an inverse was found

        if (fx > 1.0E-6):
            print("WARNING: numerical quantile function possibly failed to converge")
            print("   CDF(x)-p = ", fx)
        
        return(xx)

    
    def CDF(self, params, x):
        # Returns cumulative distribution function
        # Must be overridden
        print("WARNING: base class CDF function called.")
        raise NotImplementedError()
        return(None)
    
    def sortData(self):
        self.data = np.sort(self.data)
        self.isSorted = True
        
    def adtest(self):
        # Computes Anderson Darling test for the distribution
        # This is work in progress - the cutoff values need to be figured out specifically for each distribution
        # and maybe this is something that will actually appear in overloaded functions instead for that reason
        nn = float(len(self.data))
        if(self.isSorted == False):
            self.sortData()

        rev = np.flip(self.data)

        # uses python equivalent of ECDF()
        # see https://stackoverflow.com/questions/15792552/numpy-scipy-equivalent-of-r-ecdfxx-function
        frac = (2.0*np.arange(1, nn+1) - 1) / nn
        t1 = np.log(self.CDF(self.estimates, self.data))
        t2 = np.log(1.0 - self.CDF(self.estimates, rev))
                
        SS = np.sum( frac * (t1 + t2) )
        
        AA2 = -nn - SS
        # Returns A-squared
        return(AA2)

    
    def kstest(self):
        # Computes the Kolmogorov-Smirnov test statistic for the curve
        # This is here for completeness, but there are better ones to use for sure
        # The CDF might be different from the regular CDF for a particular curve type, so this should be
        # overloaded in that case for any given curve type
        nn = float(len(self.data))
        if(self.isSorted == False):
            self.sortData()

        cdfy = self.CDF(self.estimates, self.data)
        ecdfy = np.arange(1, nn+1) / nn

        fig = plt.plot(self.data, cdfy)
        plt.plot(self.data, ecdfy)
        plt.show()
        print("plotted")

        dif = np.absolute(cdfy - ecdfy)
        ks = np.amax(dif)
        return(ks)
    
    
    def verifyMaximum(self):
        # Verify that the solution is a maximum: the second derivative should be negative
        grad2 = self.ddllcurve()
        #print(grad2)
        result = True
        if(np.any(grad2 >= 0.0)):
            result = False
        return(result)
    
    def uncertainty(self):
        # Calculates the Cramer-Rao bound from the Fisher Information
        # We have all of it already as part of the maximum likelihood estimation process
        diag = self.ddllcurve() # these are the diagonal elements of the hessian matrix
        
        
        rt = np.sqrt(fisher2)
        return(1.0/rt)
    
    def plotFit(self, logarithmic=True, nbins=50):
        # Generates a matplotlib graph of the fit
        lw = np.amin(self.data)        
        hi = np.amax(self.data)
            
        slic = (hi-lw)/(nbins+1)
        
        #if(logarithmic ==False):
        hbins = np.arange(lw, hi, slic) 
        #else:
        #    hbins=np.logspace(np.log10(lw),np.log10(hi), 50)
        
        hst = np.histogram(self.data, bins=hbins)

        # The histogram is shifted by half a bin width of course, when we want to plot the centre values on the linear error bar plot
        if(logarithmic == False):
            rl = np.roll(hbins,-1)
            xvals = 0.5*(hbins + rl)
            xvals = np.delete(xvals,-1)
        else:
            xvals = np.delete(hbins,-1)
            
        errors=np.sqrt(hst[0])
        yvals=hst[0]
        
        hnorm = np.sum(yvals)
        fitvals = self.curve(self.estimates, xvals)
        fnorm = np.sum(fitvals)
        fitvals = fitvals * hnorm/fnorm

        fig,ax = plt.subplots()
        ax.errorbar(xvals, yvals, errors, fmt='o', mfc='none')
        ax.plot(xvals, fitvals, color='black')
        if(logarithmic==True):
            plt.yscale('log')
            plt.xscale('log')
        ax.set_xlabel('Q (Å-1)')
        ax.set_ylabel('Intensity')
        plt.show()
        
        
class gaussianCurve(MLECurve):
    
    # This gaussian curve object is a thorough test case of the methodology
    # It is well known and well-behaved for all methods 
    # (assuming mu and 3*sigma are such that the whole curve is visible within the sampled range of x values)
    # All three methods are implemented:
    # (1) An analytical solution
    # (2) A semi-numerical solution, where the differential and second differential are both supplied analytically,
    #     but the search is done numerically by the base class
    # (3) A numerical solution, where the base class functions are all called to calculate the derivatives numerically
    #
    # This is because, for many curve types, an analytical solution might not be available
    
    def init(self, data):
        self.nparams=2
        
        if(len(data) > 0):
            self.setupGuesses()
        
    def mleAnalytic(self):
        self.method = "analytically"
        N = len(self.data)
        mu = (1.0/N) * np.sum(self.data)
        sigma2 = (1.0/N) * np.sum((self.data - mu)**2.0)
        sigma = np.sqrt(sigma2)
        self.estimates = np.array([mu, sigma])
        
    def mle(self):
        #self.mleAnalytic()
        MLECurve.mle(self)
        
    def curve(self, params, dat=np.array(None)):
        # Return the basic likelihood curve
        mu = params[0]
        sigma = params[1]
        if(np.any(dat == None)):
            xx = self.data
        else:
            xx = dat
        normf = (1.0 / (sigma * np.sqrt(2.0*np.pi))) * np.exp( -( ((xx-mu)**2.0)/ (2.0 * sigma**2.0)))
        return(normf)
        
    def llcurve(self, params):
        # Return the sum of the log likelihood for the curve shape
        normf = self.curve(params)
        lg = np.log(normf, out=np.full_like(normf, 1.0E-30), where=(normf!=0)) # this stops warnings trying to do log(0.0)
        return(np.sum(lg))
    
    def dllcurveAnalytic(self, params=np.array([None, None])):
        # differential of the log likelihood 
        
        if(np.any(params == None)):
            params = self.estimates
        
        mu = params[0]
        sigma = params[1]
        data = self.data
        
        d1 = np.sum( -2.0 * (mu - data)/sigma**2.0 )
        d2 = np.sum( -1.0/sigma + ((self.data - mu)**2.0) / sigma**3.0 )

        return ( np.array([d1, d2]))
    
    def ddllcurveAnalytic(self, params=np.array([None, None])):
        # second differential of the log likelihood 
        
        if(np.any(params == None)):
            params = self.estimates
        
        mu = params[0]
        sigma = params[1]
        data = self.data
        
        dd1 = np.sum( -len(data)/sigma*2.0 )
        dd2 = np.sum( (1.0/sigma**2.0) - 3.0*((data-mu)**2.0) /sigma**4.0 )


        return ( np.array([dd1, dd2]))
        
    def setupGuesses(self):
        self.guesses = np.array([np.mean(self.data), 0.0]) # This is actually the analytic solution for mu
        
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
        
    def CDF(self, params, x):
        mu = params[0]
        sigma= params[1]
        root2 = np.sqrt(2.0)
        cdf = 0.5 * ( 1.0 + erf((x - mu)/(sigma*root2)) )
        return(cdf)
    
    def Quantile(self, params, p):
        mu = params[0]
        sigma = params[1]
        root2 = np.sqrt(2.0)
        qn = mu + sigma * root2 * erfinv(2.0 * p - 1.0)
        return(qn)
    
    def generateTestSamples(self, params, xrange, nsamples, verbose=True):
        pmin = self.CDF(params, xrange[0])
        pmax = self.CDF(params, xrange[1])
            
        uniform = np.random.uniform(pmin, pmax, nsamples)
        self.data = self.Quantile(params, uniform)
        self.setupGuesses()
        if(verbose):
            print("Generated", nsamples, "samples using parameters", params)

    def report(self):
        print("Gaussian curve maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guesses (mu, sigma)")
        print(self.estimates, "solution obtained", self.method)
        print("That a maximum was found is", self.verifyMaximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")
        

class lorentzianCurve(MLECurve):

    def init(self, data):
        self.nparams=1
        
        if(len(data) > 0):
            self.setupGuesses()
            
    def setupGuesses(self):
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
        MLECurve.mle(self)
            
    def curve(self, params, dat=np.array(None)):
        # Return the basic likelihood curve
        kappa = params[0]
        if(np.any(dat == None)):
            xx = self.data
        else:
            xx = dat
        lorf = (kappa / np.pi) * 1.0 / (xx**2.0 + kappa**2.0)
        return(lorf)
            
    def llcurve(self, params):
        # Return the sum of the log likelihood for the curve shape
        lorf = self.curve(params)
        lg = np.log(lorf, out=np.full_like(lorf, 1.0E-30), where=(lorf!=0)) # this stops warnings trying to do log(0.0)
        return(np.sum(lg))
    
    def dllcurve(self, params=np.array([None])):
        # differential of the log likelihood 
        
        if(np.any(params == None)):
            params = self.estimates
        
        kappa = params[0]
        data2 = self.data**2.0
        
        grad = np.sum( (1.0/kappa) - 2.0 * kappa / (kappa**2.0 + data2))
        return ( np.array([grad]))
    
    def ddllcurve(self, params=np.array([None])):
        # second differential of the log likelihood 
        
        if(np.any(params == None)):
            params = self.estimates
        
        kappa = params[0]
        data2 = self.data**2.0
        
        grad = np.sum( -(1.0 / kappa**2.0) + (4.0*kappa**2.0 / (kappa**2.0 + data2)**2.0) - 2.0/(kappa**2.0 + data2))
        
        return ( np.array([grad]))
    
    
    def CDF(self, params, x):
        kappa = params[0]
        cdf = (1.0/np.pi) * np.arctan( x / kappa ) + 0.5
        return(cdf)

    def halfCDF(self, params, x):
        # This is the CDF for half curve assuming the centre is at x=0
        kappa = params[0]
        cdf = (2.0/np.pi) * np.arctan( x / kappa ) 
        return(cdf)

    
    def Quantile(self, params, p):
        kappa = params[0]
        qn = kappa * np.tan(np.pi * (p - 0.5))
        return(qn)

    def nQuantile(self, params, p):
        cdf = super().Quantile(params, p)
        return(cdf)    
    
    def gridECDF(self, dat):
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
        if(self.isSorted == False):
            self.sortData()

        minx = np.amin(self.data)
        maxx = np.amax(self.data)

        pmin = self.CDF(self.estimates, minx)
        pmax = self.CDF(self.estimates, maxx)

        uniform = np.random.uniform(pmin, pmax, nni)
        synth = self.Quantile(self.estimates, uniform)

        synth = np.sort(synth)
                    
        ecdfx, ecdfy = self.gridECDF(self.data) # self.CDF(self.estimates, self.data)
        #ecdfy = np.arange(1, nn+1) / nn
        scdfx, scdfy = self.gridECDF(synth)
        
        dif = np.absolute(scdfy - ecdfy)
        ks = np.amax(dif)
        return(ks)
    
    def generateTestSamples(self, params, xrange, nsamples, verbose=True):
        pmin = self.CDF(params, xrange[0])
        pmax = self.CDF(params, xrange[1])
            
        uniform = np.random.uniform(pmin, pmax, nsamples)
        self.data = self.Quantile(params, uniform)
        self.setupGuesses()
        if(verbose):
            print("Generated", nsamples, "samples using parameters", params)
    
    def report(self):
        print("Lorentzian curve maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guess (kappa)")
        print(self.estimates, "solution obtained", self.method)
        print("That a maximum was found is", self.verifyMaximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")
        
        
        
        
class lorentzianSquaredCurve(MLECurve):

    def CDF(self, params, x):
        kappa = params[0]
        ss    = params[1]

        t1 = np.pi * (2.0 + ss)/kappa
        t2 = (2.0 * ss * x)/(x*x + kappa*kappa)
        t3 = 2.0 * (2.0 + ss) * np.arctan(x / kappa) / kappa

        cdf = 0.25 * (t1 + t2 + t3)
        return(cdf)

    def curve(self, params, dat=np.array(None)):
        kappa = params[0]
        ss = params[1]

        if(np.any(dat == None)):
            xx = self.data
        else:
            xx = dat

        t1 = 1.0 / (kappa*kappa + xx * xx)
        t2 = kappa*kappa*ss / (kappa*kappa + xx*xx)**2.0
        return(t1 + t2)

    def llcurve(self, params):
        crv = self.curve(params)
        lg = np.log(crv, out=np.full_like(crv, 1.0E-30), where=(crv!=0)) 
        return(np.sum(lg))

    
        
        
        
class hardSphereCurve(MLECurve):
    
    def CDF(self, params, x):
        A = params[0]
        R = params[1]
        cdf = (A*(-3 - 5.0*R**2.0 * x**2.0 + 2.0*np.pi*R**5.0 * x**5.0 + \
                  (3.0 - R**2.0 * x**2.0 + 2.0*R**4.0 * x**4.0) * np.cos(2.0*R*x) - \
                  R*x*(6.0 + R**2.0 * x**2.0)*np.sin(2.0*R*x) + 4.0*R**5.0 * x**5.0 \
                  *sinintegral(2.0*R*x)))/(30.0*R**6.0 * x**5.0)
    
    def curve(self, params, dat=np.array(None)):
        # Return the basic likelihood curve
        A = params[0]
        R = params[1]
        
        if(np.any(dat == None)):
            xx = self.data
        else:
            xx = dat
        
        hrd = (A*(-(Q*R*np.cos(Q*R)) + np.sin(Q*R))**2.0)/(Q**6.0 * R**6.0)
        return(hrd)
            
    def llcurve(self, params):
        # Return the sum of the log likelihood for the curve shape
        hrd = self.curve(params)
        lg = np.log(hrd, out=np.full_like(hrd, 1.0E-30), where=(hrd!=0)) # this stops warnings trying to do log(0.0)
        return(np.sum(lg))
    
    
    

# TO DO:

#  P value tests
#  Goodness of fit metrics - "Is the fit good, or valid?"
#  Bayesian information criterion - "Which is the best model that fits my data"

