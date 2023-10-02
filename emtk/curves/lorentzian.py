"""A lorentzian curve is based on classical theory of correlation
lengths when a system approaches a phase transition.  It also arises
in some fuzzy systems.  This class inherits from the MLE / bayesian
inference base class.

Another name for lorentzian curve is a Cauchy distribution - the two
curves are mathematically identical.

Unlike for a gaussian curve, the mean and the variance are infinite,
because the tails of a lorentzian are broad.  This is the easiest way
of understanding why there are no good numerical maximum likelihood
estimates for the parameters of this curve.  The next best solution is
to define the differentials analytically, and then call the numerical
solver in the base class.  

The first and second derivatives of the log likelihood do exist, as do
the CDF and Quantile, which makes life easy.

The implementation below assumes small angle scattering with a
properly treated data set centred on Q=0.  Any systematic error on the
zero point will translate into an inaccuracy in the estimates of kappa
(the width parameter).

The function has the form:

                   gamma
     I(Q) = -----------------------
            pi * Q^2 + pi * gamma^2


A fit of the data attempts to identify gamma, the spatial extent of
the correlations in the system.  A least squares fit would also
determine the amplitude of the curve, proportional to the number
density and contrast.  These can be computed numerically knowing the
total number of neutrons counted and the gamma parameter, and the
uncertainty on that parameter likewise from the variance on gamma.

In addition to overloading various base class methods, the following
are exemplified:
    
    Methods:
        setupGuesses:
            Intialises the guess values to be used for iteration
        gridECDF:
            Histograms and computes an empirical cumulative 
            distribution function for statistical test metrics

"""


from . import curve as base

import numpy as np

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt


class LorentzianCurve(base.Curve):

    def init(self, data):
        self.nparams=1

        if len(data) > 0:
            self.setupGuesses()

    def setup_guesses(self):
        """Creates initial guesses of the parameters before numerical MLE
        method is applied.

        """

        self.guesses = np.array([np.mean(self.data)])

        # now generate a random sample over sigma and pick the best one as the initial estimate
        mindata = np.amin(self.data)
        maxdata = np.amax(self.data)
        nguess = 50
        kappas = np.logspace(-4,-1, nguess)
        guess_vals = np.zeros_like(kappas)

        for ii in range(nguess):
            pars = np.array([self.guesses[0], kappas[ii]])
            guess_vals[ii] = self.llcurve(pars)

        best_estimate = np.argmax(guess_vals)
        self.guesses[0] = kappas[best_estimate]
        self.estimates = self.guesses
        print(self.estimates)

    def mle(self, verbose=False):
        """Passes through to the base class to perform numerical MLE.

        """

        if (self.guesses == None).any():
            self.setup_guesses()

        base.Curve.mle(self, verbose)
        self.calc_variances()

    def curve(self, params, dat=None):
        """Returns the basic likelihood curve for a Lorentzian function.

        """
        dat = np.asarray(dat)

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

    def d_llcurve(self, params=None):
        """The analytical first derivative of the log-likelihood of a
        Lorentzian distribution.

        """

        params = np.asarray(params)
        
        if np.any(params == None):
            params = self.estimates

        kappa = params[0]
        data2 = self.data**2.0

        grad = np.sum( (1.0/kappa) - 2.0 * kappa / (kappa**2.0 + data2))
        return np.array([grad])

    def dd_llcurve(self, params=None):
        """The analytical second derivative of the log-likelihood of a
        Lorentzian distribution.

        """

        params = np.asarray(params)

        if np.any(params == None):
            params = self.estimates

        kappa = params[0]
        data2 = self.data**2.0

        grad = np.sum( -(1.0 / kappa**2.0) +\
                       (4.0*kappa**2.0 / (kappa**2.0 + data2)**2.0) - 2.0/(kappa**2.0 + data2))

        return np.array([grad])

    def cdf(self, params, x):

        kappa = params
        cdf = (1.0/np.pi) * np.arctan( x / kappa ) + 0.5
        return cdf

    def half_cdf(self, params, x):
        # This is the CDF for half curve assuming the centre is at x=0
        kappa = params[0]
        cdf = 0.5 + np.arctan( x / kappa ) / np.pi
        return cdf


    def quantile(self, params, p):
        kappa = params
        qn = kappa * np.tan(np.pi * (p - 0.5))
        return qn

    def n_quantile(self, params, p):
        cdf = super().quantile(params, p)
        return cdf

    def grid_ecdf(self, dat):
        # Calculates the empirical CDF on a grid of 100 points
        xmin = np.amin(dat)
        xmax = np.amax(dat)

        ns = 100

        stp = (xmax - xmin)/float(ns)


        nn = float(len(dat))
        xs = np.arange(xmin, xmax, stp)

        # With float rounding error, sometimes the length splills over
        # to 101 pts.  If so, remove the last point
        if xs.size > 100:
            xs = np.delete(xs, -1)

        #print ("siz", xs.size, xmin, xmax, stp, (xmax-xmin)/stp)

        ecdfy = np.zeros_like(xs)

        for i in range(ns):
            ecdfy[i] = np.sum( dat < xs[i] ) / nn

        return (xs, ecdfy)





    def ks_test(self):
        # Overloading the base class ks-test and doing a numerical KS test with two eCDFs
        nn = float(len(self.data))
        nni = int(nn)
        if self.is_sorted is False:
            self.sort_data()

        minx = np.amin(self.data)
        maxx = np.amax(self.data)

        pmin = self.cdf(self.estimates, minx)
        pmax = self.cdf(self.estimates, maxx)

        uniform = np.random.uniform(pmin, pmax, nni)
        synth = self.quantile(self.estimates, uniform)

        synth = np.sort(synth)

        ecdfx, ecdfy = self.grid_ecdf(self.data) # self.CDF(self.estimates, self.data)
        #ecdfy = np.arange(1, nn+1) / nn
        scdfx, scdfy = self.grid_ecdf(synth)

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
        print("That a maximum was found is", self.verify_maximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")


    def calc_variances(self, crit = 1.0):
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
            sensible result.  This means that calc_variances() might be unstable 
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
        
        second_diff = self.dd_llcurve(self.estimates)
        fisherI = -second_diff/self.data.size
        
        variance = 1.0 / fisherI
        self.variances = np.array([variance])
        
        

    def infer(self, plot=False, background=None, verbose=True):

        # Bayesian inference

        background = np.asarray(background)
        if np.any(background == None):
            # No background events were supplied
            marginal_likelihood = np.full_like(self.data, 1.0)
        else:
            # Background events were supplied
            print("Using background as marginal likelihood")
            nbg = background.size
            ndt = self.data.size
            scale = ndt / nbg
            # scikit-learn:
            #kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            #kde.fit(background[:, np.newaxis])
            #marginal_likelihood = kde.score_samples(self.data[:, np.newaxis])

            #scipy:
            kde = gaussian_kde(background, bw_method='silverman')
            marginal_likelihood = kde.evaluate(self.data)
            
            mlsum = np.sum(marginal_likelihood)
            marginal_likelihood = scale * marginal_likelihood / mlsum

            inverse_ml = 1.0 / marginal_likelihood

            if verbose:
                xmin = np.amin(background)
                xmax = np.amax(background)

                slic = (xmax-xmin)/51.0
                hbins = np.arange(xmin, xmax, slic)

                hst = np.histogram(background, bins=hbins)
                xvals = np.delete(hbins,-1)
                yvals=hst[0]
                errors=np.sqrt(hst[0])

                fig, ax = plt.subplots()
                ax.errorbar(xvals, yvals, errors, fmt='o', mfc='none')
                plt.yscale('log')
                plt.xscale('log')
                ax.set_xlabel('Q (Å-1)')
                ax.set_ylabel('Intensity')
                plt.xlim([xmin, xmax])
                plt.show()
                
                
                fig, ax = plt.subplots()
                ax.scatter(self.data, marginal_likelihood)
                ax.set_xlabel('Q')
                ax.set_ylabel('Marginal likelihood')
                plt.yscale('log')
                plt.xscale('log')
                plt.xlim([xmin, xmax])
                plt.show()
                
                fig, ax = plt.subplots()
                ax.scatter(self.data, inverse_ml)
                ax.set_xlabel('Q')
                ax.set_ylabel('Inverse Marginal likelihood')
                plt.yscale('log')
                plt.xscale('log')
                plt.xlim([xmin, xmax])
                plt.show()
            
            
        
        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.data), np.amax(self.data) ])
        
        logmean = np.mean(np.log10( xrange)) 
        loghw =  np.std(np.log10(xrange))
        rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/rrange
        rrange = np.round(rrange)

        print("rrange", rrange)

        rvals = np.arange(rrange[1], rrange[0], 1.0)

        revr = np.flip(rvals)

        #print(rvals)

        kappas = 1.0 / revr
        print("kappas", kappas[0], kappas[-1])
        

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
            posterior = prior + np.log10((1.0/marginal_likelihood[neutron]) * \
                                         (kappas / np.pi) / (self.data[neutron]**2.0 + kappas2))
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









class LorentzianBGCurve(base.Curve):

    def init(self, data):
        self.nparams=2

        if len(data) > 0:
            self.setupGuesses()

    def setup_guesses(self):
        """Creates initial guesses of the parameters before numerical MLE
        method is applied.

        """

        print("Setting up guesses k and bg")
        
        self.guesses = np.array([np.mean(self.data), 0.1])

        # now generate a random sample over sigma and pick the best one as the initial estimate
        #mindata = np.amin(self.data)
        #maxdata = np.amax(self.data)
        #nguess = 50
        #kappas = np.logspace(-4,-1, nguess)
        #guess_vals = np.zeros_like(kappas)

        #for ii in range(nguess):
        #    pars = np.array([self.guesses[0], kappas[ii]])
        #    guess_vals[ii] = self.llcurve(pars)

        #best_estimate = np.argmax(guess_vals)
        #self.guesses[0] = kappas[best_estimate]
        #print(self.guesses)

    def mle(self, verbose=False):
        """Passes through to the base class to perform numerical MLE.

        """

        base.Curve.mle(self, verbose)
        self.calc_variances()

    def curve(self, params, dat=None):
        """Returns the basic likelihood curve for a Lorentzian function.

        """
        dat = np.asarray(dat)

        xx = dat
        
        kappa = params[0]
        bb = params[1]
        if np.any(dat == None):
            xx = self.data

        lorf = bb + (kappa / np.pi) * 1.0 / (xx**2.0 + kappa**2.0)
        return lorf

    def llcurve(self, params):
        """Returns the log-likelihood for a Lorentzian distribution.

        """

        lorf = self.curve(params)
        # this stops warnings trying to do log(0.0)
        lg = np.log(lorf, out=np.full_like(lorf, 1.0E-30), where= lorf!=0 )
        return np.sum(lg)






    def infer(self, r_range=None, bg_range=None, plot=True):

        # Bayesian inference
        r_range = np.asarray(r_range)
        bg_range = np.asarray(bg_range)

        # First, figure out range of kappa values
        if (r_range == None).any():
            xrange = np.array( [ np.amin(self.data), np.amax(self.data) ])
        
            logmean = np.mean(np.log10( xrange)) 
            loghw = 0.5*np.std(np.log10(xrange))
            rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
            rrange = 1.0/rrange
            rrange = np.round(rrange)
        else:
            rmax = np.amax(r_range)
            rmin = np.amin(r_range)

            rrange = np.array([rmin, rmax])
            
        print(rrange)

        rmin = np.amin(rrange)
        rmax = np.amax(rrange)

        rstep = (rmax - rmin) / 100.0

        rvals = np.arange(rmin, rmax, rstep)
        revr = np.flip(rvals)
        kappas = 1.0 / revr


        if (bg_range == None).any():
            bgmax = 2.0
            bgmin = 0.1
        else:
            bgmax = np.amax(bg_range)
            bgmin = np.amin(bg_range)

        bgstep = (bgmax - bgmin) / 100
        bgs = np.arange(bgmin, bgmax, bgstep)

        
        kappa, bg = np.meshgrid(kappas, bgs)

        prior_value = 1.0E-04
        
        prior = np.full_like(kappa, prior_value)
        posterior = np.full_like(kappa, prior_value)

        # Flip into logspace

        prior = np.log10(prior)
        posterior = np.log10(posterior)


        kappa2 = kappa**2.0


        # Infer

        for neutron in np.arange(0, self.data.size, 1):
            posterior = prior + np.log10(bg + (kappa / np.pi) / (self.data[neutron]**2.0 + kappa2))
            prior = np.copy(posterior) # this is copied over for the next iteration

            
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
            plt.contourf(kappa, bg, posterior)
            plt.axis('scaled')
            cbar = plt.colorbar()
            cbar.set_label('Relative likelihood')
            ax.set_xlabel('kappa')
            ax.set_ylabel('bg')
            ax.set_aspect('auto')
            plt.show




    



    def report(self):
        print("Lorentzian curve with background maximum likelihood estimation")
        print(len(self.data), "data points")
        print(self.guesses, "as initial guess (kappa, BG)")

        
        if not np.any(self.variances) == None :
            errstr = " +/- " + np.array2string(self.variances[0]) + "?"
        else:
            errstr = ""
            
        print(self.estimates, errstr, "solution obtained", self.method)

        print(self.estimates, "solution obtained", self.method)
        print("That a maximum was found is", self.verify_maximum(), "via second derivative")
        #print(self.uncertainty(), "uncertainty sigma (=root-variance)")


    def calc_variances(self, crit = 1.0):
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
            sensible result.  This means that calc_variances() might be unstable 
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
        
        second_diff = self.dd_llcurve(self.estimates)
        fisherI = -second_diff/self.data.size
        
        variance = 1.0 / fisherI
        self.variances = np.array([variance])
        
        



