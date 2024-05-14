"""The EventModeAnalyser object is the main API for EMTK for users to perform event mode data analysis.

import emtk.EventModeAnalyser as ema
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

import emcee

class Analyser:
    """Main object with which users will interact.

    """

    def __init__(self, set_data=None, set_weights=None):
        self.data = np.asarray(set_data)
        self.weights = np.asarray(set_weights)
        self.likelihood_pmf_function = None
        self.log_likelihood_function = None
        self.least_squares_pmf_function = None
        self.least_squares_parameters = None
        self.least_squares_model = None
        self.lse_result = None
        self.histo = None
        self.histx = None
        self.histy = None
        self.histe = None
        
        self.kde = None

        self.theta_seed = None
        self.nwalkers = 32 # just leave this alone probably
        self.ndims = None
        self.lpf = None
        self.pmf = None
        self.llf = None

        self.mcmc_parameter_values = None
        self.mcmc_parameter_sigmas = None

        self.xmin = np.amin(self.data)
        self.xmax = np.amax(self.data)

        self.n_events = self.data.size

        if self.weights is None:
            print("Analyser object created with", self.n_events, "events in range", self.xmin, "-", self.xmax)
        else:
            print("Analyser object created with", self.n_events, "weighted events in range", self.xmin, "-", self.xmax)


    def help(self):
        helpstr =\
            """The analyser object is created with a numpy array of events
(x-values) and their weights:\n

            import emtk.EventModeAnalyser.Analyser as ema
            ema1 = ema.Analyser(events, weights)

There are two provided workflows with the same API.  The first API
does regular histogram and least squares fit analysis with lmfit as a
backend.  So you need to define a function and parameters, exactly
according to lmfit documentation, which are passed through to lmfit.
These include:

            set_lse_function()
            make_lse_params()
            lse_fit()

And to get the results of those fits:
            plot_LSE_fit()
            get_lse_param_values()
            get_lse_param_sigmas()
            get_lse_param_names()

The last one takes the names as specified in the lmfit interface to
make the parameters, as passed to the function above.

The main event mode analysis part is a Markov-Chain Monte-Carlo
sampling method using emcee as a backend.  If you want to subsample
randomly the events to reduce the data size (e.g. for testing) you can
do so - in this example we subsample a data size of 6 million events
to only 50,000 for testing:

            ema2 = ema1.subsample(50000)

You can then visualise the data using either a histogram or a kernel
density plot: 

            ema2.plot_histogram() ema2.plot_kde()

If necessary you can specify the y-axis limits for your KDE plot:

            ema2.plot_kde([0.1, 20])

Note that with event mode analysis, the background needs to be
parameterised and fit as a contribution to the curve.  It is not
subtracted.  However, all the other corrections (detector efficiency,
solid angle etc) are quantified using the event weight array.

You need to specify a log-prior function that takes one 'theta'
argument as an array of variables; a pmf function that again takes
'theta' and also an array of x-values as events, along with xmin,
xmax, and an array of weights; and a log-likelihood function that
again takes 'theta' and an array of x-values (events), xmin, xmax,
weights, and the log-prior function as arguments.  This is a bit
long-winded so look at the worked example notebook to see how to do
it.  The API is then assigned these functions to use:

            ema2.lpf = log_prior_function
            ema2.pmf = probability_mass_function
            ema2.llf = log_likelihood_function

You then need to seed the search space with a starting point:

            ema2.theta_seed = np.array([theta1, theta2, ... , thetaN])

At this point you could do both fitting methods:
           
            ema2.lse_fit()
            ema2.MCMC_fit()

And to compare the fitting methods.  The MCMC fit can be shown using
either a histogram or KDE plot for the events:
       
            ema2.plot_LSE_fit()
            ema2.plot_MCMC_fit()
            ema2.plot_MCMC_fit(method="histo")
            ema2.plot_MCMC_fit(method="kde")


To see the sampled distribution of the nth parameter, you can plot it
with:

            ema2.plot_MCMC_parameter_distribution(N)

To compare with LSE parameter values (if they exist):

            ema2.plot_MCMC_parameter_distribution(N, compare=True)

To see a convergence trace of all parameters:
            
            ema2.plot_MCMC_convergences()

Get the parameters and sigmas as determined by MCMC:

            pvals, sigs = ema2.get_MCMC_parameters()
            
         

"""

        print(helpstr)
            

    def simplex_weights(self, Qraw):
        """
        Computes the simplex of n+1 weighting factors given n mixture parameters

        Raw Q values can run between 0-1 to keep things simple
        The sum is assumed to be 1, preventing out of gamut values
        That condition of unity summation must be enforced elsewhere, e.g. in the 
        the log_prior.

        Note that Qraw has one dimension fewer
        than the number of parameters, like this:
        https://en.m.wikipedia.org/wiki/Ternary_plot
        """

        Qraw = np.asarray(Qraw)
    
        Qsum = np.sum(Qraw)    
        Qlast = 1.0 - Qsum
        
        Qvals = np.append(Qraw, Qlast)
                
        return Qvals



    
            
    def optimal_n_bins(self):
        """Calculate optimal number of bins from Freedman-Diaconis rule
        https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
        https://en.wikipedia.org/wiki/Freedman–Diaconis_rule

        """

        if self.data is None:
            raise ValueError(
                f"attempt to find optimal number of data points with no data defined."
                )

        iqr = np.subtract(*np.percentile(self.data, [75, 25]))
        if iqr == 0.0:
            print("WARNING: interquartile range is zero.")
            return 0
        return int((self.xmax - self.xmin)*self.n_events**(1.0/3.0)/(2.0*iqr))

    def calculate_histogram(self):
        if self.data is None:
            raise ValueError(
                f"attempt to compute histogram with no data defined."
                )
        
        # If we get here, we have events
        self.xmin=np.amin(self.data)
        self.xmax=np.amax(self.data)
        
        opt_n_bin = self.optimal_n_bins()

        slic=(self.xmax - self.xmin)/(opt_n_bin+1)
        hbins = np.arange(self.xmin, self.xmax, slic)

        if self.weights is None:
            hst = np.histogram(study_data, bins=hbins, density=True)
        else:
            hst = np.histogram(self.data, bins=hbins, density=True, weights=self.weights)

        self.histo = hst

        x_hist = hst[1]
        x_hist = x_hist[:-1]
        y_hist = hst[0]
        e_hist = np.sqrt(y_hist)

        self.histx = x_hist
        self.histy = y_hist
        self.histe = e_hist
            
        
    def plot_histogram(self):
        """ Plots a histogram of the events in the scipp fashion
        """
        self.calculate_histogram()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_LSE_fit(self):
        self.calculate_histogram()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')
        plt.plot(self.histx, self.lse_result.best_fit, color='black', label="LSE fit")
        
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()
        plt.legend()
        plt.show()





    def calculate_kde(self):
        print("Calculating KDE")
        reshaped = self.data.reshape(-1, 1) # sklearn needs this for some reason
        # still compute optimal number of grid points
        nx = self.optimal_n_bins()
        slic=(self.xmax - self.xmin)/(nx+1)
        
        xgrid = np.arange(self.xmin, self.xmax, slic)

        kde = gaussian_kde(self.data, bw_method=20*slic, weights=self.weights)
        xgrid_reshape = xgrid.reshape(-1, 1)
        kde_line = kde.evaluate(xgrid)

        end_delta = xgrid[-1] - xgrid[-2]
        fakepoint = xgrid[-1] + end_delta
        x_extra = np.append(xgrid, fakepoint)
        xshift = np.delete(x_extra, 0)

        dx = xshift - xgrid

        slices = kde_line * dx

        integral = np.sum(slices)

        self.kdex = xgrid
        self.kdey = kde_line / integral
        self.kde = kde

        
    def plot_kde(self, yspan=[None, None]):


        yr = np.asarray(yspan)

        self.calculate_kde()

        fig,ax = plt.subplots()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        plt.plot(self.kdex, self.kdey, label='Optimal KDE')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()

        if not yr.any() is None:
            ax.set_ylim(yr)
        plt.legend()
        plt.show()
        

    def plot_MCMC_fit_with_histo(self):

        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)

        inds = np.random.randint(len(flat_samples), size=30)

        plt.rcParams["figure.figsize"] = (5.75,3.5)
        fig, ax = plt.subplots()

        self.calculate_histogram()
        pt_sum = np.sum(self.histy)

        
        for ind in inds:
            sample = flat_samples[ind]
            yfit = self.pmf(sample, self.histx, self.xmin, self.xmax, None)
            ysum = np.sum(yfit)
            scale = pt_sum / ysum
            yfit = yfit * scale

            if ind == inds[0]:
                ax.plot(self.histx, yfit, color='black', alpha=0.2, label='Population of MCMC walkers')
            else:
                ax.plot(self.histx, yfit, color='black', alpha=0.2)
        
        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()
        plt.legend()
        plt.show()


        
    def plot_MCMC_fit_with_kde(self):

        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)

        inds = np.random.randint(len(flat_samples), size=30)
        pt_sum = np.sum(self.kdey)

        plt.rcParams["figure.figsize"] = (5.75,3.5)
        fig, ax = plt.subplots()

        for ind in inds:
            sample = flat_samples[ind]
            yfit = self.pmf(sample, self.kdex, self.xmin, self.xmax, None)
            ysum = np.sum(yfit)
            scale = pt_sum / ysum
            yfit = yfit * scale

            if ind == inds[0]:
                ax.plot(self.kdex, yfit, color='black', alpha=0.2, label='Population of MCMC walkers')
            else:
                ax.plot(self.kdex, yfit, color='black', alpha=0.2)
        
        self.calculate_kde()
        ax.plot(self.kdex, self.kdey, color='blue', label='Optimal KDE')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()
        plt.legend()
        plt.show()


    def plot_MCMC_fit(self, method="kde"):
        if method=="kde":
            self.plot_MCMC_fit_with_kde()
        if method=="histo":
            self.plot_MCMC_fit_with_histo()

            

    def plot_MCMC_convergences(self):
        # Plot the theta curves during sampling
        if self.ndim is None or self.ndim < 1:
            raise ValueError(
                f"ndims is not defined so cannot subplot parameters."
                )
            
        fig, axes = plt.subplots(self.ndim, figsize=(8, 10), sharex=True)
        samples = self.sampler.get_chain(discard=100)

        lsp = np.asarray(self.least_squares_parameters)

        if lsp.any() is None or lsp.size < 1:
            refLSE = False
        else:
            refLSE = True
            pnams = self.get_lse_param_names()
            pnams = pnams[1:]
        
        #truevals=np.array([true_kappa, np.log10(porod_events.size / (porod_events.size + curv.data.size))])
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            if refLSE:
                ax.hlines(lsp[i], 0, samples[:,:,i].size, color='r', ls='--', label='Least squares estimate')
                labeltxt = pnams[i]
            else:
                labeltxt = "$\\theta$[" + str(i) + "]"
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labeltxt)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.legend()
        plt.show()



        
        
    def set_lse_function(self, func):
        self.least_squares_pmf_function = func
        self.least_squares_model = Model(self.least_squares_pmf_function)
        print("Least squares model function defined.")


    def make_lse_params(self, **kwgs):
        if self.least_squares_model is None:
            raise ValueError(
                f"attempt to create parameters for an undefined model.  Define the model function first."
                )
        self.least_squares_parameters = self.least_squares_model.make_params(**kwgs)

    def lse_fit(self):
        if self.histx.any() is None or self.histy.any() is None:
            self.calculate_histogram()

        if self.least_squares_model is None:
            raise ValueError(
                f"attempt to fit events with an undefined model.  Define the model function first."
                )
        
        self.lse_result = self.least_squares_model.fit(self.histy, self.least_squares_parameters, x=self.histx)
            
    def get_lse_param_values(self):
        # returns numpy array of best fit parameter values
        valdict = self.lse_result.best_values
        vals = np.zeros(len(valdict))

        i = 0
        for key in valdict:
            vals[i] = valdict[key]
            i=i+1
        
        return vals

    def get_lse_param_names(self):
        # returns array of parameter names
        valdict = self.lse_result.best_values
        vals = [None]*len(valdict)

        i = 0
        for key in valdict:
            vals[i] = key
            i=i+1
        
        return vals


    def get_lse_param_sigmas(self):
        # returns numpy array of best fit parameter sigmas
        if  self.lse_result.result.uvars != None:
            uvars = self.lse_result.result.uvars
            sigmas =  np.zeros(len(uvars))
            
            i = 0
            for key in uvars:
                sigmas[i] = uvars[key].std_dev
                i=i+1

        else:
            if self.lse_result.best_values != None:
                sigmas = np.zeros(len(self.lse_result.best_values))
            else:
                sigmas = np.asarray(0.0)
        
        return sigmas

    

    def subsample(self, subsample_size):
        # Randomly subsamples the events to a smaller data size.
        # Returns a deep copy of the object.

        copyobj = copy.deepcopy(self)

        rng = np.random.default_rng()
        elements = rng.choice(copyobj.n_events, subsample_size)

        subsample = copyobj.data[elements]
        subweights = copyobj.weights[elements]

        copyobj.data = subsample
        copyobj.weights = subweights
        copyobj.n_events = subsample_size


        
        copyobj.xmin = np.amin(copyobj.data)
        copyobj.xmax = np.amax(copyobj.data)

        copyobj.lse_result = None
        copyobj.histo = None
        copyobj.kde = None

        return(copyobj)
        

    def MCMC_fit(self, nburn=50, niter=200):
        p0 = np.asarray(self.theta_seed)
        self.ndim = p0.size
        
        if p0.any() is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined initial theta_seed parameter values.  Define that first."
                )

        if self.lpf is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined log prior function (llf).  Define that first."
                )

        if self.llf is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined log likelihood function (llf).  Define that first."
                )

        print("MCMC launch")



        p0 = [p0 + 1e-5 * np.random.randn(self.ndim) for k in range(self.nwalkers)]

        myllf = self.llf
        nwk = self.nwalkers
        ndm = self.ndim

        if self.weights.any() is None:
            self.weights = np.ones_like(self.data)
        
        # Set up the sampler.
        self.sampler = emcee.EnsembleSampler(nwk, ndm, myllf, args=[self.data, self.xmin, self.xmax, self.weights, self.lpf])
        
        # Run a burn-in chain and save the final location
        print("Burn in:")
        state = self.sampler.run_mcmc(p0, nburn, progress=True)
    
        # Run the production chain.
        self.sampler.reset()
        print("Sampling:")
        self.sampler.run_mcmc(state, niter, progress=True);


    def get_MCMC_parameters(self):
        samples=self.sampler.get_chain(flat=True)

        self.mcmc_parameter_values = np.zeros(self.ndim)
        self.mcmc_parameter_sigmas = np.zeros(self.ndim)

        rge = range(self.ndim)
        
        for i in rge:
            self.mcmc_parameter_values[i] = np.mean(samples[:,i])
            self.mcmc_parameter_sigmas[i] = np.std(samples[:,i])

        return self.mcmc_parameter_values, self.mcmc_parameter_sigmas
        

    def plot_MCMC_parameter_distribution(self, item, compare=False, log=False, loglog=False):

        samps = self.sampler.get_chain(flat=True)

        npars = samps.shape[1]

        if item < 0 or item >= npars:
            raise ValueError(
                f"attempt to analyse a parameter with an index that is out of the range of the number of available parameters."
                )
        
        p_mean = np.mean(samps[:,item])
        p_stddev = np.std(samps[:,item])

        barmin = p_mean - p_stddev
        barmax = p_mean + p_stddev


        lsp = np.asarray(self.least_squares_parameters)
        lsee = self.get_lse_param_sigmas()

        if lsp.any() is None or lsp.size < 1 or compare==False:
            refLSE = False
            pnam= "parameter [" + str(item) + "]"
            xlab = pnam
            ylab = "p(" + xlab + ")"
        else:
            refLSE = True
            pnams = self.get_lse_param_names()
            pnams = pnams[1:]
            pnam = pnams[item]
            lse_pval = lsp[item]
            lse_eval = lsee[item]
            lstxt   = "LSE value " + str(round(lse_pval,4))
            xlab = pnam
            ylab = "p(" + xlab + ")"

        fittxt = "MCMC value " + str(round(p_mean,4))

        hst=plt.hist(samps[:,item], bins='auto', color='k', histtype="step")
        ytop = np.amax(hst[0])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        #plt.vlines(true_kappa, 0, ytop*0.8, color='g')            
        plt.vlines(p_mean, 0, ytop, color="red")
        plt.hlines(y=ytop*0.5, xmin=barmin, xmax=barmax, color="red")
        #plt.xlim([true_kappa*0.7, true_kappa*1.3])
        #plt.text(true_kappa*1.005, ytop*0.77, truetxt, color="g")
        
        plt.text(p_mean, ytop*0.97, fittxt, color="red")

        if refLSE:
            plt.vlines(lse_pval, 0, ytop*0.9, ls='--', color='b')
            plt.hlines(y=ytop*0.9*0.5, xmin=lse_pval-lse_eval, xmax=lse_pval+lse_eval, ls='--', color='b')
            plt.text(lse_pval*1.005, ytop*0.87, lstxt, color="b")

        if log or loglog:
            plt.xscale('log')

        if loglog:
            plt.yscale('log')

            
        plt.show()


