"""The EventModeAnalyser object is the main API for EMTK for users to perform event mode data analysis.

import emtk.EventModeAnalyser.EMAnalyser as ema
"""

import copy

import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
from lmfit import Model

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

import emcee

from typing import Tuple # we are going to return tuples from functions, need that

# To put in type hinting of returning a class object from a class member function you need to
# persuade python to not suck.
# This fiddle is constantly being tweaked by the python gods and breaks from sub-release to sub-release.
# And lo... we should do:
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
# Thanks yuanzz on stackoverflow 


class EMAnalyser:
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

        self.sampler = None
        self.sampler_state = None

        self.theta_seed = None
        self.nwalkers = 32 # just leave this alone probably
        self.ndims = None
        self.lpf = None
        self.pmf = None
        self.llf = None

        self.mcmc_parameter_values = None
        self.mcmc_parameter_errors = None

        self.xmin = np.amin(self.data)
        self.xmax = np.amax(self.data)

        self.n_events = self.data.size

        if self.weights is None:
            print("EMAnalyser object created with", self.n_events, "events in range", self.xmin, "-", self.xmax)
        else:
            print("EMAnalyser object created with", self.n_events, "weighted events in range", self.xmin, "-", self.xmax)



            
    def help(self):
        helpstr =\
            """The analyser object is created with a numpy array of events
(x-values) and their weights:\n

            import emtk.EventModeAnalyser.EMAnalyser as ema
            ema1 = ema.EMAnalyser(events, weights)

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
            get_lse_param_errors()
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

            ema2.plot_histogram() 
            ema2.plot_kde()

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

The standard method (method=mean) assumes that the parameter distribution is a 
normal distribution / gaussian with a central mean and a standard
deviation from which the parameter value and standard error can be 
easily computed.  It could be that the distribution is sharply peaked
with long tails either side, which will throw off the mean and standard error
significantly if the tails are asymmetric.  Therefore, you can pass method='peak'
which uses scipy signal analysis to extract the most prominent peak and estimate
its FWHM, from which the standard error is extracted assuming that little part, at
least, is roughly gaussian even if the tails are something else.  In other words:

            pvals, sigs = ema2.get_MCMC_parameters(method='peak')
            
Personally, I found that in testing the mean method works just fine, but as soon
as you throw real data at this then the peak method was more reliable for the examples
that I was working on.


"""

        print(helpstr)
            

    def simplex_weights(self, Qraw: np.ndarray) -> np.ndarray:
        """
        Computes the simplex of n+1 weighting factors given n mixture parameters.

        Raw Q values can run between 0-1 to keep things simple.
        The sum is assumed to be 1, preventing out of gamut values.
        Whilst it would be possible to normalise the input array and return array,
        this is most conveniently enforced with the log_prior function.

        Note that Qraw has one dimension fewer
        than the number of parameters, like this:
        https://en.m.wikipedia.org/wiki/Ternary_plot
        """

        # Force the input to be a numpy array
        Qraw = np.asarray(Qraw)

        # TODO: Error checking for nonsense input...

        # The final term is one minus the sum
        Qsum = np.sum(Qraw)    
        Qlast = 1.0 - Qsum

        # Add the final element to the array
        Qvals = np.append(Qraw, Qlast)
                
        return Qvals


    def self_check(self):
        # Checks object self consistency and returns errors if necessary
        if self.data is None:
            print("WARNING: self check: data is not defined.")
            return False

        if not np.isfinite(self.data).all():
            print("WARNING: self check: data has non-finite elements.")
            return False

        # If we get to this point, all is good
        return True
    

            
    def optimal_n_bins(self) -> int:
        """Calculates the optimal number of bins from Freedman-Diaconis rule.
        See for example:
        https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
        https://en.wikipedia.org/wiki/Freedman–Diaconis_rule

        """

        # Protect against calling when there are no data points
        if self.data is None:
            raise ValueError(
                f"attempt to find optimal number of data points with no data defined."
                )


        # Apply the Freedman-Diaconis calculation
        # First calculate the interquartile range of the data
        iqr = np.subtract(*np.percentile(self.data, [75, 25]))

        # If all the data points are equal (or maybe there is only one data point)
        # then the IQR is zero and that makes no sense for anything that comes after
        if iqr == 0.0:
            print("WARNING: interquartile range is zero.")
            return 0

        # If we get to this point it's probably OK, return the Freedman-Diaconis value
        return int((self.xmax - self.xmin)*self.n_events**(1.0/3.0)/(2.0*iqr))

    

    def calculate_histogram(self):
        """Calculates a histogram of the weighted events using
        numpy.histogram.  Just prepares the data, does not plot.  The
        actual plotting is done by plot_histogram().

        """

        # Protect against no data points
        if self.data is None:
            raise ValueError(
                f"attempt to compute histogram with no data defined."
                )
        
        # If we get here, we have events

        #  Get the range of values for the events
        self.xmin=np.amin(self.data)
        self.xmax=np.amax(self.data)

        # Calculate the optimum number of histogram bins
        opt_n_bin = self.optimal_n_bins()

        # Create that number of bins spanning the range of event values
        slic=(self.xmax - self.xmin)/(opt_n_bin+1)
        hbins = np.arange(self.xmin, self.xmax, slic)

        # Maybe the events are weighted, maybe they aren't.  Handle both scenarios.
        if self.weights is None:
            hst = np.histogram(study_data, bins=hbins, density=True)
        else:
            hst = np.histogram(self.data, bins=hbins, density=True, weights=self.weights)

        # Assign the object values that we'll need later on using the
        # created numpy.histogram object.
        self.histo = hst

        # The way that numpy makes histograms is not x-y pairs but x bins
        # We'll remove the last point and plot the histogram as a matplotlib step
        # later with the step at the beginning of the point so it is correct.
        x_hist = hst[1]
        x_hist = x_hist[:-1]

        # Grab the y values
        y_hist = hst[0]

        # Error values are square root of y values (poisson statistics)
        e_hist = np.sqrt(y_hist)

        # Save the results in the class variables for later use
        self.histx = x_hist
        self.histy = y_hist
        self.histe = e_hist


        
        
    def plot_histogram(self, loglog=True, log=True, xlabel='Q (Å$^{-1}$)'):
        """ Plots a histogram of the events in the scipp fashion.
        Uses matplotlib of course.
        Setting log=True makes the y-axis logarithmic.
        Setting loglog=True makes both y- and x-axes logarithmic.
        """

        # First we re-calculate the histogram data
        self.calculate_histogram()

        # Force the shape of the plot to be close to scipp for convenient comparisons.
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        # Plot the histogram as a step plot
        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')

        # Maybe make the graph a log plot or log-log plot as appropriate
        if log or loglog:
            plt.yscale('log')
        if loglog:
            plt.xscale('log')

        # Label the axes, add a legend, and show.
        # TODO: maybe the units of x-axis are not Q...
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()
        plt.show()


        
        
    def plot_LSE_fit(self, loglog=True, log=True, xlabel='Q (Å$^{-1}$)', save=""):
        """ Plots a histogram of the events in the scipp fashion,
        and overlays the least-squares fit of the data.
        Uses matplotlib.
        Setting log=True makes the y-axis logarithmic.
        Setting loglog=True makes the x- and y-axes logarithmic.
        """
        
        self.calculate_histogram()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        # Plot the histogram as a matplotlib step plot
        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')
        # Plot the fit as a regular matplotlib plot
        plt.plot(self.histx, self.lse_result.best_fit, color='black', label="LSE fit")

        # Apply logarithmic axes as requested
        if log or loglog:
            plt.yscale('log')
        if loglog:
            plt.xscale('log')

        # Rest of the nice features
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()
        if save != "":
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')

        plt.show()



    def plot_LSE_initial(self, loglog=True, log=True, xlabel='Q (Å$^{-1}$)', save=""):
        """ Plots a histogram of the events in the scipp fashion,
        and overlays the least-squares initial PDF using the starting
        parameter values.
        Uses matplotlib.
        Setting log=True makes the y-axis logarithmic.
        Setting loglog=True makes the x- and y-axes logarithmic.
        """
        self.calculate_histogram()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        evaly = self.least_squares_model.eval(x=self.histx, params=self.least_squares_parameters)

        # Plot the histogram as a matplotlib step plot
        plt.step(self.histx, self.histy, where='post', label='Optimal Histo')
        # Plot the fit as a regular matplotlib plot
        plt.plot(self.histx, evaly, color='black', label="LSE starting parameters")

        # Apply logarithmic axes as requested
        if log or loglog:
            plt.yscale('log')
        if loglog:
            plt.xscale('log')

        # Rest of the nice features
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()
        if save != "":
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')
        plt.show()



    def calculate_kde(self, logarithmic=True, method="fdr"):
        """ Computes the kernel density estimate of the weighted events.
        Uses scipy's KDE method.  Scikit learn has more options for kernels,
        but scipy has weighted points.  Some of the code here is legacy from
        sklearn testing.
        The plot of the kde is not done by this function, that is handled by 
        plot_kde().
        """
        
        print("Calculating KDE")

        # If a logarithmic x-axis is requested, then the data must be
        # put on a logarithmic axis first
        if logarithmic:
            print("   - logarithmic scale conversion")
            self.data = np.log10(self.data)
            self.xmax = np.amax(self.data)
            self.xmin = np.amin(self.data)
            pmb_bandwidth_scale = 1.0 # to be continued...
            pmb_grid_scale = 1.0
        else:
            print("   - linear scale")
            pmb_grid_scale = 0.1
            pmb_bandwidth_scale = 0.01 # to be continued...
            
        #reshaped = self.data.reshape(-1, 1) # sklearn needs this for some reason

        # Compute optimal number of grid points in the same way as for
        # histogram.  We should check this at some point, mabye this
        # assumption is invalid.
        nx = self.optimal_n_bins()
        slice_size=pmb_grid_scale * (self.xmax - self.xmin)/(nx+1)
        xgrid = np.arange(self.xmin, self.xmax, slice_size)

        print("nx", nx)
        print("slice size", slice_size)
        #print("xgrid", xgrid)

        # Call scipy's gaussian_kde method.
        # In testing I find that a bandwidth of
        # 20x the histogram bin width looks right on a linear binning

        # TODO: a parameterisation of the bandwidth method

        if method=="fdr":
            kde = gaussian_kde(self.data, bw_method=pmb_bandwidth_scale*slice_size, weights=self.weights)

        else:
            # could be method="silverman" or method="scott"
            kde = gaussian_kde(self.data, bw_method=method, weights=self.weights)
        
        # xgrid_reshape = xgrid.reshape(-1, 1) # scikit learn again
        kde_line = kde.evaluate(xgrid)

        # Now we need to normalise the curve correctly.
        # We need a dx value for each data point, so we'll add one at the end
        end_delta = xgrid[-1] - xgrid[-2]
        fakepoint = xgrid[-1] + end_delta
        x_extra = np.append(xgrid, fakepoint)
        xshift = np.delete(x_extra, 0)

        dx = xshift - xgrid

        # Now compute a numerical integral of the KDE curve
        slices = kde_line * dx
        integral = np.sum(slices)

        # ... and store the normalised KDE curve in class variables for later use
        self.kdey = kde_line
        self.kdex = xgrid
        self.kde = kde
        self.kdey = self.kdey / integral
        

        # If we were working on a log scale, we need to reverse that
        # transform so the data is linear again
        if logarithmic:
            tens = np.full_like(self.data, 10.0)
            self.data = np.power(tens, self.data)
            tens = np.full_like(self.kdex, 10.0)
            self.kdex = np.power(tens, self.kdex)
            self.xmin = np.amin(self.data)
            self.xmax = np.amax(self.data)
             # scale the distribution with the derivative -
             # we transformed the x-axis so the new values have all
             # been modified by the derivative, in this case it's a log
             # function
            self.kdey = self.kdey * 1.0 / (self.kdex * np.log(10.0))


        
        
    def plot_kde(self, ylimits=[None, None], log=True, loglog=True, xlabel='Q (Å$^{-1}$)', method='fdr'):
        """ Plots the kernel density estimate of the data set.
        Setting ylimits adds a manual range to the plot on the y-axis.
        Setting log=True plots the y-axis on a log scale.
        Setting loglog=True plots the y- and x-axes on a log scale.
        
        """
        # Force the limits to be a numpy array
        yr = np.asarray(ylimits)

        uselogx = False
        if loglog:
            uselogx = True

        # Re-calculate the KDE data
        self.calculate_kde(logarithmic=uselogx, method=method)

        # Create matplotplib objects
        fig,ax = plt.subplots()

        # Force a figure scale similar to scipp
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        # Plot the KDE
        plt.plot(self.kdex, self.kdey, label='Optimal KDE')

        # If necessary, apply logarithmic axes
        if log or loglog:
            plt.yscale('log')
        if loglog:
            plt.xscale('log')

        # Label axes and trim off the fat on the figure
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()

        # Apply manual limits if specified
        if not yr.any() is None and yr.size == 2:
            ax.set_ylim(yr)

        # Add legend and show figure
        plt.legend()
        plt.show()
        


        
    def plot_MCMC_fit_with_histo(self, log=True, loglog=True, xlabel='Q (Å$^{-1}$)', save=""):
        """ Plots a histogram of the weighted events.
        Overlays a plot of the model PDF where each set of parameters
        is obtained from each of the the MCMC walkers.  A converged
        fit usually results in a single black line for the fit.  A poor
        fit that is not well converged will show many gray lines.
        """

        

        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)

        inds = np.random.choice(len(flat_samples), size=30, replace=False)

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
        if log or loglog:
            plt.yscale('log')
        if log:
            plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()

        if save != "":
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')

        plt.show()

        
    def plot_MCMC_fit_with_kde(self, log=True, loglog=True, xlabel='Q (Å$^{-1}$)', save=""):
        """ Plots a KDE of the weighted events.
        Overlays a plot of the model PDF where each set of parameters
        is obtained from each of the the MCMC walkers.  A converged
        fit usually results in a single black line for the fit.  A poor
        fit that is not well converged will show many gray lines.
        """
        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)

        inds = np.random.choice(len(flat_samples), size=30, replace=False)
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

        uselogx = False
        if loglog:
            uselogx = True
        
        self.calculate_kde(logarithmic=uselogx, method='fdr')
        
        ax.plot(self.kdex, self.kdey, color='blue', label='Optimal KDE')
        if log or loglog:
            plt.yscale('log')
        if log:
            plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()

        if save != "":
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')

        plt.show()

        

    def plot_MCMC_fit(self, method="kde", log=True, loglog=True, xlabel='Q (Å$^{-1}$)', save=""):
        """ A convenience function that calls one of two methods,
        either histogram or KDE.  See the documentation on those
        two methods for more details.
        """
        if method=="kde":
            self.plot_MCMC_fit_with_kde(log=log, loglog=loglog, xlabel=xlabel, save=save)
        if method=="histo":
            self.plot_MCMC_fit_with_histo(log=log, loglog=loglog, xlabel=xlabel, save=save)

            

    def plot_MCMC_convergences(self):
        """
        Plot the theta curves during the final stages of sampling.
        Each parameter is plotted as a separate graph of 
        parameter value vs iteration.  You can then see whether the fit
        has converged well, since all walkers are basically moving
        around the final value in a gaussian distribution.  If the
        walkers are gradually increasing or decreasing you can
        conclude that the fit has not converged.
        """
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
        """ Assigns the least squares probability mass function.
        This is passed through to the lmfit module and is called at 
        each iteration of the least squares fit.
        """
        self.least_squares_pmf_function = func
        self.least_squares_model = Model(self.least_squares_pmf_function)
        print("Least squares model function defined.")


    def make_lse_params(self, **kwgs):
        """ Makes the LSE parameters according to the method of lmfit.
        Basically this is an interface for that.  **kwgs are just passed
        directly through.  See the lmfit documentation for more details.
        """
        if self.least_squares_model is None:
            raise ValueError(
                f"attempt to create parameters for an undefined model.  Define the model function first."
                )
        self.least_squares_parameters = self.least_squares_model.make_params(**kwgs)

    def lse_fit(self):
        """ Calls lmfit to find a least squares fit to the pmf model.
        """
        self.calculate_histogram()

        if self.least_squares_model is None:
            raise ValueError(
                f"attempt to fit events with an undefined model.  Define the model function first."
                )
        
        self.lse_result = self.least_squares_model.fit(self.histy, self.least_squares_parameters, x=self.histx)



        
    def get_lse_param_values(self) -> np.ndarray:
        """ returns numpy array of best fit parameter values as 
        determined by lmfit.
        """
        valdict = self.lse_result.best_values
        vals = np.zeros(len(valdict))

        i = 0
        for key in valdict:
            vals[i] = valdict[key]
            i=i+1
        
        return vals
    

    def get_lse_param_names(self) -> list[str]:
        """ returns array of parameter names as defined in the 
        lmfit parameters object.  Saves you having to keep
        track of those parameter names yourself when plotting
        graphs.
        """
        valdict = self.lse_result.best_values
        vals = [None]*len(valdict)

        i = 0
        for key in valdict:
            vals[i] = key
            i=i+1
        
        return vals


    def get_lse_param_errors(self) -> np.ndarray:
        """ returns numpy array of sigma values for the fit parameters
        from lmfit.
        """

        # Before we get into the meat of this, if the lmfit is not
        # a good fit then the sigmas are not defined.

        try:
            uvars = self.lse_result.result.uvars
            sigmas =  np.full(len(uvars), np.inf)

            i = 0
            for key in uvars:
                sigmas[i] = uvars[key].std_dev
                i=i+1

        except AttributeError: # 'MinimizerResult' object has no attribute 'uvars'            
            # Here the fit was in some way bad.
            # If the fit returned parameter values, this means that
            # the uncertainties are infinite for each parameter.
            # If there was no parameter best values, then the fit didn't work.
            # In either case, we return infinity with an appropriate shape.
            print("WARNING: least squares uncertainties are not defined.  Fit is probably not good.")
            
            if self.lse_result.best_values != None:
                sigmas = np.full(len(self.lse_result.best_values), np.inf)
            else:
                sigmas = np.asarray(np.inf)
        
        return sigmas





    
    def shuffle(self):
        # Shuffles the data
        # In case there is something systematically wrong and you want to smoothe over that
        perm = np.arange(self.data.size)
        np.random.shuffle(perm)
        
        ndata = self.data[perm]
        nweights = self.weights[perm]

        self.data = ndata
        self.weights = nweights
        
    

    def subsample(self, subsample_size: int, randomize=True) -> Self:
        # Subsamples the events to a smaller data size.
        # Returns a deep copy of the object.

        # Make a deep copy of the self object
        copyobj = copy.deepcopy(self)

        if subsample_size < copyobj.n_events:
            if randomize:
                # Choose a random subset of the data elements
                print("Subsampling randomly without replacements.")
                rng = np.random.default_rng()
                elements = rng.choice(copyobj.n_events, subsample_size, replace=False)
            else:
                # Take the first elements in order
                print("Subsampling in order.")
                elements = np.arange(subsample_size)

            # get a local copy of those data elements
            subsample = copyobj.data[elements]
            subweights = copyobj.weights[elements]
 
            # Write them over to the copy object
            copyobj.data = subsample
            copyobj.weights = subweights
            copyobj.n_events = subsample_size

        else:
            # save time and just leave the copy intact.
            # A user is unlikely to do this but maybe
            # they are scripting something that tries to
            # subsample a larger number of events from a data set
            # than exists

            print("WARNING: attempt to subsample", subsample_size, "events from a data set with", copyobj.n_events, "elements")
            print("Subsample will return a direct copy of the data instead.")

        

        # Overwite the copy object data range
        copyobj.xmin = np.amin(copyobj.data)
        copyobj.xmax = np.amax(copyobj.data)

        
            
        # reset the copy object class variables
        copyobj.lse_result = None
        copyobj.histo = None
        copyobj.kde = None

        # return the copy object
        return(copyobj)
        


    
    def MCMC_fit(self, nburn=100, niter=200, convergence="None"):
        """Performs the weighted MCMC fit of the event data.

        nburn is the number of iterations to use for burn-in.

        niter is the number of iterations to use for the actual
        analysis.

        NOTE: if the parameter samples are noisy, undersampled, and or
        posterized, as seen in the plot_MCMC_parameter_distribution()
        then increasing niter does NOT improve the statistics.
        Increasing the number of EVENTS improves the statistics of the
        parameters.

        """

        # Check input parameters
        if not (convergence == "None" or convergence == "Autocorrelation" or convergence == "Gelman-Rubin"):
            print('ERROR: convergence must be one of "None", "Autocorrelation" or "Gelman-Rubin"')
            return

        # Force the parameter seed values to be a numpy array
        p0 = np.asarray(self.theta_seed)

        # Protect against badly defined seed values
        self.ndim = p0.size
        if p0.any() is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined initial theta_seed parameter values.  Define that first."
                )

        # Protect against an undefined log prior function
        if self.lpf is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined log prior function (llf).  Define that first."
                )

        # Protect against an undefined log likelihood function.
        if self.llf is None:
            raise ValueError(
                f"attempt to launch MCMC with undefined log likelihood function (llf).  Define that first."
                )

        # If we get here, we will attempt to launch an MCMC analysis.
        print("MCMC launch")


        # Create local copies of the class variables we need to pass to MCMC
        # MCMC cannot see inside objects
        myllf = self.llf
        nwk = self.nwalkers
        ndm = self.ndim

        # Avoid a runtime error in MCMC where the number of walkers is less than
        # 2x the number of dimensions
        if nwk < 2*ndm:
            nwk = 2*ndm+1
            self.nwalkers = nwk

        # Create a spread of randomised starting points for each walker, that is
        # clustered around the seed position
        p0 = [p0 + 1e-5 * np.random.randn(self.ndim) for k in range(self.nwalkers)]

            
        # If the weights array is badly defined or not defined, use unity weights.
        if self.weights.any() is None:
            self.weights = np.ones_like(self.data)
        # otherwise we assume the weights supplied by the user are correct and just proceed.


        # The first option is that there is no convergence criterion, so we just do a normal
        # burn and sample

        if convergence == "None":

            self.sampler = emcee.EnsembleSampler(nwk, ndm, myllf, args=[self.data, self.xmin, self.xmax, self.weights, self.lpf])
            # Run a burn-in chain and save the final location
            print("Burn in:")
            self.sampler_state = self.sampler.run_mcmc(p0, nburn, progress=True)
 
            # Run the production chain.
            self.sampler.reset()
            print("Sampling:")
            self.sampler_state = self.sampler.run_mcmc(self.sampler_state, niter, progress=True)
 
            print("MCMC sampling complete.")
 
 
        # The second option is that the user wants to use autocorrelation.
        # This is under development and doesn't actually work yet
         
        if convergence == "Autocorrelation":
 
            print("WARNING: Autocorrelation convergence monitoring is an option under development and doesn't actually work yet...")
             
            # Set up the sampler.
            backend = emcee.backends.HDFBackend("autocorrelation_sampler_backend.h5")
            backend.reset(nwk, ndm)
         
            self.sampler = emcee.EnsembleSampler(nwk, ndm, myllf, backend=backend, args=[self.data, self.xmin, self.xmax, self.weights, self.lpf])
         
            # Run the production chain.
            self.sampler.reset()
            print("Sampling:")
 
            # Uses the autocorrelation metric as shown on the emcee website
             
            index = 0
            autocorr = np.empty(niter)
 
            old_tau = np.inf
 
            for sample in self.sampler.sample(p0, iterations=niter, progress=True, store=True):
                # Only check every 100 steps
                if self.sampler.iteration % 100:
                    continue
                
                tau = self.sampler.get_autocorr_time(discard=50)
                
                print(tau)
             
                autocorr[index] = np.mean(tau)
                index += 1
 
                converged = np.all(tau * 100 < self.sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
 
 
                 
        # The third option is to use the Gelman-Rubin statistic
        # This is under development and doesn't work yet
 
        if convergence == "Gelman-Rubin":
            
            print("WARNING: Gelman-Rubin convergence monitoring is an option under development and doesn't actually work yet...")
 
            # Set up the sampler.
            backend = emcee.backends.HDFBackend("gelman-rubin_sampler_backend.h5")
            backend.reset(nwk, ndm)
         
            self.sampler = emcee.EnsembleSampler(nwk, ndm, myllf, backend=backend, args=[self.data, self.xmin, self.xmax, self.weights, self.lpf])


            # Run the production chain.
            self.sampler.reset()
            print("Sampling:")

            for sample in self.sampler.sample(p0, iterations=niter, progress=True, store=True):
                # Only check every 100 steps
                if self.sampler.iteration % 100:
                    continue
                
                grs_array = self.gelman_rubin_statistic()

                grs = np.amax(grs_array)

                print("Current Gelman-Rubin statistic:", grs)
            
                converged = (grs < 1.2)
                terminate = (self.sampler.iteration > niter)

                if converged or terminate:
                    break

                # Can give feedback on whether it was a convergence or a termination...
        

            
                
    def get_MCMC_parameters(self, method='mean') -> Tuple[np.ndarray, np.ndarray]:
        """Returns the mean and standard deviation of each parameter as
        determined by MCMC.  The results are returned as a tuple.
        
        parameters:
            mode:
                (string) can be 'mean' or 'mode'.  Mode discards lowest
                histogram quartile entries to eliminate the skew caused by broad tails,
                but this of course only applies to the central value, not the error bar

        """

        # Grab a flat markov chain from the sampler
        samples=self.sampler.get_chain(flat=True)

        nn = samples.shape[0]
        rootn = np.sqrt(nn)

        # initialise the result arrays
        self.mcmc_parameter_values = np.zeros(self.ndim)
        self.mcmc_parameter_errors = np.zeros(self.ndim)

        # Figure out how many parameters there are
        rge = range(self.ndim)

        # For each parameter, get the mean and standard error from the markov chain samples
        for i in rge:
            if method=='peak':
                # construct a histogram and extract the mode (peak) from it
                hst=np.histogram(samples[:,i], bins='fd')
                peakval=np.amax(hst[0])
                peakind=np.where(hst[0]==peakval)[0][0]
                peakx=hst[1][peakind]

                self.mcmc_parameter_values[i] = peakx

                # Use scipy signal to extract the FWHM of the peak
                sigdat = hst[0]
                peaks, _ = spsig.find_peaks(sigdat)
                widths = spsig.peak_widths(sigdat, peaks, rel_height=0.5)
                
                prominences = spsig.peak_prominences(sigdat, peaks)
                maxprom = np.amax(prominences[0])
                maxind = np.where(prominences[0]==maxprom)[0][0]
                fwhm = widths[0][maxind]

                print('fwhm', fwhm)

                # fwhm = 2 root 2 ln 2 times sigma

                sigma = fwhm / 2.0 * np.sqrt(2.0 * np.log(2.0))

                stderr = sigma / rootn
                
                self.mcmc_parameter_errors[i] = stderr

            else: # mode='mean'
                # The parameter value is the mean of each parameter dimension
                # in the chain
                self.mcmc_parameter_values[i] = np.mean(samples[:,i])

                # The standard error is the standard deviation of each parameter
                # dimension in the chain divided by the square root of the
                # number of samples
                stdd = np.std(samples[:,i])
                print('stdd', stdd)
            
                self.mcmc_parameter_errors[i] = stdd/rootn

        # Return the means and standard deviations as a tuple
        return self.mcmc_parameter_values, self.mcmc_parameter_errors


    

    def plot_MCMC_parameter_distribution(self, item, compare=False, log=False, loglog=False, method='mean', savename=''):
        """Plots the distribution of the <item>th parameter sampled by MCMC.
        This can be useful to see the quality of the convergence for the
        parameter in question.  Specifically we are interested to know:

        1) Is it sampled enough (i.e. does it look like a smooth-ish gaussian)?
        2) Is it posterized (i.e. is it a few points repeated over and over)?  
        3) Is it noisy?

        If any of those are true, in means that the parameter space is
        too discrete and undersampled.  We fix it not with the number
        of iterations, which feels logical, but with the number of
        EVENTS which makes the parameter space less discrete.

        """

        # Get a flat chain 
        samps = self.sampler.get_chain(flat=True)

        # Figure out how many parameters there are
        npars = samps.shape[1]

        # Protect against bad user input for which parameter to investigate.
        if item < 0 or item >= npars:
            raise ValueError(
                f"attempt to analyse a parameter with an index that is out of the range of the number of available parameters."
                )

        pvals, svals = self.get_MCMC_parameters(method=method)

        # Get the mean and standard deviation of the markov chain samples
        #p_mean = np.mean(samps[:,item])
        #p_stddev = np.std(samps[:,item])

        p_mean = pvals[item]
        p_stderr = svals[item]

        print(p_mean, '+/-', p_stderr, 'using method', method)
        
        # calculate the size graphical error bar to put on the plot
        barmin = p_mean - p_stderr
        barmax = p_mean + p_stderr

        # We might also like to compare against least squares, so lets get those results
        lsp = np.asarray(self.least_squares_parameters)
        if compare:
            lsee = self.get_lse_param_errors()
            # that produces warnings if the fit is not defined / not good
            # only run it if we actually want to do the comparison

        # if there are no valid least squares results, or the user does not want a comparison,
        # then our results are simple
        if lsp.any() is None or lsp.size < 1 or compare==False:
            # we dont' do the comparison
            refLSE = False
            # Create text labels based on the parameters
            pnam= "parameter [" + str(item) + "]"
            xlab = pnam
            ylab = "p(" + xlab + ")"
        else:
            # there are valid least squares results and the user wants to have a comparison
            refLSE = True
            # get the parameter names
            pnams = self.get_lse_param_names()

            # extract the relevant element from the name, value, and sigma
            # Don't forget that the LSE results have an amplitude parameter!
            pnam = pnams[item+1]
            lse_pval = lsp[item+1]
            lse_eval = lsee[item+1]
            # create a label texts from that info 
            lstxt   = "LSE value " + str(round(lse_pval,4))
            xlab = pnam
            ylab = "p(" + xlab + ")"

        # Create a text label from the MCMC result
        fittxt = "MCMC value " + str(round(p_mean,4))


        # Make a histogram of the data points
        hst = np.histogram(samps[:,item], bins='fd')
        #hst=plt.hist(samps[:,item], bins='auto', color='k', histtype="step")
        hx = hst[1]
        hx = hx[:-1]
        plt.step(hx, hst[0], where='mid')
        # figure out the maximum y value of that histogram
        ytop = np.amax(hst[0])
        # Label the axes accordingly with the previously established texts
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        # The centre point and standard deviation are overlayed as a cross
        plt.vlines(p_mean, 0, ytop, color="red")
        plt.hlines(y=ytop*0.5, xmin=barmin, xmax=barmax, color="red")

        # The parameter value is appended at the top of the centre bar
        plt.text(p_mean, ytop*0.97, fittxt, color="red")

        
        if refLSE:
            # if the comparison with LSE is done, we also make a blue cross marking the estimate
            # centre value, standard deviation, and text label at the top
            plt.vlines(lse_pval, 0, ytop*0.9, ls='--', color='b')
            plt.hlines(y=ytop*0.9*0.5, xmin=lse_pval-lse_eval, xmax=lse_pval+lse_eval, ls='--', color='b')
            plt.text(lse_pval*1.005, ytop*0.87, lstxt, color="b")

        # Apply logarithmic axis scaling if required:
        if log or loglog:
            plt.xscale('log')

        if loglog:
            plt.yscale('log')

        if savename != '':
            plt.savefig(savename, dpi=600, bbox_inches='tight')

        # Show the plot
        plt.show()





           
    def gelman_rubin_statistic(self) -> np.ndarray :

        chains = self.sampler.get_chain()
            
        nwk = self.nwalkers # "J"
        ndm = self.ndim
        nsamps = chains.shape[0] #"L"

        nkeep = int(nsamps / 2.0)
        
        chains = chains[-nkeep::]
        
        
        # Now compute the statistic with the remaining chains
        chain_mean = np.mean(chains, axis=0)        
        grand_mean = np.mean(chain_mean, axis=0)

        within_chain_variance = 1.0 / (nsamps - 1.0) * np.sum( (chains - grand_mean)**2.0, axis=0)
        between_chain_variance = nsamps/(nwk - 1.0) * np.sum( (chain_mean - grand_mean)**2.0, axis=0)
        
        ww = (1.0 / nwk) * np.sum(within_chain_variance, axis=0)
        
        t1 = (nsamps-1.0)*ww / nsamps
        t2 = between_chain_variance / nsamps
        
        grstat = (t1 + t2) / ww
        
        return grstat
