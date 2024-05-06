"""The EventModeAnalyser object is the main API for EMTK for users to perform event mode data analysis.

import emtk.EventModeAnalyser as ema
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

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

        self.xmin = np.amin(self.data)
        self.xmax = np.amax(self.data)

        self.n_events = self.data.size

        if self.weights is None:
            print("Analyser object created with", self.n_events, "events in range", self.xmin, "-", self.xmax)
        else:
            print("Analyser object created with", self.n_events, "weighted events in range", self.xmin, "-", self.xmax)


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
            hst = np.histogram(study_data, bins=hbins)
        else:
            hst = np.histogram(self.data, bins=hbins, weights=self.weights)

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

        kde = gaussian_kde(self.data, bw_method="silverman", weights=self.weights)
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
        self.kdey = kde_line * integral
        self.kde = kde

        
    def plot_kde(self):

        self.calculate_kde()
        
        plt.rcParams["figure.figsize"] = (5.75,3.5)

        plt.plot(self.kdex, self.kdey, label='Optimal KDE')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Intensity')
        plt.xlabel('Q (Å$^{-1}$)')
        plt.tight_layout()
        plt.legend()
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
        uvars = self.lse_result.result.uvars
        sigmas =  np.zeros(len(uvars))

        i = 0
        for key in uvars:
            sigmas[i] = uvars[key].std_dev
            i=i+1
        
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
        
