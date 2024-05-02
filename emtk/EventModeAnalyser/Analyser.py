"""The EventModeAnalyser object is the main API for EMTK for users to perform event mode data analysis.

import emtk.EventModeAnalyser as ema
"""

import numpy as np
import matplotlib.pyplot as plt


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
        self.histo = None
        self.kde = None

        self.xmin = np.amin(self.data)
        self.xmax = np.amax(self.data)

        self.n_events = self.data.size

        if self.weights is None:
            print("Analyser object created with", self.n_events, "events in range", self.xmin, "-", self.xmax)
        else:
            print("Analyser object created with", self.n_events, "weighted events in range", self.xmin, "-", self.xmax)


            
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

        
            
        
    def plot_histogram(self):
        """ Plots a histogram of the events in the scipp fashion
        """

        if self.data is None:
            raise ValueError(
                f"attempt to plot histogram with no data defined."
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

        x_hist = hst[1]
        x_hist = x_hist[:-1] # eh?
        y_hist = hst[0]
        e_hist = np.sqrt(y_hist)

        plt.rcParams["figure.figsize"] = (5.75,3.5)

        plt.step(x_hist, y_hist, where='post', label='Optimal Histo')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('[numpy histo]')
        plt.tight_layout()
        plt.show()

