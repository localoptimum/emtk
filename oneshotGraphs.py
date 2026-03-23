import numpy as np

import matplotlib.pyplot as plt



def p_conv_plot(item, param_name='', Bayes=True, LSE=True, savename='', correct=np.inf, ylims=[np.inf, np.inf]):

    pnam = pnams[item+1]

    wtop = -np.inf
    wbot = np.inf

    xmin = np.inf
    xmax = -np.inf
    
    # Least squares parameters
    if LSE == True:
        lsnevs = n_evs
        lsvalues = overall_lse_results[:,item+1]
        lserrs = overall_lse_errors[:,item+1]

        lsmask = lsvalues != 0.0
        lsnevs = lsnevs[lsmask]
        lsvalues = lsvalues[lsmask]
        lserrs = lserrs[lsmask]
    
        lsbtop = lsvalues+lserrs
        lsbbot = lsvalues-lserrs
        lswtop = np.amax(lsvalues)
        lswbot = np.amin(lsvalues)

        if lswtop > wtop:
            wtop = lswtop

        if lswbot < wbot:
            wbot = lswbot

        if np.amax(lsbtop) > wtop:
            wtop = np.amax(lsbtop)

        if np.amin(lsbbot) < wbot:
            wbot = np.amin(lsbbot)

        if np.amin(lsnevs) < xmin:
            xmin = np.amin(lsnevs)

        if np.amax(lsnevs) > xmax:
            xmax = np.amax(lsnevs)

    # Bayesian parameters

    if Bayes == True:
        mcnevs = n_evs
        mcvalues = overall_mcmc_results[:,item]
        mcerrs = overall_mcmc_errors[:,item]

        mcmask = mcvalues != 0.0
        mcvalues = mcvalues[mcmask]
        mcerrs = mcerrs[mcmask]
        mcnevs = mcnevs[mcmask]
    
    
        mcbtop = mcvalues+mcerrs
        mcbbot = mcvalues-mcerrs
        mcwtop = np.amax(mcvalues)
        mcwbot = np.amin(mcvalues)

        if mcwtop > wtop:
            wtop = mcwtop

        if mcwbot < wbot:
            wbot = mcwbot

        if np.amin(mcnevs) < xmin:
            xmin = np.amin(mcnevs)

        if np.amax(mcnevs) > xmax:
            xmax = np.amax(mcnevs)

    if np.isfinite(correct):
        if correct > wtop:
            wtop = correct

        if correct < wbot:
            wbot = correct
        
    wspan = wtop - wbot
    wpad = wspan * 0.1
    wtop = wtop + np.abs(wpad)
    wbot = wbot - np.abs(wpad)

        
    fig, ax = plt.subplots()
    if LSE:
        ax.plot(lsnevs, lsvalues, color='blue', label='LSE')
        ax.fill_between(lsnevs, lsbbot, lsbtop, color='blue', alpha=0.15, label='LSE std. err')
        ax.set_ylim([wbot, wtop])
        ax.set_xlabel("# events")
        ax.set_ylabel(pnam + "Value")
        if param_name != '':
            ax.set_ylabel(param_name)

    if Bayes:
        ax.plot(mcnevs, mcvalues, color='red', label='Bayes')
        ax.fill_between(mcnevs, mcbbot, mcbtop, color='red', alpha=0.15, label='Bayes std. err')

    if np.isfinite(correct):
        plt.hlines(correct, xmin, xmax, label='Visual estimate', ls='--', color='orange')

    if np.isfinite(ylims).all():
        plt.ylim(ylims)
        
    plt.xscale('log')
    plt.yscale('linear')
    ax.legend()

    if savename != '':
        plt.savefig(savename, dpi=600, bbox_inches='tight')


with open('arcs_parameters.npy', 'rb') as f:
    n_evs = np.load(f)
    pnams = np.load(f)
    overall_lse_results = np.load(f)
    overall_lse_errors = np.load(f) 
    overall_mcmc_results = np.load(f)
    overall_mcmc_errors = np.load(f)




# Elastic Line Convergence
parameter_no = 0
p_conv_plot(parameter_no, param_name='Elastic Line $\mu$ (meV)', Bayes=False, savename='convergence/arcs_elastic_mu.png', correct=0.0)

parameter_no = 7
p_conv_plot(parameter_no, param_name='Elastic $\sigma$ (meV)',  Bayes=False, savename='convergence/arcs_elastic_sigma.png', correct=22.0, ylims=[9, 26])

parameter_no = 14
p_conv_plot(parameter_no, param_name='Fractional Elastic Mixing Amplitude', Bayes=False, savename='convergence/arcs_elastic_amplitude.png')


# Line 1 Convergence

parameter_no = 1
p_conv_plot(parameter_no, param_name='Ex1 $\mu$ (meV)',  Bayes=False,savename='convergence/arcs_ex1_mu.png', correct=140.0)

parameter_no = 8
p_conv_plot(parameter_no, param_name='Ex1 $\sigma$ (meV)',  Bayes=False,savename='convergence/arcs_ex1_sigma.png', correct=21)

parameter_no = 15
p_conv_plot(parameter_no, param_name='Fractional Ex1 Mixing Amplitude',  Bayes=False,savename='convergence/arcs_ex1_amplitude.png')



# Line 2 Convergence

parameter_no = 2
p_conv_plot(parameter_no, param_name='Ex2 $\mu$ (meV)',  Bayes=False, savename='convergence/arcs_ex2_mu.png', correct=277.0, ylims=[270, 286])

parameter_no = 9
p_conv_plot(parameter_no, param_name='Ex2 $\sigma$ (meV)',  Bayes=False, savename='convergence/arcs_ex2_sigma.png', correct=23.8)

parameter_no = 16
p_conv_plot(parameter_no, param_name='Fractional Ex2 Mixing Amplitude',  Bayes=False, savename='convergence/arcs_ex2_amplitude.png')



# Line 3 Convergence

parameter_no = 3
p_conv_plot(parameter_no, param_name='Ex3 $\mu$ (meV)',Bayes=False,  savename='convergence/arcs_ex3_mu.png', correct=413.0)

parameter_no = 10
p_conv_plot(parameter_no, param_name='Ex3 $\sigma$ (meV)', Bayes=False, savename='convergence/arcs_ex3_sigma.png', correct=32.0, ylims=[17, 34])

parameter_no = 17
p_conv_plot(parameter_no, param_name='Fractional Ex3 Mixing Amplitude', Bayes=False, savename='convergence/arcs_ex3_amplitude.png')


# Line 4 Convergence

parameter_no = 4
p_conv_plot(parameter_no, param_name='Ex4 $\mu$ (meV)', Bayes=False, savename='convergence/arcs_ex4_mu.png', correct=561.0, ylims=[543, 575])

parameter_no = 11
p_conv_plot(parameter_no, param_name='Ex4 $\sigma$ (meV)', Bayes=False, savename='convergence/arcs_ex4_sigma.png', correct=44.0)

parameter_no = 18
p_conv_plot(parameter_no, param_name='Fractional Ex4 Mixing Amplitude', Bayes=False, savename='convergence/arcs_ex4_amplitude.png')



# Background 1

parameter_no = 5
p_conv_plot(parameter_no, param_name='BG1 $\mu$ (meV)', Bayes=False, savename='convergence/arcs_BG1_mu.png')

parameter_no = 12
p_conv_plot(parameter_no, param_name='BG1 $\sigma$ (meV)', Bayes=False, savename='convergence/arcs_BG1_sigma.png')

parameter_no = 19
p_conv_plot(parameter_no, param_name='Fractional BG1 Mixing Amplitude', Bayes=False, savename='convergence/arcs_BG1_amplitude.png')



# Background 2

parameter_no = 6
p_conv_plot(parameter_no, param_name='BG2 $\mu$ (meV)', Bayes=False, savename='convergence/arcs_BG2_mu.png')

parameter_no = 13
p_conv_plot(parameter_no, param_name='BG2 $\sigma$ (meV)', Bayes=False, savename='convergence/arcs_BG2_sigma.png')



