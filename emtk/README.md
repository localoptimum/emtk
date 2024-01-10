# emtk - Event Mode Toolkit

## About
This project will produce a set of tools for analysing ESS data, initially in the large scale structures category (SANS, Reflectometry...) that are not covered by the traditional methods such as least squares regression.

It has the following objectives / requirements:

1. To run in event mode, without the need of intermediate steps such as histograms (although perhaps a lightweight filter is needed on the front end to convert events into individual Q points)
2. To extract the maximum possible amount of information out of neutron scattering data, and the closely related next point:
3. To answer the question as to what is the minimum amount of useful data that can be reliably analysed: could useful information be extracted even from a single pulse?
4. To be customisable for new curve types when the need arises
5. Runs in python and jupyter notebooks
6. Is interfaced with other ESS projects

Note that this is an experimental project, it is not in any way some kind of production-ready tool for data analysis.

## Input Knowledge

My PhD work was in SANS, and whilst I am generally happy with the main results (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.227202) with the knowledge I now have I would probably not use least squares regression to fit curves like power laws.  The conclusions wouldn't change, but for rigour I would probably try something like maximum likelihood.  Analysis that I did in R of the COVID case data convinced me that maximum likelihood would probably be a better option (it also convinced me that the swedish medical officer, Anders Tegnel, wasn't actually looking at the data before giving his reassuring news conferences that the covid cases would soon drop - at the time they were perfectly exponential across Europe in countries that were not locking down).

Thomas also suggests mixture models and kernel density estimation may play a role and I'm inclined to agree, it is especially useful in improving the accuracy of classifiers (e.g. some of those in scikit-learn)

Maximum entropy and bayesian methods are another interesting avenue to explore here.  Along these lines, Lucy-Richardson deconvolution has been looked into.

## Project Monitoring
The project is managed by Thomas Rød from DMSC.  The [kanban board is here](https://jira.esss.lu.se/secure/RapidBoard.jspa?projectKey=EMTK&rapidView=1206).

## Prerequisites
The project environment can be recreated by installing the following.
* Anaconda, which you can [get here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
* Optionally, create a conda environment with "conda create --name whatever"
* conda install jupyterlab

Then the following packages have been used for the most recent tests, notebook files etc.
* conda install scikit-learn
* conda install powerlaw
* conda install scipy
* conda install matplotlib
* conda install pymc
* conda install pytorch
* conda install numpyro
* conda install seaborn
* conda install tqdm
* conda install jupyterlab_widgets
* conda install ipywidgets

* pip3 install lmfit

On a mac, I was playing with JAX running on the apple M1 GPU (pip3 install jax-metal) but there are errors and version problems so that's work in progress.  You can also do JAX on CUDA, these speed up the MCMC sampling in PyMC but not the MAP.  You can [read more about it here](https://jax.readthedocs.io/en/latest/installation.html).

## Most Recent Reading

* 

## Packages

* [PyMC](https://www.pymc.io/welcome.html): Currently **fails** due to dependency errors with numpy, possibly coming from legacy issues related to the discontinuation of theano, on which PyMC is based, which is a pity because it has a really clean API and all the scipy distributions.
* [EMCEE](https://emcee.readthedocs.io/en/stable/) Brilliant.  Have to do everything by hand, it's just the sampler.  It's not quite working yet because I don't understand the marginal integration properly and there is a 90% chance I have coded that bit wrong.  It's therefore worth getting another more blackbox distro to do it right, so we can compare the outputs and fix the manual method.
* [Pomegranate](https://pomegranate.readthedocs.io/en/latest/#) (and [its github](https://github.com/jmschrei/pomegranate)): **Best next option**.  Probabilistic programming based on pytorch.  It is the lightest API of all of them, but the distribution library is small.  Setting up a custom distribution has been removed from the documentation of the current version.  This might be the quickest way through the project if a custom distribution can be easily defined as per the old method, so it's what I'm going to do next.
* [Tensorflow Probability](https://www.tensorflow.org/probability): self explanatory.  Second best option.  Relies on JAX also, but other backends are supported.  Also does the marginalisation but it's a bit obscure.
* [Edward](http://edwardlib.org): probabilistic programming library in python.  It looks promising, in that it does the mixture weight marginalisation for you, but the API is a lot more involved than some of the others.  Looks like the scipy distributions are used.

