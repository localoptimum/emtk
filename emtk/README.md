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


## Input Knowledge

My PhD work was in SANS, and whilst I am generally happy with the main results (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.227202) with the knowledge I now have I would probably not use least squares regression to fit curves like power laws.  The conclusions wouldn't change, but for rigour I would probably try something like maximum likelihood.  Analysis that I did in R of the COVID case data convinced me that maximum likelihood would probably be a better option (it also convinced me that the swedish medical officer, Anders Tegnel, wasn't actually looking at the data before giving his reassuring news conferences that the covid cases would soon drop - at the time they were perfectly exponential across Europe in countries that were not locking down).

Thomas also suggests mixture models and kernel density estimation may play a role and I'm inclined to agree, it is especially useful in improving the accuracy of classifiers (e.g. some of those in scikit-learn)

Maximum entropy and bayesian methods are another interesting avenue to explore here.  Along these lines, Lucy-Richardson deconvolution has been looked into.

## Project Monitoring
The project is managed by Thomas Rød from DMSC.  The [kanban board is here](https://jira.esss.lu.se/secure/RapidBoard.jspa?projectKey=EMTK&rapidView=1206).


