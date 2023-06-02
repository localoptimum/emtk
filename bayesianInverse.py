import numpy as np
import matplotlib.pyplot as plt

class bayesianHisto:
    def __init__(self):
        self.nparams=0
        self.estimates = np.array([None])
        self.variances = np.array([None])
        self.datax = np.empty(0)
        self.datay = np.empty(0)

        self.rvals = np.empty(0)
        self.pr = np.empty(0)

        self.psf = None

    def x2(self, yvals, fity):
        '''Basic calculation of chi-squared'''
        diff = yvals-fity
        diff2= diff**2.0
        frac = diff2/fity
        x2 = np.sum(frac)
        return x2

    def getfity(self, uj, pij):
        fity = pij.dot(uj)
        return(fity)


    def errorBarCalculation(self, datay, pij, uj):
        # We just figure out how far the individual bin moves
        # in order to increase x2 by one.
        # First order estimate overestimates the error bars significantly.
        # In least squares regression I think it's a parabolic surface
        # strictly speaking, so we should do that.
    
        testuj = np.copy(uj)
        testujs= np.sum(testuj)
        testuj = testuj / testujs
        
        fity = self.getfity(testuj, pij)
        sumf = np.sum(fity)
        sumd = np.sum(datay)
        fity = fity * sumd/sumf
        
        y1 = self.x2(datay, fity) #/ datay.size
        
        dx = np.zeros_like(uj)
        errors = np.zeros_like(uj)
        
        for i in range(uj.size):
            testuj = np.copy(uj)
            if testuj[i] == 0.0:
                delta = 0.001
            else:
                delta = testuj[i]/1000.0
            
            xx1 = testuj[i]
            xx2 = xx1 + delta
            xx3 = xx2 + delta
            
            testuj[i] = xx2
        
            #testujs= np.sum(testuj)
            #testuj = testuj / testujs
        
            fity = self.getfity(testuj, pij)
            sumf = np.sum(fity)
            sumd = np.sum(datay)
            fity = fity * sumd/sumf
        
            y2 = self.x2(datay, fity) #/ datay.size
        
            dx[i] = xx2-xx1

            dxdu = np.abs((xx2-xx1) / delta)
        
            errors[i] = 1.0 / dxdu
        
        #    print("dx", dx)
        #    print("errors", errors)
        
        return errors
            
    def LR_deconv(self, niter =100, calcErrors=False):

        if self.psf is None :
            self.psf, self.rvals = self.calc_psf()

        di = self.datay
        pij = self.psf
               
        uj = np.ones(pij.shape[1])
    
        #print("di is", di.shape)
        #print("pij is", pij.shape)
        #print("uj is", uj.shape)
        uj = uj / uj.size
        fity = self.getfity(uj, pij)
        
        sumf = np.sum(fity)
        sumd = np.sum(self.datay)
        fity = fity * sumd/sumf
        
        chisq = self.x2(self.datay, fity)/self.datay.size
    
        #print("x2=", chisq)
    
        converged = False
        iterdone = 0
    
        for _ in range(niter):
    
            #ci = np.sum(pij.dot(uj))
            ci = pij.dot(uj)
        
            #print("ci:", ci)
    
            dici = (di/ci)
        
            pijt = np.transpose(pij)
    
            mult = pijt.dot(dici)
    
            ujp = uj * mult
        
            uj = np.copy(ujp)
        
            testuj = np.copy(uj)
            testujs= np.sum(testuj)
            testuj = testuj / testujs
            
            fity = self.getfity(testuj, pij)
            sumf = np.sum(fity)
            sumd = np.sum(self.datay)
            fity = fity * sumd/sumf
        
            chisq = self.x2(self.datay, fity) / self.datay.size
        
            #print("x2=", chisq)
        
            iterdone = iterdone + 1
        
            if(chisq <= 1.0):
               converged = True
               break
            else:
               continue
        
            #print(uj[0])
        
        # Maybe now we need to calculate errors?
        if calcErrors:
            errors = self.errorBarCalculation(self.datay, pij, uj)    
    
        
        ujs = np.sum(uj)
        uj = uj / ujs
    
        if calcErrors:
            errors = errors / ujs
        
        if converged:
            print("Lucy-Richardson fit converged after", iterdone, "iterations.")
        else:
            print("Lucy-Richardson fit failed to converge after", iterdone, "iterations.")

        print("x2=", chisq)
        
        if calcErrors:
            return uj, errors
        else:
            return uj


class bayesianLorentzian(bayesianHisto):

    def lorentzian(self, qq, kappa):
        lr = (kappa / np.pi) / (np.power(qq, 2.0) + np.power(kappa, 2.0))
        return lr
        
    
    def calc_psf(self):
        #datax = data[:,0]
        #datay = data[:,1]
        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.datax), np.amax(self.datax) ])
        rrange = 1.0/xrange
        rrange = np.round(np.flip(rrange))
        #print(xrange)
        #print(rrange)
        rvals = np.arange(rrange[0], 2.0*rrange[1], 1.0)
        nx = self.datax.size
        nr = rvals.size
    
        #print(nx)
        #print(nr)
        
        pmatrix = np.zeros((nx, nr))
    
        for i in range(nx):
            for j in range(nr):
                pmatrix[i,j] = pmatrix[i,j] + self.lorentzian(self.datax[i], 1.0/rvals[j])
        return pmatrix, rvals

            
    def infer(self, plotk=False, plotr=False):

        # Bayesian inference

        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.datax), np.amax(self.datax) ])

        #print(xrange)
        
        #logmean = np.mean(np.log10( xrange)) 
        #loghw = 0.5*np.std(np.log10(xrange))
        #rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/xrange
        
        #rrange = np.round(rrange)
        rrange = np.round(np.flip(rrange))
        #print(rrange)
        
        self.rvals = np.arange(rrange[0], rrange[1], 1.0)


        #print(self.rvals)
        
        revr = np.flip(self.rvals)

        #print(self.datax)

        kappas = 1.0 / revr


        self.kappas = kappas
        
        # Now setup prior

        prior = np.full_like(kappas, 1.0/kappas.size)
        posterior = np.full_like(kappas, 1.0/kappas.size)
        
        #print(kappas)

        kappas2 = kappas**2.0

        # flip into log space - combining probabilities is then a sum, powers
        # become multiplication
        # This is pretty much how you have to do it with a histogram because
        # you can't apply each neutron one by one, but each bar one by one
        prior = np.log10(prior)
        posterior = np.log10(posterior)

        for bin in np.arange(0, self.datax.size, 1):
            lr = (kappas / np.pi) / (self.datax[bin]**2.0 + kappas2)
            posterior = prior + self.datay[bin] * np.log10(lr)
            prior = np.copy(posterior)

        # flip back out of log space
        #print(posterior)
        posterior = posterior - np.amax(posterior)
        #print(posterior)
        posterior = 10**posterior

        # normalise probability curve (WARNING - assumes the whole curve has been sampled!
        total = np.sum(posterior)
        posterior = posterior / total

        self.posterior = posterior
        
        centre =  np.sum( posterior * kappas)

        diffs = (kappas - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 

        revpost = np.flip(posterior)
        self.pr = revpost
        
        if plotr:
            fig, ax = plt.subplots()
            ax.plot(self.rvals, self.pr)
            ax.set_xlabel('Correlation Length (Angstroms)')
            ax.set_ylabel('Inferred probability')

        if plotk:
            fig, ax = plt.subplots()
            ax.plot(kappas, posterior)
            ax.set_xlabel('Kappa (inverse Angstroms)')
            ax.set_ylabel('Inferred probability')
            
        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates

    
    def plotfit(self):
        errors = np.sqrt(self.datay)

        fig, ax = plt.subplots()

        plt.errorbar(self.datax, self.datay, errors, fmt='o', mfc='none')
        plt.xscale('log')
        plt.yscale('log')

        fity = np.full_like(self.datax, 0.0)


        kappas = self.kappas
        kappas2 = self.kappas**2.0
        x2 = self.datax**2.0
        
        for kk in np.arange(0, self.kappas.size, 1):
            lor = (kappas[kk] / np.pi) / (x2 + kappas2[kk])
            fity = fity + lor*self.posterior[kk]

        # normalise
        datsum = np.sum(self.datay)
        linsum = np.sum(fity)

        scale = datsum / linsum

        fity = fity * scale

        #print(self.datax)
        #print(fity)

            
        plt.plot(self.datax, fity, color='black')
        ax.set_xlabel('Q (Å-1)')
        ax.set_ylabel('Intensity')



class bayesianSpheres(bayesianHisto):
        
    def infer(self, plotr=False):

        # Bayesian inference

        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.datax), np.amax(self.datax) ])

        #print(xrange)
        
        #logmean = np.mean(np.log10( xrange)) 
        #loghw = 0.5*np.std(np.log10(xrange))
        #rrange = np.array([10**(logmean-loghw), 10**(logmean+loghw)])
        rrange = 1.0/xrange
        
        #rrange = np.round(rrange)
        rrange = np.round(np.flip(rrange))
        #print(rrange)
        
        self.rvals = np.arange(rrange[0], rrange[1], 1.0)

        self.rvals = np.arange(1.0, 500.0, 5.0)

        #print(self.rvals)
        # Now setup prior

        prior = np.full_like(self.rvals, 1.0/self.rvals.size)
        posterior = np.full_like(self.rvals, 1.0/self.rvals.size)
        

        # flip into log space - combining probabilities is then a sum
        prior = np.log10(prior)
        posterior = np.log10(posterior)

        # Weighting is applied in the summing stage, not inside the log XD
        for bin in np.arange(0, self.datax.size, 1):
            xr = self.datax[bin] * self.rvals
            xr6= xr ** 6.0
            # REMOVE factor of 15, it's doing no good and we are going to scale everything anyway?
            # hrd = 15.0 * self.rvals * ( np ...
            hrd =  15.0 * self.rvals * ( np.sin(xr) - xr * np.cos(xr))**2.0  /  xr6
            #sm = np.sum(hrd)
            posterior = prior + self.datay[bin] * np.log10(hrd)# / sm
            prior = np.copy(posterior)

        # flip back out of log space
        #print(posterior)
        posterior = posterior - np.amax(posterior)
        #print(posterior)


        posterior = pow(10.0, posterior)
        #print(posterior)

        # normalise probability curve (WARNING - assumes the whole curve has been sampled!
        #total = np.sum(posterior)
        #posterior = posterior / total
        
        centre =  np.sum( posterior * self.rvals)

        diffs = (self.rvals - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 

        self.pr = posterior
        
        if plotr:
            fig, ax = plt.subplots()
            ax.plot(self.rvals, self.pr)
            ax.set_xlabel('Particle Size (Angstroms)')
            ax.set_ylabel('Inferred probability')
            
        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates


    
    def plotfit(self):
        fig = plt.plot(self.datax, self.datay, 'bs')
        plt.xscale('log')
        plt.yscale('log')

        fity = np.full_like(self.datax, 0.0)

        for rr in np.arange(0, self.rvals.size, 1):
            xr = self.datax * self.rvals[rr]
            xr6= xr ** 6.0
            hrd = 15.0 * rr * ( np.sin(xr) - xr * np.cos(xr))**2.0  / (2.0 * np.pi * xr6)
            fity = fity + hrd*self.pr[rr]

        # normalise
        datsum = np.sum(self.datay)
        linsum = np.sum(fity)

        scale = datsum / linsum

        fity = fity * scale

        #print(self.datax)
        #print(fity)

            
        plt.plot(self.datax, fity)




        

class bayesianGuinier(bayesianHisto):

    def guinier(self, qq, rr):
            return np.exp(-qq*qq*rr*rr / 3.0)

    def calc_psf(self):
        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.datax), np.amax(self.datax) ])
        rrange = 1.0/xrange
        rrange = np.round(np.flip(rrange))
        #print(xrange)
        #print(rrange)
        rvals = np.arange(rrange[0], 2.0*rrange[1], 1.0)
        nx = self.datax.size
        nr = rvals.size
        
        pmatrix = np.zeros((nx, nr))
    
        for i in range(nx):
            for j in range(nr):
               pmatrix[i,j] = pmatrix[i,j] + self.guinier(self.datax[i], 1.0/rvals[j])
        return pmatrix, rvals
        
    def infer(self, plotr=False):

        # Bayesian inference

        # First figure out range of kappa values
        xrange = np.array( [ np.amin(self.datax), np.amax(self.datax) ])

        rrange = 1.0/xrange
        
        #rrange = np.round(rrange)
        rrange = np.round(np.flip(rrange))
        print(rrange)
        
        self.rvals = np.arange(0.0, rrange[1], 1.0)

        #print(self.rvals)
        # Now setup prior

        prior = np.full_like(self.rvals, 1.0/self.rvals.size)
        posterior = np.full_like(self.rvals, 1.0/self.rvals.size)


        # flip into log space - combining probabilities is then a sum
        prior = np.log10(prior)
        posterior = np.log10(posterior)

        # Weighting is applied in the summing stage, not inside the log XD
        for bin in np.arange(0, self.datax.size, 1):
            xr2 = self.datax[bin] * self.datax[bin] * self.rvals * self.rvals
            gnr = np.exp(-xr2 / 3.0)
            posterior = prior + self.datay[bin] * np.log10(gnr)# / sm
            prior = np.copy(posterior)

        # flip back out of log space
        #print(posterior)
        posterior = posterior - np.amax(posterior)
        posterior = pow(10.0, posterior)

        # normalise probability curve (WARNING - assumes the whole curve has been sampled!
        total = np.sum(posterior)
        posterior = posterior / total

        #print(posterior)
        
        centre =  np.sum( posterior * self.rvals)

        diffs = (self.rvals - centre)**2.0
        stddev = np.sqrt( np.sum(diffs*posterior)) 

        self.pr = posterior
        
        if plotr:
            fig, ax = plt.subplots()
            ax.plot(self.rvals, self.pr)
            ax.set_xlabel('Particle Size (Angstroms)')
            ax.set_ylabel('Inferred probability')
            
        self.estimates = np.array([centre])
        self.variances = np.array([stddev])
        
        self.method = 'Bayesian inference'

        return self.estimates


    
    def plotfit(self):
        x2 = self.datax ** 2.0
        logy = np.log(self.datay)
        fig,ax = plt.subplots()
        ax.plot(x2, logy, 'bs')
        #plt.xscale('log')
        #plt.yscale('log')

        fity = np.full_like(self.datax, 0.0)

        for rr in np.arange(0, self.rvals.size, 1):
            xr = self.datax * self.rvals[rr]
            xr2= xr * xr
            gnr = np.exp(-xr2 / 3.0)
            fity = fity + gnr*self.pr[rr]

        # normalise
        datsum = np.sum(self.datay)
        linsum = np.sum(fity)

        scale = datsum / linsum

        fity = fity * scale

        #print(self.datax)
        #print(fity)
        logfit = np.log(fity)
            
        ax.plot(x2, logfit, color='black')
        ax.set_xlabel('Q2')
        ax.set_ylabel('Log(I)')
        ax.set_title('Guinier Plot of Truncated Data')
