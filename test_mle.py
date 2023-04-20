
import emtk
import numpy as np
import pytest

def test_gauss():
    gauss = emtk.gaussianCurve()

    params = np.array([5.0, 1.0])

    # Test the inversion of CDF and Quantile functions
    
    xx = 2.0

    pp = gauss.cdf(params, xx)

    revx = gauss.Quantile(params, pp)

    assert xx == pytest.approx(revx)

    
    # Create a small test distribution and fit it analytically, making
    # sure that the answers are roughly right

    xrg = np.array([0.0, 10.0])
    
    gauss.generateTestSamples(params, xrg, 1000)
    gauss.mleAnalytic()

    diffs = np.absolute(gauss.estimates - params)
    tolerances = np.full_like(diffs, 0.1)
    
    np.testing.assert_array_less(diffs, tolerances)

    # Fit the same numerically, and same checks

    gauss.mle()
    diffs = np.absolute(gauss.estimates - params)
    tolerances = np.full_like(diffs, 0.1)
    
    np.testing.assert_array_less(diffs, tolerances)

    

    
def test_lorentzian():
    loren = emtk.lorentzianCurve()
    
    clength = 90.0 # correlation length in system
    kappa = 1.0 / clength
    params = np.array([kappa])
    xrange = np.array([0.001, 0.1])

    # Test the inversion of CDF and Quantile functions
    xx = 0.01
    pp = loren.cdf(params, xx)
    revx = loren.Quantile(params, pp)

    assert xx == pytest.approx(revx)


    # Create a test distribution and fit it
    loren.generateTestSamples(params, xrange, 2000)
    loren.mle()

    diffs = np.absolute(loren.estimates - params)
    tolerances = np.full_like(diffs, 0.1)
    np.testing.assert_array_less(diffs, tolerances)
    


    
def test_hardSpheres():
    # This can be a little unstable sometimes...!  it has more samples
    # than the lorentzian for that reason
    sph = emtk.hardSphereCurve()
    params = np.array([75.0])
    xrange = np.array([0.001, 0.1])


    # Test the inversion of CDF and Quantile functions
    xx = 0.01
    pp = sph.cdf(params, xx)
    revx = sph.Quantile(params, pp)

    assert xx == pytest.approx(revx)


    # Create a test distribution and fit it
    sph.generateTestSamples(params, xrange, 4000)
    sph.guesses = np.array([80.0])
    sph.mle()

    diffs = np.absolute(sph.estimates - params)
    tolerances = np.full_like(diffs, 5.0)
    np.testing.assert_array_less(diffs, tolerances)

    
def test_porod():
    # This one can also be occasionally unstable I increased the
    # number of samples to quite a bit more than the other curves
    curv = emtk.porodCurve()
    pvalues = np.array([4.0, 0.001])
    xrange = np.array([0.001, 0.01])

    # Test the inversion of CDF and Quantile functions
    xx = 0.01
    pp = curv.cdf(pvalues, xx)
    revx = curv.Quantile(pvalues, pp)

    assert xx == pytest.approx(revx)
    
    # Create a test distribution and fit it
    
    curv.generateTestSamples(pvalues, xrange, 6000)
    curv.guesses = np.array([3.6, 0.005])
    curv.mle()
    
    diffs = np.absolute(curv.estimates - pvalues)
    tolerances = np.array([0.2, 1.0E-04])
    np.testing.assert_array_less(diffs, tolerances)
