import numpy as np
from time import time
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import tplquad, dblquad, quad, simps
import yt; yt.enable_parallelism(); print(yt.is_root()); is_root = yt.is_root();
from scipy.special import spherical_jn as spjn
import argparse
from scipy.interpolate import interpn, interp2d
from scipy.optimize import minimize

# Custom Plot Formatting
plt.rcParams['figure.facecolor'] ='white'
plt.rcParams['figure.figsize']   = (12,8)
plt.rcParams['xtick.labelsize']  = 15
plt.rcParams['ytick.labelsize']  = 15
plt.rcParams['axes.titlesize']  = 15
plt.rcParams['axes.labelsize']  = 15

debug = False
debug_factor = 10 if debug else 1


# Define Quijote Cosmology
Om0 = 0.3175
quijote = {'flat': True, 'H0': 67.11, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
cosmology.addCosmology('Quijote', quijote)
quijote = cosmology.setCosmology('Quijote')

# Define Abacus Cosmology
# uses planck18
abacus = cosmology.setCosmology('planck18')

cosmo = abacus

z = 1.025
f = cosmo.Om0**0.6

def mu(k,t=0):
    return np.cos(t)

def TwoEllipsoidCaps(d, t, s, R):
    dmax = (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))
    if d < dmax:
        return (np.pi*(64*R**3 - (24*np.sqrt(2)*d*R**2*np.sqrt(1 + s**3 - (-1 + s**3)*np.cos(2*t)))/s + (np.sqrt(2)*d**3*(1 + s**3 - (-1 + s**3)*np.cos(2*t))**1.5)/s**3))/48.
    else:
        return 0 

def DoubleSphereVolume(R,d):
    if d < 2*R:
        return np.pi/12 * (d - 2*R)**2 * (d + 4*R)
    else:
        return 0
    
# Donghui Jeong Liang Dai, Marc Kamionkowski, and Alexander S. Szalay 2014 (Eq 2, 3)
def XiRSDApprox(r, rmu, z=z, b=1, separateterms=False):

    # Define Legendre Polynomials
    P0 = lambda x: 1
    P2 = lambda x: ( 3*x**2 - 1)/2
    P4 = lambda x: (35*x**4 - 30*x**2 + 3)/8

    # Compute Other Moments
    # First: Power Spectrum
    Pk = lambda k: cosmo.matterPowerSpectrum(k=k, z=z)
    def xin(r, n):
    
        integrand = lambda k: k**2 * Pk(k) / (2 * np.pi**2) * spjn(n, k*r)
        
        # integrate log-grid
        # smaller r's result in larger periods for k --> need wider grid so 1/r dependence
        # r in Mpc/h
        Npts = 10000//debug_factor
        kgrid = np.sort(np.concatenate([np.logspace(-6,0,Npts), np.linspace(1 + 1/Npts, np.maximum(50/r,20), Npts)]))
        integrand_grid = integrand(kgrid)
        result = simps(integrand_grid, kgrid)
        assert not np.isnan(result), "xin result is nan"
        return result

    if not separateterms:
        result = (b*b + 2/3*b*f + f*f/5)*xin(r, 0)*P0(rmu) - (4/3*b*f + 4/7*f*f)*xin(r, 2)*P2(rmu) + 8/35*f*f*xin(r, 4)*P4(rmu)

        return result
    else:
        
        # Only Compute Power Spectrum Moments Once
        xi0 = xin(r, 0)
        xi2 = xin(r, 2)
        xi4 = xin(r, 4)

        # Squared, Linear, and Constant terms (with respect to bias parameter)
        #b2term = b*b*(xi0*P0(rmu))
        #b1term = b*(2/3*f*xi0*P0(rmu) + 4/3*f*xi2*P2(rmu))
        #b0term = f*f/5*xi0*P0(rmu) + 4/7*f*f*xi2*P2(rmu) + 8/35*f*f*xi4*P4(rmu)
        ## without the b's so we can optimize later
        b2term = xi0*P0(rmu)
        b1term = 2/3*f*xi0*P0(rmu) + 4/3*f*xi2*P2(rmu)
        b0term = f*f/5*xi0*P0(rmu) + 4/7*f*f*xi2*P2(rmu) + 8/35*f*f*xi4*P4(rmu)

        return np.array([b2term, b1term, b0term])
    
def XiIntegral(R, s, R1=1e-3, RSD=2, separateterms=True):
    
    # We add a 2*pi here from the azimuthal integral that would need to be done in the following lines
    weight = lambda r, t: 2*np.pi*TwoEllipsoidCaps(r, t, s, R)

    if RSD == 0:
        integrand = lambda r, t: weight(r,t)*cosmo.correlationFunction(r,z) * r*r*np.sin(t)
    elif RSD == 1:
        if is_root:
            print("Using Dipole + Quadruple O(f^2) Approximation")
        if not separateterms:
            integrand = lambda r, t: weight(r,t)*XiRSDApprox(r,np.cos(t)) * r*r*np.sin(t)
        else:
            integrand = lambda r, t: weight(r,t)*XiRSDApprox(r,np.cos(t), separateterms=separateterms) * r*r*np.sin(t)
    elif RSD not in [0,1]:
        assert False, "invalid RSD mode"

    dlims = lambda t: [R1, (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))]

    # 1D Integral, other done w/ simps
    if not separateterms or RSD == 0:
        def new_integrand(t): 
            lims = dlims(t)
            rgrid = np.linspace(lims[0], lims[1], 1000//debug_factor) # Linear Grid
            #rgrid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 1000) # Logarithmic Grid
            result = simps([integrand(rg, t) for rg in rgrid], rgrid)
            assert not np.isnan(result), "new_integrand result is nan"
            return result

        # Simpsons Rule
        thgrid = np.linspace(0, np.pi, 200//debug_factor)
        result = simps([new_integrand(th) for th in thgrid], thgrid)
        return result
    else:
        def new_integrand(t):
            lims = dlims(t)
            rgrid = np.linspace(lims[0], lims[1], 1000//debug_factor) # Linear Grid
            #rgrid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 1000) # Logarithmic Grid
            result = simps([integrand(rg, t) for rg in rgrid], rgrid, axis=0)
            assert not np.isnan(result).all(), "new_integrand result is nan"; assert result.size == 3, "wrong size"
            return result

        # Simpsons Rule
        thgrid = np.linspace(0, np.pi, 100//debug_factor)
        result = simps([new_integrand(th) for th in thgrid], thgrid, axis=0)
        return result


S = [1.0, 2.0, 4.0]
R = np.logspace(np.log10(1),np.log10(62),200)
RSD = 0

params = [(s, r) for s in S for r in R]

xis = {}
for sto, (s, r) in yt.parallel_objects(params, 0, dynamic=True, storage=xis):

    j = params.index((s,r))

    print(f"working on item {j}, which is r={r:.3f} Mpc/h for s={s:.3f}");
    start = time()

    sto.result = XiIntegral(r, s, RSD=RSD, separateterms=True)
    sto.result_id = f"{j}"

    print(f"finished in {time()-start:.2f} seconds; result is {sto.result}")

if yt.is_root():

    bterms = []
    for s in range(len(S)):

        try:
            results = np.array([xis[f"{len(R)*s + j}"] for j in range(len(R))])
        except:
            # Sometimes only works with integer indices instead of strings??
            results = np.array([xis[len(R)*s + j] for j in range(len(R))])

        bterms.append(results)

    np.savez("theory_ellipsoidal.npz", bterms=bterms, S=S, R=R, RSD=np.array(RSD))