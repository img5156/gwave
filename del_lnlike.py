#!/usr/local/bin/python
import emcee
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import signal
import scipy.optimize as opt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as pl
import corner
from matplotlib.ticker import MaxNLocator
from emcee.utils import MPIPool
import argparse
import scipy.interpolate as si
import pickle
#import time

# provide sample waveform model
from waveform import *
# routines for binning, summary data, etc.
from binning import *

# ---- sample python code for:
#
#   processing GW170817 data
#   constructing frequency binning
#   prepare summary data
#   compute likelihood using summary data

# number of detectors (Livingston and Hanford)
ndtct = 1


# Since loading large data file is actually slow, we could cheat here by loading pre-computed data
# This is only needed for testing
#_, LFT, psd_L = pickle.load(open('GW170817_L1_f_LFT_psd_taper.pckl','rb'), encoding='latin1')
#_, HFT, psd_H = pickle.load(open('GW170817_H1_f_HFT_psd_taper.pckl','rb'), encoding='latin1')

# To start from scratch, one should download the data from the LIGO website
# and put the data file in the data directory

# We use 2048 seconds of strain data at sampling rate of 4096 Hz (noise subtracted)
n_sample = 2**23
T = 2048.0
n_conv = 20

# load LIGO strain data (time domain)
L1 = np.loadtxt('data/L-L1_LOSC_CLN_4_V1-1187007040-2048.txt')
H1 = np.loadtxt('data/H-H1_LOSC_CLN_4_V1-1187007040-2048.txt')

print('Finished loading LIGO data.')

# time-domain and frequency-domain grid
t = np.linspace(0, T, n_sample+1)[:-1]
f = np.linspace(0, 1.0/T*n_sample/2.0, n_sample//2+1)

# apply a Tukey window function to eliminate the time-domain boundary ringing
tukey = sp.signal.tukey(n_sample, alpha=0.1)
LFT = np.fft.rfft(L1*tukey)/n_sample
HFT = np.fft.rfft(H1*tukey)/n_sample

# estimate PSDs for both L1 and H1
psd_L = 2.0*np.convolve(np.absolute(LFT)**2, np.ones((n_conv))/n_conv, mode='same')*T
psd_H = 2.0*np.convolve(np.absolute(HFT)**2, np.ones((n_conv))/n_conv, mode='same')*T
psd = [psd_L, psd_H]

# frequency domain data
sFT = [LFT, HFT]

print('Finished estimating the PSD.')

# -------- parameters used to generate the fiducial waveform ----------- #
# change the values if needed
# note that once summary data is constructed and efficient likelihood evaluation is available, these can be adjusted to (nearly) best-fit parameters
# we assume D_eff = 1.0 [Mpc] and phi_c = 0.0 for the fiducial waveforms at both detectors; values do not matter because we analytically maximize with respect them.
# we provide (approximate merger times); in practice, they can be easily found using slicing matched filter with FFT
# we neglect in-plane spin components and the asymmetric tidal deformation parameter

MC = 1.1976                                      # detector frame chirp mass [Msun]
ETA = 0.244                                      # symmetric mass ratio m1*m2/(m1 + m2)**2
DELTA = np.sqrt(1.0 - 4.0*ETA)                   # asymmetric mass ratio (m1 - m2)/(m1 + m2)
M = MC/ETA**0.6                                  # total mass [Msun]
M1 = 0.5*M*(1.0 + DELTA)                        # primary mass [Msun]
M2 = 0.5*M*(1.0 - DELTA)                        # second mass [Msun]
S1Z = 0.0                                        # aligned spin component for the primary
S2Z = 0.0                                        # aligned spin component for the secondary
CHIA = 0.5*(S1Z - S2Z)
CHIS = 0.5*(S1Z + S2Z)
CHIEFF = CHIS + DELTA*CHIA
LAM = 0.0                                      # reduced tidal deformation parameter
TC1 = -205.5556                                  # merger time (L1)
TC2 = -205.5521                                  # merger time (H1)

# allowed bounds for parameters
# change or further refine if desired
Mc_bounds = [1.1973, 1.1979]
eta_bounds = [0.2, 0.24999]
chieff_bounds = [-0.2, 0.2]
chia_bounds = [-0.999, 0.999]
Lam_bounds = [0.0, 1000.0]
dtc_bounds = [-0.005, 0.005]
par_bounds = [Mc_bounds, eta_bounds, chieff_bounds, chia_bounds, Lam_bounds] + [dtc_bounds for k in range(ndtct)]

#Average value for all the bounds
Mc_avg = 0.5*(Mc_bounds[0]+Mc_bounds[1])
eta_avg = 0.5*(eta_bounds[0]+eta_bounds[1])
chieff_avg = 0.5*(chieff_bounds[0]+chieff_bounds[1])
chia_avg = 0.5*(chia_bounds[0]+chia_bounds[1])
Lam_avg = 0.5*(Lam_bounds[0]+Lam_bounds[1])
tc1_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])
tc2_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])

# fiducial waveforms sampled at full frequency resolution
h0_L = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h0_H = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H*np.exp(-2.0j*np.pi*f*TC2)]

print('Constructed fiducial waveforms.')

# prepare frequency binning
# range of frequency to be used in the computation of likelihood [f_lo, f_hi] [Hz]
f_lo = 23.0
f_hi = 1000.0

Nbin, fbin, fbin_ind = setup_bins(f_full=f, f_lo=f_lo, f_hi=f_hi, chi=1.0, eps=0.5)
#print(Nbin, fbin, fbin_ind)

print("Frequency binning done: # of bins = %d"%(Nbin))

# next prepare summary data
sdat = compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0)

print("Prepared summary data.")

# find (nearly) best-fit parameters by maximizing the likelihood
par_bf = get_best_fit(sdat, par_bounds, h0_0, fbin, fbin_ind, ndtct, atol=1e-10, verbose=True)

# update the best-fit parameters
MC = par_bf[0]                                       # detector frame chirp mass [Msun]
ETA = par_bf[1]                                      # symmetric mass ratio m1*m2/(m1 + m2)**2
DELTA = np.sqrt(1.0 - 4.0*ETA)                       # asymmetric mass ratio (m1 - m2)/(m1 + m2)
M = MC/ETA**0.6                                      # total mass [Msun]
M1 = 0.5*M*(1.0 + DELTA)                        # primary mass [Msun]
M2 = 0.5*M*(1.0 - DELTA)                        # second mass [Msun]
CHIEFF = par_bf[2]
CHIA = par_bf[3]
CHIS = CHIEFF - DELTA*CHIA
S1Z = CHIS + CHIA                                        # aligned spin component for the primary
S2Z = CHIS - CHIA                                # aligned spin component for the secondary
LAM = par_bf[4]                                      # reduced tidal deformation parameter
TC1 += par_bf[5]                                  # merger time (L1)
#TC2 += par_bf[6]                                  # merger time (H1)

print('Updated parameters for the fiducial waveform')

# now update fiducial waveforms
h0_L = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h0_H = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
#h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H*np.exp(-2.0j*np.pi*f*TC2)]

print('Updated fiducial waveforms.')

# next prepare summary data
sdat = compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0_0)

print('Updated summary data.')


def lnlikelihood(Mc, eta, chieff, chia, lam, tc1):
    par_best = [Mc, eta, chieff, chia, lam, tc1]
    return(-lnlike(par_best, sdat, h0_0, fbin, fbin_ind, ndtct))

# Uniform prior on all parameter in their respective range
def lnprior(Mc, eta, chieff, chia, lam, tc1):
    if 1.1973<Mc<1.1979 and 0.2<eta<0.24999 and -0.2<chieff<0.2 and -0.999<chia<0.999 and 0<lam<1000 and -0.005<tc1<0.005:
        l = 0.0
    else:
        l =-np.inf
    return l


# Multiplying likelihood with prior
def lnprob(Mc, eta, chieff, chia, lam, tc1):
	lp = lnprior(Mc, eta, chieff, chia, lam, tc1)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlikelihood(Mc, eta, chieff, chia, lam, tc1)


# Defining a function just for minimization routine to find a point to start
def func(theta):
	Mc, eta, chieff, chia, lam, tc1, tc2 = theta
	return -2.*lnprob(Mc, eta, chieff, chia, lam, tc1)

def lnp(theta):
	Mc, eta, chieff, chia, lam, tc1= theta
	return lnprob(Mc, eta, chieff, chia, lam, tc1)


#result = opt.minimize(func, [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg])
result = [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg]


#start_time = time.perf_counter()
#print("Started time.")
# Set up the sampler.
print("Setting up sampler for binning algorithm.")
ndim, nwalkers = 6, 20
pos = [result + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos

# Clear and run the production chain.
print("Running MCMC for binning algorithm...")
sampler.run_mcmc(pos, 50)
#print (pos)
print("Done.")


# Removing first 100 points as chain takes some time to stabilize
burnin = 10
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
print("Saving data in file.")
np.savetxt("test_bin_del_lnlike_005k_1w.dat",samples,fmt='%f',  header="Mc eta chieff chia lam tc1 tc2")

#print("--- %s seconds ---" % (time.perf_counter() - start_time))

print("Loading the saved data.")
pars = np.loadtxt('test_bin_del_lnlike_005k_1w.dat')

Mc = pars[:, 0]
eta = pars[:, 1]
chieff = pars[:, 2]
chia = pars[:, 3]
lam = pars[:, 4]
tc1 = pars[:, 5]
#tc2 = pars[:, 6]

lnlk_bin = np.zeros(len(Mc))

for i in range(len(Mc)):
    lnlk_bin[i] = lnlikelihood(Mc[i], eta[i], chieff[i], chia[i], lam[i], tc1[i])

print("Created lnlikelihood array using binning.")
df = 1./128./4.
#def sh(f):
#    s = 5.623746655206207e-51 + 6.698419551167371e-50*f**(-0.125) + 7.805894950092525e-31/f**20. + 4.35400984981997e-43/f**6. + 1.630362085130558e-53*f + 2.445543127695837e-56*f**2 + 5.456680257125753e-66*f**5
#    return s

#Calculating the innerproduct
def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/psd).sum()))*df
    return summ

# Define log likelihood
def lnlike_real(Mc, eta, chieff, chia, lam, tc1):
    M = Mc / eta ** 0.6
    delta = np.sqrt(1.0 - 4.0 * eta)
    chis = chieff - delta * chia
    s1Z = chis + chia
    s2Z = chis - chia
    h1_L1 = hf3hPN(f, M, eta, s1z=s1Z, s2z=s2Z, Lam=lam)
    h1_L = h1_L*np.exp(-2.0j*np.pi*f*tc1)
    #h1_H = hf3hPN(f, M, eta, s1z=s1z, s2z=s2z, Lam=lam)
    # these are NOT shifted to the right merger times
    #h1_1 = np.append(h1_L, h1_H)
    # these are shifted to the right merger times
    #sFT = np.append(LFT, HFT)
    #h1 = np.append(h1_L*np.exp(-2.0j*np.pi*f*tc1), h1_H*np.exp(-2.0j*np.pi*f*tc2))
    #print(len(np.asarray(h1_L)), len(np.asarray(LFT)), len(psd_L))
    #print(np.amax(np.asarray(h1_L)), np.amax(np.asarray(LFT)), np.amax(np.asarray(psd_L)))
    return -0.5*overlap(np.asarray(LFT)-np.asarray(h1_L), np.asarray(LFT)-np.asarray(h1_L), f)


def lnprob_real(Mc, eta, chieff, chia, lam, tc1):
	lp = lnprior(Mc, eta, chieff, chia, lam, tc1)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real(Mc, eta, chieff, chia, lam, tc1)

def lnp_real(theta):
	Mc, eta, chieff, chia, lam, tc1 = theta
	return lnprob(Mc, eta, chieff, chia, lam, tc1)

result2 = [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg]

print("Setting up sampler for waveform.")
ndim, nwalkers = 6, 20
pos = [result2 + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnp_real)
#print pos

# Clear and run the production chain.
print("Running MCMC for waveform...")
sampler2.run_mcmc(pos, 50)
#print (pos)
print("Done.")

burnin = 10
samples2 = sampler2.chain[:, burnin:, :].reshape((-1, ndim))
# saving data in file
np.savetxt("test_real_del_lnlike_005k_1w.dat",samples2,fmt='%f',  header="Mc eta chieff chia lam tc1 tc2")

print("Loading the saved data.")
pars_real = np.loadtxt('test_real_del_lnlike_005k_1w.dat')

Mc_real = pars_real[:, 0]
eta_real = pars_real[:, 1]
chieff_real = pars_real[:, 2]
chia_real = pars_real[:, 3]
lam_real = pars_real[:, 4]
tc1_real = pars_real[:, 5]
#tc2_real = pars_real[:, 6]

lnlk_real = np.zeros(len(Mc_real))

print("Creating likelihood array for waveform.")

for i in range(len(Mc_real)):
    lnlk_real[i] = lnlike_real(Mc_real[i], eta_real[i], chieff_real[i], chia_real[i], lam_real[i], tc1_real[i])

print("Created lnlikelihood array using waveform.")

del_lnlk = np.zeros(len(lnlk_bin))

for i in range(len(lnlk_bin)):
    del_lnlk[i] = abs(abs(lnlk_bin[i])-abs(lnlk_real[i]))

print("Created del_lnlikelihood array.")
pl.scatter(lnlk_bin, del_lnlk)
pl.savefig('plot_test_del_lnlike_005k_02w.pdf')
pl.close()
