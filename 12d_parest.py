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
#import pickle
#import time

# provide sample waveform model
from waveform_rb import *
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
#TC2 = -205.5521                                   # merger time (H1)
THETA = -0.4                                      #declination
PSI = np.pi
PHI = 3.44
DL = 40 
I = np.pi/6
#TC = 0.0
PHI_C = 0.0

# allowed bounds for parameters
# change or further refine if desired
Mc_bounds = [1.1973, 1.1979]
eta_bounds = [0.2, 0.24999]
chieff_bounds = [-0.2, 0.2]
chia_bounds = [-0.999, 0.999]
Lam_bounds = [0.0, 1000.0]
dtc_bounds = [-0.005, 0.005]
theta_bounds = [-np.pi/2, np.pi/2]
psi_bounds = [0, 2*np.pi]
phi_bounds = [0, 2*np.pi]
Dl_bounds = [10, 200]
i_bounds = [0, np.pi]
#tc_bounds = [-5, 5]
phi_c_bounds = [-np.pi, np.pi]
par_bounds = [Mc_bounds, eta_bounds, chieff_bounds, chia_bounds, Lam_bounds, theta_bounds, psi_bounds, phi_bounds, Dl_bounds, i_bounds, phi_c_bounds] + [dtc_bounds for k in range(ndtct)]

#Average value for all the bounds
Mc_avg = 0.5*(Mc_bounds[0]+Mc_bounds[1])
eta_avg = 0.5*(eta_bounds[0]+eta_bounds[1])
chieff_avg = 0.5*(chieff_bounds[0]+chieff_bounds[1])
chia_avg = 0.5*(chia_bounds[0]+chia_bounds[1])
Lam_avg = 0.5*(Lam_bounds[0]+Lam_bounds[1])
tc1_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])
#tc2_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])
theta_avg = 0.5*(theta_bounds[0]+theta_bounds[1])
phi_avg = 0.5*(phi_bounds[0]+phi_bounds[1])
psi_avg = 0.5*(psi_bounds[0]+psi_bounds[1])
Dl_avg = 0.5*(Dl_bounds[0]+Dl_bounds[1])
i_avg = 0.5*(i_bounds[0]+i_bounds[1])
#tc_avg = 0.5*(tc_bounds[0]+tc_bounds[1])
phi_c_avg = 0.5*(phi_c_bounds[0]+phi_c_bounds[1])

# fiducial waveforms sampled at full frequency resolution
h0_L = hf3hPN_L(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM, THETA, PSI, PHI, DL, I, PHI_C)
h0_H = hf3hPN_H(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM, THETA, PSI, PHI, DL, I, PHI_C)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H]

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
LAM = par_bf[4]
THETA = par_bf[5]
PSI = par_bf[6]
PHI = par_bf[7]
DL = par_bf[8]
I = par_bf[9]
PHI_C = par_bf[10]
TC1 += par_bf[11]                                  # merger time (L1)
#TC2 += par_bf[13]                                  # merger time (H1)

print('Updated parameters for the fiducial waveform')

# now update fiducial waveforms
h0_L = hf3hPN_L(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM, THETA, PSI, PHI, DL, I, PHI_C)
h0_H = hf3hPN_H(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM, THETA, PSI, PHI, DL, I, PHI_C)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H]

print('Updated fiducial waveforms.')

# next prepare summary data
sdat = compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0)

print('Updated summary data.')
#print(sdat)

# example of likelihood evaluation: check the likelihood of the new fiducial waveform
#par_best = [MC, ETA, CHIEFF, CHIA, LAM, 0.0, 0.0]
#print(-lnlike(par_best, sdat, h0_0, fbin, fbin_ind, ndtct))

def lnlikelihood(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1):
    par_best = [Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1]
    return(-lnlike(par_best, sdat, h0_0, fbin, fbin_ind, ndtct))

# Uniform prior on all parameter in their respective range
def lnprior(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1):
    if 1.1973<Mc<1.1979 and 0.2<eta<0.24999 and -0.2<chieff<0.2 and -0.999<chia<0.999 and 0<lam<1000 and -0.005<tc1<0.005 and -np.pi/2<theta<np.pi/2 and 0.0<psi<2.0*np.pi and 0.0<phi<2.0*np.pi and 10<Dl<200 and 0<i<np.pi and -np.pi<phi_c<np.pi:
        l = 0.0
    else:
        l =-np.inf
    return l

#Calculating the innerproduct
#def overlap(A, B, f):
#    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/sh(f)).sum()))*df
#    return summ

# Define log likelihood
#def lnlike(lnA, tc, phic, Mc, eta, e2):
#    h = htilde(lnA, tc, phic, Mc, eta, mu2, mu3, mu4, mu5, e2, e3, e4, ff)
#    print(-0.5*overlap(Data-h, Data-h, ff))

# Multiplying likelihood with prior
def lnprob(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1):
	lp = lnprior(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlikelihood(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1)


# Defining a function just for minimization routine to find a point to start
def func(theta):
	Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1 = theta
	return -2.*lnprob(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1)

def lnp(theta):
	Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1 = theta
	return lnprob(Mc, eta, chieff, chia, lam, theta, psi, phi, Dl, i, phi_c, tc1)


#result = opt.minimize(func, [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg])
#result = [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg]
result = [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, theta_avg, psi_avg, phi_avg, Dl_avg, i_avg, phi_c_avg, tc1_avg]
#Mc_ml, eta_ml, chieff_ml, chia_ml, lam_ml, tc1_ml, tc2_ml = result['x']


#start_time = time.perf_counter()
#print("Started time.")
# Set up the sampler.
print("Setting up sampler.")
ndim, nwalkers = 12, 30
pos = [result + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 500)
#print (pos)
print("Done.")

# Removing first 1000 points as chain takes some time to stabilize
burnin = 100
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
# saving data in file
np.savetxt("12d_emcee_sampler_rb3_05k_03w.dat",samples,fmt='%f',  header="Mc eta chieff chia lam theta psi phi Dl i phi_c tc1")

quit()
# Plot for progression of sampler for each parameter
pl.clf()
fig1, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(result[0], color="#888888", lw=2)
axes[0].set_ylabel(r"$MC$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(result[1], color="#888888", lw=2)
axes[1].set_ylabel(r"$Eta$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(result[2], color="#888888", lw=2)
axes[2].set_ylabel(r"$CHI_eff$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(result[3], color="#888888", lw=2)
axes[3].set_ylabel(r"$CHIa$")


axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(result[4], color="#888888", lw=2)
axes[4].set_ylabel(r"$LAMBDA$")

axes[5].plot(sampler.chain[:, :, 11].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(result[11], color="#888888", lw=2)
axes[5].set_ylabel(r"$TC_1$")


fig1.tight_layout(h_pad=0.0)
fig1.savefig("12d_line-time-plot_ext_005k_03w.pdf")

pl.clf()
fig1, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(result[5], color="#888888", lw=2)
axes[0].set_ylabel(r"$Theta$")

axes[1].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(result[6], color="#888888", lw=2)
axes[1].set_ylabel(r"$Psi$")

axes[2].plot(sampler.chain[:, :, 7].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(result[7], color="#888888", lw=2)
axes[2].set_ylabel(r"$Phi$")

axes[3].plot(sampler.chain[:, :, 8].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(result[8], color="#888888", lw=2)
axes[3].set_ylabel(r"$D_l$")

axes[4].plot(sampler.chain[:, :, 9].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(result[9], color="#888888", lw=2)
axes[4].set_ylabel(r"$i$")

axes[5].plot(sampler.chain[:, :, 10].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(result[10], color="#888888", lw=2)
axes[5].set_ylabel(r"$Phi_c$")


fig1.tight_layout(h_pad=0.0)
fig1.savefig("12d_line-time-plot_int_005k_03w.pdf")
#fig.show()
