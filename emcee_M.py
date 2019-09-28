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
from waveform_m import *
# routines for binning, summary data, etc.
from binning import *

# ---- sample python code for:
#
#   processing GW170817 data
#   constructing frequency binning
#   prepare summary data
#   compute likelihood using summary data

# number of detectors (Livingston and Hanford)
ndtct = 2
ndim = 7

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

PHIC = 0.1
LNA = -46.0
E2 = 1
MC = 1.1976                                      # detector frame chirp mass [Msun]
ETA = 0.244                                      # symmetric mass ratio m1*m2/(m1 + m2)**2
DELTA = np.sqrt(1.0 - 4.0*ETA)                   # asymmetric mass ratio (m1 - m2)/(m1 + m2)
M = MC/ETA**0.6                                  # total mass [Msun]
M1 = 0.5*M*(1.0 + DELTA)                        # primary mass [Msun]
M2 = 0.5*M*(1.0 - DELTA)                        # second mass [Msun]
S1Z = 0.0                                        # aligned spin component for the primaryi
S2Z = 0.0                                        # aligned spin component for the secondary
CHIA = 0.5*(S1Z - S2Z)
CHIS = 0.5*(S1Z + S2Z)
CHIEFF = CHIS + DELTA*CHIA
LAM = 0.0                                      # reduced tidal deformation parameter
TC1 = -205.5556                                  # merger time (L1)
TC2 = -205.5521                                  # merger time (H1)

# allowed bounds for parameters
# change or further refine if desired
Mc_bounds = [1.1970, 1.1980]
eta_bounds = [0.1, 0.2499]
lnA_bounds = [-52, -44]
dtc_bounds = [-0.005, 0.005]
phic_bounds = [-np.pi,np.pi]
e2_bounds = [-9.0,11.0]
par_bounds = [lnA_bounds, phic_bounds, Mc_bounds, eta_bounds, e2_bounds] + [dtc_bounds for k in range(ndtct)]

#Average value for all the bounds
Mc_avg = 0.5*(Mc_bounds[0]+Mc_bounds[1])
eta_avg = 0.5*(eta_bounds[0]+eta_bounds[1])
lnA_avg = 0.5*(lnA_bounds[0]+lnA_bounds[1])
phic_avg = 0.5*(phic_bounds[0]+phic_bounds[1])
e2_avg = 0.5*(e2_bounds[0]+e2_bounds[1])
tc1_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])
tc2_avg = 0.5*(dtc_bounds[0]+dtc_bounds[1])

# fiducial waveforms sampled at full frequency resolution
h0_L = hf3hPN(f, LNA, PHIC, MC, ETA, E2, TC1)
h0_H = hf3hPN(f, LNA, PHIC, MC, ETA, E2, TC2)
# these are NOT shifted to the right merger times
h0 = [h0_L, h0_H]

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
LNA = par_bf[0] 
PHIC = par_bf[1]
MC = par_bf[2]                                       # detector frame chirp mass [Msun]
ETA = par_bf[3]                                      # symmetric mass ratio m1*m2/(m1 + m2)**2
E2 = par_bf[4]                                  # reduced tidal deformation parameter
TC1 += par_bf[5]                                  # merger time (L1)
TC2 += par_bf[6]                                  # merger time (H1)

print('Updated parameters for the fiducial waveform')

# now update fiducial waveforms
# fiducial waveforms sampled at full frequency resolution
h0_L = hf3hPN(f, LNA, PHIC, MC, ETA, E2, TC1)
h0_H = hf3hPN(f, LNA, PHIC, MC, ETA, E2, TC2)
h0 = [h0_L, h0_H]

print('Updated fiducial waveforms.')

# next prepare summary data
sdat = compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0)

print('Updated summary data.')
#print(sdat)

def lnlikelihood(lnA, phic, Mc, eta, e2, tc1, tc2):
    par_best = [lnA, phic, Mc, eta, e2, tc1, tc2]
    return(-lnlike(par_best, sdat, h0, fbin, fbin_ind, ndtct))

# Uniform prior on all parameter in their respective range
def lnprior(lnA, phic, Mc, eta, e2, tc1, tc2):
    if phic_bounds[0]<phic<phic_bounds[1] and Mc_bounds[0]<Mc<Mc_bounds[1] and eta_bounds[0]<eta<eta_bounds[1] and lnA_bounds[0]<lnA<lnA_bounds[1] and e2_bounds[0]<e2<e2_bounds[1] and dtc_bounds[0]<tc1<dtc_bounds[1] and dtc_bounds[0]<tc2<dtc_bounds[1]:
        l = 0.0
    else:
        l =-np.inf
    return l


# Multiplying likelihood with prior
def lnprob(lnA, phic, Mc, eta, e2, tc1, tc2):
	lp = lnprior(lnA, phic, Mc, eta, e2, tc1, tc2)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlikelihood(lnA, phic, Mc, eta, e2, tc1, tc2)


# Defining a function just for minimization routine to find a point to start
def func(theta):
	lnA, phic, Mc, eta, e2, tc1, tc2 = theta
	return -2.*lnprob(lnA, phic, Mc, eta, e2, tc1, tc2)

def lnp(theta):
	lnA, phic, Mc, eta, e2, tc1, tc2 = theta
	return lnprob(lnA, phic, Mc, eta, e2, tc1, tc2)


#result = opt.minimize(func, [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg])
result = [lnA_avg, phic_avg, Mc_avg, eta_avg, e2_avg, tc1_avg, tc2_avg]
#result = [Mc_bounds[0], eta_bounds[0], chieff_bounds[0], chia_bounds[0], Lam_bounds[0], dtc_bounds[0], dtc_bounds[0]]
#Mc_ml, eta_ml, chieff_ml, chia_ml, lam_ml, tc1_ml, tc2_ml = result['x']


#start_time = time.perf_counter()
#print("Started time.")
# Set up the sampler.
print("Setting up sampler.")
ndim, nwalkers = 7, 200
pos = [result + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 10000)
#print (pos)
print("Done.")



# Plot for progression of sampler for each parameter
pl.clf()
fig, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(result[0], color="#888888", lw=2)
axes[0].set_ylabel(r"$lnA$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(result[1], color="#888888", lw=2)
axes[1].set_ylabel(r"$PHI_c$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(result[2], color="#888888", lw=2)
axes[2].set_ylabel(r"$Mc$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(result[3], color="#888888", lw=2)
axes[3].set_ylabel(r"$ETA$")


axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(result[4], color="#888888", lw=2)
axes[4].set_ylabel(r"$E2$")

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(result[5], color="#888888", lw=2)
axes[5].set_ylabel(r"$TC_1$")

#axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
#axes[6].yaxis.set_major_locator(MaxNLocator(5))
#axes[6].axhline(tc2_ml, color="#888888", lw=2)
#axes[6].set_ylabel(r"$TC_2$")

fig.tight_layout(h_pad=0.0)
fig.savefig("figures/M_line-time-plot_changed_bounds_10k_2w.pdf")
#fig.show()

# Removing first 100 points as chain takes some time to stabilize
burnin = 1000
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

parser = argparse.ArgumentParser(description = '')
parser.add_argument("--lnA", help = "injected lnA value", type=float)
parser.add_argument("--phic", help = "injected phic value", type=float)
parser.add_argument("--Mc", help = "injected Mc value", type=float)
parser.add_argument("--eta", help = "injected eta value", type=float)
parser.add_argument("--e2", help = "injected e2 value", type=float)
parser.add_argument("--Mc", help = "injected Mc value", type=float)
parser.add_argument("--tc1", help = "injected tc1 value", type=float)
parser.add_argument("--tc2", help = "injected tc2 value", type=float)


#parser.add_argument("--nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

lnA_true=args.lnA
phic_true=args.phic
Mc_true=args.Mc
eta_true=args.eta
e2_true=e2.Mc
tc1_true=args.tc1
tc2_true=args.tc2
Mtotal = np.array([10, 20])

#Corner plot
fig1 = corner.corner(samples, labels=["lnA", "\phi_c", "M_c", "$\eta$", "e_2", "$tc_1$", "$tc_2$"],
		truths=[lnA_true, phic_true, Mc_true, eta_true, e2_true, tc1_true, tc2_true])
fig1.suptitle("one-sigma levels")
fig1.savefig('figures/M_plot_emcee_sampler_10k_2w.pdf')

Mc_mcmc, eta_mcmc, chieff_mcmc, chia_mcmc, lam_mcmc, tc1_mcmc, tc2_mcmc= map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print("True values of the parameters:")
print("lnA= ", lnA_true)
print("phic= ", phic_true)
print("Mc= ", Mc_true)
print("eta= ", eta_true)
print("e2= ", e2_true)
print("tc1= ", tc1_true)
print("tc2= ", tc2_true)


print("")
print("median and error in the parameters")
print("lnA: ", lnA_mcmc[0], lnA_mcmc[2]-lnA_mcmc[1])
print("phic: ", phic_mcmc[0], phic_mcmc[2]-phic_mcmc[1])
print("Mc: ", Mc_mcmc[0], Mc_mcmc[2]-Mc_mcmc[1])
print("eta: ", eta_mcmc[0], eta_mcmc[2]-eta_mcmc[1])
print("e2: ", e2_mcmc[0], e2_mcmc[2]-e2_mcmc[1])
print("tc1: ", tc1_mcmc[0], tc1_mcmc[2]-tc1_mcmc[1])
print("tc2: ", tc2_mcmc[0], tc2_mcmc[2]-tc2_mcmc[1])
#print("--- %s seconds ---" % (time.perf_counter() - start_time))
