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
ndtct = 2


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
chieff_bounds = [-0.2, 0.2]
chia_bounds = [-0.6, 0.6]
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
TC2 += par_bf[6]                                  # merger time (H1)

print('Updated parameters for the fiducial waveform')

# now update fiducial waveforms
h0_L = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h0_H = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H*np.exp(-2.0j*np.pi*f*TC2)]

print('Updated fiducial waveforms.')

# next prepare summary data
sdat = compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0)

print('Updated summary data.')
#print(sdat)

# example of likelihood evaluation: check the likelihood of the new fiducial waveform
#par_best = [MC, ETA, CHIEFF, CHIA, LAM, 0.0, 0.0]
#print(-lnlike(par_best, sdat, h0_0, fbin, fbin_ind, ndtct))

def lnlikelihood(Mc, eta, chieff, chia, lam, tc1, tc2):
    par_best = [Mc, eta, chieff, chia, lam, tc1, tc2]
    return(-lnlike(par_best, sdat, h0_0, fbin, fbin_ind, ndtct))

# Uniform prior on all parameter in their respective range
def lnprior(Mc, eta, chieff, chia, lam, tc1, tc2):
    if Mc_bounds[0]<Mc<Mc_bounds[1] and eta_bounds[0]<eta<eta_bounds[1] and chieff_bounds[0]<chieff<chieff_bounds[1] and chia_bounds[0]<chia<chia_bounds[1] and Lam_bounds[0]<lam<Lam_bounds[1] and dtc_bounds[0]<tc1<dtc_bounds[1] and dtc_bounds[0]<tc2<dtc_bounds[1]:
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
def lnprob(Mc, eta, chieff, chia, lam, tc1, tc2):
	lp = lnprior(Mc, eta, chieff, chia, lam, tc1, tc2)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlikelihood(Mc, eta, chieff, chia, lam, tc1, tc2)


# Defining a function just for minimization routine to find a point to start
def func(theta):
	Mc, eta, chieff, chia, lam, tc1, tc2 = theta
	return -2.*lnprob(Mc, eta, chieff, chia, lam, tc1, tc2)

def lnp(theta):
	Mc, eta, chieff, chia, lam, tc1, tc2 = theta
	return lnprob(Mc, eta, chieff, chia, lam, tc1, tc2)


#result = opt.minimize(func, [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg])
result = [Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg, tc1_avg, tc2_avg]
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

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(result[5], color="#888888", lw=2)
axes[5].set_ylabel(r"$TC_1$")

#axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
#axes[6].yaxis.set_major_locator(MaxNLocator(5))
#axes[6].axhline(tc2_ml, color="#888888", lw=2)
#axes[6].set_ylabel(r"$TC_2$")

fig.tight_layout(h_pad=0.0)
fig.savefig("figures/line-time-plot_changed_bounds_10k_2w.pdf")
#fig.show()

# Removing first 100 points as chain takes some time to stabilize
burnin = 1000
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

parser = argparse.ArgumentParser(description = '')
parser.add_argument("--Mc", help = "injected Mc value", type=float)
parser.add_argument("--eta", help = "injected eta value", type=float)
parser.add_argument("--chieff", help = "injected chieff value", type=float)
parser.add_argument("--chia", help = "injected chia value", type=float)
parser.add_argument("--lam", help = "injected lam value", type=float)
parser.add_argument("--theta", help = "injected theta value", type=float)
parser.add_argument("--psi", help = "injected psi value", type=float)
parser.add_argument("--phi", help = "injected phi value", type=float)
parser.add_argument("--Dl", help = "injected Dl value", type=float)
parser.add_argument("--i", help = "injected i value", type=float)
parser.add_argument("--phi_c", help = "injected phi_c value", type=float)
parser.add_argument("--tc1", help = "injected tc1 value", type=float)
parser.add_argument("--tc2", help = "injected tc2 value", type=float)


#parser.add_argument("--nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

Mc_true=args.Mc
eta_true=args.eta
chieff_true=args.chieff
chia_true=args.chia
lam_true=args.lam
theta_true=args.theta
psi_true=args.psi
phi_true=args.phi
Dl_true=args.Dl
i_true=args.i
phi_c_true=args.phi_c
tc1_true=args.tc1
tc2_true=args.tc2
Mtotal = np.array([10, 20])

#Corner plot
fig1 = corner.corner(samples, labels=["M_c", "$\eta$", "$\chi_eff$", "$\chi_a$", "$\lambda$", "$theta$", "$\psi$", "$\phi$", "$D_l$", "$i$", "$\phi_c$", "$tc_1$", "$tc_2$"],
		truths=[Mc_true, eta_true, chieff_true, chia_true, lam_true, theta_true, psi_true, phi_true, Dl_true, i_true, phi_c_true, tc1_true, tc2_true])
fig1.suptitle("one-sigma levels")
fig1.savefig('figures/plot_emcee_sampler_bounds_changed_10k_2w.pdf')

Mc_mcmc, eta_mcmc, chieff_mcmc, chia_mcmc, lam_mcmc, theta_mcmc, psi_mcmc, phi_mcmc, Dl_mcmc, i_mcmc, phi_c_mcmc, tc1_mcmc, tc2_mcmc= map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#fig2 = corner.corner(samples, labels=["$theta$", "$\psi$", "$\phi$", "$D_l$", "$i$", "$\phi_c$"],
#		truths=[theta_true, psi_true, phi_true, Dl_true, i_true, phi_c_true])
#fig2.suptitle("one-sigma levels")
#fig2.savefig('plot_12d_emcee_sampler_int_rb3_5k_1w.pdf')

#theta_mcmc, psi_mcmc, phi_mcmc, Dl_mcmc, i_mcmc, phi_c_mcmc= map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print("True values of the parameters:")
print("Mc= ", Mc_true)
print("eta= ", eta_true)
print("chieff= ", chieff_true)
print("chia= ", chia_true)
print("lam= ", lam_true)
print("theta= ", theta_true)
print("psi= ", psi_true)
print("phi= ", phi_true)
print("Dl= ", Dl_true)
print("i= ", i_true)
print("phi_c= ", phi_c_true)
print("tc1= ", tc1_true)
print("tc2= ", tc2_true)


print("")
print("median and error in the parameters")
print("Mc: ", Mc_mcmc[0], Mc_mcmc[2]-Mc_mcmc[1])
print("eta: ", eta_mcmc[0], eta_mcmc[2]-eta_mcmc[1])
print("chieff: ", chieff_mcmc[0], chieff_mcmc[2]-chieff_mcmc[1])
print("chia: ", chia_mcmc[0], chia_mcmc[2]-chia_mcmc[1])
print("lam: ", lam_mcmc[0], lam_mcmc[2]-lam_mcmc[1])
print("theta: ", theta_mcmc[0], theta_mcmc[2]-theta_mcmc[1])
print("psi: ", psi_mcmc[0], psi_mcmc[2]-psi_mcmc[1])
print("theta: ", theta_mcmc[0], theta_mcmc[2]-theta_mcmc[1])
print("psi: ", psi_mcmc[0], psi_mcmc[2]-psi_mcmc[1])
print("phi: ", phi_mcmc[0], phi_mcmc[2]-phi_mcmc[1])
print("Dl: ", Dl_mcmc[0], Dl_mcmc[2]-Dl_mcmc[1])
print("i: ", i_mcmc[0], i_mcmc[2]-i_mcmc[1])
print("phi_c: ", phi_c_mcmc[0], phi_c_mcmc[2]-phi_c_mcmc[1])
print("tc1: ", tc1_mcmc[0], tc1_mcmc[2]-tc1_mcmc[1])
print("tc2: ", tc2_mcmc[0], tc2_mcmc[2]-tc2_mcmc[1])
#print("--- %s seconds ---" % (time.perf_counter() - start_time))
