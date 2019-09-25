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

# provide sample waveform model
from waveform import *
# routines for binning, summary data, etc.
from binning import *

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
f1 = np.linspace(0, 1.0/T*n_sample/2.0, n_sample//2+1)
f = np.zeros(len(f1)-1)
for i in range(len(f)):
	f[i] = f1[i+1]
# apply a Tukey window function to eliminate the time-domain boundary ringing
tukey = sp.signal.tukey(n_sample, alpha=0.1)
LFT1 = np.fft.rfft(L1*tukey)/n_sample
LFT = np.zeros(len(LFT1)-1)
for i in range(len(LFT)):
	LFT[i] = LFT1[i+1]
HFT = np.fft.rfft(H1*tukey)/n_sample
HFT = np.zeros(len(HFT1)-1)
for i in range(len(HFT)):
	HFT[i] = HFT1[i+1]
# estimate PSDs for both L1 and H1
psd_L = 2.0*np.convolve(np.absolute(LFT)**2, np.ones((n_conv))/n_conv, mode='same')*T
psd_H = 2.0*np.convolve(np.absolute(HFT)**2, np.ones((n_conv))/n_conv, mode='same')*T
psd = [psd_L, psd_H]
psd2 = np.append(psd_L,psd_H)

print("PSD calculated")
# frequency domain data
sFT = [LFT, HFT]
#sFT.append(LFT)
#sFT.append(HFT)

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
ndtct = 1
df = 1./T

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

h0_L = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h0_H = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
# these are NOT shifted to the right merger times
h0_0 = [h0_L, h0_H]
# these are shifted to the right merger times
h0 = [h0_L*np.exp(-2.0j*np.pi*f*TC1), h0_H*np.exp(-2.0j*np.pi*f*TC2)]

print('UPDATED fiducial waveforms.')

def lnprior(Mc, eta, chieff, chia, lam, tc1):
    if 1.1973<Mc<1.1979 and 0.2<eta<0.24999 and -0.2<chieff<0.2 and -0.999<chia<0.999 and 0<lam<1000 and -0.005<tc1<0.005:
        l = 0.0
    else:
        l =-np.inf
    return l

def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/psd_L).sum()))*df
    return summ

def lnlike_real(Mc, eta, chieff, chia, lam, tc1):
    M = Mc / eta ** 0.6
    delta = np.sqrt(1.0 - 4.0 * eta)
    chis = chieff - delta * chia
    s1Z = chis + chia
    s2Z = chis - chia
    h1_L = hf3hPN(f, M, eta, s1z=s1z, s2z=s2z, Lam=lam)
    #h1_H = hf3hPN(f, M, eta, s1z=s1z, s2z=s2z, Lam=lam)
    # these are NOT shifted to the right merger times
    #h1_1 = np.append(h1_L, h1_H)
    # these are shifted to the right merger times
    #sFT = np.append(LFT, HFT)
    #h1 = np.append(h1_L*np.exp(-2.0j*np.pi*f*tc1), h1_H*np.exp(-2.0j*np.pi*f*tc2))
    pl.plot(f,(h1_L))
    pl.savefig('plot_test_h1_l.pdf')
    pl.close()
    print(len(np.asarray(h1_L)), len(np.asarray(LFT)), len(psd_L))
    print(np.amax(np.asarray(h1_L)), np.amax(np.asarray(LFT)), np.amax(np.asarray(psd_L)))
    a = overlap(np.asarray(h1_L),np.asarray(h1_L),f)
    b = overlap(np.asarray(h0_L),np.asarray(h0_L),f)
    c = overlap(np.asarray(h0_L),np.asarray(h1_L),f)
    return c/np.sqrt(a*b)

def lnprob_real(Mc, eta, chieff, chia, lam, tc1):
	lp = lnprior(Mc, eta, chieff, chia, lam, tc1)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real(Mc, eta, chieff, chia, lam, tc1)

def lnp_real(theta):
	Mc, eta, chieff, chia, lam, tc1 = theta
	return lnprob(Mc, eta, chieff, chia, lam, tc1)

print("Calculating overlap.")
print(lnlike_real(MC, ETA, CHIEFF, CHIA, LAM, TC1))
#print(lnlike_real(Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg))
