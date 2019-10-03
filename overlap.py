#!/usr/local/bin/python

import numpy as np
import scipy as sp

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

#par = [Mc,eta, chieff, chia, lam, tc1, tc2]
r = compute_rf(par_bf, h0, fbin, fbin_ind)

h_int = np.zeros(len(f))

for i in range(len(fbin)):
  for j in range(len(f)):
    while f(j)<=fbin(i):
      h_int[j] = (r[0][i] + (1.0/T)*r[1][i])*h0[j]
      
print(len(h_int))
print(len(h0))
print(len(f))
      
 





