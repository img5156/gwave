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
n_sample = 2**20
T = 256.0
n_conv = 20

# load LIGO strain data (time domain)
L1 = np.loadtxt('data/L.txt')
H1 = np.loadtxt('data/H.txt')


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
TC1 = -127.5556                                  # merger time (L1)
TC2 = -127.5521  

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
print(Nbin, fbin, fbin_ind)

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
h0[0][0] = 0
print('Updated fiducial waveforms.')

#print("h=",h0[0][0:100])
#print("psd_L",psd_L[0:100])
#exit()
parL = [MC, ETA, CHIEFF, CHIA, LAM, TC1]
parH = [MC, ETA, CHIEFF, CHIA, LAM, TC2]
rL = compute_rf(parL, h0[0], fbin, fbin_ind)
rH = compute_rf(parH, h0[1], fbin, fbin_ind)

#h_int = [np.zeros(len(f)), np.zeros(len(f))]
h_int = np.array([np.zeros(len(f)), np.zeros(len(f))], dtype=np.complex128)

for i in range(len(fbin)-1):
  for j in range(fbin_ind[0],len(f)):
    if f[j]<=fbin[i]:
      h_int[0][j] = (rL[0][i] + (f[j]-0.5*(f[fbin_ind[i]]+f[fbin_ind[i+1]]))*rL[1][i])*h0[0][j]
      h_int[1][j] = (rH[0][i] + (f[j]-0.5*(f[fbin_ind[i]]+f[fbin_ind[i+1]]))*rH[1][i])*h0[1][j]
    else:
      break
  print(i)
  
def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/psd_L).sum()))*(1.0/T)
    return summ
h0[0][:fbin_ind[0]] = 0 
h0[0][fbin_ind[-1]:] = 0
a = np.absolute(overlap(h0[0],h0[0],f))
b = np.absolute(overlap(h_int[0],h_int[0],f))
c = np.absolute(overlap(h0[0],h_int[0],f))

d = c/(np.sqrt(a)*np.sqrt(b))

print("h=",h0[0][42000:43000])
print("h_int=",h_int[0][42000:43000])
print("psd_L",psd_L[0:100])
print("Overlap1=",a)
print("Overlap2=",b)
print("Overlap3=",c)
print("Overlap4=",d)
#print(np.shape(h_int))
#print(np.shape(h0))
#print(len(f))
 
