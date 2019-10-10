#!/usr/local/bin/python

import numpy as np
import scipy as sp
import math as mt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as pl
# provide sample waveform model
from waveform import *
# routines for binning, summary data, etc.
from binning import *

# number of detectors (Livingston and Hanford)
#ndtct = 1

n_sample = 2**20
T = 256.0
#n_conv = 20

# load LIGO strain data (time domain)
#L1 = np.loadtxt('data/L.txt')
#H1 = np.loadtxt('data/H.txt')


print('Finished loading LIGO data.')

# time-domain and frequency-domain grid
t = np.linspace(0, T, n_sample+1)[:-1]
f = np.linspace(0, 1.0/T*n_sample/2.0, n_sample//2+1)

# apply a Tukey window function to eliminate the time-domain boundary ringing
#tukey = sp.signal.tukey(n_sample, alpha=0.1)
#pl.plot(t,tukey)
#L1T = L1*tukey
#pl.plot(t,L1T)
#pl.plot(t,L1)
#LFT = np.fft.rfft(L1T)/n_sample

# estimate PSDs for both L1 and H1
#psd_L = 2.0*np.convolve(np.absolute(LFT)**2, np.ones((n_conv))/n_conv, mode='same')*T

def sh(f):
    s = 5.623746655206207e-51 + 6.698419551167371e-50*f**(-0.125) + 7.805894950092525e-31/f**20. + 4.35400984981997e-43/f**6. + 1.630362085130558e-53*f + 2.445543127695837e-56*f**2 + 5.456680257125753e-66*f**5
    return s

print('Finished estimating the PSD.')

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


# fiducial waveforms sampled at full frequency resolution
h0 = hf3hPN(f, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h0[0] = 0
h1 = h0*np.exp(-2.0j*np.pi*f*TC1)
print('Constructed fiducial waveforms.')

# prepare frequency binning
# range of frequency to be used in the computation of likelihood [f_lo, f_hi] [Hz]
f_lo = 23.0
f_hi = 1000.0
tau = (5.0/(256.*np.pi*f_lo))*((np.pi*MC*5.*(10.**(-6.))*f_lo)**(-5./3.))
print("tau=",tau)
Nbin, fbin, fbin_ind = setup_bins(f_full=f, f_lo=f_lo, f_hi=f_hi, chi=1.0, eps=0.5)
print(Nbin, fbin, fbin_ind)

print("Frequency binning done: # of bins = %d"%(Nbin))


parL = [MC, ETA, CHIEFF, CHIA, LAM, TC1]
rL = compute_rf(parL, h0, fbin, fbin_ind)

#h_int = [np.zeros(len(f)), np.zeros(len(f))]
h_int = np.array(np.zeros(len(f)), dtype=np.complex128)
j = fbin_ind[0]
res = 1./(2.*tau)
df = 1./T
ad = res/df
print("ad=",ad)
#ad = 1
fp = np.array(np.zeros(len(f)))
#m = 0
#for i in range(len(fbin)-1):
#  for fn in np.arange(f[fbin_ind[i]], f[fbin_ind[i+1]], 1/(2*tau)):
#    fp[m] = fn
#    m+=1

print("Frequency array prepared")

k = 0
for i in range(len(fbin)-1):
  fmid = 0.5*(fbin[i] + fbin[i+1])
  for fn in np.arange(fbin[i], fbin[i+1], res):    
    fp[k] = fn
    fh = fbin_ind[i]+int((j-fbin_ind[i])*ad)
    h = 0.5*(h1[fh]+h1[fh+1])
    h_int[k] = (rL[0][i] + (fn-fmid)*rL[1][i])*h
    k+=1  
    j+=1
  #print(i)

#Truncating the array to appropriate size
fp = fp[:k] 
h_int = h_int[:k]

h_int2 = np.array(np.zeros(len(f)), dtype=np.complex128)
k = 0
j = fbin_ind[0]
for i in range(len(fbin)-1):
  fmid = 0.5*(fbin[i] + fbin[i+1])
  for fn in np.arange(fbin[i], fbin[i+1], 1./T):
    h_int2[fbin_ind[0]+k] = (rL[0][i] + (fn-fmid)*rL[1][i])*h1[fbin_ind[0]+k]
    j+=1
    k+=1  
  print(i)

print("interpolated waveform created")


def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/psd).sum()))*res
    return summ
  
#Now computing exact waveform at the new resolution 'res'

h2_0 = hf3hPN(fp, M, ETA, s1z=S1Z, s2z=S2Z, Lam=LAM)
h2 = h2_0*np.exp(-2.0j*np.pi*fp*TC1)
psd = sh(fp)

print(fp[1000])
pl.plot(fp[:1000],h_int[:1000],'r')
pl.plot(f[fbin_ind[0]:(fbin_ind[0]+int(1000*ad))],h_int2[fbin_ind[0]:(fbin_ind[0]+int(1000*ad))],'g')
pl.savefig("figures/overlap_comp2.pdf")
a = np.absolute(overlap(h2,h2,fp))
b = np.absolute(overlap(h_int,h_int,fp))
c = np.absolute(overlap(h2,h_int,fp))

d = c/(np.sqrt(a)*np.sqrt(b))

#print("h=",h2[1:1000])
#print("h_int=",h_int[1:1000])
#print("psd_L",psd[0:100])
print("Overlap1=",a)
print("Overlap2=",b)
print("Overlap3=",c)
print("Overlap4=",d)
#print(np.shape(h_int))
#print(np.shape(h0))
#print(len(f))
 
