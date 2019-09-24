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
f = np.linspace(0, 1.0/T*n_sample/2.0, n_sample//2+1)

# apply a Tukey window function to eliminate the time-domain boundary ringing
tukey = sp.signal.tukey(n_sample, alpha=0.1)
LFT = np.fft.rfft(L1*tukey)/n_sample
HFT = np.fft.rfft(H1*tukey)/n_sample

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
#TC2 += par_bf[6]                                  # merger time (H1)

print('Updated parameters for the fiducial waveform')

def f_isco(M):
    return 1./6.**(3./2.)/np.pi/M

def htilde(Mc, eta, mu2=1.0, mu3=1.0, mu4=1.0, mu5=1.0, e2=1.0, e3=1.0, e4=1.0):
    M = Mc/eta**(3./5.)
    v  = (np.pi*M*f)**(1./3.)
    flso = f_isco(M)
    vlso = (np.pi*M*flso)**(1./3.)
    lnA = np.log(Af3hPN(f,M,eta))

    psi = -2.0j*np.pi*f*tc1 + (3./(128.*v**5*mu2**2*eta))*(1 - 16.*np.pi*v**3 + v**2*(1510./189.+e2**2*(-5./81. + 20.*eta/81.)/mu2**2 - 130.*eta/21. + mu3**2*(-6835./2268.+6835.*eta/567.)/mu2**2)
                                    + v**4*(242245./5292.+4525.*eta/5292.+145445.*eta**2/5292.+mu4**2*(-89650./3969. + 179300.*eta/1323.-89650.*eta**2/441.)/mu2**2+e3**2*(-50./63.+100.*eta/21.-50.*eta**2/7.)/mu2**2 +e2**2*(-785./378. +7115.*eta/756. -835.*eta**2/189.)/mu2**2+e2**4*(5./648.-5.*eta/81. +10.*eta**2/81.)/mu2**4+mu3**4*(9343445./508032.-9343445.*eta/63504.+9343445.*eta**2/31752.)/mu2**4 + mu3**2*((-66095./7056.+170935.*eta/3024.-403405.*eta**2/5292.)/mu2**2+e2**2*(6835./9072. -6835.*eta/1134. +6835.*eta**2/567.)/mu2**4))
                                    + v**7*(484490.*np.pi/1323. - 141520.*np.pi*eta/1323. + 442720.*np.pi*eta**2/1323. + e2**2*(-1570.*np.pi/63. + 7220.*np.pi*eta/63. - 3760.*np.pi*eta**2/63.)/mu2**2 + e3**2*(-400.*np.pi/63. + 800.*np.pi*eta/21. - 400.*np.pi*eta**2/7.)/mu2**2 + mu4**2*(-400.*np.pi/3969. + 800.*np.pi*eta/1323. - 400.*np.pi*eta**2/441.)/mu2**2 + mu3**4*(6835.*np.pi/254016. - 6835.*np.pi*eta/31752. + 6835.*np.pi*eta**2/15876.)/mu2**4 + e2**4*(10.*np.pi/81. - 80.*np.pi*eta/81. + 160.*np.pi*eta**2/81.)/mu2**4 + mu3**2*((-88205.*np.pi/2352. + (63865.*np.pi*eta)/252. - 182440.*np.pi*eta**2/441.)/mu2**2 + e2**2*(54685.*np.pi/9072. - 54685.*np.pi*eta/1134. + 54685.*np.pi*eta**2/567.)/mu2**4))
                                    + v**6*((-82403040211200. + 2646364089600.*np.pi**2)*eta/14082647040. + 123839990.*eta**2/305613. + 18300845.*eta**3/1222452. + mu3**6*(12772489315./256048128. - 12772489315.*eta/21337344. + 12772489315.*eta**2/5334336. - 12772489315.*eta**3/4000752.)/mu2**6 + e4**2*(5741./1764.-11482.*eta/441.+28705.*eta**2/441.-22964.*eta**3/441.)/mu2**2 + e3**2*(27730./3969.-179990.*eta/3969.+341450.*eta**2/3969.-51050.*eta**3/1323.)/mu2**2 + e2**6*(5./11664. - 5*eta/972. + 5.*eta**2/243.-20.*eta**3/729.)/mu2**6+e2**4*(-265./1512. + 20165.*eta/13608.-5855.*eta**2/1701.+310.*eta**3/243.)/mu2**4 + (1002569.*mu5**2/12474.-4010276.*mu5**2*eta/6237.+10025690.*mu5**2*eta**2/6237.-8020552.*mu5**2*eta**3/6237.)/mu2**2 + e2**2*((3638245./190512. -2842015.*eta/31752.+760985.*eta**2/13608.-328675.*eta**3/23814.)/mu2**2 + e3**2*(-(50./567.) + 500.*eta/567. -550.*eta**2/189. +200.*eta**3/63.)/mu2**4) + mu3**4*((e2**2*(9343445./3048192. -9343445.*eta/254016.+9343445.*eta**2/63504.-9343445.*eta**3/47628.))/mu2**6 + (868749005./10668672.-2313421945.*eta/3556224.+191974645.*eta**2/148176. +9726205.*eta**3/666792.)/mu2**4) + mu4**2*((-(86554310./916839.) + (553387330.*eta)/916839. - (289401650.*eta**2)/305613. - (4322750.*eta**3)/101871.)/mu2**2 + (e2**2*(-(89650./35721.) + (896500.*eta)/35721. - (986150.*eta**2)/11907. + (358600.*eta**3)/3969.))/mu2**4) + mu3**2*((-(4809714655./29338848.) + (8024601785.*eta)/9779616. - (19149203695.*eta**2)/29338848. - (190583245.*eta**3)/7334712.)/mu2**2 + (e2**4*(6835./108864. - (6835.*eta)/9072. + (6835.*eta**2)/2268. - (6835.*eta**3)/1701.))/mu2**6 + (e2**2*(-(656195./95256.) + (229475.*eta)/3888. - (3369935.*eta**2)/23814. + (82795.*eta**3)/1323.))/mu2**4 + (e3**2*(-(34175./7938.) + (170875.*eta)/3969. - (375925.*eta**2)/2646. + (68350.*eta**3)/441.))/mu2**4 + (mu4**2*(-(61275775./500094.) + (306378875.*eta)/250047. - (674033525.*eta**2)/166698. + (122551550.*eta**3)/27783.))/mu2**4) + (1./14082647040.)*(36871736640768. - 4592284139520.*GammaE - 3004298035200.*np.pi**2 - 9184568279040.*np.log(2) - 4592284139520.*np.log(v)))
                             + v**5*(12080.*np.pi/189. - 3680.*np.pi*eta/63. + e2**2*(-20.*np.pi/27. + 80.*np.pi*eta/27.)/mu2**2 + mu3**2*(-9115.*np.pi/756. + 9115.*np.pi*eta/189.)/mu2**2)*(1 + 3.*np.log(v/vlso)))
    h = np.exp(lnA)*mu2*f**(-7./6.)*np.exp(1j*psi)
    return h

def lnprior(Mc, eta, chieff, chia, lam, tc1):
    if 1.1973<Mc<1.1979 and 0.2<eta<0.24999 and -0.2<chieff<0.2 and -0.999<chia<0.999 and 0<lam<1000 and -0.005<tc1<0.005:
        l = 0.0
    else:
        l =-np.inf
    return l

def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/psd_L).sum()))*df
    return summ

# Define log likelihood
def lnlike_real(Mc, eta, chieff, chia, lam, tc1):
    M = Mc / eta ** 0.6
    delta = np.sqrt(1.0 - 4.0 * eta)
    chis = chieff - delta * chia
    s1Z = chis + chia
    s2Z = chis - chia
    h1_L = htilde(Mc, eta)
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
    return -0.5*overlap(np.asarray(LFT)-np.asarray(h1_L), np.asarray(LFT)-np.asarray(h1_L), f)

def lnprob_real(Mc, eta, chieff, chia, lam, tc1):
	lp = lnprior(Mc, eta, chieff, chia, lam, tc1)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real(Mc, eta, chieff, chia, lam, tc1)

def lnp_real(theta):
	Mc, eta, chieff, chia, lam, tc1 = theta
	return lnprob(Mc, eta, chieff, chia, lam, tc1)

print("Calculating likelihood.")
print(lnlike_real(MC, ETA, CHIEFF, CHIA, LAM, TC1))
#print(lnlike_real(Mc_avg, eta_avg, chieff_avg, chia_avg, Lam_avg))
