#!/usr/bin/env python
import emcee
import numpy as np
import scipy.optimize as opt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as pl
import corner
from matplotlib.ticker import MaxNLocator
from emcee.utils import MPIPool
import argparse
import scipy.interpolate as si

parser = argparse.ArgumentParser(description = '')
parser.add_argument("nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

#Various constants
MTSUN_SI = 4.92549102554*1e-6
PC_SI = 3.086*1e16
C_SI = 2.998*1e8


# defining the last stable orbit
def f_isco(M):
    return 1./6.**(3./2.)/np.pi/M


# number of dimension
ndim = 6

# defining constants
PI =  np.pi
GammaE = 0.577215664901532

# extrinsic parameters
tc_true = 0;  phic_true = 0;

# non-GR parameters
mu2 = 1;  mu3 = 1;  mu4 = 1; mu5 = 1;  e2_true = 1;  e3 = 1; e4 = 1

# Distance
Dist = 100.*1e6*PC_SI/C_SI

# frequency parameters
fl =  5.
df =  1./128./4.


# mass parameters
q=1.2

Mtotal = np.array([10, 20])

nodenum = args.nodenum

m1 = q*Mtotal[nodenum]/(1.+q)
m2 = Mtotal[nodenum]/(1.+q)
Mc_true = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
eta_true = (m1*m2)/(m1+m2)**2.
fh = f_isco(Mtotal[nodenum]*MTSUN_SI)

lnA_true=np.log((1./30.**0.5/PI**(2./3.))*((Mc_true*MTSUN_SI)**(5./6.)/Dist))


#waveform
def htilde(lnA, tc, phic, Mc, eta, mu2, mu3, mu4, mu5, e2, e3, e4, f):
    M = Mc*MTSUN_SI/eta**(3./5.)
    v  = (PI*M*f)**(1./3.)
    flso = f_isco(M)
    vlso = (PI*M*flso)**(1./3.)


    psi = 2*f*PI*tc - phic - PI/4. + (3./(128.*v**5*mu2**2*eta))*(1 - 16.*PI*v**3 + v**2*(1510./189.+e2**2*(-5./81. + 20.*eta/81.)/mu2**2 - 130.*eta/21. + mu3**2*(-6835./2268.+6835.*eta/567.)/mu2**2)
                                    + v**4*(242245./5292.+4525.*eta/5292.+145445.*eta**2/5292.+mu4**2*(-89650./3969. + 179300.*eta/1323.-89650.*eta**2/441.)/mu2**2+e3**2*(-50./63.+100.*eta/21.-50.*eta**2/7.)/mu2**2 +e2**2*(-785./378. +7115.*eta/756. -835.*eta**2/189.)/mu2**2+e2**4*(5./648.-5.*eta/81. +10.*eta**2/81.)/mu2**4+mu3**4*(9343445./508032.-9343445.*eta/63504.+9343445.*eta**2/31752.)/mu2**4 + mu3**2*((-66095./7056.+170935.*eta/3024.-403405.*eta**2/5292.)/mu2**2+e2**2*(6835./9072. -6835.*eta/1134. +6835.*eta**2/567.)/mu2**4))
                                    + v**7*(484490.*PI/1323. - 141520.*PI*eta/1323. + 442720.*PI*eta**2/1323. + e2**2*(-1570.*PI/63. + 7220.*PI*eta/63. - 3760.*PI*eta**2/63.)/mu2**2 + e3**2*(-400.*PI/63. + 800.*PI*eta/21. - 400.*PI*eta**2/7.)/mu2**2 + mu4**2*(-400.*PI/3969. + 800.*PI*eta/1323. - 400.*PI*eta**2/441.)/mu2**2 + mu3**4*(6835.*PI/254016. - 6835.*PI*eta/31752. + 6835.*PI*eta**2/15876.)/mu2**4 + e2**4*(10.*PI/81. - 80.*PI*eta/81. + 160.*PI*eta**2/81.)/mu2**4 + mu3**2*((-88205.*PI/2352. + (63865.*PI*eta)/252. - 182440.*PI*eta**2/441.)/mu2**2 + e2**2*(54685.*PI/9072. - 54685.*PI*eta/1134. + 54685.*PI*eta**2/567.)/mu2**4))
                                    + v**6*((-82403040211200. + 2646364089600.*PI**2)*eta/14082647040. + 123839990.*eta**2/305613. + 18300845.*eta**3/1222452. + mu3**6*(12772489315./256048128. - 12772489315.*eta/21337344. + 12772489315.*eta**2/5334336. - 12772489315.*eta**3/4000752.)/mu2**6 + e4**2*(5741./1764.-11482.*eta/441.+28705.*eta**2/441.-22964.*eta**3/441.)/mu2**2 + e3**2*(27730./3969.-179990.*eta/3969.+341450.*eta**2/3969.-51050.*eta**3/1323.)/mu2**2 + e2**6*(5./11664. - 5*eta/972. + 5.*eta**2/243.-20.*eta**3/729.)/mu2**6+e2**4*(-265./1512. + 20165.*eta/13608.-5855.*eta**2/1701.+310.*eta**3/243.)/mu2**4 + (1002569.*mu5**2/12474.-4010276.*mu5**2*eta/6237.+10025690.*mu5**2*eta**2/6237.-8020552.*mu5**2*eta**3/6237.)/mu2**2 + e2**2*((3638245./190512. -2842015.*eta/31752.+760985.*eta**2/13608.-328675.*eta**3/23814.)/mu2**2 + e3**2*(-(50./567.) + 500.*eta/567. -550.*eta**2/189. +200.*eta**3/63.)/mu2**4) + mu3**4*((e2**2*(9343445./3048192. -9343445.*eta/254016.+9343445.*eta**2/63504.-9343445.*eta**3/47628.))/mu2**6 + (868749005./10668672.-2313421945.*eta/3556224.+191974645.*eta**2/148176. +9726205.*eta**3/666792.)/mu2**4) + mu4**2*((-(86554310./916839.) + (553387330.*eta)/916839. - (289401650.*eta**2)/305613. - (4322750.*eta**3)/101871.)/mu2**2 + (e2**2*(-(89650./35721.) + (896500.*eta)/35721. - (986150.*eta**2)/11907. + (358600.*eta**3)/3969.))/mu2**4) + mu3**2*((-(4809714655./29338848.) + (8024601785.*eta)/9779616. - (19149203695.*eta**2)/29338848. - (190583245.*eta**3)/7334712.)/mu2**2 + (e2**4*(6835./108864. - (6835.*eta)/9072. + (6835.*eta**2)/2268. - (6835.*eta**3)/1701.))/mu2**6 + (e2**2*(-(656195./95256.) + (229475.*eta)/3888. - (3369935.*eta**2)/23814. + (82795.*eta**3)/1323.))/mu2**4 + (e3**2*(-(34175./7938.) + (170875.*eta)/3969. - (375925.*eta**2)/2646. + (68350.*eta**3)/441.))/mu2**4 + (mu4**2*(-(61275775./500094.) + (306378875.*eta)/250047. - (674033525.*eta**2)/166698. + (122551550.*eta**3)/27783.))/mu2**4) + (1./14082647040.)*(36871736640768. - 4592284139520.*GammaE - 3004298035200.*PI**2 - 9184568279040.*np.log(2) - 4592284139520.*np.log(v)))
                             + v**5*(12080.*PI/189. - 3680.*PI*eta/63. + e2**2*(-20.*PI/27. + 80.*PI*eta/27.)/mu2**2 + mu3**2*(-9115.*PI/756. + 9115.*PI*eta/189.)/mu2**2)*(1 + 3.*np.log(v/vlso)))
    h = np.exp(lnA)*mu2*f**(-7./6.)*np.exp(1j*psi)
    return h


#Frequency range
ff = np.arange(fl, fh, df)

#CE-WB noise
def sh(f):
    s = 5.623746655206207e-51 + 6.698419551167371e-50*f**(-0.125) + 7.805894950092525e-31/f**20. + 4.35400984981997e-43/f**6. + 1.630362085130558e-53*f + 2.445543127695837e-56*f**2 + 5.456680257125753e-66*f**5
    return s

#Coloured gaussian (Adv LIGO)(generating the noise array---shilpa)

Noise=(np.random.normal(0., (sh(ff)/(4.*df))**(0.5)) +1j*np.random.normal(0.,(sh(ff)/(4.*df))**(0.5)))


#Defining the Data (d(f)=h(f)+n(f))(adding the noise array to the signal array---Shilpa)

Data=htilde(lnA_true, tc_true, phic_true, Mc_true, eta_true, mu2, mu3, mu4, mu5, e2_true, e3, e4, ff) + Noise

# Uniform prior on all parameter in their respective range
def lnprior(lnA, tc, phic, Mc, eta, e2):
    if -52.<lnA<-44. and -.1<tc<.1 and -PI<phic<PI and Mc_true-4.<Mc<Mc_true+6. and 0.1<eta<0.25 and -9<e2<11:
        l = 0.0
    else:
        l =-np.inf
    return l

#Calculating the innerproduct
def overlap(A, B, f):
    summ = 2.*np.real((((A*np.conjugate(B)+np.conjugate(A)*B)/sh(f)).sum()))*df
    return summ

# Define log likelihood
def lnlike(lnA, tc, phic, Mc, eta, e2):
    h = htilde(lnA, tc, phic, Mc, eta, mu2, mu3, mu4, mu5, e2, e3, e4, ff)
    return -0.5*overlap(Data-h, Data-h, ff)


# Multiplying likelihood with prior
def lnprob(lnA, tc, phic, Mc, eta, e2):
	lp = lnprior(lnA, tc, phic, Mc, eta, e2)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(lnA, tc, phic, Mc, eta, e2)


# Defining a function just for minimization routine to find a point to start
def func(theta):
	lnA, tc, phic, Mc, eta, e2 = theta
	return -2.*lnprob(lnA, tc, phic, Mc, eta, e2)

def lnp(theta):
	lnA, tc, phic, Mc, eta, e2 = theta
	return lnprob(lnA, tc, phic, Mc, eta, e2)


result = opt.minimize(func, [lnA_true, tc_true, phic_true, Mc_true, eta_true, e2_true])
lnA_ml, tc_ml, phic_ml, Mc_ml, eta_ml, e2_ml = result['x']


# Set up the sampler.
ndim, nwalkers = 6, 16
pos = [result['x'] + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 100)
#print (pos)
print("Done.")



# Plot for progression of sampler for each parameter
pl.clf()
fig, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(lnA_ml, color="#888888", lw=2)
axes[0].set_ylabel(r"$log(A)$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(tc_ml, color="#888888", lw=2)
axes[1].set_ylabel(r"$t_c$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(phic_ml, color="#888888", lw=2)
axes[2].set_ylabel(r"$phi_c$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(Mc_ml, color="#888888", lw=2)
axes[3].set_ylabel(r"$Mc$")


axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(eta_ml, color="#888888", lw=2)
axes[4].set_ylabel(r"$eta$")

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(e2_ml, color="#888888", lw=2)
axes[5].set_ylabel(r"$\epsilon_2$")

fig.tight_layout(h_pad=0.0)
fig.savefig("line-time-plot_of_the_prams_e2_5000_CEWB_q1.2_M%s.png"%(Mtotal[nodenum]))
#fig.show()

# Removing first 100 points as chain takes some time to stabilize
burnin = 10
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
# saving data in file
np.savetxt("emcee_sampler_e2_5000_CEWB_q1.2_M%s.dat"%(Mtotal[nodenum]),samples,fmt='%f', header="lnA tc phic Mc eta e2")
