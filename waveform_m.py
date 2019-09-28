import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.optimize import minimize, differential_evolution


# need to provide a waveform-generating function which takes an array of frequencies as input and generate h(f)
# one can replace this function with whatever waveform generating function one would like to use

# ---------- Analytical TaylorF2 frequency-domain waveform -------------- #
# including aligned spin effects
# tidal effects added
# circular binary 3.5PN GW phase
# wave frequency f [Hz]
# total mass M [Msun]
# eta is the symmetric mass ratio
# we adopt the convention that m1 >= m2
# Lam is the reduced tidal parameter \tilde\Lam
# aligned spin components: s1z and s2z
# chieff is the effective spin parameters: chieff = (m1*s1z + m2*s2z)/(m1 + m2)
# delta is the asymmetric mass ratio: delta = (m1 - m2)/(m1 + m2)
# By default we neglect spin and tidal deformation

# step function
def myHeaviside(x):
    return np.heaviside(x, 1.0)

MTSUN_SI = 4.92549102554*1e-6
PC_SI = 3.086*1e16
C_SI = 2.998*1e8
PI =  np.pi
GammaE = 0.577215664901532
    
def f_isco(M):
    return 1./6.**(3./2.)/np.pi/M

# combine the modulus and the phase of h(f)
def hf3hPN(f, lnA, phic, Mc, eta, e2, tc, mu2=1.0, mu3=1.0, mu4=1.0, mu5 = 1.0, e3=1.0, e4=1.0):

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


# compute matched filter SNR^2
def mfSNR2_amp_phic_max_sdat(sdat, k, rdata):
    """
    Compute matched filtering SNR^2 for the k-th detector using summary data: <sFT[k]|hFT>^2 / <hFT|hFT>
    This is maximized with respect to an arbitrary normalization of the template, as well as an arbitrary phase
    Summary data is passed through: sdat = [sdat_A0, sdat_A1, sdat_B0, sdat_B1]; each entry is a list for all detectors
    rdata = [r0, r1] is the relative waveform sampled at the bin edges
    """
    r0, r1 = rdata
    denom = np.sum(sdat[2][k] * np.absolute(r0) ** 2 + sdat[3][k] * 2.0 * (r0 * np.conjugate(r1)).real)
    X = np.sum(sdat[0][k] * np.conjugate(r0) + sdat[1][k] * np.conjugate(r1))
    res = np.absolute(X) ** 2 / denom
    # print('denom = ', denom)
    # print('overlap = ', res)
    return res


# compute relative waveform r(f) = h(f)/h0(f)
def compute_rf(par, h0, fbin, fbin_ind):
    """
    compute the ratio r(f) = h(f)/h0(f) where h0(f) is some fiducial waveform and h(f) correspond to parameter combinations par
    h0: fiducial waveform (it is important to pass one that is NOT shifted to the right merger time)
    fbin: frequency bin edges
    par is list of parameters: [Mc, eta, chieff, chia, Lam, dtc]
    tc: best-fit time
    """

    f = fbin  # only evaluate at frequency bin edges
    h0_bin = h0[fbin_ind]

    lnA, phic, Mc, eta, e2, tc = par

    # waveform ratio
    r = hf3hPN(f, lnA, phic, Mc, eta, e2, tc) / h0_bin * np.exp(-2.0j * np.pi * f * tc)
    r0 = 0.5 * (r[:-1] + r[1:])
    r1 = (r[1:] - r[:-1]) / (f[1:] - f[:-1])

    return np.array([r0, r1], dtype=np.complex128)


# compute minus log-likelihood
def lnlike(par, sdat, h0, fbin, fbin_ind, ndtct):
    """
    Return log of the likelihood function
    par: list of parameters [Mc, eta, chieff, chia, Lam, dtc1, dtc2] (two time constants tc1 and tc2 for two detectors)
    sdat: summary data
    h0: a list of fiducial waveforms
    fbin: bin edge frequencies
    fbin_ind: bin edge positions in the full frequency grid
    ndtct: number of detectors
    """

    lnl = 0.0
    lnA, phic, Mc, eta, e2= par[0:5]
    tc = par[5:]
    par_notc = [lnA, phic, Mc, eta, e2]

    # relative waveform
    rf = [compute_rf(par_notc + [tc[k]], h0[k], fbin, fbin_ind) for k in range(ndtct)]

    # Total log-likelihood is the sum of all detectors
    for k in range(ndtct):
        # The log-likelihood function is analytically maximized with respect to the phase constant and the overall amplitude
        lnl -= 0.5 * mfSNR2_amp_phic_max_sdat(sdat, k, rf[k])

    return lnl


# ----------- find best-fit parameters using minimization ------------- #
def get_best_fit(sdat, par_bounds, h0, fbin, fbin_ind, ndtct, maxiter=100, atol=1e-3, verbose=False):
    """
    Find the best-fit binary parameters that fit the data (with detector noise added)
    sdat is the summary data
    par_bounds is a list of allowed range for all parameters
    """
    # use differential evolution from scipy
    output = differential_evolution(lnlike, bounds=par_bounds, \
                                    args=(sdat, h0, fbin, fbin_ind, ndtct), atol=atol, maxiter=maxiter)
    res = output['x']
    lnl = -output['fun']

    # best-fit parameters
    lnA_bf = res[0]
    phic_bf = res[1]
    Mc_bf = res[2]
    eta_bf = res[3]
    e2_bf = res[4]
    tc_bf = res[5:]
    
    # output best-fit parameters if requested
    if verbose is True:
        print('log-likelihood = ', lnl)
        print('lnA_bf =', lnA_bf)
        print('phic_bf =', phic_bf)
        print('Mc_bf =', Mc_bf)
        print('eta_bf =', eta_bf)
        print('e2_bf =', e2_bf)
        print('tc_bf =', tc_bf)

    return res
