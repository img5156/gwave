#!/usr/local/bin/python

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


# phase of h(f)
def Phif3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, tc, phi_c):
    gt = 4.92549094830932e-6  # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    vlso = 1.0 / np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = (np.pi * M * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta ** 2
    eta3 = eta ** 3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    chi1sq = s1x * s1x + s1y * s1y + s1z * s1z
    chi2sq = s2x * s2x + s2y * s2y + s2z * s2z
    chi1dotchi2 = s1x * s2x + s1y * s2y + s1z * s2z
    SL = m1M * m1M * chi1L + m2M * m2M * chi2L
    dSigmaL = delta * (m2M * chi2L - m1M * chi1L)

    # phase correction due to spins
    sigma = eta * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1L * chi1L + m2M * m2M * chi2L * chi2L)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (13915.0 / 84.0 - 10.0 * eta / 3.0) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1L * chi2L
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M) + (
                -4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)) * m1M * m1M * chi1sq
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M) + (
                -4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)) * m2M * m2M * chi2sq
    phis_3PN = np.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (-8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0) * SL \
                - (170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0) * dSigmaL

    # tidal correction to phase
    # Lam is the reduced tidal deformation parameter \tilde\Lam
    # dLam is the asymmetric reduced tidal deformation parameter, which is usually negligible
    #tidal = Lam * v10 * (- 39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12

    return 2*np.pi*f*tc - phi_c - np.pi/4 + (3.0 / 128.0 / eta / v5 * (
                1.0 + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2 + (phis_15PN - 16.0 * np.pi) * v3 \
                + 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma) * v4 \
                + (38645.0 / 756.0 * np.pi - 65.0 / 9.0 * eta * np.pi - ga) * (1.0 + 3.0 * np.log(v / vlso)) * v5 \
                + (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * np.pi ** 2 - 6848.0 / 21.0 * (
                    EulerGamma + np.log(4.0 * v)) + \
                   (
                               -15737765635.0 / 3048192.0 + 2255.0 * np.pi ** 2 / 12.0) * eta + 76055.0 / 1728.0 * eta2 - 127825.0 / 1296.0 * eta3 + phis_3PN) * v6 \
                + (np.pi * (
                    77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta ** 2) + phis_35PN) * v7)) # +tidal)


# correction to modulus of h(f)
# normalization should be C * (GN*Mchirp/c^3)^5/6 * f^-7/6 / D / pi^2/3 * (5/24)^1/2
def Af3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    gt = 4.92549094830932e-6  # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = (np.pi * M * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta ** 2
    eta3 = eta ** 3

    # modulus correction due to aligned spins
    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    be = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta * chis)
    sigma = chia ** 2 * (81.0 / 16.0 - 20.0 * eta) + 81.0 / 8.0 * chia * chis * delta + chis ** 2 * (
                81.0 / 16.0 - eta / 4.0)
    eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta) + chis * (
                5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta + 502429.0 / 16128.0)

    return 1.0 + v2 * (11.0 / 8.0 * eta + 743.0 / 672.0) + v3 * (be / 2.0 - 2.0 * np.pi) + v4 * (
                1379.0 / 1152.0 * eta2 + 18913.0 / 16128.0 * eta + 7266251.0 / 8128512.0 - sigma / 2.0) \
           + v5 * (57.0 / 16.0 * np.pi * eta - 4757.0 * np.pi / 1344.0 + eps) \
           + v6 * (
                       856.0 / 105.0 * EulerGamma + 67999.0 / 82944.0 * eta3 - 1041557.0 / 258048.0 * eta2 - 451.0 / 96.0 * np.pi ** 2 * eta + 10.0 * np.pi ** 2 / 3.0 \
                       + 3526813753.0 / 27869184.0 * eta - 29342493702821.0 / 500716339200.0 + 856.0 / 105.0 * np.log(
                   4.0 * v)) \
           + v7 * (- 1349.0 / 24192.0 * eta2 - 72221.0 / 24192.0 * eta - 5111593.0 / 2709504.0) * np.pi

def Af3hPN_rb(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, theta, psi, phi):
    gt = 4.92549094830932e-6  # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = (np.pi * M * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta ** 2
    eta3 = eta ** 3
    gmst  = 40566.97325177817
    
    # modulus correction due to aligned spins
    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    be = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta * chis)
    sigma = chia ** 2 * (81.0 / 16.0 - 20.0 * eta) + 81.0 / 8.0 * chia * chis * delta + chis ** 2 * (
                81.0 / 16.0 - eta / 4.0)
    eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta) + chis * (
                5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta + 502429.0 / 16128.0)
    
    Fp_H = 0.5*(-(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))*(-0.4949984203281415*np.cos(theta)*np.cos(psi) - 0.155295733350767*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
           + np.cos(gmst - phi)*np.sin(psi)) - 0.7851296988257865*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) - np.cos(theta)*np.cos(psi)*(0.1466757550558504*np.cos(theta)*np.cos(psi)
           + 0.45615443587480603*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi)) - 0.4949984203281415*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi)))
           - (np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi))*(0.45615443587480603*np.cos(theta)*np.cos(psi) + 0.6384539437699361*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
           + np.cos(gmst - phi)*np.sin(psi)) - 0.155295733350767*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi)))
           + (-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))*(-0.4949984203281415*np.cos(theta)*np.sin(psi) - 0.7851296988257865*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))
           - 0.155295733350767*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))+ np.cos(theta)*np.sin(psi)*(0.1466757550558504*np.cos(theta)*np.sin(psi)
           - 0.4949984203281415*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) + 0.45615443587480603*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))
           + (-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))*(0.45615443587480603*np.cos(theta)*np.sin(psi) - 0.155295733350767*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))
           + 0.6384539437699361*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))))

    Fx_H = 0.5*((-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))*(-0.4949984203281415*np.cos(theta)*np.cos(psi) - 0.155295733350767*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
          + np.cos(gmst - phi)*np.sin(psi)) - 0.7851296988257865*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) + np.cos(theta)*np.sin(psi)*(0.1466757550558504*np.cos(theta)*np.cos(psi)
          + 0.45615443587480603*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi)) - 0.4949984203281415*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi)))
          + (-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))*(0.45615443587480603*np.cos(theta)*np.cos(psi) + 0.6384539437699361*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
          + np.cos(gmst - phi)*np.sin(psi)) - 0.155295733350767*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) + (-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta)
          + np.sin(gmst - phi)*np.sin(psi))*(-0.4949984203281415*np.cos(theta)*np.sin(psi) - 0.7851296988257865*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))
          - 0.155295733350767*(-np.cos(gmst - phi)*np.cos(psi) +np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))) + np.cos(theta)*np.cos(psi)*(0.1466757550558504*np.cos(theta)*np.sin(psi)
          - 0.4949984203281415*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) + 0.45615443587480603*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))
          + (np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi))*(0.45615443587480603*np.cos(theta)*np.sin(psi) - 0.155295733350767*(-np.cos(psi)*np.sin(gmst - phi)
          - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) + 0.6384539437699361*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))))

    Fp_L= 0.5*(-(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi))*(-0.36346460885209697*np.cos(theta)*np.cos(psi) - 0.2184495631041813*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
          + np.cos(gmst - phi)*np.sin(psi)) + 0.28082805583982157*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) - np.cos(theta)*np.cos(psi)*(-0.6041263080383013*np.cos(theta)*np.cos(psi)
          - 0.36346460885209697*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi)) + 0.49433707673015326*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi)))
          - (-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))*(0.49433707673015326*np.cos(theta)*np.cos(psi) + 0.28082805583982157*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
          + np.cos(gmst - phi)*np.sin(psi)) + 0.8225758711424823*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) + np.cos(theta)*np.sin(psi)*(-0.6041263080383013*np.cos(theta)*np.sin(psi)
          + 0.49433707673015326*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) - 0.36346460885209697*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))
          + (-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))*(-0.36346460885209697*np.cos(theta)*np.sin(psi)
          + 0.28082805583982157*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) - 0.2184495631041813*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))
          + (-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))*(0.49433707673015326*np.cos(theta)*np.sin(psi) + 0.8225758711424823*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))
          + 0.28082805583982157*(-np.cos(gmst- phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))))

    Fx_L = 0.5*((-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))*(-0.36346460885209697*np.cos(theta)*np.cos(psi) - 0.2184495631041813*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi)
          + np.cos(gmst - phi)*np.sin(psi)) + 0.28082805583982157*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) + np.cos(theta)*np.sin(psi)*(-0.6041263080383013*np.cos(theta)*np.cos(psi)
          - 0.36346460885209697*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi)) + 0.49433707673015326*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi)))
          + (-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))*(0.49433707673015326*np.cos(theta)*np.cos(psi) + 0.28082805583982157*(np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi))
          + 0.8225758711424823*(-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))) + np.cos(theta)*np.cos(psi)*(-0.6041263080383013*np.cos(theta)*np.sin(psi)
          + 0.49433707673015326*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) - 0.36346460885209697*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi)))
          + (np.cos(psi)*np.sin(theta)*np.sin(gmst - phi) + np.cos(gmst - phi)*np.sin(psi))*(-0.36346460885209697*np.cos(theta)*np.sin(psi) + 0.28082805583982157*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi))
          - 0.2184495631041813*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))) + (-np.cos(gmst - phi)*np.cos(psi)*np.sin(theta) + np.sin(gmst - phi)*np.sin(psi))*(0.49433707673015326*np.cos(theta)*np.sin(psi)
          + 0.8225758711424823*(-np.cos(psi)*np.sin(gmst - phi) - np.cos(gmst - phi)*np.sin(theta)*np.sin(psi)) + 0.28082805583982157*(-np.cos(gmst - phi)*np.cos(psi) + np.sin(theta)*np.sin(gmst - phi)*np.sin(psi))))
    
    return [Fp_H, Fx_H, Fp_L, Fx_L]


# combine the modulus and the phase of h(f)
def hf3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, Deff=1.0):
    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    Mchirp = M * eta ** 0.6
    A0 = Mchirp ** (5.0 / 6.0) / (f + 1e-100) ** (7.0 / 6.0) / Deff / np.pi ** (2.0 / 3.0) * np.sqrt(5.0 / 24.0)

    Phi = Phif3hPN(f, M, eta, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z, Lam=Lam, dLam=dLam)
    A = Af3hPN(f, M, eta, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z, Lam=Lam, dLam=dLam)

    # note the convention for the sign in front of the phase
    return pre * A0 * A * np.exp(-1.0j * Phi)
            
def hf3hPN_H(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, Deff=1.0, theta, psi, phi, Dl, i, tc, phi_c):
 
    A0 = Af3hPN_rb(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, theta, psi, phi)
    # note the convention for the sign in front of the phase
    
    phase = Phif3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, tc, phi_c)
    
    h_p = 0.5*(1+(np.cos(i))**2.0)*(f**(-7.0/6.0))*np.exp(1j*phase)/Dl
    h_x = -0.5*(np.cos(i)/Dl)*(f**(-7.0/6.0))*np.exp(1j*phase)
    
    return A0[0]*h_p + A0[1]*h_x
            
def hf3hPN_L(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, Deff=1.0, theta, psi, phi, Dl, i, tc, phi_c):
    
    A0 = Af3hPN_rb(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, theta, psi, phi)
    # note the convention for the sign in front of the phase
    
    phase = Phif3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, tc, phi_c)
    
    h_p = 0.5*(1+(np.cos(i))**2.0)*(f**(-7.0/6.0))*np.exp(1j*phase)/Dl
    h_x = -0.5*(np.cos(i)/Dl)*(f**(-7.0/6.0))*np.exp(1j*phase)
    
    return A0[2]*h_p + A0[3]*h_x


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

    Mc, eta, chieff, chia, Lam, theta, psi, phi, Dl, i, tc, phi_c = par
    M = Mc / eta ** 0.6
    delta = np.sqrt(1.0 - 4.0 * eta)
    chis = chieff - delta * chia
    s1z = chis + chia
    s2z = chis - chia

    # waveform ratio
    r = hf3hPN_L(f, M, eta, s1z=s1z, s2z=s2z, Lam=Lam, theta, psi, phi, Dl, i, tc, phi_c) / h0_bin * np.exp(-2.0j * np.pi * f * dtc)
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

    Mc, eta, chieff, chia, Lam, theta, psi, phi, Dl, i, tc, phi_c = par[0:12]
    dtc = par[12:]
    par_notc = [Mc, eta, chieff, chia, Lam, theta, psi, phi, Dl, i, tc, phi_c]

    # relative waveform
    rf = [compute_rf(par_notc + [dtc[k]], h0[k], fbin, fbin_ind) for k in range(ndtct)]

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
    Mc_bf = res[0]
    eta_bf = res[1]
    chieff_bf = res[2]
    chia_bf = res[3]
    Lam_bf = res[4]
    theta_bf = res[5]
    psi_bf = res[6]
    phi_bf = res[7]
    Dl_bf = res[8]
    i_bf = res[9]
    tc_bf = res[10]
    phi_c_bf = res[11]
    dtc_bf = res[12:]

    # output best-fit parameters if requested
    if verbose is True:
        print('log-likelihood = ', lnl)
        print('Mc_bf =', Mc_bf)
        print('eta_bf =', eta_bf)
        print('chieff_bf =', chieff_bf)
        print('chia_bf =', chia_bf)
        print('Lam_bf =', Lam_bf)
        print('theta_bf =', theta_bf)
        print('psi_bf =', psi_bf)
        print('phi_bf =', phi_bf)
        print('Dl_bf =', Dl_bf)
        print('i_bf =', i_bf)
        print('tc_bf =', tc_bf)
        print('phi_c_bf =', phi_c_bf)
        print('dtc_bf =', dtc_bf)

    return res
