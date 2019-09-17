#!/usr/local/bin/python

import numpy as np
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import signal


# construct frequency bins for relative binning
def setup_bins(f_full, f_lo, f_hi, chi=1.0, eps=0.5):
    """
    construct frequency binning
    f_full: full frequency grid
    [f_lo, f_hi] is the frequency range one would like to use for matched filtering
    chi, eps are tunable parameters [see Barak, Dai & Venumadhav 2018]
    return the number of bins, a list of bin-edge frequencies, and their positions in the full frequency grid
    """
    f = np.linspace(f_lo, f_hi, 10000)
    # f^ga power law index
    ga = np.array([-5.0 / 3.0, -2.0 / 3.0, 1.0, 5.0 / 3.0, 7.0 / 3.0])
    dalp = chi * 2.0 * np.pi / np.absolute(f_lo ** ga - f_hi ** ga)
    dphi = np.sum(np.array([np.sign(ga[i]) * dalp[i] * f ** ga[i] for i in range(len(ga))]), axis=0)
    Dphi = dphi - dphi[0]
    # now construct frequency bins
    Nbin = int(Dphi[-1] // eps)
    Dphi2f = interp1d(Dphi, f, kind='slinear', bounds_error=False, fill_value=0.0)
    Dphi_grid = np.linspace(Dphi[0], Dphi[-1], Nbin + 1)
    # frequency grid points
    fbin = Dphi2f(Dphi_grid)
    # indices of frequency grid points in the FFT array
    fbin_ind = np.array([np.argmin(np.absolute(f_full - ff)) for ff in fbin])
    # make sure grid points are precise
    fbin = np.array([f_full[i] for i in fbin_ind])

    return (Nbin, fbin, fbin_ind)


# compute summary data given a bin partition and fiducial waveforms
def compute_sdat(f, fbin, fbin_ind, ndtct, psd, sFT, h0):
    """
    Compute summary data
    Need to compute for each detector
    Parameters:
        f is the full frequency grid (regular grid; length = n_sample/2 + 1)
        fbin is the bin edges
        fbin_ind gives the positions of bin edges in the full grid
        ndtct is the number of detectors
        psd is a list of PSDs
        sFT is a list of frequency-domain strain data
        h0  is a list of fiducial waveforms
    Note that sFT and h0 need to be provided with the full frequency resolution
    """
    # total number of frequency bins
    Nbin = len(fbin) - 1
    # total duration of time-domain sequence
    T = 1.0 / (f[1] - f[0])

    # arrays to store summary data
    sdat_A0 = []
    sdat_A1 = []
    sdat_B0 = []
    sdat_B1 = []

    # loop over detectors
    for k in range(ndtct):
        a0 = np.array([4.0 * np.sum(sFT[k][fbin_ind[i]:fbin_ind[i + 1]] \
                                    * np.conjugate(h0[k][fbin_ind[i]:fbin_ind[i + 1]]) \
                                    * T / psd[k][fbin_ind[i]:fbin_ind[i + 1]]) for i in range(Nbin)])

        b0 = np.array([4.0 * np.sum(np.absolute(h0[k][fbin_ind[i]:fbin_ind[i + 1]]) ** 2 \
                                    * T / psd[k][fbin_ind[i]:fbin_ind[i + 1]]) for i in range(Nbin)])

        a1 = np.array([4.0 * np.sum(sFT[k][fbin_ind[i]:fbin_ind[i + 1]] \
                                    * np.conjugate(h0[k][fbin_ind[i]:fbin_ind[i + 1]]) \
                                    * T / psd[k][fbin_ind[i]:fbin_ind[i + 1]] \
                                    * (f[fbin_ind[i]:fbin_ind[i + 1]] - 0.5 * (fbin[i] + fbin[i + 1]))) for i in
                       range(Nbin)])

        b1 = np.array([4.0 * np.sum(np.absolute(h0[k][fbin_ind[i]:fbin_ind[i + 1]]) ** 2 \
                                    * T / psd[k][fbin_ind[i]:fbin_ind[i + 1]] \
                                    * (f[fbin_ind[i]:fbin_ind[i + 1]] - 0.5 * (fbin[i] + fbin[i + 1]))) for i in
                       range(Nbin)])

        sdat_A0.append(a0)
        sdat_A1.append(a1)
        sdat_B0.append(b0)
        sdat_B1.append(b1)

    return [sdat_A0, sdat_A1, sdat_B0, sdat_B1]
