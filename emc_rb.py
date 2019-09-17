import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import corner
import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument("--Mc", help = "injected Mc value", type=float)
parser.add_argument("--eta", help = "injected eta value", type=float)
parser.add_argument("--chieff", help = "injected chieff value", type=float)
parser.add_argument("--chia", help = "injected chia value", type=float)
parser.add_argument("--lam", help = "injected lam value", type=float)

parser.add_argument("--tc1", help = "injected tc1 value", type=float)
parser.add_argument("--tc2", help = "injected tc2 value", type=float)
#parser.add_argument("--nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

Mc_true=args.Mc
eta_true=args.eta
chieff_true=args.chieff
chia_true=args.chia
lam_true=args.lam
tc1_true=args.tc1
tc2_true=args.tc2

Mtotal = np.array([10, 20])

samples = np.loadtxt('emcee_sampler_rb3_5k_1w_noopt_bound0.dat')

#Corner plot
fig = corner.corner(samples, labels=["M_c", "$\eta$", "$\chi_eff$", "$\chi_a$", "$\lambda$", "$tc_1$", "$tc_2$"],
		truths=[Mc_true,eta_true, chieff_true, chia_true, lam_true, tc1_true, tc2_true])
fig.suptitle("one-sigma levels")
fig.savefig('emcee_posterior_rb3_5k_1w_noopt_bound0.png')

Mc_mcmc, eta_mcmc, chieff_mcmc, chia_mcmc, lam_mcmc, tc1_mcmc, tc2_mcmc = map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print("True values of the parameters:")
print("Mc= ", Mc_true)
print("eta= ", eta_true)
print("chieff= ", chieff_true)
print("chia= ", chia_true)
print("lam= ", lam_true)
print("tc1= ", tc1_true)
print("tc2= ", tc2_true)

print("")

print("median and error in the parameters")
print("Mc: ", Mc_mcmc[0], Mc_mcmc[2]-Mc_mcmc[1])
print("eta: ", eta_mcmc[0], eta_mcmc[2]-eta_mcmc[1])
print("chieff: ", chieff_mcmc[0], chieff_mcmc[2]-chieff_mcmc[1])
print("chia: ", chia_mcmc[0], chia_mcmc[2]-chia_mcmc[1])
print("lam: ", lam_mcmc[0], lam_mcmc[2]-lam_mcmc[1])
print("tc1: ", tc1_mcmc[0], tc1_mcmc[2]-tc1_mcmc[1])
print("tc2: ", tc2_mcmc[0], tc2_mcmc[2]-tc2_mcmc[1])