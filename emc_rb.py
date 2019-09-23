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
parser.add_argument("--theta", help = "injected theta value", type=float)
parser.add_argument("--psi", help = "injected psi value", type=float)
parser.add_argument("--phi", help = "injected phi value", type=float)
parser.add_argument("--Dl", help = "injected Dl value", type=float)
parser.add_argument("--i", help = "injected i value", type=float)
parser.add_argument("--phi_c", help = "injected phi_c value", type=float)
parser.add_argument("--tc1", help = "injected tc1 value", type=float)
#parser.add_argument("--nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

Mc_true=args.Mc
eta_true=args.eta
chieff_true=args.chieff
chia_true=args.chia
lam_true=args.lam
theta_true=args.theta
psi_true=args.psi
phi_true=args.phi
Dl_true=args.Dl
i_true=args.i
phi_c_true=args.phi_c
tc1_true=args.tc1

Mtotal = np.array([10, 20])

samples = np.loadtxt('12d_emcee_sampler_rb3_10k_1w.dat')

#Corner plot
fig1 = corner.corner(samples, labels=["M_c", "$\eta$", "$\chi_eff$", "$\chi_a$", "$\lambda$", "$theta$", "$\psi$", "$\phi$", "$D_l$", "$i$", "$\phi_c$", "$tc_1$"],
		truths=[Mc_true, eta_true, chieff_true, chia_true, lam_true, theta_true, psi_true, phi_true, Dl_true, i_true, phi_c_true, tc1_true])
fig1.suptitle("one-sigma levels")
fig1.savefig('plot_12d_emcee_sampler_rb3_10k_1w.pdf')

Mc_mcmc, eta_mcmc, chieff_mcmc, chia_mcmc, lam_mcmc, theta_mcmc, psi_mcmc, phi_mcmc, Dl_mcmc, i_mcmc, phi_c_mcmc, tc1_mcmc= map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#fig2 = corner.corner(samples, labels=["$theta$", "$\psi$", "$\phi$", "$D_l$", "$i$", "$\phi_c$"],
#		truths=[theta_true, psi_true, phi_true, Dl_true, i_true, phi_c_true])
#fig2.suptitle("one-sigma levels")
#fig2.savefig('plot_12d_emcee_sampler_int_rb3_5k_1w.pdf')

#theta_mcmc, psi_mcmc, phi_mcmc, Dl_mcmc, i_mcmc, phi_c_mcmc= map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

quit()

print("True values of the parameters:")
print("Mc= ", Mc_true)
print("eta= ", eta_true)
print("chieff= ", chieff_true)
print("chia= ", chia_true)
print("lam= ", lam_true)
print("dd= ", tc1_true)
print("tc1= ", tc1_true)
print("tc1= ", tc1_true)
print("tc1= ", tc1_true)
print("tc1= ", tc1_true)
print("tc1= ", tc1_true)
print("tc1= ", tc1_true)


print("")

print("median and error in the parameters")
print("Mc: ", Mc_mcmc[0], Mc_mcmc[2]-Mc_mcmc[1])
print("eta: ", eta_mcmc[0], eta_mcmc[2]-eta_mcmc[1])
print("chieff: ", chieff_mcmc[0], chieff_mcmc[2]-chieff_mcmc[1])
print("chia: ", chia_mcmc[0], chia_mcmc[2]-chia_mcmc[1])
print("lam: ", lam_mcmc[0], lam_mcmc[2]-lam_mcmc[1])
print("tc1: ", tc1_mcmc[0], tc1_mcmc[2]-tc1_mcmc[1])
print("tc2: ", tc2_mcmc[0], tc2_mcmc[2]-tc2_mcmc[1])
