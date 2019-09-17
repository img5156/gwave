import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import corner
import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument("--lnA", help = "injected lnA value", type=float)
parser.add_argument("--tc", help = "injected tc value", type=float)
parser.add_argument("--phic", help = "injected phic value", type=float)
parser.add_argument("--Mc", help = "injected Mc value", type=float)
parser.add_argument("--eta", help = "injected eta value", type=float)
parser.add_argument("--e2", help = "injected e2 value", type=float)
parser.add_argument("--nodenum", help = "Node number for the condor runs", type=int)
args = parser.parse_args()

lnA_true=args.lnA
tc_true=args.tc
phic_true=args.phic
Mc_true=args.Mc
eta_true=args.eta
e2_true=args.e2

Mtotal = np.array([10, 20])

samples = np.loadtxt('emcee_sampler_e2_5000_%s.dat'%(Mtotal[args.nodenum]))

#Corner plot
fig = corner.corner(samples, labels=["lnA", r"$t_c$", "$\phi_c$", "$Mc$", "$\eta$", "$\epsilon_2$"],
		truths=[lnA_true,tc_true, phic_true, Mc_true, eta_true, e2_true])
fig.suptitle("one-sigma levels")
fig.savefig('emcee_posterior_e2_5000_M%s.png'%(Mtotal[args.nodenum]))

lnA_mcmc, tc_mcmc, phic_mcmc, Mc_mcmc, eta_mcmc, e2_mcmc = map(lambda v: (v[1], v[0],v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print ("True values of the parameters:")
print ("lnA= ", lnA_true)
print ("tc= ", tc_true)
print "phic= ", phic_true 
print "Mc= ", Mc_true
print "eta= ", eta_true
print "e2= ", e2_true


print ""

print "median and error in the parameters" 
print "tc: ", lnA_mcmc[0], lnA_mcmc[2]-lnA_mcmc[1]
print "tc: ", tc_mcmc[0], tc_mcmc[2]-tc_mcmc[1]
print "phic: ", phic_mcmc[0], phic_mcmc[2]-phic_mcmc[1]
print "Mc: ", Mc_mcmc[0], Mc_mcmc[2]-Mc_mcmc[1]  
print "Mc: ", eta_mcmc[0], eta_mcmc[2]-eta_mcmc[1]
print "e2: ", e2_mcmc[0], e2_mcmc[2]-e2_mcmc[1] 
