"Analysis to determine the best choice of hyperparamaters."
import math
import numpy as np
import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

CAUSALS = [
    "causal_here.pkl",
    "causal_ubuntu@ec2-35-164-100-106.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-26-75-82.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-36-251-66.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-36-63-29.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-37-223-20.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-40-133-49.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-40-226-151.us-west-2.compute.amazonaws.com.pkl",
    "causal_ubuntu@ec2-52-40-234-245.us-west-2.compute.amazonaws.com.pkl"]

NONCAUSALS = [
    "noncausal_here.pkl",
    "noncausal_ubuntu@ec2-35-164-100-106.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-26-75-82.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-36-251-66.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-36-63-29.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-37-223-20.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-40-133-49.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-40-226-151.us-west-2.compute.amazonaws.com.pkl",
    "noncausal_ubuntu@ec2-52-40-234-245.us-west-2.compute.amazonaws.com.pkl"]


CAUSES = [joblib.load(resfile) for resfile in CAUSALS]
NCAUSES = [joblib.load(resfile) for resfile in NONCAUSALS]

CRES = [run for runs in CAUSES for run in runs]
NRES = [run for runs in NCAUSES for run in runs]

CNTS = [run['settings']['n_time'] for run in CRES]
NNTS = [run['settings']['n_time'] for run in NRES]

# for i, run in enumerate(CRES):
#     print("--------------------------")
#     print("Run: " + str(i))
#     print("Sharpe: "+str(run["stats"]["sharpe"]))
#     print("Max drawdown: "+str(run["stats"]["maxDD"]))
#     print("Time off peak: "+str(run["stats"]["maxTimeOffPeak"]))

STATS = ["sharpe", "returnYearly", "maxDD", "maxTimeOffPeak"]
SETS = ["n_time", "restart_variables", "lr_mult_base", "val_sharpe_threshold",
        "n_sharpe", "lr", "val_period", "lookback", "retrain_interval",
        "num_epochs", "lbd", "allow_shorting"]
CATS = STATS + SETS
ONEHOT = ["cost_type"]

# I want to just look at Sharpe vs. setting for each of the possible
# settings.
SETSNOLR = ["n_time", "restart_variables",
            "val_sharpe_threshold",
            "n_sharpe", "val_period", "lookback", "retrain_interval",
            "num_epochs", "allow_shorting"]
LRLIKE = ["lr_mult_base", "lr", "lbd"]
for stat in STATS:
    cstatvals = [run['stats'][stat] for run in CRES]
    nstatvals = [run['stats'][stat] for run in NRES]
    for setting in SETSNOLR:
        csetvals = [c['settings'][setting] for c in CRES]
        nsetvals = [n['settings'][setting] for n in NRES]
        plt.scatter(csetvals, cstatvals, c='b')
        plt.scatter(nsetvals, nstatvals, c='r')
        plt.xlabel(setting)
        plt.ylabel(stat)
        plt.savefig("cvnc_" + stat + "_vs_" + setting + ".png")
        plt.clf()
    for setting in LRLIKE:
        cvals = [math.log10(c['settings'][setting]) for c in CRES]
        nvals = [math.log10(n['settings'][setting]) for n in NRES]
        plt.scatter(cvals, cstatvals, c='b')
        plt.scatter(nvals, nstatvals, c='r')
        plt.xlabel(setting)
        plt.ylabel(stat)
        plt.savefig("cvnc_" + stat + "_vs_" + setting + ".png")
        plt.clf()

ALLRES = CRES + NRES
ALLSH = [run['stats']['sharpe'] for run in ALLRES]
np.mean(ALLSH)
np.std(ALLSH)
np.median(ALLSH)
min(ALLSH)
max(ALLSH)

plt.scatter(range(len(ALLSH)), ALLSH, c='b')
plt.xlabel('Run')
plt.ylabel('Sharpe Ratio')
plt.title('Mean: ' + str(round(np.mean(ALLSH), 2))
          + ', Standard Deviation:' + str(round(np.mean(ALLSH), 2)))
plt.savefig("allsharpe.png")
plt.clf()

MINSHR = next(run for run in ALLRES if run['stats']['sharpe'] == min(ALLSH))
FIFTHSHR = next(
    run for run in ALLRES if run['stats']['sharpe'] == sorted(ALLSH)[-5])
MAXSHR = next(run for run in ALLRES if run['stats']['sharpe'] == max(ALLSH))
MEDSHR = next(
    run for run in ALLRES if run['stats']['sharpe'] == np.median(ALLSH))

plt.plot(MINSHR['fundEquity'][200:])
plt.xlabel('Days after ' + str(MINSHR['fundDate'][900]))
plt.ylabel('Fund Equity')
plt.title('Run with Minimum Sharpe: '
          + str(round(MINSHR['stats']['sharpe'], 2)))
plt.savefig('min_sharpe_plot.png')
plt.clf()

plt.plot(MEDSHR['fundEquity'][200:])
plt.xlabel('Days after ' + str(MEDSHR['fundDate'][900]))
plt.ylabel('Fund Equity')
plt.title('Run with Median Sharpe: '
          + str(round(MEDSHR['stats']['sharpe'], 2)))
plt.savefig('med_sharpe_plot.png')
plt.clf()

plt.plot(MAXSHR['fundEquity'][200:])
plt.xlabel('Days after ' + str(MAXSHR['fundDate'][900]))
plt.ylabel('Fund Equity')
plt.title('Run with Maximum Sharpe: '
          + str(round(MAXSHR['stats']['sharpe'], 2)))
plt.savefig('max_sharpe_plot.png')
plt.clf()

plt.plot(FIFTHSHR['fundEquity'][200:])
plt.xlabel('Days after ' + str(FIFTHSHR['fundDate'][900]))
plt.ylabel('Fund Equity')
plt.title('Run with Fifth to Max Sharpe: '
          + str(round(FIFTHSHR['stats']['sharpe'], 2)))
plt.savefig('fifth_max_sharpe_plot.png')
plt.clf()


# scp -i causeai.pem ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com:/home/ubuntu/data/causehf/saved_data/max_sharpe_plot.png .
# scp -i causeai.pem ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com:/home/ubuntu/data/causehf/saved_data/min_sharpe_plot.png .
# scp -i causeai.pem ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com:/home/ubuntu/data/causehf/saved_data/med_sharpe_plot.png .
# scp -i causeai.pem ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com:/home/ubuntu/data/causehf/saved_data/fifth_max_sharpe_plot.png .


# for i, run in enumerate(ALLSETS):
#     ares = np.array([])
#     for setting in SETS:
#         ares = np.hstack((ares, np.array(run[setting])))
#     if 'hres' not in locals():
#         hres = np.array(ares)
#     else:
#         hres = np.vstack((hres, np.array(ares)))

# from matplotlib.mlab import PCA
# hpca = PCA(hres)
