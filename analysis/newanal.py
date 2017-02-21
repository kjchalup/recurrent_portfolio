"Analysis to determine the best choice of hyperparamaters."
import math
import numpy as np
import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

RESFILES = ["ubuntu@ec2-52-37-223-20.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-26-75-82.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-35-165-35-72.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-40-226-151.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-40-234-245.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-40-133-49.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-36-63-29.us-west-2.compute.amazonaws.com.pkl",
            "ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com.pkl"]

RES = [joblib.load(resfile) for resfile in RESFILES]
RESULTS = [run for runs in RES for run in runs]

# Add all of the successful runs to GOODS.
GOODS = []
BADS = []

for i, run in enumerate(RESULTS):
    worked = True
    print("--------------------------")
    print("Run: " + str(i))
    try:
        print("Yearly returns: "+str(run["stats"]["returnYearly"]))
    except KeyError:
        print("")
        print("Too much drawdown.")
        BADS.append(run)
        worked = False
    if worked:
        GOODS.append(run)
        print("Sharpe: "+str(run["stats"]["sharpe"]))
        print("Max drawdown: "+str(run["stats"]["maxDD"]))
        print("Time off peak: "+str(run["stats"]["maxTimeOffPeak"]))

STATS = ["sharpe", "returnYearly", "maxDD", "maxTimeOffPeak"]
SETS = ["n_time", "restart_variables", "lr_mult_base", "val_sharpe_threshold",
        "n_sharpe", "lr", "val_period", "lookback", "retrain_interval",
        "num_epochs", "lbd", "allow_shorting"]
CATS = STATS + SETS
ONEHOT = ["cost_type"]

# See the values of all the settings.
for setting in SETS:
    print("---------------------")
    print("Setting: " + setting)
    print([good['settings'][setting] for good in GOODS])
    #print([bad[setting] for bad in BADS])

# I want to just look at Sharpe vs. setting for each of the possible
# settings.
SHARPES = [good['stats']['sharpe'] for good in GOODS]
#NOSHARPES = [-1.75 for bad in BADS]
SETSNOLR = ["n_time", "restart_variables",
            "val_sharpe_threshold",
            "n_sharpe", "val_period", "lookback", "retrain_interval",
            "num_epochs", "allow_shorting"]
LRLIKE = ["lr_mult_base", "lr", "lbd"]
for setting in SETSNOLR:
    #badvals = [bad[setting] for bad in BADS]
    goodvals = [good['settings'][setting] for good in GOODS]
    plt.scatter(goodvals, SHARPES, c='b')
    #plt.scatter(badvals, NOSHARPES, c='r')
    plt.xlabel(setting)
    plt.ylabel("Sharpe Ratio")
    plt.savefig("sharpe_vs_" + setting + ".png")
    plt.clf()

for setting in LRLIKE:
    #badvals = [math.log10(bad[setting]) for bad in BADS]
    goodvals = [math.log10(good['settings'][setting]) for good in GOODS]
    plt.scatter(goodvals, SHARPES, c='b')
    #plt.scatter(badvals, NOSHARPES, c='r')
    plt.xlabel("log(" + setting + ")")
    plt.ylabel("Sharpe Ratio")
    plt.savefig("sharpe_vs_" + setting + ".png")
    plt.clf()

GOODSETS = [good['settings'] for good in GOODS]
ALLSETS = GOODSETS + BADS
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
