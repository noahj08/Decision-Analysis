import pandas as pd
import numpy as np
from scipy.stats import beta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import argparse

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

# settings for seaborn plotting styles
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(4.5,3)})
# Reading in data
data = {
        "year": [],
        "storm": [],
        "mold": []
}
raw_data = pd.ExcelFile("Desktop\\252shit.xlsx").parse().values.tolist()
for i in range(2, len(raw_data)):
    if (i % 5 == 2):
        data["year"].append(raw_data[i][0])
    elif (i % 5 == 4):
        data["storm"].append(raw_data[i][0])
    elif (i % 5 == 1):
        data["mold"].append(raw_data[i][0])

# Get counts
total_length = len(data["storm"])
num_rains = np.sum(data["storm"])
num_molds_while_raining = 0
num_molds_while_dry = 0
for i in range(total_length):
    if (data["mold"] == "Yes"):
        if (data["storm"] == 1):
            num_molds_while_raining += 1
        else:
            num_molds_while_dry += 1

# Calculate binomeal distributions for rains
prior_beta_rains = beta.rvs(6, 3, size=9000)
data_beta_rain = beta.rvs(6 + num_rains, 3 + total_length - num_rains, size=9000)
rain_mean = np.mean(data_beta_rain)
print("Rain Mean: {}".format(rain_mean))

# Calculate binomeal distributions for mold
prior_beta_molds = beta.rvs(4, 6, size=9000)
data_beta_molds_while_raining = beta.rvs(4 + num_molds_while_raining, 6 + total_length - num_molds_while_raining, size=9000)
mold_mean = np.mean(data_beta_molds_while_raining)
print("Mold Mean: {}".format(mold_mean))

#Optionally plots
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('plot', type=str,
                    choices = ["Rain Prior", "Rain Posterior", "Mold Prior Given Rain", "Mold Posterior Given Rain"],
                    help='Choose what distribution to graph')
args = parser.parse_args()
plot = args.plot
print(plot)
if (plot == "Rain Prior"):
    ax_prior = sns.distplot(prior_beta_rains,
                      kde=False,
                      bins=100,
                      color='blue',
                      hist_kws={"linewidth": 15,'alpha':1})
    ax_prior.set(xlabel='Beta of rain prior', ylabel='Frequency')
elif (plot == "Rain Posterior"):
    ax_posterior = sns.distplot(data_beta_rain,
                            kde=False,
                            bins=100,
                            color='red',
                            hist_kws={'linewidth':15,'alpha':1})
    ax_posterior.set(xlabel="Beta of rain posterior", ylabel="Frequency")
elif (plot == "Mold Prior Given Rain"):
    ax_posterior = sns.distplot(prior_beta_molds,
                            kde=False,
                            bins=100,
                            color='red',
                            hist_kws={'linewidth':15,'alpha':1})
    ax_posterior.set(xlabel="Beta of mold prior given rain", ylabel="Frequency")
elif (plot == "Mold Posterior Given Rain"):
    ax_posterior = sns.distplot(data_beta_molds_while_raining,
                            kde=False,
                            bins=100,
                            color='red',
                            hist_kws={'linewidth':15,'alpha':1})
    ax_posterior.set(xlabel="Beta of mold posterior given rain", ylabel="Frequency")

plt.show()
