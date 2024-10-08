import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
import random
from scipy.stats import rayleigh
from scipy.stats import beta
from mpl_axes_aligner import shift

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions

FIG_SIZE_HALF = (6, 3)
FONT_SIZE_LAB = 14
FONT_SIZE_LEG = 12
FONT_SIZE_AXI = 12

columns = ["glob_idle_dist_to_queued_dem_avg"]
df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/glob_idle_dist_to_queued_dem_avg.csv", usecols=columns)

Ydf = df["glob_idle_dist_to_queued_dem_avg"]      # {"idle_dist": glob_idle_dist_avg}
#Zdf = df["glob_st_dev_idle"]        # {"st_dev_idle": glob_st_dev_idle}
for i in range(0, len(Ydf)):
    if Ydf[i] <= 4:
        Ydf.drop(i,axis=0,inplace=True)
        #Zdf.drop(i, axis=0, inplace=True)
    else:
        Ydf[i] = Ydf[i] * 200
        #Zdf[i] = Zdf[i] * 200

#print(Ydf)
#print(Zdf)
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=Ydf, kind="hist", bins = 50)

#print((sum(Ydf) / len(Ydf)))
#print((sum(Zdf) / len(Zdf)))

x1 = []
y1 = []
x2 = []
y2 = []
d = []

for i in range(0, 10000):
    x1.append(random.randrange(0, 7000, 1))
    x2.append(random.randrange(0, 7000, 1))
    y1.append(random.randrange(0, 7000, 1))
    y2.append(random.randrange(0, 7000, 1))


for i in range(0, 10000):
    d.append(abs(x1[i] - x2[i]) + abs(y1[i] - y2[i]))

print("avg length: " + str(sum(d)/len(d)))
#print(d)

"""plt.hist(d, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()"""


f = Fitter(d, distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
f.fit()
#print(f.summary())
print(f.get_best(method = 'sumsquare_error'))
a = f.get_best(method = 'sumsquare_error').get('beta').get('a')
b = f.get_best(method = 'sumsquare_error').get('beta').get('b')
beta_list = list(np.random.beta(a, b, 10000))
for i in range(0, len(beta_list)):
    beta_list[i] = beta_list[i] * 14000
#print(beta_list)
st_dev = beta.std(2.39, 4.7145) * 14000
mean = beta.mean(2.39, 4.7145) * 14000
print("st dev: " + str(st_dev))
print("mean: " + str(mean))

d = [x / 1000 for x in d]

sns.set_style('white')
sns.set_context("paper", font_scale = 2)
ax = sns.displot(data=d, kind="hist", bins = 50)
for a in ax.axes[0]:
    a.set_xlabel("Idle distance (km)")
    a.set_ylabel("Frequency")
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
#sns.displot(data=beta_list, kind="hist", bins = 50)
#plt.show()


n = np.random.normal(2403, 386, 10000)
n = [x / 1000 for x in n]
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
ax2 = sns.displot(data=n, kind="hist", bins = 50)
for a in ax2.axes[0]:
    a.set_xlabel("Idle distance (km)")
    a.set_ylabel("Frequency")
plt.show()
