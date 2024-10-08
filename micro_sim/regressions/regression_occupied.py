#from micro_sim.main1 import *
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import matplotlib.pyplot as plt


#columns = ["glob_idle_dist_avg", "glob_st_dev_idle", "glob_demand_count_norm", "glob_veh_vacant_dens"]
columns = ["glob_occup_dist_avg", "glob_st_dev_occup", "glob_demand_count_norm", "glob_veh_vacant_dens"]
df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
print(df)

"""Y = {"idle_dist": glob_idle_dist_avg}
Z = {"st_dev_idle": glob_st_dev_idle}
X = {
    "demand_density": glob_demand_count_norm,
    "vacant_vehicle_density": glob_veh_vacant_dens
    #"request_rate": glob_request_rate_per_veh_per_min
}"""

Ydf = df["glob_occup_dist_avg"]      # {"idle_dist": glob_idle_dist_avg}
Xdf = df[["glob_demand_count_norm", "glob_veh_vacant_dens"]]
Zdf = df["glob_st_dev_occup"]

print(sum(Ydf)/len(Ydf) * 200)
print(sum(Zdf)/len(Zdf) * 200)

def poly_regression_idle_dist():
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xdf)
    model = LinearRegression().fit(x_, Ydf)

    r_sq = model.score(x_, Ydf)
    print('coefficient of determination: ', r_sq)
    #print('intercept: ', model.intercept_)
    #print('coefficients: ', model.coef_)

    #filename = '/Users/maryia/PycharmProjects/SimulationTrial/idle_dist_model'
    #pickle.dump(model, open(filename, 'wb'))

def poly_regression_st_dev():
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xdf)
    model4 = LinearRegression().fit(x_, Zdf)

    r_sq = model4.score(x_, Zdf)
    print()
    print('coefficient of determination: ', r_sq)
    #print('intercept: ', model4.intercept_)
    #print('coefficients: ', model4.coef_)

    #filename = '/Users/maryia/PycharmProjects/SimulationTrial/st_dev_idle_dist_model'
    #pickle.dump(model4, open(filename, 'wb'))

def boxplot1():
    plt.figure(1)
    data = df["glob_demand_count_norm"].values.tolist()
    plt.boxplot(data)
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    plt.title('density of demand per 40s per block of 200 m')

def boxplot2():
    plt.figure(2)
    data2 = df["glob_veh_vacant_dens"].values.tolist()
    plt.boxplot(data2)
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    plt.title('density of vacant fleet per per block of 200 m')

def boxplot3():
    plt.figure(3)
    data3 = df["glob_occup_dist_avg"].values.tolist()
    data3 = [x * 200 for x in data3]
    plt.boxplot(data3)
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    plt.title('Average service trip length')
    plt.ylabel('Distance, meters')
    plt.ylim(0, 5000)

def boxplot4():
    plt.figure(4)
    data4 = df["glob_st_dev_occup"].values.tolist()
    data4 = [x * 200 for x in data4]
    plt.boxplot(data4)
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    plt.title('Average standard deviation')

#y = Ydf.values.tolist()
#z = Zdf.values.tolist()

#poly_regression_idle_dist()
#poly_regression_st_dev()
boxplot1()
boxplot2()
boxplot3()
boxplot4()

"""plt.plot(glob_demand_count_norm, '-k', label='demand')
plt.plot(glob_veh_density, '-r', label='veh')
plt.plot(y, '-y', label='length')
plt.plot(z, '-b', label='st dev')
p = [0] * len(y)
plt.plot(p, 'r')"""


plt.show()
