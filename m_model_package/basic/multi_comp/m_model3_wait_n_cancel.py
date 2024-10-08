############################################################################################################################################################
############## At each matching: look at the idle distance. Lower than a threshold - serve, higher - put in the queue.      ################################
############## If cannot match - put in the queue.                                                                          ################################
############## If a request is in the queue for more than 3 min - cancel this request.                                      ################################
############## No cooperation.                                                                                              ################################
############################################################################################################################################################


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
from sklearn.preprocessing import PolynomialFeatures
import csv
import math

####### Defining variables of M-model (look modeling doc) ###########
companies = ['uber', 'lyft', 'didi']
n_PAS = {i: [] for i in companies}
o_PAS = {i: [] for i in companies}
n_I = {i: [] for i in companies}
o_RH = {i: [] for i in companies}
lmd_RH = {i: [] for i in companies}
lmd_RH_served = {i: [] for i in companies}

n_I_mov = {i: [] for i in companies}
n_I_stop = {i: [] for i in companies}
dist_cancel = {i: [] for i in companies}

prod_n_I_mov = {i: [] for i in companies}

n_RHI = {i: [] for i in companies}
o_RHI = {i: [] for i in companies}
o_PAS_total = {i: [] for i in companies}
m_RHI = {i: [] for i in companies}
l_RHI = {i: [] for i in companies}
v = []

n_RH = {i: [] for i in companies}
l_RH = {i: [] for i in companies}
m_RH = {i: [] for i in companies}

n_PV = []
o_PV = []
lmd_PV = []
m_PV = []
l_PV = []  # = l_RH

l_RHI_s = {i: [] for i in companies}
l_RH_s = {i: [] for i in companies}
l_PV_s = []

num_veh = {i: [] for i in companies}  # total number of RH vehicles
num_mov_veh = {i: [] for i in companies}  # number of moving RH vehicles

pass_wait_time = {i: [] for i in companies}

max_t = 14400  # time horizon
delta_t = 2  # time step

max_x = 7000  # length of Lyon in m, taken from the fact that the surface of Lyon = 42 km2
max_y = 7000  # width of Lyon

alpha = -3  # constant, taken from the PhD thesis of Mikhail Murashkin

sigma_idle = {i: [] for i in
              companies}  # standard dev for the distance run by idle vehicles, taken from the regression model
# sigma_idle.append(92)        # 0.46 * 200
sigma_occup = 60  # constant, st dev for the distance run by occup veh-s. taken from the microsimulation as the average (0.86 * 200)
sigma_pv = 60  # equal to sigma_occup

dist_cancel['uber'] = 2000  # the cancellation threshold of idle distance
dist_cancel['lyft'] = 2000
dist_cancel['didi'] = 2000

queue_wait_time = 180  # time (in sec) that a customer waits in the queue before being transferred to another company

cancel = {i: [0] * int(max_t / delta_t) for i in companies}  # number of cancelled requests
queue_common = {i: [0] * int(max_t / delta_t) for i in
                companies}  # common queue where the demand not served by the home-company is put. demand is put in this list according to it's home-company
queue_common_total = {i: [0] * int(max_t / delta_t) for i in
                companies}  # common queue where the demand not served by the home-company is put. demand is put in this list according to it's home-company


dem_served_by_competitor = {i: [0] * int(max_t / delta_t) for i in
                            companies}  # if demand from queue_common is served - it goes to the corresponding serving company in this list. p.s. it can be served not only by a competitor, but by the home-company as well

frac_n_I_mov = {i: [] for i in companies}
frac_n_I_mov['uber'] = 0
frac_n_I_mov['lyft'] = 0
frac_n_I_mov['didi'] = 0

glob_dem_dens = {i: [] for i in companies}
glob_idle_veh_dens = {i: [] for i in companies}


def calculate_V_MFD(n):  # to calculate the speed. function is taken from Louis' MFD of Lyon
    V = 0
    if n < 18000:
        V = 11.5 - n * 6 / 18000
    elif n < 55000:
        V = 11.5 - 6 - (n - 18000) * 4.5 / (55000 - 18000)
    elif n < 80000:
        V = 11.5 - 6 - 4.5 - (n - 55000) * 1 / (80000 - 55000)

    return max(V, 0.001)


def o_MFD(i):  # calculation of the outflow. includes technique to prevent negative accumulation of vehicles
    for key in companies:
        if i == delta_t:
            o_RHI[key].append(max(
                min((n_RHI[key][-1] + alpha * (m_RHI[key][-1] / l_RHI_s[key][-1] - n_RHI[key][-1])) * v[-1] /
                    l_RHI[key][
                        -1],
                    n_RHI[key][-1] / delta_t + lmd_RH[key][int(i / delta_t)]), 0))
        else:
            o_RHI[key].append(max(
                min((n_RHI[key][-1] + alpha * (m_RHI[key][-1] / l_RHI_s[key][-1] - n_RHI[key][-1])) * v[-1] / l_RHI[key][
                    -1],
                    n_RHI[key][-1] / delta_t + o_PAS[key][-1]), 0))
        print(key + " o_RHI: " + str(o_RHI[key][-1]))
        # if o_RHI[key][-1] < 0:
        #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        o_RH[key].append(
            min((n_RH[key][-1] + alpha * (m_RH[key][-1] / l_RH_s[key][-1] - n_RH[key][-1])) * v[-1] / l_RH[key][
                int(i / delta_t) - 1],
                n_RH[key][-1] / delta_t + o_RHI[key][-1]))
        print(key + " o_RH: " + str(o_RH[key][-1]))
    o_PV.append(min((n_PV[-1] + alpha * (m_PV[-1] / l_PV_s[-1] - n_PV[-1])) * v[-1] / l_PV[int(i / delta_t) - 1],
                    n_PV[-1] / delta_t + lmd_PV[-1]))
    print("o_PV: " + str(o_PV[-1]))


def sigma_idle_calcul():  # calculation of st dev of idle veh-s using a regression model from microscopic simulation
    for key in companies:
        if n_PAS[key][-1] > 0:
            # sigma_idle[key].append(sigma_idle[key][-1])
            sigma_idle[key].append(386)
            print("sigma idle: " + str(sigma_idle[key][-1]))
            demand_count_norm = (lmd_RH[key][int(i / delta_t)] / (
                    max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over one minute while here the time is calculated in sec
            veh_density = ((n_I[key][-1]) / (max_x * max_y)) * 40000  # calculate vehicle density
            glob_dem_dens[key].append(demand_count_norm)
            glob_idle_veh_dens[key].append(veh_density)
        elif n_PAS[key][-1] == 0:
            demand_count_norm = (lmd_RH[key][int(i / delta_t)] / (
                    max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over one minute while here the time is calculated in sec
            veh_density = ((n_I[key][-1]) / (max_x * max_y)) * 40000  # calculate vehicle density
            glob_dem_dens[key].append(demand_count_norm)
            glob_idle_veh_dens[key].append(veh_density)
            # request_rate_per_veh_per_dt = lmd_RH[int(i / delta_t)] / (n_RHI[-1] + n_RH[-1] + n_I[-1])     # request per veh per time step
            demand_queued_norm = n_PAS[key][-1] / (max_x * max_y)

            X = {
                "demand_density": [demand_count_norm],
                "vehicle_density": [veh_density]
                # "request_rate": [request_rate_per_veh_per_dt]
                # "queued_demand_density": [demand_queued_norm]
            }

            Xdf = pd.DataFrame(X)  # transform to dataframe
            x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
                Xdf)  # transform data to use polynomial regression
            model = pickle.load(open('/Users/maryia/PycharmProjects/SimulationTrial/estimation models/big_fleet_st_dev',
                                     'rb'))  # use regression model created in micro-simulation
            sigma_idle_ = model.predict(x_)  # prediction
            sigma_idle[key].append(
                sigma_idle_[0] * 200)  # multiplying by 200 as sigma is calculated for a block of 200 m
            print(key + " sigma idle: " + str(sigma_idle[key][-1]))
            if sigma_idle[key][-1] < 0:
                print("Error: negative standard deviation")
                exit()


def steady_dist_l_s(i):  # calculation of avg remaining distance t obe traveled in steady state
    sigma_idle_calcul()
    for key in companies:
        l_RHI_s[key].append((l_RHI[key][-1] ** 2 + sigma_idle[key][-1] ** 2) / (2 * l_RHI[key][-1]))
        l_RH_s[key].append((l_RH[key][int(i / delta_t)] ** 2 + sigma_occup ** 2) / (2 * l_RH[key][int(i / delta_t)]))
        print(key + " l_RHI_*: " + str(l_RHI_s[key][-1]))
        print(key + " l_RH_*: " + str(l_RH_s[key][-1]))

    l_PV_s.append((l_PV[int(i / delta_t)] ** 2 + sigma_pv ** 2) / (2 * l_PV[int(i / delta_t)]))
    print("l_PV_*: " + str(l_PV_s[-1]))


def idle_dist_calcul():  # calculation of avg trip length of idle veh-s using a regression model from microscopic simulation
    for key in companies:
        if n_PAS[key][-1] > 0:
            # l_RHI[key].append(l_RHI[key][-1])
            l_RHI[key].append(2403)
            print(key + " l_RHI: " + str(l_RHI[key][-1]))
        elif n_PAS[key][-1] == 0:
            demand_count_norm = (lmd_RH[key][int(i / delta_t)] / (
                    max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over one minute while here the time is calculated in sec
            veh_density = ((n_I[key][-1]) / (max_x * max_y)) * 40000  # calculate vehicle density
            # request_rate_per_veh_per_dt = lmd_RH[int(i / delta_t)] / (n_RHI[-1] + n_RH[-1] + n_I[-1])     # request per veh per time step
            demand_queued_norm = n_PAS[key][-1] / (max_x * max_y)
            print(key + " DENSITY OF DEMAND " + str(demand_count_norm))
            print(key + " DENSITY OF VEH " + str(veh_density))

            X = {
                "demand_density": [demand_count_norm],
                "vehicle_density": [veh_density]
                # "request_rate": [request_rate_per_veh_per_dt]
                # "queued_demand_density": [demand_queued_norm]
            }

            Xdf = pd.DataFrame(X)  # transform to dataframe
            x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
                Xdf)  # transform data to use polynomial regression
            model = pickle.load(open('/Users/maryia/PycharmProjects/SimulationTrial/estimation models/big_fleet_idle_dist',
                                     'rb'))  # use regression model created in micro-simulation
            idle_dist_ = model.predict(x_)  # prediction
            l_RHI[key].append(idle_dist_[0] * 200)
            print(key + " l_RHI: " + str(l_RHI[key][-1]))


def init_lmb_RH(max_time):  # demand initialization
    dem_vary_t = 7200  # time duration when demand is variable
    # for 1/4 part of dem_vary_t it's constant, for next 1/4 it's increasing, for next 1/4 it's decreasing, for last 1/4 it's constant
    for key in companies:
        for i1 in range(0, max_time // delta_t + 1):

            ####### Constant demand #######
            if i1 <= (dem_vary_t // delta_t + 1) / 4:
                lmd_RH[key].append(0.8)
                if key == 'uber':
                    lmd_PV.append(20.5)

            ####### Increase of demand - use 1/6 part of time duration when demand is increasing #############
            elif (dem_vary_t // delta_t + 1) / 4 < i1 <= (
                    (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)):
                lmd_RH[key].append(0.9)
                if key == 'uber':
                    lmd_PV.append(21)

            elif ((dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)) < i1 <= \
                    ((dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(0.9)
                if key == 'uber':
                    lmd_PV.append(21)

            elif ((dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    ((dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1)
                if key == 'uber':
                    lmd_PV.append(21.5)

            elif ((dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    ((dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1.1)
                if key == 'uber':
                    lmd_PV.append(22)

            elif ((dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    ((dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1.1)
                if key == 'uber':
                    lmd_PV.append(22)

            # Constant demand
            elif ((dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    (dem_vary_t // delta_t + 1) / 2:
                lmd_RH[key].append(1.2)
                if key == 'uber':
                    lmd_PV.append(22.5)

            ######## Decrease of demand - use 1/6 part of time duration when demand is decreasing ############
            elif (2 * (dem_vary_t // delta_t + 1) / 4) < i1 <= \
                    (2 * (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)):
                lmd_RH[key].append(1.2)
                if key == 'uber':
                    lmd_PV.append(22.5)

            elif (2 * (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)) < i1 <= \
                    (2 * (dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1.1)
                if key == 'uber':
                    lmd_PV.append(22)

            elif (2 * (dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    (2 * (dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1.1)
                if key == 'uber':
                    lmd_PV.append(22)

            elif (2 * (dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    (2 * (dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(1)
                if key == 'uber':
                    lmd_PV.append(21.5)

            elif (2 * (dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    (2 * (dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
                lmd_RH[key].append(0.9)
                if key == 'uber':
                    lmd_PV.append(21.5)

            elif (2 * (dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                    (3 * (dem_vary_t // delta_t + 1) / 4):
                lmd_RH[key].append(0.9)
                if key == 'uber':
                    lmd_PV.append(21)

            # Constant demand
            elif 3 * (dem_vary_t // delta_t + 1) / 4 < i1 <= (dem_vary_t // delta_t + 1):
                lmd_RH[key].append(0.8)
                if key == 'uber':
                    lmd_PV.append(20.5)

            ######## Constant demand for the rest of time horizon ########
            elif (dem_vary_t // delta_t + 1) < i1 <= (max_time // delta_t + 1):
                lmd_RH[key].append(0.8)
                if key == 'uber':
                    lmd_PV.append(20.5)

            ####### Check if the demand is negative ########
            if lmd_RH[key][-1] < 0:
                lmd_RH[key][-1] = 0


def n_PAS_and_n_I(i):  # Calculation of accumulation of idle veh-s and waiting passengers to be matched

    for key in companies:
        if (delta_t * o_RH[key][-1] + n_I[key][-1]) < \
                (delta_t * lmd_RH[key][int(i / delta_t)] + n_PAS[key][-1]):  # case when demand is higher than available vehicles

            n_PAS[key].append(max(delta_t * (lmd_RH[key][int(i / delta_t)] - o_RH[key][-1]) - n_I[key][-1] + n_PAS[key][-1],
                             0))  # accumulation of passengers
            o_PAS[key].append(o_RH[key][-1] + n_I[key][-1] / delta_t)  # outflow of passengers = inflow of idle moving vehicles
            dummy1 = o_PAS[key][-1]  # remember the normal outflow of matched passengers before the cancellation

            ############### Cancellation of requests where idle distance is bigger than a threshold #########
            dummy_round_outflow = math.ceil(
                o_PAS[key][-1])  # !!!!!!! attention, might be zero - rounded up outflow of matched passengers
            dummy_frac_part = o_PAS[key][-1] - math.floor(
                o_PAS[key][-1])  # !!!!!!! attention, might be zero - fraction part of the outflow of matched passengers
            dummy_int_part = math.floor(o_PAS[key][-1])  # !!!!!!! attention, might be zero

            if o_PAS[key][-1] >= 0:  # if there are matched passengers
                dummy_lengths = np.random.normal(l_RHI[key][-1], sigma_idle[key][-1],
                                                 dummy_round_outflow)  # generate the idle distance length for each demand request knowing the current idle distance and standard deviation and using normal distribution

                # return the index of elements that are bigger than a threshold, if the last element is also bigger, check if there is a fraction and reduce o_PAS by fraction, otherwise only by one unit
                def condition(x):
                    return x >= dist_cancel[key]  # function to return the values from a list that satisfy the constraint

                output = [idx for idx, element in enumerate(dummy_lengths) if condition(
                    element)]  # return the indexes of those distance values that are higher than a threshold
                if len(output) != 0 and output[-1] == (len(dummy_lengths) - 1) and len(
                        dummy_lengths) != 0 and dummy_frac_part != 0:  # check if the fraction part of the initial outflow of matched passengers has a generated distance higher than a threshold
                    o_PAS[key][-1] = o_PAS[key][
                                    -1] - dummy_frac_part  # if yes - reduce this fraction from the initial outflow of matched passengers
                    o_PAS[key][-1] = o_PAS[key][-1] - len(
                        output) + 1  # and reduce other requests that have distance higher than a threshold, but add 1 because we already reduced the fraction
                else:
                    o_PAS[key][-1] = o_PAS[key][-1] - len(
                        output)  # if no fraction parts with higher distance - than just reduce other requests that have distance higher than a threshold

            ##################################################################################################

            n_I[key].append((dummy1 - o_PAS[key][
                -1]) * delta_t)  # accumulation of idle vehicles - add those vehicles that were supposed to serve the cancelled requests
            n_PAS[key][-1] = n_PAS[key][-1] + (dummy1 - o_PAS[key][
                -1]) * delta_t  # add in the queue the difference between the initially calculated pass. outflow and the new one

        elif (delta_t * o_RH[key][-1] + n_I[key][-1]) >= \
                (delta_t * lmd_RH[key][int(i / delta_t)] + n_PAS[key][
                    -1]):  # case when demand is lower or equal than available vehicles

            o_PAS[key].append(lmd_RH[key][int(i / delta_t)] + n_PAS[key][
                -1] / delta_t)  # outflow of passengers = inflow of idle moving vehicles
            dummy1 = o_PAS[key][-1]

            ############### Cancellation of requests where idle distance is bigger than a threshold #########
            dummy_round_outflow = math.ceil(o_PAS[key][-1])  # !!!!!!! attention, might be zero
            dummy_frac_part = o_PAS[key][-1] - math.floor(o_PAS[key][-1])  # !!!!!!! attention, might be zero
            dummy_int_part = math.floor(o_PAS[key][-1])  # !!!!!!! attention, might be zero

            if o_PAS[key][-1] >= 0:
                dummy_lengths = np.random.normal(l_RHI[key][-1], sigma_idle[key][-1], dummy_round_outflow)

                # return the index of elements that are bigger than a threshold, if the last element is also bigger, check if there is a fraction and reduce o_PAS by fraction, otherwise only by one unit
                def condition(x):
                    return x >= dist_cancel[key]

                output = [idx for idx, element in enumerate(dummy_lengths) if condition(element)]
                if len(output) != 0 and output[-1] == (len(dummy_lengths) - 1) and len(
                        dummy_lengths) != 0 and dummy_frac_part != 0:
                    o_PAS[key][-1] = o_PAS[key][-1] - dummy_frac_part
                    o_PAS[key][-1] = o_PAS[key][-1] - len(output) + 1
                else:
                    o_PAS[key][-1] = o_PAS[key][-1] - len(output)

            ##################################################################################################

            n_I[key].append(max(delta_t * (o_RH[key][-1] - lmd_RH[key][int(i / delta_t)]) - n_PAS[key][-1] + n_I[key][-1],
                           0))  # accumulation of idle vehicles
            n_I[key][-1] = n_I[key][-1] + (dummy1 - o_PAS[key][-1]) * delta_t  # add also those vehicles that were supposed to serve the canceled requests

            n_PAS[key].append((dummy1 - o_PAS[key][-1]) * delta_t)      # put canceled requests in the queue

        o_PAS_total[key].append(o_PAS[key][-1])

        ################ Count passengers that are waiting for more than the specified time ######################

    if i >= queue_wait_time:
        accum_o_PAS_total = {j: [] for j in companies}
        accum_lmd_RH = {j: [] for j in companies}
        for key in companies:
            accum_o_PAS_total[key] = np.cumsum(o_PAS_total[key])
            accum_lmd_RH[key] = np.cumsum(lmd_RH[key])
            for t2 in range(0, len(accum_lmd_RH[key])):
                if (accum_o_PAS_total[key][-1] - 0.05 <= accum_lmd_RH[key][t2] <= accum_o_PAS_total[key][-1] + 0.05) and \
                        (i - t2 * delta_t > queue_wait_time) and \
                        (accum_lmd_RH[key][int((i - queue_wait_time) / 2)] - accum_o_PAS_total[key][-1] > 0):
                    # and ((accum_lmd_RH[key][int((i - queue_wait_time) / 2)] - accum_o_PAS[key][-1]) * delta_t <= n_PAS[key][-1]):
                    dum1 = accum_lmd_RH[key][int((i - queue_wait_time) / 2)]
                    dum2 = accum_o_PAS_total[key][-1]
                    cancel[key][int(i / delta_t)] = (dum1 - dum2)

                    n_PAS[key][-1] = n_PAS[key][-1] - cancel[key][int(i / delta_t)] * delta_t
                    o_PAS_total[key][-1] = o_PAS_total[key][-1] + cancel[key][int(i / delta_t)]
                    break


def normal_dist_of_l_RH(l_RH):  # visualization of normal distribution of occupied vehicle distance
    for key in companies:
        count, bins, ignored = plt.hist(l_RH[key], 30, density=True)
        plt.plot(bins, 1 / (sigma_occup * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - 8036) ** 2 / (2 * sigma_occup ** 2)),
                 linewidth=2, color='r')
        plt.title(key + " Norm distribution of l_RH")
        plt.show()


def normal_dist_of_l_PV(l_PV):  # visualization of normal distribution of private vehicle distance
    count, bins, ignored = plt.hist(l_PV, 30, density=True)
    plt.plot(bins, 1 / (sigma_pv * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - 8036) ** 2 / (2 * sigma_pv ** 2)),
             linewidth=2, color='r')
    plt.title("Norm distribution of l_PV")
    plt.show()


def visualization():  ################## Visualization #################
    def idle():  # reservoir of idle waiting vehicles and waiting passengers
        # plt.figure(1)
        plt.subplots(7, 1, constrained_layout=True)
        plt.subplot(7, 1, 1)
        x = n_I['uber']
        y = n_I['lyft']
        z = n_I['didi']
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')
        plt.plot(z, 'aqua')
        plt.title('n_I - accum. of idle stopped')

        plt.subplot(7, 1, 2)
        x = o_RH['uber']
        y = o_RH['lyft']
        u = o_RH['didi']
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')
        plt.plot(u, 'aqua')
        z = [0] * len(o_RH['uber'])
        plt.plot(z, 'r')
        plt.title('o_RH - inflow to idle stopped')

        plt.subplot(7, 1, 3)
        x = lmd_RH['uber']
        y = lmd_RH['lyft']
        u = lmd_RH['didi']
        plt.plot(x, 'b', label='uber')
        plt.plot(y, 'cornflowerblue', label='lyft')
        plt.plot(u, 'aqua', label='didi')
        plt.title('lmd_RH - arriving demand requests')
        plt.legend(loc="upper right")

        plt.subplot(7, 1, 4)
        x = n_PAS['uber']
        y = n_PAS['lyft']
        u = n_PAS['didi']
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')
        plt.plot(u, 'aqua')
        plt.title('n_PAS - accum. of waiting pass. to be matched')

        plt.subplot(7, 1, 5)
        x = o_PAS['uber']
        y = o_PAS['lyft']
        u = o_PAS['didi']

        plt.plot(u, 'aqua')
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')
        plt.title('o_PAS - outflow of matched passengers')

        plt.subplot(7, 1, 6)
        x = cancel['uber']
        y = cancel['lyft']
        u = cancel['didi']
        plt.plot(u, 'aqua')
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')

        #plt.locator_params(axis='x', nbins=15)
        plt.title('# of canceled requests')

        plt.subplot(7, 1, 7)
        x = o_PAS_total['uber']
        y = o_PAS_total['lyft']
        u = o_PAS_total['didi']
        plt.plot(u, 'aqua')
        plt.plot(x, 'b')
        plt.plot(y, 'cornflowerblue')
        #plt.locator_params(axis='x', nbins=15)
        plt.title('o_PAS_total - outflow of all passengers (matched+canceled)')


    def veh():
        plt.subplots(5, 1, constrained_layout=True)
        plt.subplot(5, 1, 1)
        x = num_veh['uber']
        y = num_veh['lyft']
        plt.plot(x, 'magenta')
        plt.plot(y, 'plum')
        plt.title('total # of RH vehicles in the system')

        plt.subplot(5, 1, 2)
        x = num_mov_veh['uber']
        y = num_mov_veh['lyft']
        plt.plot(x, 'magenta')
        plt.plot(y, 'plum')
        plt.title('# of moving RH vehicles in the system')

        plt.subplot(5, 1, 3)
        x = n_I_mov['uber']
        y = n_I_mov['lyft']
        plt.plot(x, 'magenta')
        plt.plot(y, 'plum')
        plt.title('# of cruising idle RH vehicles')

        plt.subplot(5, 1, 4)
        x = n_I_stop['uber']
        y = n_I_stop['lyft']
        plt.plot(x, 'magenta')
        plt.plot(y, 'plum')
        plt.title('# of non-moving idle RH vehicles')

        plt.subplot(5, 1, 5)
        x = prod_n_I_mov['uber']
        y = prod_n_I_mov['lyft']
        plt.plot(x, 'magenta')
        plt.plot(y, 'plum')
        plt.title('production of cruising idle RH vehicles per time unit')

    def rhi():  # reservoir of idle moving vehicles
        # plt.figure(2)
        plt.subplots(6, 1, constrained_layout=True)
        plt.subplot(6, 1, 1)
        x = n_RHI['uber']
        y = n_RHI['lyft']
        plt.plot(x, 'k')
        plt.plot(y, 'grey')
        plt.title('n_RHI - accum. of idle moving')

        plt.subplot(6, 1, 2)
        x = o_PAS['uber']
        y = o_PAS['lyft']
        plt.plot(x, 'k', label='uber')
        plt.plot(y, 'grey', label='lyft')
        plt.title('o_PAS - inflow of matched passengers')
        plt.legend(loc="upper right")

        plt.subplot(6, 1, 3)
        x = o_RHI['uber']
        y = o_RHI['lyft']
        plt.plot(x, 'k')
        plt.plot(y, 'grey')
        z = [0] * len(o_RHI['uber'])
        plt.plot(z, 'r')
        plt.title('o_RHI - outflow of idle moving')

        plt.subplot(6, 1, 4)
        x = m_RHI['uber']
        y = m_RHI['lyft']
        plt.plot(x, 'k')
        plt.plot(y, 'grey')
        plt.title('m_RHI - rem. dist. of idle moving')

        plt.subplot(6, 1, 5)
        x = l_RHI['uber']
        y = l_RHI['lyft']
        plt.plot(x, 'k')
        plt.plot(y, 'grey')
        plt.title('l_RHI - avg trip length of idle moving')

        plt.subplot(6, 1, 6)
        x = sigma_idle['uber']
        y = sigma_idle['lyft']
        plt.plot(x, 'k')
        plt.plot(y, 'grey')
        plt.title('standard deviation of idle distance')

        # plt.tight_layout()
        # plt.subplots(constrained_layout=True)

    def rh():  # reservoir of moving with passenger vehicles
        plt.subplots(6, 1, constrained_layout=True)
        plt.subplot(6, 1, 1)
        x = n_RH['uber']
        y = n_RH['lyft']
        plt.plot(x, 'g', label='uber')
        plt.plot(y, 'palegreen', label='lyft')
        plt.title('n_RH - accum. of moving w/ passenger')
        plt.legend(loc="upper right")

        plt.subplot(6, 1, 2)
        x = o_RHI['uber']
        y = o_RHI['lyft']
        plt.plot(x, 'g')
        plt.plot(y, 'palegreen')
        z = [0] * len(o_RHI['uber'])
        plt.plot(z, 'r')
        plt.title('o_RHI - inflow of moving w/ passenger')

        plt.subplot(6, 1, 3)
        x = o_RH['uber']
        y = o_RH['lyft']
        plt.plot(x, 'g')
        plt.plot(y, 'palegreen')
        z = [0] * len(o_RH['uber'])
        plt.plot(z, 'r')
        plt.title('o_RH - outflow of moving w/ passenger')

        plt.subplot(6, 1, 4)
        x = m_RH['uber']
        y = m_RH['lyft']
        plt.plot(x, 'g')
        plt.plot(y, 'palegreen')
        plt.title('m_RH - rem. dist. of moving w/ passenger')

        plt.subplot(6, 1, 5)
        x = v
        plt.plot(x, 'g')
        plt.title('v - network speed')

        plt.subplot(6, 1, 6)
        x = l_RH['uber']
        y = l_RH['lyft']
        plt.plot(x, 'g')
        plt.plot(y, 'palegreen')
        plt.title('l_RH - avg trip length of occup veh')

    def pv():  # reservoir of private vehicles
        plt.subplots(6, 1, constrained_layout=True)
        plt.subplot(6, 1, 1)
        x = n_PV
        plt.plot(x, 'y')
        plt.title('n_PV - accum. of PV')

        plt.subplot(6, 1, 2)
        x = lmd_PV
        plt.plot(x, 'y')
        plt.title('lmb_PV - demand requests of PV')

        plt.subplot(6, 1, 3)
        x = o_PV
        plt.plot(x, 'y')
        y = [0] * len(o_PV)
        plt.plot(y, 'r')
        plt.title('o_PV - outflow of PV')

        plt.subplot(6, 1, 4)
        x = m_PV
        plt.plot(x, 'y')
        plt.title('m_PV - rem. dist. of PV')

        plt.subplot(6, 1, 5)
        x = v
        plt.plot(x, 'y')
        plt.title('v - network speed')
        plt.subplot(6, 1, 6)
        x = l_PV
        plt.plot(x, 'y')
        plt.title('l_PV - avg trip length of PV')

    def countcurve_uber():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = np.cumsum(o_PAS['uber'])
        y = np.cumsum(o_RHI['uber'])
        z = np.cumsum(o_RH['uber'])
        j = np.cumsum(lmd_RH['uber'])
        p = np.cumsum(o_PAS_total['uber'])
        plt.plot(x, 'magenta', label='o_PAS')
        plt.plot(y, 'k', label='o_RHI')
        plt.plot(z, 'g', label='o_RH')
        plt.plot(j, 'r', label='lmb_RH')
        plt.plot(p, 'b', label='o_PAS_total')
        plt.legend(loc="upper left")
        plt.title('Uber')

    def countcurve_lyft():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = np.cumsum(o_PAS['lyft'])
        y = np.cumsum(o_RHI['lyft'])
        z = np.cumsum(o_RH['lyft'])
        j = np.cumsum(lmd_RH['lyft'])
        plt.plot(x, 'magenta', label='o_PAS')
        plt.plot(y, 'k', label='o_RHI')
        plt.plot(z, 'g', label='o_RH')
        plt.plot(j, 'r', label='lmb_RH')
        plt.legend(loc="upper left")
        plt.title('Lyft')

    def countcurve_didi():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = np.cumsum(o_PAS['didi'])
        y = np.cumsum(o_RHI['didi'])
        z = np.cumsum(o_RH['didi'])
        j = np.cumsum(lmd_RH['didi'])
        p = np.cumsum(o_PAS_total['didi'])
        plt.plot(x, 'magenta', label='o_PAS')
        plt.plot(y, 'k', label='o_RHI')
        plt.plot(z, 'g', label='o_RH')
        plt.plot(j, 'r', label='lmb_RH')
        plt.plot(p, 'b', label='o_PAS_total')
        plt.legend(loc="upper left")
        plt.title('Didi')

    # for key in companies:
    idle()
    veh()
    rhi()
    rh()
    pv()
    countcurve_uber()
    countcurve_lyft()
    countcurve_didi()

    plt.show()


def vis2():
    def accum():
        plt.subplots(7, 1, constrained_layout=True)
        plt.subplot(7, 1, 1)
        x = n_I['uber']
        y = n_I['lyft']
        plt.plot(x, 'b')
        plt.plot(y, 'g')

        plt.title('accum. of idle non-moving')

        plt.subplot(7, 1, 2)
        x = n_PAS['uber']
        y = n_PAS['lyft']
        plt.plot(x, 'b', label='uber')
        plt.plot(y, 'g', label='lyft')
        plt.legend(loc="upper right")
        plt.title('accum. of waiting pass. to be matched')

        plt.subplot(7, 1, 3)
        x = n_RHI['uber']
        y = n_RHI['lyft']
        plt.plot(x, 'b')
        plt.plot(y, 'g')
        plt.title('accum. of idle moving')

        plt.subplot(7, 1, 4)
        x = l_RHI['uber']
        y = l_RHI['lyft']
        plt.plot(x, 'b')
        plt.plot(y, 'g')
        plt.title('avg trip length of idle moving')

        plt.subplot(7, 1, 5)
        x = n_RH['uber']
        y = n_RH['lyft']
        plt.plot(x, 'b')
        plt.plot(y, 'g')
        plt.title('accum. of moving w/ passenger')

        plt.subplot(7, 1, 6)
        x = lmd_RH['uber']
        y = lmd_RH['lyft']
        plt.plot(x, 'b')
        plt.plot(y, 'g')
        plt.title('arriving demand requests (equal for both companies)')

        plt.subplot(7, 1, 7)
        x = n_PV
        plt.plot(x, 'y')
        plt.title('accum. of private vehicles')

    accum()

    plt.show()


if __name__ == "__main__":

    ########## Initialization of starting values at t = 0 ##############
    # pay attention that speed, length and time need to have the same unit of measurement (m/s)
    print("iter: 0")
    """for key in companies:
        n_PAS[key].append(0)
        n_I[key].append(100)
        n_RHI[key].append(50)
        n_RH[key].append(450)"""

    n_PAS['uber'].append(0)
    n_I['uber'].append(450)
    n_RHI['uber'].append(300)
    n_RH['uber'].append(450)

    n_PAS['lyft'].append(0)
    n_I['lyft'].append(510)
    n_RHI['lyft'].append(330)
    n_RH['lyft'].append(480)

    n_PAS['didi'].append(0)
    n_I['didi'].append(420)
    n_RHI['didi'].append(270)
    n_RH['didi'].append(390)

    n_PV.append(20659)

    for key in companies:
        n_I_mov[key].append(n_I[key][-1] * frac_n_I_mov[key])
        n_I_stop[key].append(n_I[key][-1] - n_I_mov[key][-1])

    # max_t = 7200  # time horizon

    init_lmb_RH(max_t)  # demand initialization

    # for i2 in range(0, max_t // delta_t + 1):  # private vehicles demand initialization
    #    lmd_PV.append(20)

    # for i in range(0, max_t // delta_t + 1):       # initialization of constant demand
    #    lmd_RH.append(0.15)

    i = 0  # to avoid any errors reinitialize i
    idle_dist_calcul()  # prediction of avg idle distance using regression
    # l_RHI.append(1400)     # 7 * 200, 7 is taken from microscopic model boxplot as median
    # l_RH.append(5740)      # constant, taken as avg from microscopic simulation (28.7 * 200)
    # l_PV.append(5740)      # constant, taken as avg from microscopic simulation (28.7 * 200)
    np.random.seed(4424)
    for key in companies:
        l_RH[key] = np.random.normal(4424, sigma_occup,
                                     max_t // delta_t + 1)  # random avg trip length from norm distribution

    l_PV = np.random.normal(4424, sigma_pv, max_t // delta_t + 1)  # random avg trip length from norm distribution
    # normal_dist_of_l_RH(l_RH)      # to verify the norm distribution of l_RH
    # normal_dist_of_l_PV(l_PV)      # to verify the norm distribution of l_PV
    for key in companies:
        print(key + " l_RH: " + str(l_RH[key]))
    print("l_PV: " + str(l_PV))

    v.append(calculate_V_MFD(
        n_RHI['didi'][-1] + n_RH['didi'][-1] + n_RHI['uber'][-1] + n_RH['uber'][-1] + n_RHI['lyft'][-1] + n_RH['lyft'][-1] + n_PV[-1] + n_I_mov['uber'][-1] +
        n_I_mov['lyft'][-1] + n_I_mov['didi'][-1]))  # calculation of speed

    for key in companies:
        prod_n_I_mov[key].append(n_I_mov[key][-1] * v[-1] * delta_t)

    #### Estimation of total remaining travel distance
    for key in companies:
        m_RHI[key].append(l_RHI[key][0] * n_RHI[key][0] / 2)
        m_RH[key].append(l_RH[key][0] * n_RH[key][0] / 2)
        m_PV.append(l_PV[0] * n_PV[0] / 2)
        num_veh[key].append(n_I[key][-1] + n_RHI[key][-1] + n_RH[key][-1])  # total number of RH vehicles
        num_mov_veh[key].append(n_RHI[key][-1] + n_RH[key][-1])  # # of moving RH vehicles

        print(key + " lmb_RH: " + str(lmd_RH[key][int(i / delta_t)]))
        print(key + " n_PAS: " + str(n_PAS[key][-1]))
        print(key + " n_I: " + str(n_I[key][-1]))
        print(key + " n_RHI: " + str(n_RHI[key][-1]))
        print(key + " n_RH: " + str(n_RH[key][-1]))
        #        print("n moving: " + str(n_RHI[key][-1] + n_RH[key][-1] + n_PV[-1]))
        print(key + " m_RHI: " + str(m_RHI[key][-1]))
        print(key + " m_RH: " + str(m_RH[key][-1]))
        print(key + " # of RH vehicles in the system: " + str(num_veh[key][-1]))

    print("sigma_occup: " + str(sigma_occup))
    print("n: " + str(n_I['uber'][-1] + n_RHI['uber'][-1] + n_RH['uber'][-1] +
                      n_I['didi'][-1] + n_RHI['didi'][-1] + n_RH['didi'][-1] +
                      n_I['lyft'][-1] + n_RHI['lyft'][-1] + n_RH['lyft'][-1] + n_PV[-1]))
    print("v: " + str(v[-1]))
    print("m_PV: " + str(m_PV[-1]))
    print("n_PV: " + str(n_PV[-1]))
    print("lmb_PV: " + str(lmd_PV[int(i / delta_t)]))
    print("sigma_pv: " + str(sigma_pv))
    steady_dist_l_s(i)  # avg remaining distance to be traveled in steady state

    for i in range(delta_t, max_t, delta_t):
        print()
        print("iter: " + str(i))

        o_MFD(i)

        # n_I.append(max(delta_t * (o_RH[-1] - lmd_RH[int(i / delta_t)]) + n_I[-1], 0))
        n_PAS_and_n_I(i)  # accumulation of idle stopped vehicles and passengers waiting to be matched. Must be placed after calculation of outflow and before calculation of vehicle accumulation
        for key in companies:
            print(key + " o_PAS: " + str(o_PAS[key][-1]))
            print(key + " lmb_RH: " + str(lmd_RH[key][int(i / delta_t)]))

        print("lmb_PV: " + str(lmd_PV[int(i / delta_t)]))

        ######### Calculation of accumulation. Negative accumulation substituted with 0
        for key in companies:
            n_RHI[key].append(max(delta_t * (o_PAS[key][-1] - o_RHI[key][-1]) + n_RHI[key][-1], 0))
            n_RH[key].append(max(delta_t * (o_RHI[key][-1] - o_RH[key][-1]) + n_RH[key][-1], 0))

            print(key + " n_PAS: " + str(n_PAS[key][-1]))
            print(key + " n_I: " + str(n_I[key][-1]))
            print(key + " n_RHI: " + str(n_RHI[key][-1]))
            print(key + " n_RH: " + str(n_RH[key][-1]))

        for key in companies:
            n_I_mov[key].append(n_I[key][-1] * frac_n_I_mov[key])
            n_I_stop[key].append(n_I[key][-1] - n_I_mov[key][-1])

        n_PV.append(max(delta_t * (lmd_PV[int(i / delta_t)] - o_PV[-1]) + n_PV[-1], 0))
        print("n_PV: " + str(n_PV[-1]))
        # print("n moving: " + str(n_RHI[-1] + n_RH[-1] + n_PV[-1]))
        print("n: " + str(n_I['uber'][-1] + n_RHI['uber'][-1] + n_RH['uber'][-1] +
                          n_I['didi'][-1] + n_RHI['didi'][-1] + n_RH['didi'][-1] +
                          n_I['lyft'][-1] + n_RHI['lyft'][-1] + n_RH['lyft'][-1] + n_PV[-1]))

        v.append(
            calculate_V_MFD(n_RHI['didi'][-1] + n_RH['didi'][-1] + n_RHI['uber'][-1] + n_RH['uber'][-1] + n_RHI['lyft'][-1] + n_RH['lyft'][-1] + n_PV[-1] +
                            n_I_mov['uber'][-1] + n_I_mov['lyft'][-1] + n_I_mov['didi'][-1]))
        print("v: " + str(v[-1]))

        for key in companies:
            prod_n_I_mov[key].append(n_I_mov[key][-1] * v[-1] * delta_t)

        idle_dist_calcul()
        for key in companies:
            print(key + " l_RH: " + str(l_RH[key][int(i / delta_t)]))

        print("l_PV: " + str(l_PV[int(i / delta_t)]))
        print("sigma_occup: " + str(sigma_occup))
        print("sigma_pv: " + str(sigma_pv))
        steady_dist_l_s(i)

        for key in companies:
            m_RHI[key].append(delta_t * (o_PAS[key][-1] * l_RHI[key][-1] - n_RHI[key][-1] * v[-1]) + m_RHI[key][-1])
            m_RH[key].append(
                delta_t * (o_RHI[key][-1] * l_RH[key][int(i / delta_t)] - n_RH[key][-1] * v[-1]) + m_RH[key][-1])
            print(key + " m_RHI: " + str(m_RHI[key][-1]))
            print(key + " m_RH: " + str(m_RH[key][-1]))

        m_PV.append(delta_t * (lmd_PV[int(i / delta_t)] * l_PV[int(i / delta_t)] - n_PV[-1] * v[-1]) + m_PV[-1])
        print("m_PV: " + str(m_PV[-1]))

        for key in companies:
            num_veh[key].append(n_I[key][-1] + n_RHI[key][-1] + n_RH[key][-1])
            num_mov_veh[key].append(n_RHI[key][-1] + n_RH[key][-1])  # # of moving RH vehicles
            print(key + " # of RH vehicles in the system: " + str(num_veh[key][-1]))

    o_MFD(i)
    n_PAS_and_n_I(i)  # accumulation of idle stopped vehicles and passengers waiting to be matched. Must be placed after calculation of outflow and before calculation of vehicle accumulation
    for key in companies:
        print(key + " o_PAS: " + str(o_PAS[key][-1]))
        print(key + " lmb_RH: " + str(lmd_RH[key][int(i / delta_t)]))

    ################### Comparison metrics starts here ##########################

    print("avg speed: " + str(sum(v) / len(v)))
    print("total accum of veh: " + str(sum(num_mov_veh['uber']) + sum(num_mov_veh['lyft']) + sum(num_mov_veh['didi']) + sum(n_PV)))
    print("total cancellation: " + str((sum(cancel['uber']) + sum(cancel['lyft']) + sum(cancel['didi'])) * delta_t))

    accum_o_PAS_total_uber = np.cumsum(o_PAS_total['uber'])
    accum_lmd_RH_uber = np.cumsum(lmd_RH['uber'])
    for t4 in range(0, len(accum_lmd_RH_uber)):
        for t5 in range(0, len(accum_o_PAS_total_uber)):
            if accum_o_PAS_total_uber[t5] - 0.05 <= accum_lmd_RH_uber[t4] <= accum_o_PAS_total_uber[t5] + 0.05:
                pass_wait_time['uber'].append((t5 - t4) * delta_t)

    print("avg pass wait time uber: " + str(sum(pass_wait_time['uber']) / len(pass_wait_time['uber'])))

    accum_o_PAS_total_lyft = np.cumsum(o_PAS_total['lyft'])
    accum_lmd_RH_lyft = np.cumsum(lmd_RH['lyft'])
    for t6 in range(0, len(accum_lmd_RH_lyft)):
        for t7 in range(0, len(accum_o_PAS_total_lyft)):
            if accum_o_PAS_total_lyft[t7] - 0.05 <= accum_lmd_RH_lyft[t6] <= accum_o_PAS_total_lyft[t7] + 0.05:
                pass_wait_time['lyft'].append((t7 - t6) * delta_t)

    print("avg pass wait time lyft: " + str(sum(pass_wait_time['lyft']) / len(pass_wait_time['lyft'])))

    accum_o_PAS_total_didi = np.cumsum(o_PAS_total['didi'])
    accum_lmd_RH_didi = np.cumsum(lmd_RH['didi'])
    for t8 in range(0, len(accum_lmd_RH_didi)):
        for t9 in range(0, len(accum_o_PAS_total_didi)):
            if accum_o_PAS_total_didi[t9] - 0.05 <= accum_lmd_RH_didi[t8] <= accum_o_PAS_total_didi[t9] + 0.05:
                pass_wait_time['didi'].append((t9 - t8) * delta_t)

    print("avg pass wait time didi: " + str(sum(pass_wait_time['didi']) / len(pass_wait_time['didi'])))

    print("total avg pass wait time: " + str((sum(pass_wait_time['lyft']) + sum(pass_wait_time['uber']) + sum(pass_wait_time['didi'])) / (
                len(pass_wait_time['lyft']) + len(pass_wait_time['didi']) + len(pass_wait_time['uber']))))

    print("utilisation rate of vehicles:  " + str(((max_t * 600/delta_t) - (sum(n_I['uber'])+sum(n_I['lyft'])+sum(n_I['didi'])))/(max_t * 600/delta_t)))
    print("accum of idle stopped:  " + str(sum(n_I['uber']) + sum(n_I['lyft']) + sum(n_I['didi'])))
    ################### Comparison metrics ends here ##########################

    visualization()


    # vis2()

