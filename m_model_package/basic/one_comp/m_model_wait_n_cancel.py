############################################################################################################################################################
############## At each matching: look at the idle distance. Lower than a threshold - serve, higher - put in the queue.      ################################
############## If cannot match - put in the queue.                                                                          ################################
############## If a request is in the queue for more than 3 min - cancel this request.                                      ################################
############################################################################################################################################################

import math
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
from sklearn.preprocessing import PolynomialFeatures
import csv


FIG_SIZE_HALF = (6, 3)
FONT_SIZE_LAB = 18
FONT_SIZE_LEG = 12
FONT_SIZE_AXI = 12

####### Defining variables of M-model (look modeling doc) ###########
n_PAS = []
o_PAS = []  # outflow of passengers
o_PAS_total = []
n_I = []
o_RH = []
lmd_RH = []

n_I_mov = []
n_I_stop = []

prod_n_I_mov = []

n_RHI = []
o_RHI = []
m_RHI = []
l_RHI = []
v = []

n_RH = []
l_RH = []
m_RH = []

n_PV = []
o_PV = []
lmd_PV = []
m_PV = []
l_PV = []  # = l_RH

l_RHI_s = []
l_RH_s = []
l_PV_s = []

num_veh = []  # total number of RH vehicles
num_mov_veh = []  # number of moving RH vehicles

pass_wait_time = []

delta_t = 2  # time step
max_t = 14400  # time horizon

max_x = 7000  # length of Lyon in m, taken from the fact that the surface of Lyon = 42 km2
max_y = 7000  # width of Lyon

cancel = np.zeros(int(max_t / delta_t)) # number of cancelled requests

alpha = -3  # constant, taken from the PhD thesis of Mikhail Murashkin

sigma_idle = []  # standard dev for the distance run by idle vehicles, taken from the regression model
# sigma_idle.append(92)        # 0.46 * 200
sigma_occup = 60  # constant, st dev for the distance run by occup veh-s. taken from the microsimulation as the average (0.86 * 200)
sigma_pv = 60  # equal to sigma_occup

dist_cancel = 2000  # the cancellation threshold of idle distance

queue_wait_time = 180  # time (in sec) that a customer waits in the queue before being transferred to another company

frac_n_I_mov = 0

glob_dem_dens = []
glob_idle_veh_dens = []
glob_queued_dem_norm = []


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
    if i == delta_t:
        o_RHI.append(max(min((n_RHI[-1] + alpha * (m_RHI[-1] / l_RHI_s[-1] - n_RHI[-1])) * v[-1] / l_RHI[-1],
                             n_RHI[-1] / delta_t + lmd_RH[int(i / delta_t)]), 0))
    else:
        o_RHI.append(max(min((n_RHI[-1] + alpha * (m_RHI[-1] / l_RHI_s[-1] - n_RHI[-1])) * v[-1] / l_RHI[-1],
                             n_RHI[-1] / delta_t + o_PAS[-1]), 0))
    print("o_RHI: " + str(o_RHI[-1]))
    if o_RHI[-1] < 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    o_RH.append(min((n_RH[-1] + alpha * (m_RH[-1] / l_RH_s[-1] - n_RH[-1])) * v[-1] / l_RH[int(i / delta_t) - 1],
                    n_RH[-1] / delta_t + o_RHI[-1]))
    print("o_RH: " + str(o_RH[-1]))
    o_PV.append(min((n_PV[-1] + alpha * (m_PV[-1] / l_PV_s[-1] - n_PV[-1])) * v[-1] / l_PV[int(i / delta_t) - 1],
                    n_PV[-1] / delta_t + lmd_PV[-1]))
    print("o_PV: " + str(o_PV[-1]))


def sigma_idle_calcul():  # calculation of st dev of idle veh-s using a regression model from microscopic simulation
    if n_I[-1] == 0:  # if there are queued passengers
        """demand_count_norm = (lmd_RH[int(i / delta_t)] / (max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over one minute while here the time is calculated in sec
        veh_density = ((n_I[-1]) / (max_x * max_y)) * 40000    # calculate vehicle density
        glob_dem_dens.append(demand_count_norm)
        glob_idle_veh_dens.append(veh_density)
        sigma_idle.append(sigma_idle[-1])
        print("sigma idle: " + str(sigma_idle[-1]))"""
        demand_count_norm = (lmd_RH[int(i / delta_t)] / (max_x * max_y)) * 40000 * 40
        queued_demand_norm = (n_PAS[-1] / (max_x * max_y)) * 40000 * 40
        veh_density = ((n_I[-1] + n_RHI[-1] + n_RH[-1]) / (max_x * max_y)) * 40000
        glob_queued_dem_norm.append(queued_demand_norm)
        glob_dem_dens.append(demand_count_norm)
        glob_idle_veh_dens.append(((n_I[-1]) / (max_x * max_y)) * 40000)

        """X = {
            "demand_density": [demand_count_norm],
            "queued_demand_norm": [queued_demand_norm],
            "vehicle_density": [veh_density]
            # "request_rate": [request_rate_per_veh_per_dt]
            # "queued_demand_density": [demand_queued_norm]
        }
        Xdf = pd.DataFrame(X)  # transform to dataframe
        x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
            Xdf)  # transform data to use polynomial regression
        model = pickle.load(open('/Users/maryia/PycharmProjects/SimulationTrial/queued_st_dev_idle_dist_model',
                                 'rb'))  # use regression model created in micro-simulation
        sigma_idle_ = model.predict(x_)  # prediction
        sigma_idle.append(sigma_idle_[0] * 200)  # multiplying by 200 as sigma is calculated for a block of 200 m
        print("sigma idle: " + str(sigma_idle[-1]))"""

        sigma_idle.append(386)  # average from saturated micro simulation (data stored in micro_sim_data2)

        # sigma_idle.append(beta.std(2.39, 4.7145) * 14000)
        print("sigma idle: " + str(sigma_idle[-1]))

    elif n_I[-1] > 0:  # if there are no queued passengers
        demand_count_norm = (lmd_RH[int(i / delta_t)] / (
                    max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over 40 sec while here the time is calculated in 1 sec
        veh_density = ((n_I[-1]) / (max_x * max_y)) * 40000  # calculate vacant (idle) vehicle density
        glob_dem_dens.append(demand_count_norm)
        glob_idle_veh_dens.append(veh_density)
        glob_queued_dem_norm.append(0)
        # request_rate_per_veh_per_dt = lmd_RH[int(i / delta_t)] / (n_RHI[-1] + n_RH[-1] + n_I[-1])     # request per veh per time step
        # demand_queued_norm = n_PAS[-1] / (max_x * max_y)

        X = {
            "demand_density": [demand_count_norm],
            "vacant_vehicle_density": [veh_density]
            # "request_rate": [request_rate_per_veh_per_dt]
            # "queued_demand_density": [demand_queued_norm]
        }

        Xdf = pd.DataFrame(X)  # transform to dataframe
        x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
            Xdf)  # transform data to use polynomial regression
        model = pickle.load(open('/Users/maryia/PycharmProjects/SimulationTrial/estimation models/big_fleet_st_dev',
                                 'rb'))  # use regression model created in micro-simulation
        sigma_idle_ = model.predict(x_)  # prediction
        sigma_idle.append(sigma_idle_[0] * 200)  # multiplying by 200 as sigma is calculated for a block of 200 m
        print("sigma idle: " + str(sigma_idle[-1]))
        if sigma_idle[-1] < 0:
            print("Error: negative standard deviation")
            if i == 0:
                sigma_idle[-1] = 2
            else:
                sigma_idle[-1] = sigma_idle[-2]
            #exit()



def steady_dist_l_s(i):  # calculation of avg remaining distance t obe traveled in steady state
    sigma_idle_calcul()
    l_RHI_s.append((l_RHI[-1] ** 2 + sigma_idle[-1] ** 2) / (2 * l_RHI[-1]))
    l_RH_s.append((l_RH[int(i / delta_t)] ** 2 + sigma_occup ** 2) / (2 * l_RH[int(i / delta_t)]))
    l_PV_s.append((l_PV[int(i / delta_t)] ** 2 + sigma_pv ** 2) / (2 * l_PV[int(i / delta_t)]))
    print("l_RHI_*: " + str(l_RHI_s[-1]))
    print("l_RH_*: " + str(l_RH_s[-1]))
    print("l_PV_*: " + str(l_PV_s[-1]))


def idle_dist_calcul():  # calculation of avg trip length of idle veh-s using a regression model from microscopic simulation
    if n_I[-1] == 0:
        """l_RHI.append(l_RHI[-1])
        print("l_RHI: " + str(l_RHI[-1]))"""
        demand_count_norm = (lmd_RH[int(i / delta_t)] / (max_x * max_y)) * 40000 * 40
        queued_demand_norm = (n_PAS[-1] / (max_x * max_y)) * 40000 * 40
        veh_density = ((n_I[-1] + n_RHI[-1] + n_RH[-1]) / (max_x * max_y)) * 40000

        """print("!!!!! demand_count_norm: " + str(demand_count_norm))
        print("!!!!! queued_demand_norm: " + str(queued_demand_norm))
        print("!!!!! veh_density: " + str(veh_density))

        X = {
            "demand_density": [demand_count_norm],
            "queued_demand_norm": [queued_demand_norm],
            "vehicle_density": [veh_density]
            # "request_rate": [request_rate_per_veh_per_dt]
            # "queued_demand_density": [demand_queued_norm]
        }
        Xdf = pd.DataFrame(X)  # transform to dataframe
        x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
            Xdf)  # transform data to use polynomial regression
        model = pickle.load(open('/Users/maryia/PycharmProjects/SimulationTrial/queued_idle_dist_model',
                                 'rb'))  # use regression model created in micro-simulation
        idle_dist_ = model.predict(x_)  # prediction
        l_RHI.append(idle_dist_[0] * 200)
        print("l_RHI: " + str(l_RHI[-1]))
        if l_RHI[-1] < 0:
            exit()"""

        l_RHI.append(2403)  # average from saturated micro simulation (data stored in micro_sim_data2)

        # l_RHI.append(np.random.beta(2.39, 4.7145, 1)[0] * 14000)
        print("l_RHI: " + str(l_RHI[-1]))

    elif n_I[-1] > 0:
        demand_count_norm = (lmd_RH[int(i / delta_t)] / (
                    max_x * max_y)) * 40000 * 40  # normalizing the demand (40000 = 200 * 200, which is a length of block in microsim) and multiplying by 40 as in microsim regression is done for demand normalized over one minute while here the time is calculated in sec
        veh_density = ((n_I[-1]) / (max_x * max_y)) * 40000  # calculate idle vehicle density
        # request_rate_per_veh_per_dt = lmd_RH[int(i / delta_t)] / (n_RHI[-1] + n_RH[-1] + n_I[-1])     # request per veh per time step
        # demand_queued_norm = n_PAS[-1] / (max_x * max_y)
        print("DENSITY OF DEMAND " + str(demand_count_norm))
        print("DENSITY OF VEH " + str(veh_density))

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
        l_RHI.append(idle_dist_[0] * 200)
        print("l_RHI: " + str(l_RHI[-1]))



def init_lmb(max_time):  # demand initialization
    dem_vary_t = 7200  # time duration when demand is variable
    # for 1/4 part of dem_vary_t it's constant, for next 1/4 it's increasing, for next 1/4 it's decreasing, for last 1/4 it's constant

    for i1 in range(0, max_time // delta_t + 1):

        ####### Constant demand #######
        if i1 <= (dem_vary_t // delta_t + 1) / 4:
            lmd_RH.append(2.4)
            lmd_PV.append(20.4)

        ####### Increase of demand - use 1/6 part of time duration when demand is increasing #############
        elif (dem_vary_t // delta_t + 1) / 4 < i1 <= (
                (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)):
            lmd_RH.append(2.7)
            lmd_PV.append(21)

        elif ((dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)) < i1 <= \
                ((dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(2.7)
            lmd_PV.append(21)

        elif ((dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                ((dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3)
            lmd_PV.append(21.5)

        elif ((dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                ((dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3.3)
            lmd_PV.append(22)

        elif ((dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                ((dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3.3)
            lmd_PV.append(22)

        # Constant demand
        elif ((dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                (dem_vary_t // delta_t + 1) / 2:
            lmd_RH.append(3.6)
            lmd_PV.append(22.5)

        ######## Decrease of demand - use 1/6 part of time duration when demand is decreasing ############
        elif (2 * (dem_vary_t // delta_t + 1) / 4) < i1 <= \
                (2 * (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)):
            lmd_RH.append(3.6)
            lmd_PV.append(22.5)

        elif (2 * (dem_vary_t // delta_t + 1) / 4 + (dem_vary_t // delta_t + 1) / (4 * 6)) < i1 <= \
                (2 * (dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3.3)
            lmd_PV.append(22)

        elif (2 * (dem_vary_t // delta_t + 1) / 4 + (2 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                (2 * (dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3.3)
            lmd_PV.append(22)

        elif (2 * (dem_vary_t // delta_t + 1) / 4 + (3 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                (2 * (dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(3)
            lmd_PV.append(21.5)

        elif (2 * (dem_vary_t // delta_t + 1) / 4 + (4 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                (2 * (dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)):
            lmd_RH.append(2.7)
            lmd_PV.append(21.5)

        elif (2 * (dem_vary_t // delta_t + 1) / 4 + (5 * (dem_vary_t // delta_t + 1)) / (4 * 6)) < i1 <= \
                (3 * (dem_vary_t // delta_t + 1) / 4):
            lmd_RH.append(2.7)
            lmd_PV.append(21)

        # Constant demand
        elif 3 * (dem_vary_t // delta_t + 1) / 4 < i1 <= (dem_vary_t // delta_t + 1):
            lmd_RH.append(2.4)
            lmd_PV.append(20.5)

        ######## Constant demand for the rest of time horizon ########
        elif (dem_vary_t // delta_t + 1) < i1 <= (max_time // delta_t + 1):
            lmd_RH.append(2.4)
            lmd_PV.append(20.5)

        ####### Check if the demand is negative ########
        if lmd_RH[-1] < 0:
            lmd_RH[-1] = 0


def n_PAS_and_n_I(i):  # Calculation of accumulation of idle veh-s and waiting passengers to be matched
    if (delta_t * o_RH[-1] + n_I[-1]) < \
            (delta_t * lmd_RH[int(i / delta_t)] + n_PAS[-1]):   # case when demand is higher than available vehicles

        n_PAS.append(max(delta_t * (lmd_RH[int(i / delta_t)] - o_RH[-1]) - n_I[-1] + n_PAS[-1], 0))     # accumulation of passengers
        o_PAS.append(o_RH[-1] + n_I[-1] / delta_t)              # outflow of passengers = inflow of idle moving vehicles
        dummy1 = o_PAS[-1]                  # remember the normal outflow of matched passengers before the cancellation

        ############### Cancellation of requests where idle distance is bigger than a threshold #########
        dummy_round_outflow = math.ceil(o_PAS[-1])              # !!!!!!! attention, might be zero - rounded up outflow of matched passengers
        dummy_frac_part = o_PAS[-1] - math.floor(o_PAS[-1])     # !!!!!!! attention, might be zero - fraction part of the outflow of matched passengers
        dummy_int_part = math.floor(o_PAS[-1])                  # !!!!!!! attention, might be zero

        if o_PAS[-1] >= 0:                      # if there are matched passengers
            dummy_lengths = np.random.normal(l_RHI[-1], sigma_idle[-1], dummy_round_outflow)    # generate the idle distance length for each demand request knowing the current idle distance and standard deviation and using normal distribution
            # return the index of elements that are bigger than a threshold, if the last element is also bigger, check if there is a fraction and reduce o_PAS by fraction, otherwise only by one unit
            def condition(x): return x >= dist_cancel      # function to return the values from a list that satisfy the constraint
            output = [idx for idx, element in enumerate(dummy_lengths) if condition(element)]       # return the indexes of those distance values that are higher than a threshold
            if len(output) != 0 and output[-1] == (len(dummy_lengths) - 1) and len(dummy_lengths) != 0 and dummy_frac_part != 0:    # check if the fraction part of the initial outflow of matched passengers has a generated distance higher than a threshold
                o_PAS[-1] = o_PAS[-1] - dummy_frac_part     # if yes - reduce this fraction from the initial outflow of matched passengers
                o_PAS[-1] = o_PAS[-1] - len(output) + 1     # and reduce other requests that have distance higher than a threshold, but add 1 because we already reduced the fraction
            else:
                o_PAS[-1] = o_PAS[-1] - len(output)         # if no fraction parts with higher distance - than just reduce other requests that have distance higher than a threshold

        ##################################################################################################

        n_I.append((dummy1 - o_PAS[-1]) * delta_t)          # accumulation of idle vehicles - add those vehicles that were supposed to serve the cancelled requests
        n_PAS[-1] = n_PAS[-1] + (dummy1 - o_PAS[-1]) * delta_t                   # add in waiting list the difference between the initially calculated pass. outflow and the new one

    elif (delta_t * o_RH[-1] + n_I[-1]) >= \
            (delta_t * lmd_RH[int(i / delta_t)] + n_PAS[-1]):            # case when demand is lower or equal than available vehicles

        o_PAS.append(lmd_RH[int(i / delta_t)] + n_PAS[-1] / delta_t)     # outflow of passengers = inflow of idle moving vehicles
        dummy1 = o_PAS[-1]

        ############### Cancellation of requests where idle distance is bigger than a threshold #########
        dummy_round_outflow = math.ceil(o_PAS[-1])              # !!!!!!! attention, might be zero
        dummy_frac_part = o_PAS[-1] - math.floor(o_PAS[-1])     # !!!!!!! attention, might be zero
        dummy_int_part = math.floor(o_PAS[-1])                  # !!!!!!! attention, might be zero

        if o_PAS[-1] >= 0:
            dummy_lengths = np.random.normal(l_RHI[-1], sigma_idle[-1], dummy_round_outflow)
            # return the index of elements that are bigger than a threshold, if the last element is also bigger, check if there is a fraction and reduce o_PAS by fraction, otherwise only by one unit
            def condition(x): return x >= dist_cancel
            output = [idx for idx, element in enumerate(dummy_lengths) if condition(element)]
            if len(output) != 0 and output[-1] == (len(dummy_lengths) - 1) and len(dummy_lengths) != 0 and dummy_frac_part != 0:
                o_PAS[-1] = o_PAS[-1] - dummy_frac_part
                o_PAS[-1] = o_PAS[-1] - len(output) + 1
            else:
                o_PAS[-1] = o_PAS[-1] - len(output)

        ##################################################################################################

        n_I.append(max(delta_t * (o_RH[-1] - lmd_RH[int(i / delta_t)]) - n_PAS[-1] + n_I[-1], 0))       # accumulation of idle vehicles
        n_I[-1] = n_I[-1] + (dummy1 - o_PAS[-1]) * delta_t          # add also those vehicles that were supposed to serve

        n_PAS.append((dummy1 - o_PAS[-1]) * delta_t)

    o_PAS_total.append(o_PAS[-1])

    ################ Count passengers that are waiting for more than the specified time ######################

    if i >= queue_wait_time:
        accum_o_PAS_total = np.cumsum(o_PAS_total)
        accum_lmd_RH = np.cumsum(lmd_RH)
        for t2 in range(0, len(accum_lmd_RH)):
            if (accum_o_PAS_total[-1] - 0.05 <= accum_lmd_RH[t2] <= accum_o_PAS_total[-1] + 0.05) and \
                    (i - t2 * delta_t > queue_wait_time) and \
                    (accum_lmd_RH[int((i - queue_wait_time) / 2)] - accum_o_PAS_total[-1] > 0):
                # and ((accum_lmd_RH[key][int((i - queue_wait_time) / 2)] - accum_o_PAS[key][-1]) * delta_t <= n_PAS[key][-1]):
                dum1 = accum_lmd_RH[int((i - queue_wait_time) / 2)]
                dum2 = accum_o_PAS_total[-1]
                cancel[int(i / delta_t)] = (dum1 - dum2)

                n_PAS[-1] = n_PAS[-1] - cancel[int(i / delta_t)] * delta_t
                o_PAS_total[-1] = o_PAS_total[-1] + cancel[int(i / delta_t)]
                break


def normal_dist_of_l_RH(l_RH):  # visualization of normal distribution of occupied vehicle distance
    count, bins, ignored = plt.hist(l_RH, 30, density=True)
    plt.plot(bins, 1 / (sigma_occup * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - 8036) ** 2 / (2 * sigma_occup ** 2)),
             linewidth=2, color='r')
    plt.title("Norm distribution of l_RH")
    plt.show()


def normal_dist_of_l_PV(l_PV):  # visualization of normal distribution of private vehicle distance
    count, bins, ignored = plt.hist(l_PV, 30, density=True)
    plt.plot(bins, 1 / (sigma_pv * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - 8036) ** 2 / (2 * sigma_pv ** 2)),
             linewidth=2, color='r')
    plt.title("Norm distribution of l_PV")
    plt.show()


def visualization():  ################## Visualization #################
    tick = []
    for a in range(0, 14401):
        tick.append(math.floor(a / 3600))

    def idle():  # reservoir of idle waiting vehicles and waiting passengers
        # plt.figure(1)
        plt.subplots(7, 1, constrained_layout=True)
        plt.subplot(7, 1, 1)
        x = n_I
        plt.plot(x, 'b')
        plt.title('n_I - accum. of idle stopped')

        plt.subplot(7, 1, 2)
        x = o_RH
        plt.plot(x, 'b')
        y = [0] * len(o_RH)
        plt.plot(y, 'r')
        plt.title('o_RH - inflow to idle stopped')

        plt.subplot(7, 1, 3)
        x = lmd_RH
        plt.plot(x, 'b')
        plt.title('lmd_RH - arriving demand requests')

        plt.subplot(7, 1, 4)
        x = n_PAS
        plt.plot(x, 'b')
        plt.title('n_PAS - accum. of waiting pass. to be matched')

        plt.subplot(7, 1, 5)
        x = o_PAS
        plt.plot(x, 'b')
        plt.title('o_PAS - outflow of matched passengers')

        plt.subplot(7, 1, 6)
        x = cancel
        plt.plot(x, 'b')
        plt.title('# of request cancellations')

        plt.subplot(7, 1, 7)
        x = o_PAS_total
        plt.plot(x, 'b')
        # plt.locator_params(axis='x', nbins=15)
        plt.title('o_PAS_total - outflow of all passengers (matched+canceled)')

    def veh():
        plt.subplots(5, 1, constrained_layout=True)
        plt.subplot(5, 1, 1)
        x = num_veh
        plt.plot(x, 'magenta')
        plt.title('total # of RH vehicles in the system')

        plt.subplot(5, 1, 2)
        x = num_mov_veh
        plt.plot(x, 'magenta')
        plt.title('# of moving RH vehicles in the system')

        plt.subplot(5, 1, 3)
        x = n_I_mov
        plt.plot(x, 'magenta')
        plt.title('# of cruising idle RH vehicles')

        plt.subplot(5, 1, 4)
        x = n_I_stop
        plt.plot(x, 'magenta')
        plt.title('# of non-moving idle RH vehicles')

        plt.subplot(5, 1, 5)
        x = prod_n_I_mov
        plt.plot(x, 'magenta')
        plt.title('production of cruising idle RH vehicles per time unit')

    def rhi():  # reservoir of idle moving vehicles
        # plt.figure(2)
        plt.subplots(6, 1, constrained_layout=True)
        plt.subplot(6, 1, 1)
        x = n_RHI
        plt.plot(x, 'k')
        plt.title('n_RHI - accum. of idle moving')

        plt.subplot(6, 1, 2)
        x = o_PAS
        plt.plot(x, 'k')
        plt.title('o_PAS - inflow of matched passengers')

        plt.subplot(6, 1, 3)
        x = o_RHI
        plt.plot(x, 'k')
        y = [0] * len(o_RHI)
        plt.plot(y, 'r')
        plt.title('o_RHI - outflow of idle moving')

        plt.subplot(6, 1, 4)
        x = m_RHI
        plt.plot(x, 'k')
        y = [0] * len(o_RHI)
        plt.plot(y, 'r')
        plt.title('m_RHI - rem. dist. of idle moving')

        plt.subplot(6, 1, 5)
        x = l_RHI
        plt.plot(x, 'k')
        plt.title('l_RHI - avg trip length of idle moving')

        plt.subplot(6, 1, 6)
        x = sigma_idle
        plt.plot(x, 'k')
        plt.title('standard deviation of idle distance')



        # plt.tight_layout()
        # plt.subplots(constrained_layout=True)

    def rh():  # reservoir of moving with passenger vehicles
        plt.subplots(6, 1, constrained_layout=True)
        plt.subplot(6, 1, 1)
        x = n_RH
        plt.plot(x, 'g')
        plt.title('n_RH - accum. of moving w/ passenger')

        plt.subplot(6, 1, 2)
        x = o_RHI
        plt.plot(x, 'g')
        y = [0] * len(o_RHI)
        plt.plot(y, 'r')
        plt.title('o_RHI - inflow of moving w/ passenger')

        plt.subplot(6, 1, 3)
        x = o_RH
        plt.plot(x, 'g')
        y = [0] * len(o_RH)
        plt.plot(y, 'r')
        plt.title('o_RH - outflow of moving w/ passenger')

        plt.subplot(6, 1, 4)
        x = m_RH
        plt.plot(x, 'g')
        plt.title('m_RH - rem. dist. of moving w/ passenger')

        plt.subplot(6, 1, 5)
        x = v
        plt.plot(x, 'g')
        plt.title('v - network speed')

        plt.subplot(6, 1, 6)
        x = l_RH
        plt.plot(x, 'g')
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

    def accum():
        plt.subplots(7, 1, constrained_layout=True)
        plt.subplot(7, 1, 1)
        x = n_I
        plt.plot(x, 'k')
        plt.title('accum. of idle non-moving')

        plt.subplot(7, 1, 2)
        x = n_PAS
        plt.plot(x, 'k')
        plt.title('accum. of waiting pass. to be matched')

        plt.subplot(7, 1, 3)
        x = n_RHI
        plt.plot(x, 'k')
        plt.title('accum. of idle moving')

        plt.subplot(7, 1, 4)
        x = l_RHI
        plt.plot(x, 'k')
        plt.title('avg trip length of idle moving')

        plt.subplot(7, 1, 5)
        x = n_RH
        plt.plot(x, 'k')
        plt.title('accum. of moving w/ passenger')

        plt.subplot(7, 1, 6)
        x = lmd_RH
        plt.plot(x, 'k')
        plt.title('arriving demand requests')

        plt.subplot(7, 1, 7)
        x = n_PV
        plt.plot(x, 'y')
        plt.title('accum. of private vehicles')

    def countcurve():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = np.cumsum(o_PAS)
        y = np.cumsum(o_RHI)
        z = np.cumsum(o_RH)
        j = np.cumsum(lmd_RH)
        p = np.cumsum(o_PAS_total)
        plt.plot(x, 'magenta', label='o_PAS')
        plt.plot(y, 'k', label='o_RHI')
        plt.plot(z, 'g', label='o_RH')
        plt.plot(j, 'r', label='lmb_RH')
        plt.plot(p, 'b', label='o_PAS_total')
        plt.legend(loc="upper left")

    def regression_param():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        data = [glob_dem_dens]
        plt.boxplot(data)
        plt.title('density of demand used for regression')

    def regression_param2():
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        data = [glob_queued_dem_norm]
        plt.boxplot(data)
        plt.title('density of queued demand used for regression')

    def demand_curve_PV():
        labels = ["0", "1", "2", "3", "4"]
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = lmd_PV
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'red')
        plt.ylabel("Demand rate", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)
        #plt.title('Private vehicles demand curve')

    def demand_curve_RH():
        labels = ["0", "1", "2", "3", "4"]
        plt.subplots(1, 1, constrained_layout=True)
        plt.subplot(1, 1, 1)
        x = lmd_RH
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'k')
        plt.ylabel("Demand rate", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)
        #plt.title('Ride-hailing vehicles demand curve')

    def visual_maas():
        plt.subplots(4, 1, constrained_layout=True)
        plt.subplot(4, 1, 4)
        x = n_I
        plt.plot(x, 'b')
        plt.title('n_I - accumulation of vacant non-moving vehicles')
        plt.xlabel("time")

        plt.subplot(4, 1, 1)
        x = lmd_RH
        plt.plot(x, 'b')
        plt.title('lmd_RH - arriving demand requests')

        plt.subplot(4, 1, 2)
        x = n_PAS
        plt.plot(x, 'b')
        plt.title('n_PAS - accumulation of waiting passengers to be matched')


        plt.subplot(4, 1, 3)
        x = cancel
        plt.plot(x, 'b')
        plt.title('number of canceled requests')

    def big_vehicles1():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        #f.set_figheight(3)
        labels = ["0", "1", "2", "3", "4"]
        x = n_PAS
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'b')
        plt.ylabel("Accumulation\nof passengers", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles2():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = cancel
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'b')
        plt.ylabel("Number of\ncanceled requests", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles3():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = n_I
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'b')
        plt.ylabel("Accumulation of\nvacant vehicles", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles4():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = n_RHI
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'b')
        plt.ylabel("Accumulation of idle\nmoving vehicles", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles5():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = l_RHI
        x = [val for val in x for _ in (0, 1)]
        x = [k / 1000 for k in x]
        plt.plot(x, 'b')
        plt.ylabel("Mean idle\ndistance (km)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles6():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = sigma_idle
        x = [val for val in x for _ in (0, 1)]
        x = [k / 1000 for k in x]
        plt.plot(x, 'b')
        plt.ylabel("Standard deviation\nof idle distance (km)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles7():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = n_RH
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'b', label='Company 1')
        plt.ylabel("Accumulation of\nservice vehicles", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles8():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = v
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'g')
        plt.ylabel("Network speed (m/s)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles9():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = n_PV
        x = [val for val in x for _ in (0, 1)]
        x = [k / 10000 for k in x]
        plt.plot(x, 'y', label='Private vehicles')
        plt.ylabel("Accumulation of\nprivate vehicles ($10^4$ veh)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
        plt.yticks(fontsize=FONT_SIZE_AXI)

    def big_vehicles10():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = lmd_PV
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'red')
        plt.ylabel("Demand rate\n(requests per sec)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_LAB)
        plt.yticks(fontsize=FONT_SIZE_LAB)

    def big_vehicles11():
        plt.subplots(1, 1, constrained_layout=True, figsize=(15, 3))
        plt.subplot(1, 1, 1)
        labels = ["0", "1", "2", "3", "4"]
        x = lmd_RH
        x = [val for val in x for _ in (0, 1)]
        plt.plot(x, 'k')
        plt.ylabel("Demand rate\n(requests per sec)", fontsize=FONT_SIZE_LAB)
        plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
        plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_LAB)
        plt.yticks(fontsize=FONT_SIZE_LAB)

    #idle()
    #veh()
    #rhi()
    #rh()
    #pv()
    #accum()
    #countcurve()
    #regression_param()
    #regression_param2()
    #demand_curve_RH()
    #demand_curve_PV()
    #visual_maas()
    big_vehicles1()
    big_vehicles2()
    big_vehicles3()
    big_vehicles4()
    big_vehicles5()
    big_vehicles6()
    big_vehicles7()
    big_vehicles8()
    big_vehicles9()
    big_vehicles10()
    big_vehicles11()
    plt.show()
    #print('density of queued demand used for regression: ')
    #print(glob_queued_dem_norm)


if __name__ == "__main__":

    ########## Initialization of starting values at t = 0 ##############
    # pay attention that speed, length and time need to have the same unit of measurement (m/s)
    print("iter: 0")
    n_PAS.append(0)
    n_I.append(1800)
    n_RHI.append(600)
    n_RH.append(1200)
    n_PV.append(20659)

    n_I_mov.append(round(n_I[-1] * frac_n_I_mov))
    n_I_stop.append(n_I[-1] - n_I_mov[-1])

    #max_t = 14400  # time horizon in sec

    init_lmb(max_t)  # demand initialization

    # for i2 in range(0, max_t // delta_t + 1):       # private vehicles demand initialization
    #    lmd_PV.append(20)

    # for i in range(0, max_t // delta_t + 1):       # initialization of constant demand
    #    lmd_RH.append(0.3)

    i = 0  # to avoid any errors reinitialize i
    idle_dist_calcul()  # prediction of avg idle distance using regression
    # l_RHI.append(500)       #
    # l_RH.append(5740)      # constant, taken as avg from microscopic simulation (28.7 * 200)
    # l_PV.append(5740)      # constant, taken as avg from microscopic simulation (28.7 * 200)
    np.random.seed(4424)
    l_RH = np.random.normal(4424, sigma_occup, max_t // delta_t)  # random avg trip length from norm distribution
    l_PV = np.random.normal(4424, sigma_pv, max_t // delta_t)  # random avg trip length from norm distribution
    # normal_dist_of_l_RH(l_RH)      # to verify the norm distribution of l_RH
    # normal_dist_of_l_PV(l_PV)      # to verify the norm distribution of l_PV

    print("l_RH: " + str(l_RH[0]))
    print("l_PV: " + str(l_PV[0]))

    v.append(calculate_V_MFD(n_RHI[-1] + n_RH[-1] + n_PV[-1] + n_I_mov[-1]))  # calculation of speed
    prod_n_I_mov.append(n_I_mov[-1] * v[-1] * delta_t)

    #### Estimation of total remaining travel distance
    m_RHI.append(l_RHI[0] * n_RHI[0] / 2)
    m_RH.append(l_RH[0] * n_RH[0] / 2)
    m_PV.append(l_PV[0] * n_PV[0] / 2)

    num_veh.append(n_I[-1] + n_RHI[-1] + n_RH[-1])  # total number of RH vehicles
    num_mov_veh.append(n_RHI[-1] + n_RH[-1])

    print("lmb_RH: " + str(lmd_RH[int(i / delta_t)]))
    print("lmb_PV: " + str(lmd_PV[int(i / delta_t)]))
    print("n_PAS: " + str(n_PAS[-1]))
    print("n_I: " + str(n_I[-1]))
    print("n_RHI: " + str(n_RHI[-1]))
    print("n_RH: " + str(n_RH[-1]))
    print("n_PV: " + str(n_PV[-1]))
    print("n moving: " + str(n_RHI[-1] + n_RH[-1] + n_PV[-1]))
    print("n: " + str(n_I[-1] + n_RHI[-1] + n_RH[-1] + n_PV[-1]))
    print("v: " + str(v[-1]))
    print("m_RHI: " + str(m_RHI[-1]))
    print("m_RH: " + str(m_RH[-1]))
    print("m_PV: " + str(m_PV[-1]))
    print("sigma_occup: " + str(sigma_occup))
    print("sigma_pv: " + str(sigma_pv))
    steady_dist_l_s(i)  # avg remaining distance to be traveled in steady state

    print("# of RH vehicles in the system: " + str(num_veh[-1]))

    for i in range(delta_t, max_t, delta_t):
        print()
        print("iter: " + str(i))

        o_MFD(i)

        # n_I.append(max(delta_t * (o_RH[-1] - lmd_RH[int(i / delta_t)]) + n_I[-1], 0))
        n_PAS_and_n_I(i)  # accumulation of idle stopped vehicles and passengers waiting to be matched. Must be placed after calculation of outflow and before calculation of vehicle accumulation

        n_I_mov.append(round(n_I[-1] * frac_n_I_mov))
        n_I_stop.append(n_I[-1] - n_I_mov[-1])

        print("o_PAS: " + str(o_PAS[-1]))
        print("lmb_RH: " + str(lmd_RH[int(i / delta_t)]))
        print("lmb_PV: " + str(lmd_PV[int(i / delta_t)]))

        ######### Calculation of accumulation. Negative accumulation substituted with 0
        n_RHI.append(max(delta_t * (o_PAS[-1] - o_RHI[-1]) + n_RHI[-1], 0))
        n_RH.append(max(delta_t * (o_RHI[-1] - o_RH[-1]) + n_RH[-1], 0))
        n_PV.append(max(delta_t * (lmd_PV[int(i / delta_t)] - o_PV[-1]) + n_PV[-1], 0))
        print("n_PAS: " + str(n_PAS[-1]))
        print("n_I: " + str(n_I[-1]))
        print("n_RHI: " + str(n_RHI[-1]))
        print("n_RH: " + str(n_RH[-1]))
        print("n_PV: " + str(n_PV[-1]))
        print("n moving: " + str(n_RHI[-1] + n_RH[-1] + n_PV[-1]))
        print("n: " + str(n_I[-1] + n_RHI[-1] + n_RH[-1] + n_PV[-1]))

        v.append(calculate_V_MFD(n_RHI[-1] + n_RH[-1] + n_PV[-1] + n_I_mov[-1]))
        print("v: " + str(v[-1]))
        prod_n_I_mov.append(n_I_mov[-1] * v[-1] * delta_t)

        idle_dist_calcul()

        print("l_RH: " + str(l_RH[int(i / delta_t)]))
        print("l_PV: " + str(l_PV[int(i / delta_t)]))

        print("sigma_occup: " + str(sigma_occup))
        print("sigma_pv: " + str(sigma_pv))
        steady_dist_l_s(i)

        m_RHI.append(delta_t * (o_PAS[-1] * l_RHI[-1] - n_RHI[-1] * v[-1]) + m_RHI[-1])
        m_RH.append(delta_t * (o_RHI[-1] * l_RH[int(i / delta_t)] - n_RH[-1] * v[-1]) + m_RH[-1])
        m_PV.append(delta_t * (lmd_PV[int(i / delta_t)] * l_PV[int(i / delta_t)] - n_PV[-1] * v[-1]) + m_PV[-1])
        print("m_RHI: " + str(m_RHI[-1]))
        print("m_RH: " + str(m_RH[-1]))
        print("m_PV: " + str(m_PV[-1]))

        num_veh.append(n_I[-1] + n_RHI[-1] + n_RH[-1])
        num_mov_veh.append(n_RHI[-1] + n_RH[-1])
        print("# of RH vehicles in the system: " + str(num_veh[-1]))
        # if num_veh[-1] < num_veh[-2] and len(num_veh) > 2:
        # if (num_veh[-2] - num_veh[-1] > 0.1) and len(num_veh) > 2:
        # if i == 36:
        #    exit()

    o_MFD(max_t)
    n_PAS_and_n_I(i)  # accumulation of idle stopped vehicles and passengers waiting to be matched. Must be placed after calculation of outflow and before calculation of vehicle accumulation
    n_I_mov.append(round(n_I[-1] * frac_n_I_mov))
    n_I_stop.append(n_I[-1] - n_I_mov[-1])

    print("avg # of PV veh over time: " + str(sum(n_PV)/len(n_PV)))
    print("total RH demand: " + str(sum(lmd_RH)))
    print("total PV demand: " + str(sum(lmd_PV)))
    visualization()

    ################### Comparison metrics starts here ##########################

    print("avg speed: " + str(sum(v)/len(v)))
    print("total accum of veh: " + str(sum(num_mov_veh) + sum(n_PV)))
    print("total cancellation: " + str(sum(cancel) * delta_t))

    accum_o_PAS_total = np.cumsum(o_PAS_total)
    accum_lmd_RH = np.cumsum(lmd_RH)
    for t4 in range(0, len(accum_lmd_RH)):
        for t5 in range(0, len(accum_o_PAS_total)):
            if accum_o_PAS_total[t5] - 0.05 <= accum_lmd_RH[t4] <= accum_o_PAS_total[t5] + 0.05:
                pass_wait_time.append((t5 - t4) * delta_t)

    print("avg pass wait time: " + str(sum(pass_wait_time)/len(pass_wait_time)))



    data_new = pd.read_csv('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/strategy_comparison/speed.csv')
    data_new['v_coop'] = v
    data_new.to_csv('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/strategy_comparison/speed.csv', index=False)

    """
    ################### Comparison metrics ends here ##########################

    # open the file in the write mode
    with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/one_comp/length_1_comp.csv',
              'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(l_RHI)

    with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/one_comp/speed_1_comp.csv',
              'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(v)
    """



