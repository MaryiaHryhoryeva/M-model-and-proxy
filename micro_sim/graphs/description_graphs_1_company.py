#from micro_sim.main1 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_SIZE_HALF = (6, 3)
FONT_SIZE_LAB = 14
FONT_SIZE_LEG = 12
FONT_SIZE_AXI = 12

########################################
#### Average occupied distance run #####
########################################
def boxplot1():
    columns = ["glob_occup_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    data = df["glob_occup_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data = [k * 200 for k in data]
    plt.figure(1)
    #data = [glob_occup_dist_avg]
    plt.boxplot(data)
    plt.ylabel("Distance (m)")
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    #plt.gca().set_ylim([20, 50])
    plt.title('Average service trip length')

####################################
#### Average idle distance run #####
####################################
def boxplot2():
    columns = ["glob_idle_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    data2 = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data2 = [k * 200 for k in data2]
    plt.figure(2)
    #data2 = [glob_idle_dist_avg]
    plt.boxplot(data2)
    plt.ylabel("Distance (m)")
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    #plt.gca().set_ylim([0, 40])
    plt.title('Average idle trip length')

def boxplot3():
    columns = ["glob_st_dev_idle"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)

    data2 = df["glob_st_dev_idle"].values.tolist()
    data2 = [x * 200 for x in data2]

    #data2 = df["glob_st_dev_idle"]  # {"idle_dist": glob_idle_dist_avg}
    plt.figure(3)
    #data2 = [glob_idle_dist_avg]
    plt.boxplot(data2)
    plt.ylabel("Distance (m)")
    #plt.gca().xaxis.set_ticklabels([str(num_veh['uber']) + ' cars'])
    #plt.gca().set_ylim([0, 40])
    plt.title('Average st dev of idle trip length')


####################################
#### Boxplots to histogram #####
####################################

def hist1():
    columns = ["glob_occup_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    data = df["glob_occup_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data = [k * 200/1000 for k in data]
    #data = [(k - 0)/(max(data) - 0) for k in data]
    plt.figure(16)
    plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.xlabel('Average service distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_LAB)
    plt.xticks(fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.title('Average service trip length')

def hist2():
    columns = ["glob_idle_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv",
                     usecols=columns)
    data = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data = [k * 200/1000 for k in data]
    #data = [(k - 0) / (max(data) - 0) for k in data]
    plt.figure(17)
    plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.xlabel('Average idle distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_LAB)
    plt.xticks( fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.title('Average idle trip length')

def hist3():
    columns = ["glob_occup_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    data = df["glob_occup_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data = [k * 200/1000 for k in data]
    data = [(k - sum(data)/len(data))/(sum(data)/len(data)) for k in data]
    plt.figure(18)
    plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.xlabel('Average service distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_LAB)
    plt.xticks(fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.title('Average service trip length')

def hist4():
    columns = ["glob_idle_dist_avg"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv",
                     usecols=columns)
    data = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    data = [k * 200 / 1000 for k in data]
    data = [(k - sum(data)/len(data))/(sum(data)/len(data)) for k in data]
    plt.figure(19)
    plt.hist(x=data, bins='auto', color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.xlabel('Average idle distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_LAB)
    plt.xticks(fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    # plt.title('Average idle trip length')

#############################################
#### Occupied distance vs demand density #####
#############################################

def occup_vs_dem_dens():
    columns = ["glob_demand_count_norm", "glob_occup_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x1 = df["glob_demand_count_norm"]  # {"idle_dist": glob_idle_dist_avg}
    y1 = df["glob_occup_dist_avg"]
    z1 = df["glob_demand_rejection_rate"]
    #x1 = glob_demand_count_norm
    #y1 = glob_occup_dist_avg

    plt.figure(3)
    plt.title('Occupied distance vs demand density per time unit')
    #plt.plot(x1, y1, 'ko')
    #plt.scatter(x1, y1, c=z1, cmap='cool')
    plt.scatter(x1, y1, c=z1)
    #plt.scatter(x1, y1, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('demand density')
    plt.ylabel('occupied distance')
    #plt.gca().set_ylim([20, 50])
    #plt.legend()
    #plt.colorbar(orientation="horizontal", label="demand rejection rate")

    #z = np.polyfit(x1, y1, 1)
    #p = np.poly1d(z)
    #plt.plot(x1, p(x1), "k-")



#############################################
#### Idle distance vs demand density #####
#############################################

def idle_vs_dem_dens():
    columns = ["glob_demand_count_norm", "glob_idle_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x2 = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y2 = df["glob_demand_count_norm"]
    #z2 = df["glob_demand_rejection_rate"]
    #x2 = glob_demand_count_norm
    #y2 = glob_idle_dist_avg
    x2 = [k * 200/1000 for k in x2]
    y2 = [k * 37.5 for k in y2]         # (k / (200 * 200 * 40)) * 1000 * 1000 * 60 - per sq km per min

    plt.figure(4)
    plt.title('Idle distance vs demand density per time unit')
    #plt.plot(x2, y2, 'ko')
    #plt.scatter(x2, y2, c=z2, cmap='cool')
    plt.scatter(x2, y2)
    #plt.scatter(x2, y2, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('Idle distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Demand density per sq km', fontsize=FONT_SIZE_LAB)      # per min
    plt.xticks(fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.legend()
    #plt.gca().set_ylim([0, 30])
    #plt.colorbar(label='demand rejection rate')

    #z = np.polyfit(x2, y2, 1)
    #p = np.poly1d(z)
    #plt.plot(x2, p(x2), "k-")


####################################################
#### Idle distance vs request rate #####
####################################################

def idle_vs_request_rate():
    columns = ["glob_request_rate_per_veh_per_min", "glob_idle_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x3 = df["glob_request_rate_per_veh_per_min"]  # {"idle_dist": glob_idle_dist_avg}
    y3 = df["glob_idle_dist_avg"]
    z3 = df["glob_demand_rejection_rate"]

    #x3 = glob_request_rate_per_veh_per_min
    #y3 = glob_idle_dist_avg

    plt.figure(5)
    plt.title('Idle distance vs request rate per veh per min')
    #plt.plot(x3, y3, 'ko')
    plt.scatter(x3, y3, c=z3, cmap='cool')
    #plt.scatter(x3, y3, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('request rate per veh per min')
    plt.ylabel('idle distance')
    #plt.legend()
    plt.gca().set_ylim([0, 30])
    plt.colorbar(label='demand rejection rate')

    #z = np.polyfit(x3, y3, 1)
    #p = np.poly1d(z)
    #plt.plot(x3, p(x3), "k-")

####################################################
#### Occupied distance vs request rate #####
####################################################

def occup_vs_request_rate():
    columns = ["glob_request_rate_per_veh_per_min", "glob_occup_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x4 = df["glob_request_rate_per_veh_per_min"]  # {"idle_dist": glob_idle_dist_avg}
    y4 = df["glob_occup_dist_avg"]
    z4 = df["glob_demand_rejection_rate"]

    #x4 = glob_request_rate_per_veh_per_min
    #y4 = glob_occup_dist_avg

    plt.figure(6)
    plt.title('Occupied distance vs request rate per veh per min')
    #plt.plot(x4, y4, 'ko')
    plt.scatter(x4, y4, c=z4, cmap='cool')
    #plt.scatter(x4, y4, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('request rate per veh per min')
    plt.ylabel('occupied distance')
    #plt.legend()
    #plt.gca().set_ylim([20, 50])
    plt.colorbar(orientation="horizontal", label="demand rejection rate")

    z = np.polyfit(x4, y4, 1)
    p = np.poly1d(z)
    plt.plot(x4, p(x4), "k-")

#############################################
#### Occupied distance vs vehicle density #####
#############################################

def occup_vs_veh_dens():
    columns = ["glob_veh_vacant_dens", "glob_occup_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x5 = df["glob_veh_vacant_dens"]  # {"idle_dist": glob_idle_dist_avg}
    y5 = df["glob_occup_dist_avg"]
    z5 = df["glob_demand_rejection_rate"]

    #x = glob_veh_vacant_dens
    #y = glob_occup_dist_avg

    plt.figure(7)
    plt.title('Occupied distance vs vacant vehicle density')
    #plt.plot(x, y, 'ko')
    #plt.scatter(x5, y5, c=z5, cmap='cool')
    plt.scatter(x5, y5, c=z5)
    #plt.scatter(x, y, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('vehicle density')
    plt.ylabel('occupied distance')
    #plt.gca().set_ylim([20, 50])
    #plt.legend()
    #plt.colorbar(orientation="horizontal", label="demand rejection rate")

    #z = np.polyfit(x5, y5, 1)
    #p = np.poly1d(z)
    #plt.plot(x5, p(x5), "k-")


#############################################
#### Idle distance vs vehicle density #####
#############################################

def idle_vs_veh_dens():
    columns = ["glob_veh_vacant_dens", "glob_idle_dist_avg", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x6 = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y6 = df["glob_veh_vacant_dens"]
    #z6 = df["glob_demand_rejection_rate"]

    x6 = [k * 200/1000 for k in x6]
    y6 = [k * 37.5 for k in y6]

    #x = glob_veh_density
    #y = glob_idle_dist_avg

    plt.figure(8)
    plt.title('Idle distance vs vacant vehicle density')
    #plt.plot(x, y, 'ko')
    #plt.scatter(x6, y6, c=z6, cmap='cool')
    plt.scatter(x6, y6)
    #plt.scatter(x, y, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('Idle distance (km)', fontsize=FONT_SIZE_LAB)
    plt.ylabel('Vacant vehicle density per sq km', fontsize=FONT_SIZE_LAB)      # per min
    plt.xticks(fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.legend()
    #plt.colorbar(label='demand rejection rate')
    #plt.gca().set_ylim([0, 30])

    #z = np.polyfit(x, y, 1)
    #p = np.poly1d(z)
    #plt.plot(x, p(x), "k-")

def dem_vs_veh_dens():
    columns = ["glob_veh_vacant_dens", "glob_demand_count_norm"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x6 = df["glob_veh_vacant_dens"]  # {"idle_dist": glob_idle_dist_avg}
    y6 = df["glob_demand_count_norm"]
    #z6 = df["glob_demand_rejection_rate"]

    #x = glob_veh_density
    #y = glob_idle_dist_avg

    plt.figure(14)
    plt.title('Demand density vs vacant vehicle density')
    #plt.plot(x, y, 'ko')
    #plt.scatter(x6, y6, c=z6, cmap='cool')
    plt.scatter(x6, y6)
    #plt.scatter(x, y, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('vacant vehicle density')
    plt.ylabel('demand density')
    #plt.legend()
    #plt.colorbar(label='demand rejection rate')
    #plt.gca().set_ylim([0, 30])

    #z = np.polyfit(x, y, 1)
    #p = np.poly1d(z)
    #plt.plot(x, p(x), "k-")


#######################################################
#### Demand rejection rate vs Demand request rate #####
#######################################################

def rej_rate_vs_req_rate():
    columns = ["glob_demand_rejected_per_veh", "glob_request_rate_per_veh", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x7 = df["glob_demand_rejected_per_veh"]  # {"idle_dist": glob_idle_dist_avg}
    y7 = df["glob_request_rate_per_veh"]
    z7 = df["glob_demand_rejection_rate"]

    #x = glob_demand_rejected_per_veh
    #y = glob_request_rate_per_veh

    plt.figure(9)
    plt.title('Demand rejections vs Demand requests')
    #plt.plot(x, y, 'ko')
    plt.scatter(x7, y7, c=z7, cmap='cool')
    #plt.scatter(x, y, c=glob_non_moving_time_rate, cmap='cool')
    plt.xlabel('demand rejections per veh')
    plt.ylabel('demand requests per veh')
    #plt.legend()
    plt.colorbar(label='demand rejection rate, %')
    #plt.gca().set_ylim([0, 25])

    #z = np.polyfit(x, y, 1)
    #p = np.poly1d(z)
    #plt.plot(x, p(x), "k-")

####################
#### 3D graphs #####
####################

def three_d_1():
    #columns = ["glob_idle_dist_avg", "glob_veh_vacant_dens", "glob_demand_count_norm", "glob_demand_rejection_rate"]
    columns = ["glob_idle_dist_avg", "glob_veh_vacant_dens", "glob_demand_count_norm"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y = df["glob_veh_vacant_dens"]
    z = df["glob_demand_count_norm"]
    #u = df["glob_demand_rejection_rate"]
    x = [k * 200/1000 for k in x]
    #x = glob_idle_dist_avg
    #y = glob_veh_vacant_dens
    #z = glob_demand_count_norm
    y = [k * 37.5 for k in y]
    z = [k * 37.5 for k in z]

    fig = plt.figure(10)
    ax = fig.add_subplot(projection='3d')
    plt.title('Idle distance vs Vacant vehicle density vs Demand density')
    #scatter_plot = ax.scatter(x, y, z, c=u, cmap='cool')
    ax.scatter(x, y, z)
    ax.set_xlabel(' \nIdle distance (km)', fontsize=FONT_SIZE_LAB)
    ax.set_ylabel(' \nVacant vehicle\ndensity per sq km', fontsize=FONT_SIZE_LAB)      # per min
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Demand density\nper sq km\n', fontsize=FONT_SIZE_LAB, rotation=90)       # per min
    plt.tick_params(axis='both', labelsize=12)
    #plt.colorbar(scatter_plot, label='demand rejection rate')


def three_d_2():
    columns = ["glob_idle_dist_avg", "glob_veh_vacant_dens", "glob_request_rate_per_veh_per_min", "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y = df["glob_veh_vacant_dens"]
    z = df["glob_request_rate_per_veh_per_min"]
    u = df["glob_demand_rejection_rate"]

    #x = glob_idle_dist_avg
    #y = glob_veh_vacant_dens
    #z = glob_request_rate_per_veh_per_min

    fig = plt.figure(11)
    ax = fig.add_subplot(projection='3d')
    plt.title('Idle distance vs Vacant vehicle density vs Request rate per veh per min')
    scatter_plot = ax.scatter(x, y, z, c=u, cmap='cool')
    ax.set_xlabel('idle distance')
    ax.set_ylabel('vehicle density')
    ax.set_zlabel('request rate')
    plt.colorbar(scatter_plot, label='demand rejection rate')


def three_d_3():
    columns = ["glob_idle_dist_avg", "glob_request_rate_per_veh_per_min", "glob_demand_count_norm",
               "glob_demand_rejection_rate"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x = df["glob_idle_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y = df["glob_request_rate_per_veh_per_min"]
    z = df["glob_demand_count_norm"]
    u = df["glob_demand_rejection_rate"]

    #x = glob_idle_dist_avg
    #y = glob_request_rate_per_veh_per_min
    #z = glob_demand_count_norm

    fig = plt.figure(12)
    ax = fig.add_subplot(projection='3d')
    plt.title('Idle distance vs Request rate vs Demand density')
    scatter_plot = ax.scatter(x, y, z, c=u, cmap='cool')
    ax.set_xlabel('idle distance')
    ax.set_ylabel('request rate')
    ax.set_zlabel('demand density')
    plt.colorbar(scatter_plot, label='demand rejection rate')

def occup_three_d_1():

    # columns = ["glob_idle_dist_avg", "glob_veh_vacant_dens", "glob_demand_count_norm", "glob_demand_rejection_rate"]
    columns = ["glob_occup_dist_avg", "glob_veh_vacant_dens", "glob_demand_count_norm"]
    df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/micro_sim_data.csv", usecols=columns)
    x = df["glob_occup_dist_avg"]  # {"idle_dist": glob_idle_dist_avg}
    y = df["glob_veh_vacant_dens"]
    z = df["glob_demand_count_norm"]

    # u = df["glob_demand_rejection_rate"]
    x = [k * 200 / 1000 for k in x]
    # x = glob_idle_dist_avg
    # y = glob_veh_vacant_dens
    # z = glob_demand_count_norm
    y = [k * 37.5 for k in y]
    z = [k * 37.5 for k in z]

    fig = plt.figure(13)
    ax = fig.add_subplot(projection='3d')
    plt.title('Service distance vs Vacant vehicle density vs Demand density')
    # scatter_plot = ax.scatter(x, y, z, c=u, cmap='cool')
    ax.scatter(x, y, z)
    ax.set_xlabel(' \nService distance (km)', fontsize=FONT_SIZE_LAB)
    ax.set_ylabel(' \nVacant vehicle\ndensity per sq km', fontsize=FONT_SIZE_LAB)  # per min
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Demand density\nper sq km\n', fontsize=FONT_SIZE_LAB, rotation=90)  # per min
    plt.tick_params(axis='both', labelsize=12)
    # plt.colorbar(scatter_plot, label='demand rejection rate')

boxplot1()
boxplot2()
boxplot3()
#occup_vs_dem_dens()
idle_vs_dem_dens()
#idle_vs_request_rate()
#occup_vs_request_rate()
idle_vs_veh_dens()
#occup_vs_veh_dens()
#rej_rate_vs_req_rate()
#dem_vs_veh_dens()
three_d_1()
#three_d_2()
#three_d_3()
occup_three_d_1()
#hist1()
#hist2()
hist3()
hist4()

plt.show()
