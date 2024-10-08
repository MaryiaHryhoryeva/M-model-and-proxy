from micro_sim.main_queued import *

import matplotlib.pyplot as plt
import numpy as np


########################################
#### Average occupied distance run #####
########################################
plt.figure(1)
data = [uber_occup_100, lyft_occup_50]
plt.boxplot(data)
plt.gca().xaxis.set_ticklabels(['100 cars', '50 cars'])
plt.title('Average occupied distance run')

####################################
#### Average idle distance run #####
####################################
plt.figure(2)
data2 = [uber_idle_100, lyft_idle_50]
plt.boxplot(data2)
plt.gca().xaxis.set_ticklabels(['100 cars', '50 cars'])
plt.title('Average idle distance run')


#############################################
#### Occupied distance vs demand density #####
#############################################
x11 = []
x12 = []
for i in uber_demand_count:
    x11.append(i / (max_x * max_y))

y11 = uber_occup_100

for i in lyft_demand_count:
    x12.append(i / (max_x * max_y))

y12 = lyft_occup_50

plt.figure(3)
plt.title('Occupied distance vs demand density')
plt.plot(x11, y11, 'ko', label='uber, 100 vehicles')
plt.plot(x12, y12, 'bo', label='lyft, 50 vehicles')
plt.xlabel('demand density')
plt.ylabel('occupied distance')
plt.legend()

z = np.polyfit(x11, y11, 1)
p = np.poly1d(z)
plt.plot(x11, p(x11), "k-")

z = np.polyfit(x12, y12, 1)
p = np.poly1d(z)
plt.plot(x12, p(x12), "b-")

#############################################
#### Idle distance vs demand density #####
#############################################

x21 = x11
x22 = x12
y21 = uber_idle_100
y22 = lyft_idle_50

plt.figure(4)
plt.title('Idle distance vs demand density')
plt.plot(x21, y21, 'ko', label='uber, 100 vehicles')
plt.plot(x22, y22, 'bo', label='lyft, 50 vehicles')
plt.xlabel('demand density')
plt.ylabel('idle distance')
plt.legend()

z = np.polyfit(x21, y21, 1)
p = np.poly1d(z)
plt.plot(x21, p(x21), "k-")

z = np.polyfit(x22, y22, 1)
p = np.poly1d(z)
plt.plot(x22, p(x22), "b-")

####################################################
#### Idle distance vs proportion (demand/fleet) #####
####################################################

x31 = []
x32 = []
y31 = uber_idle_100
y32 = lyft_idle_50
for i in uber_demand_count:
    x31.append(i / num_veh['uber'])

y11 = uber_occup_100

for i in lyft_demand_count:
    x32.append(i / num_veh['lyft'])

plt.figure(5)
plt.title('Idle distance vs proportion (demand/fleet)')
plt.plot(x31, y31, 'ko', label='uber, 100 vehicles')
plt.plot(x32, y32, 'bo', label='lyft, 50 vehicles')
plt.xlabel('proportion (demand/fleet)')
plt.ylabel('idle distance')
plt.legend()

z = np.polyfit(x31, y31, 1)
p = np.poly1d(z)
plt.plot(x31, p(x31), "k-")

z = np.polyfit(x32, y32, 1)
p = np.poly1d(z)
plt.plot(x32, p(x32), "b-")

plt.show()
