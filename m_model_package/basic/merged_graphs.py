import matplotlib.pyplot as plt
import csv

distance = []
v = []
with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/one_comp/length_1_comp.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        distance.append(row)

with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/two_comp/length_2_comp.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        distance.append(row)

with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/one_comp/speed_1_comp.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        v.append(row)

with open('/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/two_comp/speed_2_comp.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        v.append(row)

"""x = m1.l_RHI
y = m2.l_RHI['uber']
z = m2.l_RHI['lyft']"""

"""print(x)
v1 = m1.v
v2 = m2.v"""
distance1 = distance[0]
distance2 = distance[1]
distance3 = distance[2]
distance1 = [float(i) for i in distance1]
distance2 = [float(i) for i in distance2]
distance3 = [float(i) for i in distance3]
v1 = v[0]
v2 = v[1]
v1 = [float(i) for i in v1]
v2 = [float(i) for i in v2]

#print(distance1)
#print(distance2)
#print(distance3)

#plt.subplots(1, 1, constrained_layout=True)
#plt.subplot(1, 1, 1)
#plt.plot(distance1, 'b', label='1 comp')
#plt.plot(distance2, 'k', label='2 comp - uber')
#plt.plot(distance3, 'grey', label='2 comp - lyft')
plt.title('avg trip length of idle moving')
plt.plot(distance1, 'b', label='1 comp')
plt.plot(distance2, 'k', label='2 comp - uber')
plt.plot(distance3, 'grey', label='2 comp - lyft')
plt.legend(loc="upper right")

"""plt.subplot(2, 1, 2)
plt.plot(v1, 'b')
plt.plot(v2, 'k')"""


plt.show()

plt.title('speed')
plt.plot(v1, 'b', label='1 comp')
plt.plot(v2, 'k', label='2 comp')

plt.legend(loc="upper right")

"""plt.subplot(2, 1, 2)
plt.plot(v1, 'b')
plt.plot(v2, 'k')"""


plt.show()
