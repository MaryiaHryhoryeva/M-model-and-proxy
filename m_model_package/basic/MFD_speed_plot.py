import matplotlib.pyplot as plt
import numpy as np

def calculate_V_MFD(n):  # to calculate the speed. function is taken from Louis' MFD of Lyon
    V = 0
    if n < 18000:
        V = 11.5 - n * 6 / 18000
    elif n < 55000:
        V = 11.5 - 6 - (n - 18000) * 4.5 / (55000 - 18000)
    elif n < 80000:
        V = 11.5 - 6 - 4.5 - (n - 55000) * 1 / (80000 - 55000)

    return max(V, 0.001)

n = []
v = []
for i in range(1, 79999):
    n.append(i)

for i in range(0, len(n)):
    v.append(calculate_V_MFD(n[i]))

FONT_SIZE_LAB = 14
FONT_SIZE_AXI = 12
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
plt.plot(v, 'k')
plt.xlabel("Accumulation ($10^4$ veh)", fontsize=FONT_SIZE_LAB)
plt.ylabel("Network speed (m/s)", fontsize=FONT_SIZE_LAB)
plt.xticks(np.linspace(0, 80000, len(labels)), labels, fontsize=FONT_SIZE_AXI)
plt.yticks(fontsize=FONT_SIZE_AXI)
#plt.title('n_RHI - accum. of idle moving')
plt.show()
