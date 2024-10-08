import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_SIZE_HALF = (6, 3)
FONT_SIZE_LAB = 14
FONT_SIZE_LEG = 12
FONT_SIZE_AXI = 12

df = pd.read_csv("/Users/maryia/PycharmProjects/SimulationTrial/m_model_package/basic/strategy_comparison/speed.csv")
data1 = df["v_comp"]
data2 = df["v_part_coop"]
data3 = df["v_coop"]

labels = ["0", "1", "2", "3", "4"]
plt.subplots(1, 1, constrained_layout=True)

plt.subplot(1, 1, 1)
data1 = [val for val in data1 for _ in (0, 1)]
data2 = [val for val in data2 for _ in (0, 1)]
data3 = [val for val in data3 for _ in (0, 1)]

plt.plot(data1, 'magenta', label='Competition')
plt.plot(data2, 'dodgerblue', label='Coopetition')
plt.plot(data3, 'yellowgreen', label='Cooperation')


plt.ylabel("Network speed (m/s)", fontsize=FONT_SIZE_LAB)
plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
plt.xticks(np.linspace(0, 14400, len(labels)), labels, fontsize=FONT_SIZE_AXI)
plt.yticks(fontsize=FONT_SIZE_AXI)
plt.title('(a) Arriving demand requests (same for both companies)')
plt.legend(loc="lower right")
plt.show()