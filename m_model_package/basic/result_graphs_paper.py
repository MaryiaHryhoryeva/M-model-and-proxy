import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_SIZE_HALF = (6, 3)
FONT_SIZE_LAB = 16
FONT_SIZE_LEG = 14
FONT_SIZE_AXI = 16
FONT_SIZE_TTL = 16

plt.rcParams['figure.figsize'] = [7, 4]

def speed():

    """### Changes in fleet size
    labels = ["-20%", "-10%", "0%", "+10%"]
    v_coop = [5.0146, 5.0029, 5.0076, 5.0201]
    v_comp = [5.0115, 5.0063, 5.0030, 5.0074]
    v_part_coop = [5.0093, 5.0016, 4.9996, 5.0038]"""

    """### Changes in the number of companies
    labels = ["2", "3", "4"]
    v_coop = [5.0076, 5.0076, 5.0076]
    v_comp = [5.003, 5.0017, 5.0079]
    v_part_coop = [4.9996, 5.0005, 5.004]"""

    """### Changes in market share
    labels = ["+0%", "+30%", "+50%"]
    v_coop = [5.0076, 5.0087, 5.0268]
    v_comp = [5.003, 5.0032, 4.9984]
    v_part_coop = [4.9996, 5.0003, 4.9914]"""

    """### Changes in fleet share
    labels = ["255/345", "270/330", "285/315", "300/300"]
    v_coop = [5.0076, 5.0076, 5.0076, 5.0076]
    v_comp = [5.0078, 5.003, 5.002, 5.0026]
    v_part_coop = [5.0004, 4.9996, 4.9997, 5.0007]"""

    """### Changes in demand share
    labels = ["\u00B1 0%", "\u00B1 25%", "\u00B1 50%"]
    v_coop = [5.0076, 5.0076, 5.0076]
    v_comp = [5.003, 5.0048, 5.0176]
    v_part_coop = [4.9996, 4.9993, 4.9997]"""

    ### Changes in the number of companies for 15-17% demand share
    labels = ["2", "3", "4"]
    v_coop = [5.23, 5.23, 5.23]
    v_comp = [5.1813, 5.2101, 5.2054]
    v_part_coop = [5.1934, 5.2048, 5.1815]

    plt.plot(v_coop, '-o', color='green', label='Cooperation')
    plt.plot(v_comp, '-o', color='blue', label='Competition')
    plt.plot(v_part_coop, '-o', color='darkorange', label='Coopetition')

    plt.ylabel("Average network speed (m/s)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("Changes in fleet size (%)", fontsize=FONT_SIZE_LAB)
    plt.xlabel("Number of companies", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("RH market share increase (%)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("# of vehicles: Company1/Company2", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("Demand share between the companies", fontsize=FONT_SIZE_LAB)
    #plt.xticks(np.linspace(0, 3, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.xticks(np.linspace(0, 2, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    #plt.yticks(np.arange(4.99, 5.03, 0.01), fontsize=FONT_SIZE_AXI)
    #plt.title('(a) Arriving demand requests (same for both companies)')
    plt.legend(loc="upper left", fontsize=FONT_SIZE_LEG)
    plt.title('Network speed', fontsize=FONT_SIZE_TTL)
    plt.figtext(.55, .82, "# of RH vehicles = 600", fontsize=FONT_SIZE_LEG)
    plt.figtext(.55, .77, "RH demand share = 2%", fontsize=FONT_SIZE_LEG)
    plt.show()
    #plt.savefig('/Users/maryia/Dropbox (LICIT_LAB)/Mac/Documents/PhD work/First work/TRB paper/Poster/Fig1.pdf', bbox_inches='tight')


def canceled_demand():

    """### Changes in fleet size
    labels = ["-20%", "-10%", "0%", "+10%"]
    v_coop = [28.67, 9.31, 1.82, 0]
    v_comp = [30.29, 15.57, 5.06, 1.98]
    v_part_coop = [30.67, 11.35, 4.52, 0.15]"""

    """### Changes in the number of companies
    labels = ["2", "3", "4"]
    v_coop = [1.82, 1.82, 1.82]
    v_comp = [5.06, 4.66, 8.18]
    v_part_coop = [4.52, 4.15, 4.96]"""

    """### Changes in market share
    labels = ["+0%", "+30%", "+50%"]
    v_coop = [1.82, 0, 0]
    v_comp = [5.06, 4.37, 5.05]
    v_part_coop = [4.52, 3.21, 4.21]"""

    """### Changes in fleet share
    labels = ["255/345", "270/330", "285/315", "300/300"]
    v_coop = [1.82, 1.82, 1.82, 1.82]
    v_comp = [9.6, 5.06, 3.62, 3.32]
    v_part_coop = [5.12, 4.52, 3.51, 3.30]"""

    """### Changes in demand share
    labels = ["\u00B1 0%", "\u00B1 25%", "\u00B1 50%"]
    v_coop = [1.82, 1.82, 1.82]
    v_comp = [5.06, 8.78, 34.55]
    v_part_coop = [4.52, 5.61, 6.21]"""

    ### Changes in the number of companies for 15-17% demand share
    labels = ["2", "3", "4"]
    v_coop = [0, 0, 0]
    v_comp = [2.12, 1.17, 4.13]
    v_part_coop = [0, 0, 0.07]

    plt.plot(v_coop, '-o', color='green', label='Cooperation')
    plt.plot(v_comp, '-o', color='blue', label='Competition')
    plt.plot(v_part_coop, '-o', color='darkorange', label='Coopetition')

    plt.ylabel("Canceled demand (%)", fontsize=FONT_SIZE_LAB)
    # plt.xlabel("Changes in fleet size (%)", fontsize=FONT_SIZE_LAB)
    plt.xlabel("Number of companies", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("RH market share increase (%)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("# of vehicles: Company1/Company2", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("Demand share between the companies", fontsize=FONT_SIZE_LAB)
    # plt.xticks(np.linspace(0, 3, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.xticks(np.linspace(0, 2, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    # plt.title('(a) Arriving demand requests (same for both companies)')
    #plt.legend(loc="upper left")
    plt.title('Canceled demand', fontsize=FONT_SIZE_TTL)
    plt.show()
    #plt.savefig('/Users/maryia/Dropbox (LICIT_LAB)/Mac/Documents/PhD work/First work/TRB paper/Poster/Fig2.pdf',
    #            bbox_inches='tight')


def pass_wait_time():

    """### Changes in fleet size
    labels = ["-20%", "-10%", "0%", "+10%"]
    v_coop = [89.95, 85.78, 9.16, 0]
    v_comp = [122.13, 59.01, 43.39, 8.3]
    v_part_coop = [126.65, 60.88, 45.63, 9.13]"""

    """### Changes in the number of companies
    labels = ["2", "3", "4"]
    v_coop = [9.16, 9.16, 9.16]
    v_comp = [43.39, 40.32, 60.29]
    v_part_coop = [45.63, 43.30, 66.04]"""

    """### Changes in market share
    labels = ["+0%", "+30%", "+50%"]
    v_coop = [9.16, 2.37, 0]
    v_comp = [43.39, 36.39, 31.73]
    v_part_coop = [45.63, 36.72, 34.45]"""

    """### Changes in fleet share
    labels = ["255/345", "270/330", "285/315", "300/300"]
    v_coop = [9.16, 9.16, 9.16, 9.16]
    v_comp = [44.94, 43.39, 19.95, 16.77]
    v_part_coop = [50.81, 45.63, 19.72, 17.72]"""

    """### Changes in demand share
    labels = ["\u00B1 0%", "\u00B1 25%", "\u00B1 50%"]
    v_coop = [9.16, 9.16, 9.16]
    v_comp = [43.39, 33.88, 33.5]
    v_part_coop = [45.63, 39.96, 34.57]"""

    ### Changes in the number of companies for 15-17% demand share
    labels = ["2", "3", "4"]
    v_coop = [0, 0, 0]
    v_comp = [6.35, 3.24, 15.50]
    v_part_coop = [3.53, 3.34, 16.25]

    plt.plot(v_coop, '-o', color='green', label='Cooperation')
    plt.plot(v_comp, '-o', color='blue', label='Competition')
    plt.plot(v_part_coop, '-o', color='darkorange', label='Coopetition')

    plt.ylabel("Average user waiting time (s)", fontsize=FONT_SIZE_LAB)
    # plt.xlabel("Changes in fleet size (%)", fontsize=FONT_SIZE_LAB)
    plt.xlabel("Number of companies", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("RH market share increase (%)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("# of vehicles: Company1/Company2", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("Demand share between the companies", fontsize=FONT_SIZE_LAB)
    # plt.xticks(np.linspace(0, 3, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.xticks(np.linspace(0, 2, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    # plt.title('(a) Arriving demand requests (same for both companies)')
    #plt.legend(loc="upper left")
    plt.title('User waiting time', fontsize=FONT_SIZE_TTL)
    plt.show()
    #plt.savefig('/Users/maryia/Dropbox (LICIT_LAB)/Mac/Documents/PhD work/First work/TRB paper/Poster/Fig3.pdf',
    #            bbox_inches='tight')


def pass_wait_time_comp():
    """ ### Changes in fleet size
    labels = ["-20%", "-10%", "0%", "+10%"]
    v_coop = [89.95, 85.78, 9.16, 0]
    v_comp_c1 = [138.93, 115.53, 106.01, 17.64]
    v_part_coop_c1 = [140.45, 116.19, 104.43, 19.26]
    v_comp_c2 = [106.73, 20.18, 1.12, 0]
    v_part_coop_c2 = [114.67, 21.66, 4.73, 0]"""

    """### Changes in market share
    labels = ["+0%", "+30%", "+50%"]
    v_coop = [9.16, 2.37, 0]
    v_comp_c1 = [106.01, 102.33, 95.82]
    v_part_coop_c1 = [104.43, 101.04, 95.24]
    v_comp_c2 = [1.12, 0.15, 0]
    v_part_coop_c2 = [4.73, 1.56, 2.72]"""

    """### Changes in fleet share
    labels = ["255/345", "270/330", "285/315", "300/300"]
    v_coop = [9.16, 9.16, 9.16, 9.16]
    v_comp_c1 = [112.84, 106.01, 29.51, 17.96]
    v_part_coop_c1 = [114.3, 104.43, 28.27, 17.84]
    v_comp_c2 = [0, 1.12, 11.26, 15.56]
    v_part_coop_c2 = [5.52, 4.73, 12.13, 17.6]"""

    ### Changes in demand share
    labels = ["\u00B1 0%", "\u00B1 25%", "\u00B1 50%"]
    v_coop = [9.16, 9.16, 9.16]
    v_comp_c1 = [106.01, 0, 0]
    v_part_coop_c1 = [104.43, 2.29, 0.08]
    v_comp_c2 = [1.12, 100.82, 120.47]
    v_part_coop_c2 = [4.73, 104.41, 121.42]

    plt.plot(v_coop, '-o', color='silver', label='Cooperation')
    plt.plot(v_comp_c1, '-o', color='dodgerblue', label='Competition - Company 1')
    plt.plot(v_part_coop_c1, '-o', color='skyblue', label='Coopetition - Company 1')
    plt.plot(v_comp_c2, '-o', color='orangered', label='Competition - Company 2')
    plt.plot(v_part_coop_c2, '-o', color='orange', label='Coopetition - Company 2')

    plt.ylabel("Average user waiting time (s)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("Changes in fleet size (%)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("RH market share increase (%)", fontsize=FONT_SIZE_LAB)
    #plt.xlabel("# of vehicles: Company1/Company2", fontsize=FONT_SIZE_LAB)
    plt.xlabel("Demand share between the companies", fontsize=FONT_SIZE_LAB)
    #plt.xticks(np.linspace(0, 3, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.xticks(np.linspace(0, 2, len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    # plt.title('(a) Arriving demand requests (same for both companies)')
    plt.legend(loc="center right")
    plt.show()


speed()
canceled_demand()
pass_wait_time()
#pass_wait_time_comp()

