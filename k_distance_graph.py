import matplotlib.pyplot as plt
import numpy as np
import math

'''
self-defined functions
'''
from dbscan import loadDataSet

'''
calculate k distance graph
'''

def calculate_kn_distance(X:list, k:int) ->list :
    kn_distance = []
    # if k < 4:
    #     return kn_distance
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(math.sqrt(((X[i][0] - X[j][0]) ** 2) + ((X[i][1] - X[j][1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k - 1])
    kn_distance.sort()  # 所有点的distance排序

    return kn_distance

def draw_figure(X:list, M:list) ->None :
    # original points
    for index in range(len(X)):
        plt.scatter(index, M[index])

    # linear fit
    x = np.linspace(0, len(X) - 1, len(X), dtype = int)
    fit_ret = np.polyfit(x, M, 3)
    fit_poly = np.poly1d(fit_ret)
    # plt.plot(x, fit_poly(x), 'b+', linestyle='solid')
    plt.plot(x, fit_poly(x), 'b+')

    plt.xlabel("targets")
    plt.ylabel("k-distance")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def main():
    data_set = loadDataSet("./data/ped_percep_10.txt")
    distance_list = calculate_kn_distance(data_set, 3)   # default and recommended k-value = 4
    draw_figure(data_set, distance_list)

if __name__ == '__main__':
    main()
