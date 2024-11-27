import sys
import math
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from traj_search import coord2vert, cal_distance


def inverseweight(dist, num=1.0, const=0.1):
    return num / (dist + const)


def gaussian(dist, sigma=10.0):
    return math.e ** (- dist ** 2 / (2 * sigma ** 2))


def subtractweight(dist, const=2.0):
    if dist > const:
        return 0.001
    else:
        return const - dist


def weighted_knn(kdtree, test_point, target, k=25,
                 weight_fun=inverseweight):
    nearest_dist, nearest_ind = kdtree.query(test_point, k=k)
    avg = 0.0
    totalweight = 0.0
    for i in range(k):
        dist = nearest_dist[0][i]
        idx = nearest_ind[0][i]
        weight = weight_fun(dist)
        avg += weight * target[idx]
        totalweight += weight
    avg = round(avg / totalweight)
    return avg


def read_log(logfile, dx=100):
    rssi = []
    loc = []
    for line in open(logfile).readlines():
        line = line.split(" ")[2:]
        # user = line[1]
        t = float(line[6])
        if t > 50:
            break
        posi = coord2vert(float(line[8][1:-1]), float(line[9][:-1]), dx, dx)
        r = abs(float(line[-1].split('\n')[0]))
        loc.append(posi)
        rssi.append(r)
    df = pd.DataFrame({'rssi':rssi,'posi':loc})
    return df

def fingerprint_wknn(logpath, dx=100):
    df = read_log(logpath, dx)
    data = df['rssi'].to_numpy()
    label_ = df['posi'].to_numpy()
    #normalize label
    x_max = np.max([i[0] for i in label_])
    label = [i[0] * x_max + i[1] for i in label_]
    t0 = datetime.now().timestamp()
    data = StandardScaler().fit_transform(data.reshape((-1,1)))
    tree = KDTree(data)
    predict = []
    for i in range(len(data)):
        p = weighted_knn(tree, data[i].reshape((-1,1)), label)
        predict.append(p)
    # de-normalize label
    predict = [(i//x_max, i % x_max) for i in predict]
    res = []

    for i in range(0, len(predict)):
        res.append(cal_distance(predict[i][0], predict[i][1], label_[i][0], label_[i][1]))
    print(dx, datetime.now().timestamp()-t0,  np.mean(res),  np.median(res))


if __name__=="__main__":
    logpath = sys.argv[1]
    print(logpath)
    for dx in range(100,110,20):
        fingerprint_wknn(logpath, dx)