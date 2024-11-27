import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

class Pos_signal:
    x: float
    y: float
    rssi: float
    l_rssi: list


def vert_distance(vert0, vert1):
    return np.sqrt(np.sum(np.square([vert0[0]-vert1[0], vert0[1]-vert1[1]])))

def init_random_map(n):
    r_map = {}
    for i in range(n):
        for j in range(n):
           r_map[(i, j)] = 100/(np.sqrt(i**2 + j**2)+1) + 1*np.random.random()
    return r_map

def coord2vert(x, y, dx, dy):
    return (int(x/dx), int(y/dy))

def init_map_from_omnet(file_path, dx=10, dy=10):
    """
    todo change into vertices, add distribution
    :param file_path:
    :param x:
    :param y:
    :param dx:
    :param dy:
    :return:
    """
    r_map = {}
    lines = open(file_path, 'r').readlines()
    for l in lines:
        l = l.split(" ")[2:]
        if float(l[6]) > 50:
            break
        x = float(l[8].split("(")[1].split(",")[0])
        y = float(l[9].split(",")[0])
        vert = coord2vert(x, y, dx, dy)
        rssi = abs(float(l[-1].split('\n')[0]))
        if vert not in r_map:
            r_map[vert] = []
        r_map[vert].append(rssi)
    # calculate distribution
    r_map_dist = {}
    for k, v in r_map.items():
        r_map_dist[k] = norm.fit(v)
    return r_map_dist


def setup_tree(r_map:dict):
    """
    Converting a map to tree for search later
    Root of the tree is each vertices
    :param map:
    :return:
    """
    rssi_map = {}  # contains a list of coordinates that has same value
    for k, v in r_map.items():
        if v in rssi_map:
            rssi_map[v].append(k)
        else:
            rssi_map[v] = []
            rssi_map[v].append(k)
    return rssi_map

def search_rssi_by_likelyhood(rssi: float, rssi_map: dict, p, top_k):
    poential_list = {}
    for k, v in rssi_map.items():  # vertices: (mu, sigma)
        poential_list[k] = rssi_liklyhood(rssi, v, p)
    to_keep = sorted(poential_list.keys(), key=lambda x: poential_list[x], reverse=True)[:top_k]
    return {i:poential_list[i] for i in to_keep}

def rssi_liklyhood(rssi, rssi_dist, p):
    """
    The probability that rssi is from a certain distribution
    :param rssi: the signal
    :param rssi_dist: the u and sigma of distribution
    :return:
    """
    if rssi_dist[1] == 0:  # sigma should not be zero
        return 0
    x = (rssi-rssi_dist[0])/rssi_dist[1]
    x = norm.cdf(x+p) - norm.cdf(x-p)
    return x

def vertices_transfer_prob(m_user_traj: dict, time_gap):
    """

    :param m_user_traj: {user1: [(time, (vx,vy), rssi), (time, (vx,vy), rssi), ...], user2:[]}
    :param time_gap:
    :return:
    """
    m_v2v_prob = {}  # ()
    for k, v in m_user_traj.items():
        if len(v) < 2:
            continue
        start_v = v[0][1]  # (vx, vy)
        start_t = v[0][0]  # time
        for i in range(1, len(v)):
            if v[i][0] - start_t >= time_gap:
                transfer = (start_v, v[i][1])
                if transfer not in m_v2v_prob:
                    m_v2v_prob[transfer] = 1
                else:
                    m_v2v_prob[transfer] += 1
                start_v = v[i][1]
                start_t = v[i][0]
    t = sum([i for i in m_v2v_prob.values()])
    for k, v in m_v2v_prob.items():
        m_v2v_prob[k] = v/t
    return m_v2v_prob


def search_tree_node(rssi, rssi_map, tolerance=0.1):
    """
    Return a node (coordinate) that is strength rssi.
    :param rssi:
    :return:
    """
    if tolerance == 0:
        if rssi in rssi_map:
            return rssi_map[rssi]
        else:
            return ''
    else:
        l_nodes = []
        min_rssi = rssi*(1-tolerance)
        max_rssi = rssi*(1+tolerance)
        for r in rssi_map:
            if min_rssi <= r <= max_rssi:  # r is below zero
                l_nodes.extend(rssi_map[r])
        return l_nodes if len(l_nodes) > 0 else ''


def cal_distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2-x1)+np.square(y2-y1))


def construct_trajectory(l_l_nodes, max_distance=5, max_gap=1, total_gaps=4):
    """
    Reconstruct the trajectory from the list of nodes
    :param l_l_nodes: the list of nodes to construct
    :param max_distance: the max distance between nodes to construct
    :param max_gap: the gap allowance between two points on trajectory
    :param total_gaps: the max number of gaps allowed in trajectory
    :return:
    """
    l_trajectory = []
    # init
    for k in range(len(l_l_nodes)):
        if l_l_nodes[k] != '':
            for n in l_l_nodes[k]:
                l_trajectory.append([n])
            break
    # the l_nodes left
    for k in range(k+1, len(l_l_nodes)):  # per rssi
        tmp_taj = []
        for n in l_l_nodes[k]:  # per node
            for potenial_traj in l_trajectory:
                if n == '':  # no available node
                    tmp_taj.append(potenial_traj+[''])  # add the empty node
                elif n != '':
                    if potenial_traj[-1] != '':
                        if cal_distance(n[0], n[1], potenial_traj[-1][0], potenial_traj[-1][1]) <= max_distance:
                            tmp_taj.append(potenial_traj+[n])
                        else:
                            tmp_taj.append(potenial_traj+[''])  # todo, do we need to add padding
                    elif potenial_traj[-1] == '':
                        # no immeidate adjacent node found, checking with gap
                        if len([i for i in l_trajectory[0] if i == '']) >= total_gaps:
                            continue
                        # else
                        for g in range(2, max_gap + 2):  # -1 is the last one
                            if (potenial_traj[-g] != '' and
                                    cal_distance(n[0], n[1], potenial_traj[-g][0], potenial_traj[-g][1]) <= max_distance):
                                tmp_taj.append(potenial_traj+[n])   # node found, break the gap loop
                                break
        l_trajectory = tmp_taj
    return l_trajectory


def generate_a_sequence(r_map, k=10, distance=1):
    rssi_seq = []
    l_keys = list(r_map.keys())
    rkey = l_keys[np.random.randint(0, len(l_keys)-1)]
    rssi_seq.append(r_map[rkey])
    max_try = 0
    while len(rssi_seq) < k and max_try <= 1000:
        if (str(float(rkey[0]) + distance), rkey[1]) in l_keys:
            rkey = (str(float(rkey[0]) + distance), rkey[1])
            rssi_seq.append(r_map[rkey])
        elif (str(float(rkey[0]) - distance), rkey[1]) in l_keys:
            rkey = (str(float(rkey[0]) - distance), rkey[1])
            rssi_seq.append(r_map[rkey])
        elif (rkey[0] , str(float(rkey[1])+distance)) in l_keys:
            rkey = (rkey[0] , str(float(rkey[1])+distance))
            rssi_seq.append(r_map[rkey])
        elif (rkey[0], str(float(rkey[1]) - distance)) in l_keys:
            rkey = (rkey[0], str(float(rkey[1]) - distance))
            rssi_seq.append(r_map[rkey])
        else:
            for tk in l_keys[np.random.randint(0, len(l_keys)):]:
                if cal_distance(rkey[0], rkey[1], tk[0], tk[1]) < distance:
                    rkey = tk
                    rssi_seq.append(r_map[rkey])
                    break
        max_try += 1
    return rssi_seq


# def pipeline(n):  # number of potential vertices
#     # r_map = init_random_map(n)
#     r_map = init_map_from_omnet('./log.txt')  # location and rssi
#     r_tree = setup_tree(r_map)
#     # rssi_seq = [-22.5 + (.5 - np.random.rand())*20 for _ in range(10)]
#     rssi_seq = generate_a_sequence(r_map, n, distance=5)
#     l_l_nodes = []  # rssi, nodes
#     for rssi in rssi_seq:
#         l_l_nodes.append(search_tree_node(rssi, r_tree, tolerance=0.03))
#     l_trajectory = construct_trajectory(l_l_nodes, max_distance=4, total_gaps=3)
#     print("")

def pipeline_prob(logfile, dx=10, dy=10, p=0.01, time_gap=5, top_k=3, n_step = 3):
    # setup transfer probability
    r_map = init_map_from_omnet(logfile, dx, dy)  # setup verticies and signal distribution
    # print(len(r_map))
    # r_tree = setup_tree(r_map)  # setup tree for search later
    # setup user traj for test/validation user: (time, location ,rssi)
    m_user_traj = {}  # {user1: [(time, (vx,vy), rssi), (time, (vx,vy), rssi), ...], user2:[]}
    for line in open(logfile).readlines():
        line = line.split(" ")[2:]
        user = line[1]
        t = float(line[6])
        if t > 50:
            break
        posi = coord2vert(float(line[8][1:-1]), float(line[9][:-1]), dx, dy)
        rssi = abs(float(line[-1].split('\n')[0]))
        if user not in m_user_traj:
            m_user_traj[user] = []
        m_user_traj[user].append((t, posi, rssi))
    # setup vertices transfer probability
    tran_prob = vertices_transfer_prob(m_user_traj, time_gap)
    total_accuracy = {}
    # inference per user
    for k, v in m_user_traj.items():
        c_no_transfer = 0
        current_t = v[0][0]
        poential_traj_list = []
        l_real_traj = []  # for validation
        for i in range(1, len(v)):
            if v[i][0] - current_t >= time_gap:  # infer the rssi
                poential_traj_list.append(search_rssi_by_likelyhood(v[i][2], r_map, p, top_k))
                l_real_traj.append(v[i][1])
                current_t = v[i][0]
        if n_step > 0:
            l_real_traj = l_real_traj[n_step-1:]
        # transfer
        if len(poential_traj_list)<1 or len(l_real_traj) == 0:
            continue
        l_infer_traj = [list(poential_traj_list[0].keys())[0]]
        for i in range(1, len(poential_traj_list)):  # add transfer probability from index 1
            # all steps
            # l_infer_temp = poential_traj_list[0]
            # for j in range(1, i):
            #     l_infer_temp2 = {}  # node, p
            #     for prev_vert, prev_prob in l_infer_temp.items():
            #         for curr_vert, curr_prob in poential_traj_list[j].items():
            #             if (prev_vert, curr_vert) in tran_prob:
            #                 l_infer_temp2[curr_vert] = prev_prob * curr_prob * tran_prob[(prev_vert, curr_vert)]
            #             else:
            #                 pass  # if often hit here, means should increase k to add more poential vertices
            #     l_infer_temp = l_infer_temp2
            # n step
            if n_step == 0:
                l_infer_temp = poential_traj_list[i]
            else:
                if i < n_step:  # not enough steps
                    continue
                l_infer_temp = poential_traj_list[i-n_step]
                for j in range(i-n_step+1, i):
                    l_infer_temp2 = {}  # node, p
                    for prev_vert, prev_prob in l_infer_temp.items():
                        for curr_vert, curr_prob in poential_traj_list[j].items():
                            if (prev_vert, curr_vert) in tran_prob:
                                if curr_vert not in l_infer_temp2:
                                    l_infer_temp2[curr_vert] = 0
                                l_infer_temp2[curr_vert] += prev_prob * curr_prob * tran_prob[(prev_vert, curr_vert)]
                    l_infer_temp = l_infer_temp2
            if len(l_infer_temp) > 0:
                most_lky_vt = sorted(l_infer_temp, key=lambda x:l_infer_temp[x], reverse=True)[0]
                l_infer_traj.append(most_lky_vt)
            else:
                c_no_transfer += 1
                l_infer_traj.append((0,0))  # make sure same length

        # validation
        # print("No transfer ", c_no_transfer)
        if len(l_infer_traj) != len(l_real_traj):
            print("")
        accuracy = {}  # based on number of steps
        # binary accuracy is not good enough
        # correct = 0
        # for i in range(len(l_infer_traj)):
        #     if l_infer_traj[i] == l_real_traj[i]:
        #         correct +=1
        #     accuracy[(i+1)] = correct/(i+1)
        # print(accuracy)
        # distance based accuracy
        for i in range(len(l_infer_traj)):  # accuracy at the current timedelta
            distance = vert_distance(l_infer_traj[i], l_real_traj[i])
            accuracy[(i+1)] = distance

        for k2, v2 in accuracy.items():  # accuracy per timedelta for all users
            if k2 not in total_accuracy:
                total_accuracy[k2] = []
            total_accuracy[k2].append(v2)
    # total accuracy
    rs = []
    for k, v in total_accuracy.items():
        rs.extend([i for i in v])  # putting all timegaps, all users into one array
    return rs


if __name__=="__main__":
    logpath = sys.argv[1]
    print(logpath)
    # p is not very useful
    print("dx time_gap top_k n_step t0 average median")
    for time_gap in range(1, 10, 1):  # nan when over 7, smaller better due to more measures
        for dx in range(100, 110, 20):  # greater the better
            for top_k in range(30, 35, 10):
                for n_step in range(0, 1, 2):
                    t0 = datetime.now().timestamp()  # t has nothing to do with k & step,
                    # a lot w/ dx & time_gap
                    rs = pipeline_prob(logpath, dx=dx, dy=dx, time_gap=time_gap, top_k=top_k, n_step=n_step)
                    t0 = int(datetime.now().timestamp() - t0)
                    print(dx, time_gap, top_k, n_step, t0, np.mean(rs), np.median(rs))


def bottom():
    pass