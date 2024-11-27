from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.metrics import accuracy_score
import sys
import pandas as pd
import numpy as np
from traj_search import coord2vert, cal_distance, vertices_transfer_prob
from datetime import datetime


def cal_distance_two_str(grid1, grid2):
    assert isinstance(grid1, str)
    assert isinstance(grid2, str)
    grid1 = (int(grid1.split(",")[0][1:]), int(grid1.split(",")[1][1:-1]))
    grid2 = (int(grid2.split(",")[0][1:]), int(grid2.split(",")[1][1:-1]))
    return cal_distance(grid1[0], grid1[1], grid2[0], grid2[1])

class MLP_Locate():
    def read_log(self, logfile, dx, time_gap):
        rssi = []
        loc = []
        user_timestamp = {}  # to determine time gap
        m_user_traj = {}
        for line in open(logfile).readlines():
            try:
                line = line.split(" ")[2:]
                user = line[1]
                timestamp = float(line[6])
                if timestamp > 50:
                    break
                if user in user_timestamp:
                    if timestamp - user_timestamp[user] <= time_gap:
                        continue
                # else
                r = abs(float(line[-1].split('\n')[0]))
                posi = coord2vert(float(line[8][1:-1]), float(line[9][:-1]), dx, dx)
                user_timestamp[user] = timestamp
                if user not in m_user_traj:
                    m_user_traj[user] = []
                m_user_traj[user].append((timestamp, posi, r))
                loc.append(posi)
                rssi.append(r)
            except ValueError:
                continue
        df = pd.DataFrame({'rssi':rssi,'posi':loc})
        # setup vertices transfer probability
        self.TRAN_PROB = vertices_transfer_prob(m_user_traj, time_gap)
        self.TRAN_PROB_AVG = np.average([i for i in self.TRAN_PROB.values()])
        return df, self.TRAN_PROB


    def prepare_data(self, logpath, dx, time_gap, len_x):
        df, TRAN_PROB = self.read_log(logpath, dx, time_gap)
        data = df['rssi'].to_numpy()
        label_ = df['posi'].to_numpy()
        x,y = [], []
        for i in range(0, len(data)-len_x):
            x.append(data[i:i+len_x])
            y.append(label_[i+len_x-1])
        self.yencoder = OneHotEncoder()
        y = np.array([str(i) for i in y]).reshape(-1, 1)
        self.yencoder.fit(y)
        y = self.yencoder.transform(y)
        # TRAN_PROB2 = {}
        # for k, v in TRAN_PROB.items():
        #     k0 = str([i for i in self.yencoder.transform([k[0]]).toarray()[0]])
        #     k1 = str([i for i in self.yencoder.transform([k[1]]).toarray()[0]])
        #     TRAN_PROB2[(k0,k1)] = v
        x = normalize(x)
        x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=False)

        return x_train, x_test, y_train, y_test


    def mlp_predict(self, logpath, dx, time_gap, len_x=10, traj=False):
        """

        :param logpath:
        :param dx: grid size
        :param time_gap: measurement time gap
        :param len_x: number of features to use
        :return:
        """
        t0 = datetime.now().timestamp()
        x_train, x_test, y_train, y_test = self.prepare_data(logpath, dx, time_gap,len_x)
        model = MLPClassifier(max_iter=5000)
        model.fit(x_train, y_train)
        pre_y = model.predict(x_test).toarray()

        # reverse from the model.predict
        pre_y2 = []
        for i in range(pre_y.shape[0]):
            try:
                pre_y2.append(self.yencoder.inverse_transform([pre_y[i]])[0])
            except ValueError:
                pre_y2.append(np.array(['(0, 0)']))
        assert len(pre_y) == len(pre_y2)
        pre_y = pre_y2
        # reverse from the model.predict_proba
        pre_y_prob = model.predict_proba(x_test)
        for i in range(pre_y_prob.shape[0]):
            pre_y_prob[i] = np.where(pre_y_prob[i], pre_y_prob[i] >= np.max(pre_y_prob[i]), 0)
        pre_y2 = []
        for i in range(pre_y_prob.shape[0]):
            try:
                pre_y2.append(self.yencoder.inverse_transform([pre_y_prob[i]])[0])
            except ValueError:
                pre_y2.append(np.array(['(0, 0)']))
        # assert len(pre_y) == len(pre_y2)
        # reverse from model.predict_proba plus traj
        pre_y_prob = model.predict_proba(x_test)
        for i in range(1, pre_y_prob.shape[0]):
            if not ( np.max(pre_y_prob[i]) < 0.6 and np.max(pre_y_prob[i-1]) > 0.8 ):  # only if this one is not reliable and last one is reliable
                continue
            previous_p = np.where(pre_y_prob[i-1], pre_y_prob[i-1] >= np.max(pre_y_prob[i-1]), 0)
            try:
                previous_grid = self.yencoder.inverse_transform([previous_p])[0]
                previous_grid = (int(previous_grid[0].split(",")[0][1:]), int(previous_grid[0].split(",")[1][1:-1]))
            except ValueError:
                continue
            for j in range(0, pre_y_prob.shape[1]):  # each candidate
                try:
                    this_grid = np.zeros(pre_y_prob.shape[1])
                    this_grid[j] = 1
                    this_grid = self.yencoder.inverse_transform([this_grid])[0]
                    this_grid = (int(this_grid[0].split(",")[0][1:]), int(this_grid[0].split(",")[1][1:-1]))
                    if (previous_grid, this_grid) in self.TRAN_PROB:
                        pre_y_prob[i, j] = pre_y_prob[i, j] * self.TRAN_PROB[(previous_grid, this_grid)]
                    else:  # if not in traj list, otherwise bad for those who have tran probability
                        pre_y_prob[i, j] = pre_y_prob[i, j] * self.TRAN_PROB_AVG
                except ValueError:
                    continue
        for i in range(pre_y_prob.shape[0]):
            pre_y_prob[i] = np.where(pre_y_prob[i], pre_y_prob[i] >= np.max(pre_y_prob[i]), 0)
        pre_y3 = []
        for i in range(pre_y_prob.shape[0]):
            try:
                pre_y3.append(self.yencoder.inverse_transform([pre_y_prob[i]])[0])
            except ValueError:
                pre_y2.append(np.array(['(0, 0)']))
        assert len(pre_y) == len(pre_y3)

        y_test = self.yencoder.inverse_transform(y_test)
        # a_s = accuracy_score(pre_y, y_test)
        res = []
        res_p = []
        res_p_traj = []
        for i in range(0, len(pre_y)):
            res.append(cal_distance_two_str(pre_y[i][0], y_test[i][0]))
            res_p.append(cal_distance_two_str(pre_y2[i][0], y_test[i][0]))
            res_p_traj.append(cal_distance_two_str(pre_y3[i][0], y_test[i][0]))
        c_better = [0, 0, 0 ]
        for i in range(len(res_p)):
            d_p = cal_distance_two_str(pre_y2[i][0], y_test[i][0])
            d_traj = cal_distance_two_str(pre_y3[i][0], y_test[i][0])
            if d_traj < d_p:
                c_better[0] += 1
            elif d_traj == d_p:
                c_better[1] += 1
            elif d_traj > d_p:
                c_better[2] += 1
        print(c_better, len(res_p))
        print(dx, datetime.now().timestamp() - t0, np.mean(res), np.median(res))
        print(dx, datetime.now().timestamp() - t0, np.mean(res_p), np.median(res_p))
        print(dx, datetime.now().timestamp() - t0, np.mean(res_p_traj), np.median(res_p_traj))
        return ""



if __name__=="__main__":
    logpath = sys.argv[1]
    print(logpath)
    for dx in range(50, 55, 10):
        for time_gap in range(1, 5, 1):
            for len_x in range(100, 110, 20):
                a_s = MLP_Locate().mlp_predict(logpath, dx, time_gap, len_x= len_x)