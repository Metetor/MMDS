import math
import pickle
import random
from collections import defaultdict
from math import sqrt
import time
from tqdm import tqdm

'''
Rating matrix (Sparse matrix) format:[uid,iid,score]
'''
import cProfile
from preprocessing import load_train_data, rmse, split_train_test, getResult
import numpy as np


class FunkSVD:
    """
    F:nums of Factors
    alpha:learning rate
    lambda_p,lamdba_q:
    err:error rate
    max_iterations:max iter
    """

    def __init__(self, factors, lr, lr_reg, rating_dict):
        """
        :param factors:
        :param lr:
        :param l1:
        :param l2:
        :param R:
        """
        self.mu = 50.0
        self.F = factors
        self.alpha = lr
        self.lr_reg = lr_reg
        self.max_iterations = 5
        self.R = rating_dict

        self.Q, self.P = self._init_matrix()

    def _init_matrix(self):
        """
        init P,Q matrix
        :return:
        """
        Q = dict()
        P = dict()

        with open("../tmp/Q_funk.pkl", "rb") as f:
            Q = pickle.load(f)
        with open("../tmp/P_funk.pkl", "rb") as f:
            P = pickle.load(f)
        """
        cnt = 0
        for user, rates in self.R.items():
            Q[user] = [sqrt(50 / self.F) for _ in range(self.F)]
            for item, rate in rates.items():
                self.mu += rate
                if item not in P:
                    P[item] = [sqrt(50 / self.F) for _ in range(self.F)]
                cnt += 1
        self.mu /= cnt
        """
        return Q, P

    def train(self):
        prefix_path = '../tmp/log/'
        # 开始迭代

        for step in range(self.max_iterations):
            '''
            #user-item matrix=Q*inverse(P)
            var:
                r-> rating matrix
                (x,i) belongs to rating matrix
            let F -> nums of factors,then Q->row:m,cols:F;P->row:n,cols:F
            Update P,
                P = P - eta * Nabla(P)
                Q = Q - eta * Nabla(Q)
            that Nabla(Q)=sigma(-2*(r(x,i)-qi*px)pxf+2lamda*qif)->shape(x,i)
            Organized
                P=P+sigma(alpha*((r[x][i]-qi*px)*p_xf-lamda*p_if))
                ...
                impl:
                    Q(row,col)=Q(row,col)+alpha*((r[x][i]-qi*px)*pxf-lamda*qxf)

            '''
            error = []
            for user, rates in tqdm(self.R.items()):
                for item, rate in rates.items():
                    # calc [user, item, score,err=score-np.dot(self.Q[uidx,:],self.P[iidx,:])
                    err = rate - self._predict(user, item)
                    for f in range(self.F):
                        tmp1 = self.Q[user][f]
                        tmp2 = self.P[item][f]
                        self.Q[user][f] += self.alpha * (err * tmp2 - self.lr_reg * self.Q[user][f])
                        self.P[item][f] += self.alpha * (err * tmp1 - self.lr_reg * self.P[item][f])
                        err = err - self.Q[user][f] * self.P[item][f] + tmp1 * tmp2
                    # update error
                    error.append(err ** 2)
            # print error
            rsme = np.sqrt(np.mean(error))
            print("第{0}轮训练的误差为{1}".format(step, rsme))
            # update alpha
            self.alpha *= 0.95
        with open("../tmp/P_funk.pkl", "wb") as f:
            pickle.dump(self.P, f)
        with open("../tmp/Q_funk.pkl", "wb") as f:
            pickle.dump(self.Q, f)

    def _predict(self, user, item):
        if user not in self.Q.keys() or item not in self.P.keys():
            return self.mu
        score = np.dot(self.Q[user], self.P[item])
        if score > 100:
            score = 100.0
        if score < 0:
            score = 0.0
        return score

    def predict(self, val_dict):
        val_pred_dict = defaultdict(dict)
        for user, rates in val_dict.items():
            for item, rate in rates.items():
                pred = self._predict(user, item)
                val_pred_dict[user][item] = int(pred)
        return val_pred_dict


def tune():
    data_dict = load_train_data('../data/train.txt')

    # 对sample划分训练集验证集
    train_dict, val_dict = split_train_test(data_dict)
    # SVD
    model = FunkSVD(factors=50, lr=5e-4, lr_reg=0.1, rating_dict=train_dict)

    model.train()
    """
    with open('../model/funk.pkl', 'wb') as f:
        pickle.dump(model, f)
    val_pred_dict = model.predict(val_dict)

    v_rmse = rmse(val_dict, val_pred_dict)
    print("训练的rmse值为{0}".format(v_rmse))
    """


if __name__ == '__main__':

    data_dict = load_train_data('../data/train.txt')
    train_data,val_dict=split_train_test(data_dict)
    #for lr in []
    model = FunkSVD(factors=50, lr=5e-3,lr_reg=0.1, rating_dict=data_dict)
    model.train()
    with open('../model/funk.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../model/funk.pkl','rb') as f:
        model=pickle.load(f)
    getResult(model)