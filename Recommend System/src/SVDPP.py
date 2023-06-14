# coding:utf-8
import pickle
import random
import math
from collections import defaultdict
import numpy as np
from src.preprocessing import load_train_data, rmse
import time
from tqdm import tqdm

class SVDPP(object):

    def __init__(self, num_factor=100, lr=0.002, l1=0.2, l2=0.2, l3=0.2, l4=0.2, l5=0.2, data_dict={}, num_iter=1):
        """

        :param num_factor:
        :param lr:
        :param l1:
        :param l2:
        :param l3:
        :param l4:
        :param l5:
        :param data_dict:
        :param num_iter:
        """
        self.F = num_factor
        self.alpha = lr
        self.lambda_q = l1
        self.lambda_p = l2
        self.lambda_y = l3
        self.lambda_bu = l4
        self.lambda_bi = l5
        self.num_iter = num_iter
        self.R = data_dict
        self.mu = 0.0

        self.Q = dict()
        self.P = dict()
        # Y表示每个物品所携带的隐因子属性
        self.Y = dict()
        self.Bu = dict()
        self.Bi = dict()

        self._init_matrix()
        self.count = 0

    def _init_matrix(self):
        """
        init P,Q,Bu,Bi matrix
        :return:
        """
        cnt = 0
        for user,rates in self.R.items():
            self.Q[user] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
            self.Bu[user] = 0
            cnt += len(rates)
            for item in rates.keys():
                rate = rates[item]
                self.mu += rate
                if item not in self.P:
                    self.P[item] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
                if item not in self.Y:
                    self.Y[item] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
                self.Bi[item] = 0
        self.mu /= cnt

    def getSigmaY(self, rates):
        """
        :param rates: {item1: score1, item2: score2, ......}
        :return: sigmaY
        """
        '''sigmaY = [0.0 for f in range(self.F)]
        for item, score in rates.items():
            if item not in self.Y.keys():
                continue
            for f in range(self.F):
                sigmaY[f] += self.Y[item][f]
        return sigmaY'''
        sigmaY = np.zeros(self.F)
        for item, score in rates.items():
            if item not in self.Y.keys():
                continue
            sigmaY += self.Y[item]
        return sigmaY


    def train(self):
        for step in range(self.num_iter):
            error = []
            for user, rates in tqdm(self.R.items()):
                sigmaY = self.getSigmaY(rates)
                sqrtNu = math.sqrt(1.0 * len(rates))
                sum = [0.0 for f in range(self.F)]
                for item, score in rates.items():
                    # get current error_ui
                    pre_score = self.predict(user, item, rates, sigmaY)
                    # print(pre_score)
                    # err = (score - pre_score) / 100
                    err = (score - pre_score)
                    # print(err)
                    #update Bu,Bi
                    self.Bu[user] += self.alpha * (err - self.lambda_bu * self.Bu[user])
                    self.Bi[item] += self.alpha * (err - self.lambda_bi * self.Bi[item])
                    # update Q,P
                    for f in range(self.F):
                        sum[f] += self.P[item][f] * err
                        self.Q[user][f] += self.alpha * (err * self.P[item][f] - self.lambda_q * self.Q[user][f])
                        self.P[item][f] += self.alpha * (err * (self.Q[user][f] + sigmaY[f] / sqrtNu) - self.lambda_p * self.P[item][f])
                    error.append(err ** 2)
                # update Y
                for item, _ in rates.items():
                    for f in range(self.F):
                        self.Y[item][f] += self.alpha * (sum[f] / sqrtNu - self.lambda_y * self.Y[item][f])
            print("第{0}轮训练的误差为{1}".format(step+1, np.sqrt(np.mean(error))))
            print("第{0}轮".format(step+1))
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小
        print("****************************训练结束********************************")

    def predict(self, user, item, rates, sigmaY):
        """

        :param user:
        :param item:
        :param rates: {item1: score1, item2: score2, ......}
        :param sigmaY: ∑Y
        :return:
        """
        if user not in self.Q.keys() and item not in self.P.keys():
            return self.mu
        ret = self.mu + self.Bu[user] + self.Bi[item] + \
                sum((self.Q[user][f] + sigmaY[f] / math.sqrt(1.0 * len(rates))) * self.P[item][f] for f in range(self.F))
        # np.dot(self.Q[user], self.P[item] + sigmaY / math.sqrt(len(rates)))
        return ret

    def predict_in_test(self, user, item, rates):
        """

        :param user:
        :param item:
        :param rates: {item1: score1, item2: score2, ......}
        :param sigmaY: ∑Y
        :return:
        """
        if user not in self.Bu.keys() or item not in self.Bi.keys():
            self.count += 1
            return self.mu
        if user not in self.Q.keys() or item not in self.P.keys():
            return self.mu
        sigmaY = self.getSigmaY(rates)
        ret = self.mu + self.Bu[user] + self.Bi[item] + \
              np.dot(self.Q[user], self.P[item] + sigmaY / math.sqrt(len(rates)))
        return ret


