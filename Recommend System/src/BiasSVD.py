import pickle
from collections import defaultdict
from math import sqrt
from preprocessing import load_train_data, getResult, classify, split_train_test, rmse
import numpy as np
from tqdm import tqdm


class BiasSVD:
    """
    F:nums of Factors
    alpha:learning rate
    lambda_p,lamdba_q:
    err:error rate
    max_iterations:max iter
    P
    Q
    Bu:user
    Bi:item
    mu:globalMean
    """

    def __init__(self, factors, lr, lr_reg, rating_dict):
        """

        :param factors: latent factors
        :param lr: learning rate
        :param lr_reg:
        :param rating_dict: dataset
        """
        self.mu = 49
        self.F = factors
        self.alpha = lr
        self.lr_reg = lr_reg
        self.max_iterations = 5
        # self.R = {'1': {'1': 3, '2': 4, '10': 2}, '2': {'1': 4, '2': 5, '20': 5}}
        _, self.V = split_train_test(rating_dict)
        self.R=rating_dict
        self.Q, self.P, self.Bu, self.Bi = self._init_matrix()

    def _init_matrix(self):
        """
        init P,Q,Bu,Bi matrix
        :return:
        """

        Q = dict()
        P = dict()
        Bu = dict()
        Bi = dict()
        """
        with open("../tmp/Q_bias.pkl", "rb") as f1:
            Q = pickle.load(f1)
        with open("../tmp/P_bias.pkl", "rb") as f2:
            P = pickle.load(f2)
        with open("../tmp/bu.pkl", "rb") as f3:
            Bu = pickle.load(f3)
        with open("../tmp/bi.pkl", "rb") as f4:
            Bi = pickle.load(f4)
        """
        cnt = 0
        for user, rates in self.R.items():
            Q[user] = np.random.randn(self.F)
            Bu[user] = 0.0
            for item, rate in rates.items():
                self.mu += rate
                if item not in P:
                    P[item] = np.random.randn(self.F)
                Bi[item] = 0.0
                cnt += 1
        self.mu /= cnt

        return Q, P, Bu, Bi

    def sgd(self):
        """
        随机梯度下降
        :return:
        """
        # 开始迭代
        for step in range(self.max_iterations):
            '''
            #user-item matrix=Q*inverse(P)
            var:x refers user/i refers item
            argmin sigma[(x,i) belongs to R] 
                        ((R(x,i)-(mu+bx+bi+qi*px))^2+
                        (l1*sigma(i)qi^2+l2*sigma(x)px^2+l3*sigma(x)*bx^2+l4*sigma(i)bi^2))
            var:
                r-> rating matrix
                (x,i) belongs to rating matrix
            let F -> nums of factors,then Q->row:m,cols:F;P->row:n,cols:F
            Update P,
                P = P - eta * Nabla(P)
                Q = Q - eta * Nabla(Q)
                Bu=Bu-eta*Nabla(Bu)
                Bi=Bi-eta*Nabla(Bi)
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
                    # calculate err
                    pred = self._predict(user, item)
                    err = rate - pred
                    # update Bu,Bi
                    self.Bu[user] += self.alpha * (err - self.lr_reg * self.Bu[user])
                    self.Bi[item] += self.alpha * (err - self.lr_reg * self.Bi[item])
                    # update P,Q
                    for f in range(self.F):
                        tmp1 = self.Q[user][f]
                        tmp2 = self.P[item][f]
                        self.Q[user][f] += self.alpha * (err * self.P[item][f] - self.lr_reg * self.Q[user][f])
                        self.P[item][f] += self.alpha * (err * tmp1 - self.lr_reg * self.P[item][f])
                        # update err
                        err = err - self.Q[user][f] * self.P[item][f] + tmp1 * tmp2
                    error.append(err ** 2)

            print("第{0}轮训练的误差为{1}".format(step, np.sqrt(np.mean(error))))
            # update alpha
            val_pred_dict = model.predict(self.V)

            v_rmse = rmse(self.V, val_pred_dict)
            print("训练的rmse值为{0}".format(v_rmse))
            self.alpha *= 0.9
        with open("../tmp/Q_bias.pkl", "wb") as f:
            pickle.dump(self.Q, f)
        with open("../tmp/P_bias.pkl", "wb") as f:
            pickle.dump(self.P, f)
        with open("../tmp/bu.pkl", "wb") as f:
            pickle.dump(self.Bu, f)
        with open("../tmp/bi.pkl", "wb") as f:
            pickle.dump(self.Bi, f)

    def _predict(self, user, item):
        """

        :param item:
        :param user:
        :return:
        """
        if user not in self.Q.keys() or item not in self.P.keys():
            return self.mu
        pred = np.dot(self.Q[user], self.P[item])
        # print(pred)
        bias = self.Bu[user] + self.Bi[item] + self.mu
        # print("pred:{0} bias:{1}".format(pred,bias))
        res = pred + bias
        if res>100:
            res=100.0
        if res<0:
            res=0.0
        return res

    def predict(self, val_dict):
        val_pred_dict = defaultdict(dict)
        for user, rates in val_dict.items():
            for item, rate in rates.items():
                pred = self._predict(user, item)
                val_pred_dict[user][item] = int(pred)
        return val_pred_dict


if __name__ == '__main__':
    train_path=input('请输入训练集路径')
    data_dict = load_train_data(train_path)
    print('初始化SVD类(factors=50,lr=5e-4,lr_reg=0.1)')
    model = BiasSVD(factors=50, lr=5e-4, lr_reg=0.1, rating_dict=data_dict)
    print('开始随机梯度下降训练...')
    model.sgd()
    print('正在保存模型,请确保存在../model/路径')
    with open('../model/bias.pkl', 'wb') as f:
        pickle.dump(model, f)
