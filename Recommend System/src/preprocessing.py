import math
import random
import re
from collections import defaultdict
from sklearn.model_selection import KFold, ParameterGrid


def load_train_data(filename):
    """
    file format:
    train.txt
    <user id>|<numbers of rating items>
    <item id>   <score>
    """
    data_dict = defaultdict(dict)
    user_num = 0
    item_num = 0
    rating_num = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        it = iter(lines)
        for row in it:
            # record the user id and numbers of rating items
            user_id, user_rating_num = row.split('|')
            user_num += 1
            rating_num += int(user_rating_num)
            for _ in range(int(user_rating_num)):
                row = next(it)
                # record item id and scores

                item_id, score = re.split(r'\s+', row.strip())
                # dataset.append([user_id, item_id, scores])
                # 修改返回格式为dict:format {userid:{itemid1:score1,itemid2:score2,...}}
                data_dict[user_id][item_id] = int(score)
    print("评分数据数目为{0}".format(rating_num))
    return data_dict


def load_test_data():
    pass

# 定义一个函数来划分训练集和测试集
def split_train_test(data_dict, test_user_ratio=0.5, test_item_ratio=0.2):
    # 取10%的用户作为测试集
    users = list(data_dict.keys())
    test_users = random.sample(users, int(len(users) * test_user_ratio))
    # 对每个测试用户，取10%的评分作为测试集
    test_scores = {}
    for user in test_users:
        items = list(data_dict[user].keys())
        test_items = random.sample(items, int(len(items) * test_item_ratio))
        test_scores[user] = {item: data_dict[user][item] for item in test_items}
    # 将剩余的用户和评分作为训练集
    train_scores = {}
    for user in users:
        if user not in test_users:
            train_scores[user] = data_dict[user]
        else:
            train_scores[user] = {item: data_dict[user][item] for item in data_dict[user] if
                                  item not in test_scores[user]}
    # 返回训练集和测试集
    return train_scores, test_scores


def classify(data_dict):
    for user,rates in data_dict.items():
        for item,rate in rates.items():
            if rate>=0 and rate <=20:
                data_dict[user][item]=1
            elif rate>=20 and rate <=40:
                data_dict[user][item]=2
            elif rate>=40 and rate <=60:
                data_dict[user][item]=3
            elif rate>=60 and rate <=80:
                data_dict[user][item]=4
            if rate>=80 and rate <=100:
                data_dict[user][item]=5
def restd(dict):
    for user,rates in dict.items():
        for item,rate in rates.items():
            dict[user][item]*=20.0
def rmse(val_dict, pred_dict):
    # create a list to store the squared errors
    sq_errors = []
    # loop over the keys and values of val_dict and pred_dict
    for uid, rates in val_dict.items():
        for iid, score in rates.items():
            # get the true score and the predicted score
            y_true = score
            y_pred = pred_dict[uid][iid]
            # calculate the squared error and append to the list
            sq_error = math.pow(y_true - y_pred, 2)
            sq_errors.append(sq_error)
    # calculate the mean squared error
    mse = sum(sq_errors) / len(sq_errors)
    # calculate the root mean squared error
    rmse = math.sqrt(mse)
    return rmse

def getResult(model):
    # 读取test.txt文件
    with open('../data/test.txt', 'r') as f:
        lines = f.readlines()

    user = ""
    n_items = ""
    s = ""
    for line in lines:
        if "|" in line:
            with open('../result/result_bias.txt', 'a') as f:
                f.write(s)
            s = line
            user, n_items = line.strip().split("|")
        else:
            item = line.strip()
            # item = int(item)
            score = model._predict(user, item)
            s += str(item)
            s += " "
            s += str(score)
            s += "\n"
    with open('../result/result_funk.txt', 'a') as f:
        f.write(s)