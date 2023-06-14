# coding:utf-8
import re

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False


def load_attr_data(filename):
    """
    file format:
    item.txt
    <item id>|<attribute_1>|<attribute_2>('None' means this item is not belong to any of attribute_1/2)
    """
    attr_list = []
    item_num = 0
    attr1_miss_num = 0
    attr2_miss_num = 0
    attr12_miss_num = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for row in lines:
            # record the item id and attributes
            item_id, attribute_1, attribute_2 = re.split(r'\|', row.strip())
            if attribute_1 == 'None':
                attr1_miss_num += 1
                attribute_1 = '0'
                if attribute_2 == 'None':
                    attr2_miss_num += 1
                    attr12_miss_num += 1
                    attribute_2 = '0'
            if attribute_2 == 'None':
                attr2_miss_num += 1
                attribute_2 = '0'
            item_num += 1
            # modify the format as list: [[item_id,value1,value2]]
            attr_list.append([int(item_id), int(attribute_1), int(attribute_2)])
    print("***********************itemAttribute的统计信息**********************")
    print("item数目为{0}".format(item_num))
    print("attribute_1为None的数目为{0}".format(attr1_miss_num))
    print("attribute_2为None的数目为{0}".format(attr2_miss_num))
    print("attr1和attr2都为None的数目为{0}".format(attr12_miss_num))
    return attr_list


def clustering(attr_list):
    attr_array = np.array(attr_list)
    prefix_path = "../tmp/log"
    batch_size = 5000
    X = attr_array[:, -2:]
    with open(prefix_path + "clustering_log.txt", 'w+', encoding='UTF-8') as f:
        for n_cluster in tqdm(range(50, 500)):
            # 初始化KMeans
            KM = MiniBatchKMeans(init='k-means++', n_clusters=n_cluster, batch_size=batch_size, n_init=10,
                                 max_no_improvement=10, verbose=0)
            pred = KM.fit_predict(X)
            centers = KM.cluster_centers_
            SSE = np.sqrt(np.sum((X - centers[pred]) ** 2))
            f.write("n_clusters=" + str(n_cluster) + ":    " + str(SSE) + '\n')
    plt.show()


def find_clusters(filename):
    """

    :param filename:
    :return:
    """
    x = list()
    y = list()
    with open(filename, 'r') as f:
        lines = f.readlines()
        for row in lines:
            # 使用正则表达式提取两个数字
            matches = re.findall(r'\d+\.\d+|\d+', row)
            x.append(int(matches[0]))
            y.append(float(matches[1]))
    plt.title('聚类指标可视化(x:聚类数目 y:评价指标)')
    plt.plot(x, y)
    plt.savefig('../tmp/fig/cluster.jpg')
    plt.show()


find_clusters('../tmp/log/clustering_log.txt')
