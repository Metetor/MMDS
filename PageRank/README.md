# PageRank
naive PageRank algorithm and Block_striped algorithm (C++)

- Data Format:FromNodeID/ToNodeID
## networkx实现(作业禁止使用)
调用networkx库相关函数求解，用作结果对照
### 源码
`安装networkx库,pip install networkx[all] （test时networkx.Graph()提示错误，一方面是代码文件命名不能是networkx.py,另外可能是没有[all]下载依赖文件）
`

pagerank实现在networkx.algorithms.link_analysis的pagerank_alg.py

- 最终对比C++ naive/BS的实现同nx的结果误差在1e-5，top100的结果只有些许结点存在偏差

## python 实现

## C++ 实现

options:

-h help

-v version

-i input txt(graph node)

-m mode(0 -> naive mode/1->Block_striped mode)

-r training round

## theoretical Analysis


