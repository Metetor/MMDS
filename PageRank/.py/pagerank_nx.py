# Call networkx library function to implement pagerank
import networkx as nx

# data path,please change it according to personal path
data_path = "../data.txt"
result_path = "../result/nx_weight.txt"
top100_path = "../result/nx_top100.txt"

# generate graph object
G = nx.DiGraph()
# print(help(nx.DiGraph))

# load data,data format:fromNodeID,toNodeID
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        '''
        edge=f.readline()
        print(type(edge.strip('\n').split(' '))) #<class 'list'>
        print(edge.strip('\n').split(' ')) #['1086', '579']
        [fromNode,toNode]=map(int,edge.strip('\n').split(' '))
        '''
        for raw_edge in f.readlines():
            [fromID, toID] = list(map(int, raw_edge.strip('/n').split(' ')))
            # grow by adding edges
            G.add_edge(fromID, toID)
        f.close()
except IOError as e:
    print("An IOError occured.{}".format(e.args[-1]))

# call pagerank
pr = nx.pagerank(G, alpha=0.85)

# write result to nx_result
with open(result_path, "w", encoding='utf-8') as f:
    # Nodes=pr.keys()
    # sorted(Nodes)
    for Node in sorted(pr.keys()):
        f.write(str(Node) + ' ' + str(pr[Node]) + '\n')
    f.close()

# write top100 to nx_top100
with open(top100_path, 'w', encoding='utf-8') as f:
    cnt = 0
    for Node, weight in sorted(pr.items(), key=lambda d: d[1], reverse=True):
        if cnt >= 100:
            f.close()
            break
        f.write(str(Node) + ' ' + str(weight) + '\n')
        cnt += 1
