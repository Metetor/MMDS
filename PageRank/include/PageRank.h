// encoding = GB2312
#ifndef __PAGERANK_H__
#define __PAGERANK_H__
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
using std::pair;
using std::set;
using std::string;
using std::unordered_map;
using std::vector;
class PageRank
{ // pagerank类实现
public:
	// Edge=(V,E)

	// edge的边和节点
	set<pair<int, int>> E;
	set<int> V;
	int BlockSize = 500;
	int NodeCnt, MaxPageNum;
	unordered_map<int, set<int>> matrix; // 初始稀疏矩阵(转移矩阵M)
	vector<double> w;					 // 权重矩阵w;
	double Beta = 0.85, RandomWalk, S;	 // β值,RandomWalk=(1-Beta)/NodeCnt
	double thresold = 1e-5, dev = 1;	 // 权重矩阵误差阈值

	PageRank(){};
	~PageRank(){};
	void setBeta(double beta);
	// Load Data
	void load_data(string fname);
	// Handle Dead Ends Node
	void HandleDeadEnds();	 // 将Dead Ends连接至全部节点，包括自身
	void Block_Stripe_pre(); // 分块预处理

	void WeightInit(); // 初始化权重矩阵
	void BS_WeightUpdate();
	void WeightUpdate(); // 权重矩阵迭代更新
	void WriteNode(string fname);
	void WriteTop100(string fname);
};
#endif // !__PAGERANK_H__