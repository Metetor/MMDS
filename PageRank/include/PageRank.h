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
{ // pagerank��ʵ��
public:
	// Edge=(V,E)

	// edge�ıߺͽڵ�
	set<pair<int, int>> E;
	set<int> V;
	int BlockSize = 500;
	int NodeCnt, MaxPageNum;
	unordered_map<int, set<int>> matrix; // ��ʼϡ�����(ת�ƾ���M)
	vector<double> w;					 // Ȩ�ؾ���w;
	double Beta = 0.85, RandomWalk, S;	 // ��ֵ,RandomWalk=(1-Beta)/NodeCnt
	double thresold = 1e-5, dev = 1;	 // Ȩ�ؾ��������ֵ

	PageRank(){};
	~PageRank(){};
	void setBeta(double beta);
	// Load Data
	void load_data(string fname);
	// Handle Dead Ends Node
	void HandleDeadEnds();	 // ��Dead Ends������ȫ���ڵ㣬��������
	void Block_Stripe_pre(); // �ֿ�Ԥ����

	void WeightInit(); // ��ʼ��Ȩ�ؾ���
	void BS_WeightUpdate();
	void WeightUpdate(); // Ȩ�ؾ����������
	void WriteNode(string fname);
	void WriteTop100(string fname);
};
#endif // !__PAGERANK_H__