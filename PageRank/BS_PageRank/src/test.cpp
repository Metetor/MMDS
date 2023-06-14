#include<iostream>
#include"test.h"
#include <fstream>
void test_load_data()
{
	PageRank PR;
	PR.load_data("data.txt");
	//printf("MaxNodeCnt:%d (nodecnt:%d)", PR.NodeCnt,PR.V.size());
	for (auto e : PR.E)
	{
		//printf("%d->%d\n", e.first,e.second);
	}
	for (int v=1;v<=PR.NodeCnt;v++)
	{
		if (PR.V.count(v) <= 0)
			printf("%d\n",v);
		/*printf("NodeID:%d,outDegree:%d\n", v, PR.matrix[v].size());
		for (auto to : PR.matrix[v])
		{
			printf("%d ", to);
		}*/
	}
}
void test_weight_update() {
	PageRank PR;
	PR.load_data("data.txt");
	printf("load_data finished\n");
	PR.HandleDeadEnds();
	printf("HandleDeadEnds finished\n");
	PR.Block_Stripe_pre();
	printf("Block_Stripe_pre finished\n");
	PR.WeightInit();
	printf("WeightInit finished\n");
	/*printf("\nnavie PageRank R Matrix Update\n");
	PR.WeightUpdate();
	printf("WeightUpdate end\n");*/
	printf("\nBlock Stripe PageRank R Matrix Update\n");
	PR.BS_WeightUpdate();
	printf("BS_WeightUpdate end\n");

	/*PR.WriteNode("weight.txt");
	PR.WriteTop100("Top100.txt");*/
}
void test_read_dat(string fname)
{
	int size = 8;
	std::ifstream in(fname, std::ios::binary);
	double num=0.0;
	in.seekg(2 * size);
	in.read((char*)&num, size);
	std::cout << num;
}
void dat2txt(string dat, string txt)
{
	std::ofstream fout(txt, std::ios::binary | std::ios::app);
	int size = 8;
	std::ifstream in(dat, std::ios::binary);
	double num = 0.0;
	for (int p = 0; p < 8000; p++)
	{
		in.seekg(p * size);
		in.read((char*)&num, size);
		fout << p << " " << num << std::endl;
	}
}