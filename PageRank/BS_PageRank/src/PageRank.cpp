#include"PageRank.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<algorithm>
#include"file.h"

using std::ifstream;
using std::istringstream;
using std::cout;
using std::ofstream;
void PageRank::setBeta(double beta)
{
	this->Beta = beta;
}
void PageRank::load_data(string fname)
{
	ifstream inTXT(fname);
	string line;
	while (std::getline(inTXT, line)) {//逐行读取文件
		istringstream iss(line);
		int from, to;
		if (iss >> from >> to) {//将每行字符串解析为两个整数
			//todo
			//std::cout << from << "->" << to << std::endl;
			V.insert(from);
			V.insert(to);
			E.insert({ from,to });
			matrix[from].insert(to);
		}
		else
		{
			perror("parse line error!\n");
		}
	}
	inTXT.close();//关闭文件
	//初始化PageRank的参数NodeCnt和MaxPageNum
	NodeCnt = V.size();
	MaxPageNum = *V.rbegin();//set默认排序
	RandomWalk = (1 - Beta) / NodeCnt;
	return;
}
void PageRank::HandleDeadEnds()
{
	//搜索出所有的Dead Ends Node
	for (auto v : V)
	{
		if (matrix.count(v) <= 0)
		{
			//将节点与所有节点链接
			matrix[v] = V;
		}
	}
}
void PageRank::Block_Stripe_pre() {
	//分块存储
	string blocks_path = "./Blocks";
	mk_dir(blocks_path);
	for (auto m : matrix)
	{
		int src = m.first;
		auto dsts = m.second;
		int outdeg = m.second.size();//出度
		string line = std::to_string(src) + " " + std::to_string(outdeg);
		int pre = 0,i=0;
		for (auto dst : dsts)
		{
			int blockno = dst / BlockSize;
			i = i + 1;
			if (blockno != pre && i != 1)
			{
				line += "\n";
				ofstream fout("./Blocks/Matrix" + std::to_string(pre) + ".txt",std::ios::binary|std::ios::app);
				fout << line;
				line = std::to_string(src) + " " + std::to_string(outdeg);
			}
			line = line + " " + std::to_string(dst);
			pre = blockno;
		}
		line += "\n";
		ofstream fout("./Blocks/Matrix" + std::to_string(pre) + ".txt", std::ios::binary | std::ios::app);
		fout << line;
	}
};

void PageRank::WeightInit() {
	vector<double> tmp(MaxPageNum, 1.0/NodeCnt);
	w = tmp;
	//初始化R矩阵并写入R.txt
	mk_dir("./R_training");
	ofstream fout("./R_training/R0.dat", std::ios::binary);
	fout.write((const char*)tmp.data(), tmp.size() * sizeof(double));
	fout.close();
}
void PageRank::BS_WeightUpdate() {
	
	int round = 0;//训练轮数
	dev = 1;
	while (dev > thresold&&round<6)
	{
		dev = 0.0;
		
		for (int i = 0; i <= MaxPageNum / BlockSize; i++)
		{
			//开始对每一个block进行处理
			vector<double> Rnew(BlockSize, RandomWalk);//初始化块权重矩阵
			string mat_path = "./Blocks/Matrix" + std::to_string(i) + ".txt";
			ifstream in(mat_path);
			string line;
			while (std::getline(in, line))
			{
				istringstream iss(line);
				int src, deg, dst;
				iss >> src >> deg;
				//读取R_old[src-1]
				double ri = read_dat("./R_training/R" + std::to_string(round) + ".dat", src - 1);
				while (iss >> dst)
				{
					//printf("src:%d,deg:%d,dst:%d,ri:%f", src, deg, dst,ri);
					Rnew[(dst - 1) % BlockSize] += (Beta * ri /deg);
				}
			}
			ofstream out("./R_training/R" + std::to_string(round + 1) + ".txt", std::ios::binary | std::ios::app);
			//计算dev并将其写入Rnew
			for (int j = 0; j < Rnew.size(); j++)
			{
				int real = i * BlockSize + j;
				double ri = read_dat("./R_training/R" + std::to_string(round) + ".dat", real);
				dev += abs(Rnew[j] - ri);
				/*S += Rnew[j];*/
				out << real << " " << Rnew[j] << std::endl;
			}
			out.close();
			ofstream fout("./R_training/R" + std::to_string(round + 1) + ".dat", std::ios::binary | std::ios::app);
			fout.write((const char*)Rnew.data(), Rnew.size() * sizeof(double));
			/*for (auto r:Rnew)
			{
				fout << r << std::endl;
			}*/
			fout.close();
		}
		round++;
		printf("round:%d,dev:%f\n", round,dev);
	}
}
void PageRank::WeightUpdate() {
	int round = 0;
	while (dev > thresold) {
		vector<double> w_tmp(MaxPageNum, 0.0);
		for (auto v : V)
		{
			double sigma = w[v - 1] / matrix[v].size();
			for (auto out : matrix[v])
			{
				w_tmp[out - 1] += sigma;
			}
		}
		dev = 0.0;
		for (int i = 0; i < MaxPageNum; i++)
		{
			if (abs(w_tmp[i]) > 1e-20)
			{
				w_tmp[i] = RandomWalk + Beta * w_tmp[i];
				//计算误差
				dev += abs(w[i] - w_tmp[i]);
			}
		}
		//拷贝
		w = w_tmp;
		round++;
		printf("第%d轮，误差dev=%f", round, dev);
	}
}
void PageRank::WriteNode(string fname)
{
	ofstream fout(fname, std::ios::binary);
	for (auto v : V)
	{
		fout << v << " " << w[v - 1] << std::endl;
	}
	fout.close();
}
bool cmp(pair<int, double>x, pair<int, double>y)
{
	return x.second > y.second;
}
void PageRank::WriteTop100(string fname)
{
	vector<pair<int, double>> tmp;
	for (auto v : V)
	{
		tmp.push_back({ v,w[v-1] });
	}
	sort(tmp.begin(), tmp.end(),cmp);
	ofstream fout(fname, std::ios::binary);
	for (int i = 0; i < 100; i++)
	{
		fout << tmp[i].first << " " << tmp[i].second << std::endl;
	}
	fout.close();
}