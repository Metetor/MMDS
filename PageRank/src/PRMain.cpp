#include"test.h"
int main(int argc,char *argv[])
{
	/*
	test_weight_update();
	test_read_dat("R.dat");
	dat2txt("./R_training/R2.dat", "./R_training/R1.txt");
	*/
	//�����յĲ�����������һ���Ǵ������graph node txt
	bool input_on = false;
	bool mode_on = false;
	bool r_on = false;
	string fpath="data.txt";
	int mode=0;
	int round=10;
	printf("argc:%d\n", argc);
	for (int i = 1; i < argc; ++i)
	{
		//���ܲ���
		char* pchar = argv[i];
		switch (pchar[0])
		{
		case '-': {
			switch (pchar[1])
			{
			case 'h':
				//help
				printf("-h : help\n-v : version\n-i : graph node txt\n-m :0-naive pagerank algorithmn;1-Block-Striped PageRank algorithm\n");
				break;
			case 'v':
				//version
				printf("bs_pagerank version 2.1x\n");
				break;
			case 'i':
				//input txt
				input_on = true;
				fpath = argv[i + 1];
				break;
			case 'm':
				//mode
				mode_on = true;
				mode = atoi(argv[i + 1]);
				break;
			case 'r':
				//max round
				r_on = true;
				round = atoi(argv[i + 1]);
				break;
			}
		};
		default:
			//perror("�������Ϸ�������\n");
			break;
		}
	}
	PageRank PR;
	//��������
	PR.load_data(fpath);
	printf("load data finished!\n");
	PR.HandleDeadEnds();
	printf("HandleDeadEnds finished\n");
	if (mode == 0)
	{
		//naive mode
		PR.WeightInit();
		printf("WeightInit finished\n");
		printf("\nnavie PageRank R Matrix Update\n");
		if (r_on)
			PR.WeightUpdate(round);
		else
			PR.WeightUpdate();
		printf("WeightUpdate end\n");
		PR.WriteNode("weight_naive.txt");
		printf("ȫ��ҳ�ڵ�Ȩ���������\n");
		PR.WriteTop100("Top100_naive.txt");
		printf("Top100ҳ�ڵ�Ȩ���������\n");
	}
	else if (mode == 1)
	{
		PR.Block_Stripe_pre();
		printf("Block_Stripe_pre finished\n");
		if (!PR.WeightInit())
		{
			return 0;
		};
		printf("WeightInit finished\n");
		printf("\nBlock Stripe PageRank R Matrix Update\n");
		if (r_on)
			PR.BS_WeightUpdate(round);
		else
			PR.BS_WeightUpdate();
		printf("BS_WeightUpdate finished\n");
		PR.WriteNode("weight_bs.txt","./R_training/R" + std::to_string(round) + ".dat");
		printf("ȫ��ҳ�ڵ�Ȩ���������\n");
		PR.WriteTop100("Top100_bs.txt", "./R_training/R" + std::to_string(round) + ".dat");
		printf("Top100ҳ�ڵ�Ȩ���������\n");
	}
	else
	{
		perror("the mode param is wrong,please check!\n");
	}
	return 0;
}