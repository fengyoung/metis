#include <iostream>
using namespace std; 
#include "MasterTrainer.h"
#include "MasterTrainer_Perceptron.h"
#include "MasterTrainer_MLP.h"
using namespace metis_plat; 
#include <stdio.h>
#include <string.h>


class CmdParams
{
public: 
	CmdParams()
	{
		incr = 0;
	}
	virtual ~CmdParams()
	{
	}
	void Clear()
	{
		out_model.clear();
		val_patt_file.clear(); 
		algo.clear(); 
		slave_list_file.clear();
		conf_file.clear(); 
		incr = 0;
		pri_model.clear(); 
	}
	bool IsCompleted()
	{
		if(out_model.empty())
			return false; 
		if(algo.empty())
			return false; 
		if(slave_list_file.empty())
			return false; 
		if(incr <= 0 && conf_file.empty())
			return false; 
		if(incr > 0 && pri_model.empty())
			return false; 
		return true; 
	}

public: 
	string out_model;
	string val_patt_file;	// patterns file for validation
	string algo; 
	string slave_list_file;
	string conf_file; 
	int32_t incr;
	string pri_model; 
}; 



void PrintHelp(const char* sCmd)
{
	printf("Usage: %s [options]\n", sCmd); 
	printf("Master of metis, which submits the training job.\n"); 

	printf("\n");
	printf("Options: \n"); 
	printf("\t--slave <file>\n");
	printf("\t\tslave list file consists nodes' info, REQUIRED\n"); 
	
	printf("\t--val <files>\n");
	printf("\t\tpattern file for validation\n"); 

	printf("\t--algo <txt>\n");
	printf("\t\tset training algorithm, REQUIRED.\n"); 
	printf("\t\t\"percep\", Linear Perceptron.\n"); 
	printf("\t\t\"mlp\", Multi-Layer Preceptron Neural Network.\n");

	printf("\t--out <out_model>\n");
	printf("\t\toutput model file.\n"); 

	printf("\t--incr <int>\n");
	printf("\t\ttraining the model incrementally, default 0 (not incremental), 1 for incremental training.\n"); 
	
	printf("\t--conf <file>\n");
	printf("\t\tconfig file which is needed by correspondng algorithm, it is REQUIRED when incr=0\n"); 

	printf("\t--model <file>\n");
	printf("\t\tprimary model for incremental training, it is REQUIRED when incr=1\n"); 

	printf("\n");
}



bool ParseCmdLine(CmdParams& cmdParams, int32_t nArgc, char** ssArgv)
{
	cmdParams.Clear(); 
	for(int32_t i = 1; i < nArgc; i++) 
	{
		if(ssArgv[i][0] == '-')
		{
			if(i + 1 >= nArgc)
				return false; 
			if(ssArgv[i+1][0] == '-')
				return false; 
		}
		if(strcmp(ssArgv[i], "--out") == 0)
			cmdParams.out_model = ssArgv[i+1];
		else if(strcmp(ssArgv[i], "--algo") == 0)
			cmdParams.algo = ssArgv[i+1]; 
		else if(strcmp(ssArgv[i], "--val") == 0)
			cmdParams.val_patt_file = ssArgv[i+1]; 
		else if(strcmp(ssArgv[i], "--slave") == 0)
			cmdParams.slave_list_file = ssArgv[i+1]; 
		else if(strcmp(ssArgv[i], "--conf") == 0)
			cmdParams.conf_file = ssArgv[i+1]; 
		else if(strcmp(ssArgv[i], "--incr") == 0)
			sscanf(ssArgv[i+1], "%d", &cmdParams.incr); 
		else if(strcmp(ssArgv[i], "--model") == 0)
			cmdParams.pri_model = ssArgv[i+1]; 
	}
	return cmdParams.IsCompleted(); 
}


int main(int argc, char** argv)
{
	CmdParams cmd_params; 
	if(!ParseCmdLine(cmd_params, argc, argv))
	{
		PrintHelp(argv[0]);
		return -1; 
	}
	
	printf("===================\n");
	printf("=  METIS - Train  =\n"); 
	printf("===================\n");
	
	MasterTrainer* p_master_trainer = NULL; 

	if(cmd_params.algo == "percep")
		p_master_trainer = new MasterTrainer_Perceptron(); 
	else if(cmd_params.algo == "mlp")
		p_master_trainer = new MasterTrainer_MLP(); 
	else
	{
		printf("ERROR: unsupported algorithm\n"); 
		return -2; 
	}
	
	// 载入slave列表, 并检测
	if(!p_master_trainer->LoadSlaves(cmd_params.slave_list_file.c_str()))
	{
		delete p_master_trainer; 
		return -3; 
	}

	// 查询训练pattern信息
	PattsInfoT patts_info_t = p_master_trainer->QueryPattsInfo(); 
	if(patts_info_t.x_dims < 0 || patts_info_t.y_dims < 0 || patts_info_t.patts <= 0)
	{
		delete p_master_trainer; 
		printf("ERROR: no pattern!\n"); 
		return -3; 
	}
	printf("Patts: %d; Input: %d; Output: %d\n", patts_info_t.patts, patts_info_t.x_dims, patts_info_t.y_dims);

	// 释放所有slave上的模型
	p_master_trainer->ReleaseWeightInSlaves(); 
	
	// 载入用于验证的样本
	if(!cmd_params.val_patt_file.empty())	
	{
		int32_t cnt = p_master_trainer->LoadValidatedPatts(cmd_params.val_patt_file.c_str()); 
		if(cnt <= 0)
			printf("WARNING: failed to load validated patterns form %s\n", cmd_params.val_patt_file.c_str()); 
		else	
			printf("Load %d patterns for validation\n", cnt); 
	}

	// 初始化模型
	if(cmd_params.incr <= 0)
	{
		if(!p_master_trainer->InitByLearningConfig(cmd_params.conf_file.c_str()))
		{
			delete p_master_trainer; 
			printf("ERROR: failed to initalize the Trainer by config, exit!\n"); 
			return -5; 
		}
	}
	else
	{
		if(!p_master_trainer->InitByModel(cmd_params.pri_model.c_str()))
		{
			delete p_master_trainer; 
			printf("ERROR: failed to initalize the Trainer by history model, exit!\n"); 
			return -5; 
		}
	}
	
	cout<<"--"<<endl; 
	p_master_trainer->PrintParams(cout); 

	// 执行训练
	p_master_trainer->Train(cmd_params.out_model.c_str()); 
	
	// 释放所有slave上的模型
	p_master_trainer->ReleaseWeightInSlaves(); 

	delete p_master_trainer; 
	return 0; 
}


