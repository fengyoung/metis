#include <iostream>
using namespace std; 
#include "MasterTrainer.h"
#include "MasterTrainer_Perceptron.h"
using namespace metis_plat; 
#include <stdio.h>
#include <string.h>


class CmdParams
{
public: 
	CmdParams()
	{
		threads = _DEFAULT_MAX_THREADS;
	}
	virtual ~CmdParams()
	{
	}
	void Clear()
	{
		vtr_patt_file.clear();
		slave_list_file.clear();
		threads = _DEFAULT_MAX_THREADS;
	}
	bool IsCompleted()
	{
		if(vtr_patt_file.empty())
			return false; 
		if(slave_list_file.empty())
			return false; 
		return true; 
	}

public: 
	vector<string> vtr_patt_file;
	string slave_list_file;
	int32_t threads;
}; 



void PrintHelp(const char* sCmd)
{
	printf("Usage: %s [options]\n", sCmd); 
	printf("Master of metis, which partition training patterns.\n"); 

	printf("\n");
	printf("Options: \n"); 
	printf("\t--patt <files...>\n");
	printf("\t\tpattern files for training, REQUIRED\n"); 

	printf("\t--slave <file>\n");
	printf("\t\tslave list file consists nodes' info, REQUIRED\n"); 
	
	printf("\t--threads <int>\n");
	printf("\t\tpartition thread numbers, default 32\n"); 

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
		if(strcmp(ssArgv[i], "--patt") == 0)
		{
			for(int32_t j = i + 1; j < nArgc && ssArgv[j][0] != '-'; j++) 
				cmdParams.vtr_patt_file.push_back(ssArgv[j]); 
		}
		else if(strcmp(ssArgv[i], "--slave") == 0)
			cmdParams.slave_list_file = ssArgv[i+1]; 
		else if(strcmp(ssArgv[i], "--threads") == 0)
			sscanf(ssArgv[i+1], "%d", &cmdParams.threads); 
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
	
	printf("=======================\n");
	printf("=  METIS - Partition  =\n");
	printf("=======================\n");

	MasterTrainer_Perceptron master_trainer; 
	
	// 载入slave列表, 并检测
	if(!master_trainer.LoadSlaves(cmd_params.slave_list_file.c_str()))
		return -2; 
	// 释放所有slave资源
	master_trainer.ReleaseSlaves(); 

	// mapping训练样本
	if(master_trainer.PattsMapping(cmd_params.vtr_patt_file, cmd_params.threads) <= 0)
		return -3; 

	PattsInfoT patts_info = master_trainer.MasterTrainer::QueryPattsInfo();
	printf("-----\n"); 
	printf("x_size: %d, y_size: %d\n", patts_info.x_dims, patts_info.y_dims); 
	printf("patterns: %d\n", patts_info.patts); 
	
	return 0; 
}


