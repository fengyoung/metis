#include <iostream>
#include <fstream>
using namespace std; 
#include "Worker.h"
using namespace metis_plat; 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



bool ReadSlaveList(vector<string>& vtrSlaves, const char* sSlaveListFile)
{
	ifstream ifs(sSlaveListFile); 
	if(!ifs.is_open())
		return false; 
	vtrSlaves.clear(); 
	string str; 
	Json::Value json_req_body, json_resp; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		if(str.at(0) == '#')
			continue; 
		if(Worker::Send(str.c_str(), "detect", json_req_body, json_resp))
			vtrSlaves.push_back(str); 
	}
	ifs.close(); 
	return true; 
}


void GetPattCnt(vector<string>& vtrSlaves)
{
	Json::Value json_req_body, json_resp; 
	vector<pair<string, int32_t> > vtr_pattcnt; 

	for(int32_t i = 0; i < (int32_t)vtrSlaves.size(); i++)  
	{
		if(!Worker::Send(vtrSlaves[i].c_str(), "get_patt_cnt", json_req_body, json_resp))
			continue; 
		if(json_resp["ret"].asInt() == _METIS_PLAT_SUCCESS)
			vtr_pattcnt.push_back(pair<string,int32_t>(vtrSlaves[i], json_resp["patts"].asInt()));
	}

	cout<<"--"<<endl; 
	int32_t total_cnt = 0; 
	for(int32_t k = 0; k < (int32_t)vtr_pattcnt.size(); k++) 
	{
		printf("(%d) %s | patts: %d\n", k+1, vtr_pattcnt[k].first.c_str(), vtr_pattcnt[k].second);
		total_cnt += vtr_pattcnt[k].second; 
	}
	printf("SLAVES: %d, PATTS: %d\n", (int32_t)vtr_pattcnt.size(), total_cnt); 
}


void GetStatus(vector<string>& vtrSlaves)
{
	Json::Value json_req_body, json_resp; 
	vector<pair<string, pair<int32_t, bool> > > vtr_status; 
	int32_t patt_cnt; 
	bool is_updating; 

	for(int32_t i = 0; i < (int32_t)vtrSlaves.size(); i++)  
	{
		Worker::Send(vtrSlaves[i].c_str(), "get_patt_cnt", json_req_body, json_resp); 
		patt_cnt = json_resp["patts"].asInt(); 
		Worker::Send(vtrSlaves[i].c_str(), "is_updating", json_req_body, json_resp); 
		is_updating = json_resp["updating"].asBool(); 
		vtr_status.push_back(pair<string, pair<int32_t, bool> >(vtrSlaves[i], pair<int32_t, bool>(patt_cnt, is_updating))); 
	}

	cout<<"--"<<endl; 
	int32_t total_cnt = 0; 
	bool total_updating = false; 
	for(int32_t k = 0; k < (int32_t)vtr_status.size(); k++) 
	{
		if(vtr_status[k].second.second)
		{
			printf("(%d) %s | patts: %d | updating\n", k+1, vtr_status[k].first.c_str(), vtr_status[k].second.first);
			total_updating = true; 
		}
		else
		{
			printf("(%d) %s | patts: %d | waiting\n", k+1, vtr_status[k].first.c_str(), vtr_status[k].second.first);
		}
		total_cnt += vtr_status[k].second.first; 
	}
	if(total_updating)
		printf("SLAVES: %d, PATTS: %d, UPDATING\n", (int32_t)vtr_status.size(), total_cnt); 
	else
		printf("SLAVES: %d, PATTS: %d, WAITING\n", (int32_t)vtr_status.size(), total_cnt); 

}


void Reset(vector<string>& vtrSlaves)
{
	char stmp[128]; 

	while(true)
	{
		cout<<"Do you really want to RESET all metis slaves? (yes/no): "; 
		cin>>stmp;
		if(strcmp(stmp, "yes") == 0)
		{
			Json::Value json_req_body, json_resp; 

			for(int32_t i = 0; i < (int32_t)vtrSlaves.size(); i++)  
			{
				Worker::Send(vtrSlaves[i].c_str(), "release", json_req_body, json_resp); 
				while(json_resp["ret"].asInt() != _METIS_PLAT_SUCCESS) 
					Worker::Send(vtrSlaves[i].c_str(), "release", json_req_body, json_resp); 
				printf("(%d) slave[%s] has been reset.\n", i+1, vtrSlaves[i].c_str()); 	
			}	
			break; 
		}
		else if(strcmp(stmp, "no") == 0)
		{
			break; 
		}
	}
}


int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout<<"usage: metis_assistant <slave_list_file> [--get_patt_cnt]"<<endl; 
		cout<<"                                         [--get_status]"<<endl; 
		cout<<"                                         [--reset]"<<endl; 
		return -1; 	
	}

	vector<string> vtr_slaves; 
	if(!ReadSlaveList(vtr_slaves, argv[1]))
	{
		cout<<"failed to open slave list file "<<argv[1]<<endl; 
		return -2; 
	}
	
	if(strcmp(argv[2], "--get_patt_cnt") == 0)
	{
		GetPattCnt(vtr_slaves); 
	}
	else if(strcmp(argv[2], "--get_status") == 0)
	{
		GetStatus(vtr_slaves); 
	}
	else if(strcmp(argv[2], "--reset") == 0)
	{
		Reset(vtr_slaves); 
	}
	else
	{
		cout<<"unsupported cmd!"<<endl; 
		return -3; 
	}

	return 0; 
}


