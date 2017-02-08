#include <iostream>
#include <fstream>
using namespace std; 
#include "StringArray.h"
using namespace metis_uti; 
#include "Worker.h"
#include "SlaveList.h"
using namespace metis_plat; 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void GetPattCnt(vector<string>& vtrSlaves)
{
	Json::Value json_req_body, json_resp; 
	vector<pair<string, int32_t> > vtr_pattcnt; 

	for(int32_t i = 0; i < (int32_t)vtrSlaves.size(); i++)  
	{
		if(!Worker::SendCmd(vtrSlaves[i].c_str(), "get_patt_cnt", json_req_body, json_resp))
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
		Worker::SendCmd(vtrSlaves[i].c_str(), "get_patt_cnt", json_req_body, json_resp); 
		patt_cnt = json_resp["patts"].asInt(); 
		Worker::SendCmd(vtrSlaves[i].c_str(), "is_updating", json_req_body, json_resp); 
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
				Worker::SendCmd(vtrSlaves[i].c_str(), "release", json_req_body, json_resp); 
				while(json_resp["ret"].asInt() != _METIS_PLAT_SUCCESS) 
					Worker::SendCmd(vtrSlaves[i].c_str(), "release", json_req_body, json_resp); 
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


void Test(vector<string>& vtrSlaves)
{
	Pattern patt1, patt2; 
	patt1.FromString("1,0;1,2,3,-1.1,-2.1,-3.1"); 
	patt2.FromString("0,1;-1.2,-2.2,-3.2,1,2,3"); 
	char stream[1024]; 
	int32_t len; 
	Json::Value json_req_body, json_resp; 

	int32_t cnt = 2; 
	Matrix* ws = new Matrix[2];
	ws[0].Create(7, 3);
	ws[0].Init(0.0); 
	ws[1].Create(4, 2);
	ws[1].Init(0.0); 
	

	for(int32_t i = 0; i < (int32_t)vtrSlaves.size(); i++)
	{
		json_req_body.clear(); 
		Worker::SendCmd(vtrSlaves[i].c_str(), "release", json_req_body, json_resp); 
	
		len = patt1.ToStream(stream); 	
		Worker::SendBitStream(vtrSlaves[i].c_str(), "push_patt", stream, len, json_resp); 
		cout<<JsonTransf::JsonCppToString(json_resp)<<endl; 
		
		string str = patt2.ToString(); 
		Worker::SendBitStream(vtrSlaves[i].c_str(), "push_patt_string", str.c_str(), str.length(), json_resp); 
		cout<<JsonTransf::JsonCppToString(json_resp)<<endl; 
		
		len = Matrix::MatsToStream(stream, ws, cnt); 	
		Worker::SendBitStream(vtrSlaves[i].c_str(), "set_weight", stream, len, json_resp); 
		cout<<JsonTransf::JsonCppToString(json_resp)<<endl; 
	
		Worker::GetBatchGrad(vtrSlaves[i].c_str(), stream, len, 1024*100); 
		double error = *((double*)stream); 	
		cout<<"error: "<<error<<endl; 
		Matrix* gs = Matrix::MatsFromStream(cnt, stream + sizeof(double), len); 		
		cout<<"---"<<endl; 		
		for(int32_t i = 0; i < cnt; i++) 
			cout<<gs[i].ToString()<<endl; 	


		Matrix g; 
		Matrix::ParseMatrixFromStream(g, 1, stream + sizeof(double), len - sizeof(double)); 
		cout<<"---"<<endl; 		
		cout<<g.ToString()<<endl; 
		
		Matrix::ParseMatrixFromStream(g, 0, stream + sizeof(double), len - sizeof(double)); 
		cout<<"---"<<endl; 		
		cout<<g.ToString()<<endl; 	
	}
}


int main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout<<"usage: metis_assistant <slave_list_file> [--get_patt_cnt]"<<endl; 
		cout<<"                                         [--get_status]"<<endl; 
		cout<<"                                         [--reset]"<<endl; 
		cout<<"                                         [--test]"<<endl; 
		return -1; 	
	}

	SlaveList slave_list; 
	if(!slave_list.Read(argv[1]))
	{
		cout<<"failed to open slave list file "<<argv[1]<<endl; 
		return -2; 
	}
	
	if(strcmp(argv[2], "--get_patt_cnt") == 0)
	{
		GetPattCnt(slave_list.m_vtrSlaves); 
	}
	else if(strcmp(argv[2], "--get_status") == 0)
	{
		GetStatus(slave_list.m_vtrSlaves); 
	}
	else if(strcmp(argv[2], "--reset") == 0)
	{
		Reset(slave_list.m_vtrSlaves); 
	}
	else if(strcmp(argv[2], "--test") == 0)
	{
		Test(slave_list.m_vtrSlaves); 
	}
	else
	{
		cout<<"unsupported cmd!"<<endl; 
		return -3; 
	}

	return 0; 
}


