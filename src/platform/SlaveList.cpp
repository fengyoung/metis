#include "SlaveList.h"
#include "Worker.h"
#include "JsonTransf.h"
using namespace metis_plat; 
#include "StringArray.h"
using namespace metis_uti; 
#include <fstream>
using namespace std; 


SlaveList::SlaveList()
{
}


SlaveList::~SlaveList()
{
}


bool SlaveList::Read(const char* sSlaveListFile)
{
	ifstream ifs(sSlaveListFile); 
	if(!ifs.is_open())
		return false; 
	m_vtrSlaves.clear(); 
	string str; 
	int32_t port, cnt; 
	char s_ip_port[64];  
	Json::Value json_req_body, json_resp; 
	
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		if(str.at(0) == '#')
			continue; 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			continue;
		StringArray ar2(ar.GetString(1).c_str(), "_");
		if(ar2.Count() != 2)
			continue;  
		sscanf(ar2.GetString(0).c_str(), "%d", &port);
		sscanf(ar2.GetString(1).c_str(), "%d", &cnt);
		
		for(int32_t i = 0; i < cnt; i++) 
		{
			sprintf(s_ip_port, "%s:%d", ar.GetString(0).c_str(), port);
			if(Worker::SendCmd(s_ip_port, "detect", json_req_body, json_resp))
				m_vtrSlaves.push_back(s_ip_port); 
			port++; 
		}
	}
	ifs.close(); 
	return true; 
}


