#ifndef _METIS_PLATFORM_SLAVE_LIST_H 
#define _METIS_PLATFORM_SLAVE_LIST_H 

#include <string>
#include <vector>
using namespace std; 
#include <stdint.h>


namespace metis_plat
{

class SlaveList
{
public:
	SlaveList(); 
	virtual ~SlaveList(); 

	bool Read(const char* sSlaveListFile);

public: 
	vector<string> m_vtrSlaves; 
}; 

}


#endif /* _METIS_PLATFORM_SLAVE_LIST_H */ 


