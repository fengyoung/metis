#ifndef _METIS_UTILITY_STRING_ARRAY_H 
#define _METIS_UTILITY_STRING_ARRAY_H 

#include <string> 
#include <vector> 
using namespace std; 
#include <stdint.h> 

namespace metis_uti
{

class StringArray 
{
public: 
	StringArray(const char* sStr, const char* sDilm);   
	virtual ~StringArray(); 
	
	string GetString(const int32_t nIdx) const; 
	int32_t Count() const; 

private: 
	void Decompose(const char* sStr, const char* sDilm); 
	vector<string> m_vtrString;
};  

}

#endif /* _METIS_UTILITY_STRING_ARRAY_H */ 

