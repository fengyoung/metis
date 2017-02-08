// StringArray.h
//
// Definition of class StringArray
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_UTILITY_STRING_ARRAY_H 
#define _METIS_UTILITY_STRING_ARRAY_H 

#include <string> 
#include <vector> 
using namespace std; 
#include <stdint.h> 

namespace metis_uti
{

// CLASS
//	StringArray - string array 
class StringArray 
{
public:
	// Construction & Destruction
	//
	// DESCRIPTION
	//	sStr: input string which takes some splitter for each segment
	//  sSep: splitter
	StringArray(const char* sStr, const char* sSep);   
	virtual ~StringArray(); 

	// NAME
	//	GetString - get string segment
	//
	// DESCRIPTION
	//	nIdx: index of string segment
	//
	// RETURN 
	//	indicated string segment
	string GetString(const int32_t nIdx) const;

	// NAME
	//	Count - get count of string segments
	//
	// RETURN
	//	segments count
	int32_t Count() const; 

private:
	// NAME
	//	Decompose - decompose the input string according to the splitter
	//
	// DESCRIPTION
	//	sStr: input string
	//	sSep: splitter
	void Decompose(const char* sStr, const char* sSep); 
	
private: 
	vector<string> m_vtrString;
};  

}

#endif /* _METIS_UTILITY_STRING_ARRAY_H */ 

