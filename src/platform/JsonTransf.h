#ifndef _METIS_PLATFORM_JSON_TRANSF_H 
#define _METIS_PLATFORM_JSON_TRANSF_H 

#include <string>
using namespace std; 
#include "json/json.h"


class JsonTransf
{
private:
	JsonTransf(); 
	virtual ~JsonTransf(); 

public:
	static string JsonCppToString(Json::Value& jsonValue, const bool bStyled = false); 
	static bool StringToJsonCpp(Json::Value& jsonValue, const char* sJsonStr); 
}; 


#endif /* _METIS_PLATFORM_JSON_TRANSF_H */ 


