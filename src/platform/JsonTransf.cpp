#include "JsonTransf.h"

/////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

JsonTransf::JsonTransf()
{
}


JsonTransf::~JsonTransf()
{
}


/////////////////////////////////////////////////////////////////////////////////
// Operations 

string JsonTransf::JsonCppToString(Json::Value& jsonValue, const bool bStyled)
{
	string str; 
	if(bStyled)
	{
		Json::StyledWriter writer;
		str = writer.write(jsonValue);
	}
	else
	{
		Json::FastWriter writer;
		str = writer.write(jsonValue);
		str.erase(str.length()-1, 1);
	}
	return str;
}


bool JsonTransf::StringToJsonCpp(Json::Value& jsonValue, const char* sJsonStr)
{
	if(!sJsonStr)
		return false; 

	Json::Reader reader(Json::Features::strictMode()); 
	try
	{
		return reader.parse(sJsonStr, jsonValue); 
	}
	catch(...)
	{
		return false; 		
	}
}


