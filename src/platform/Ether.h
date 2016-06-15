#ifndef _METIS_PLATFORM_ETHER_H 
#define _METIS_PLATFORM_ETHER_H 


#include <string>
using namespace std; 
#include <stdint.h>


namespace metis_plat
{

class Ether
{
private:
	// 构造函数 & 析构函数
	Ether(); 
	virtual ~Ether(); 

public:
	static bool StringToHostPort(string& strHost, int32_t& nPort, const char* sHostPort); 
	static string HostPortToString(const char* sHost, const int32_t nPort); 
	static string LocalIp(const char* ether = "eth0");
}; 

}


#endif /* _METIS_PLATFORM_ETHER_H */

