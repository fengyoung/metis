// TimeFmt.h
//
// Definition of class TimeFmt
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_UTILITY_TIME_FMT_H 
#define _METIS_UTILITY_TIME_FMT_H 

#include <string>
using namespace std;
#include <stdint.h>

namespace metis_uti
{

// Format type of time string
enum ETimeFmtType
{
	_TIME_FMT_STD,		// standard format "YYYY-MM-DD hh:mm:ss"
	_TIME_FMT_NOBLANK,	// no blank format "YYYY-MM-DD_hh:mm:ss"
	_TIME_FMT_COMPACT	// narrow format "YYYYMMDDhhmmss"
};


// Format type of date string
enum EDateFmtType
{
	_DATE_FMT_STD,		// standard format "YYYY-MM-DD"
	_DATE_FMT_COMPACT	// narrow format "YYYYMMDD"
};


// CLASS
//	TimeFmt - time format convertor 
//
// DESCRIPTION
//	This class is used for converting between timestamp in integer and time string in some string format
//
class TimeFmt
{
private:
	// Construction & Destruction
	TimeFmt();
	virtual ~TimeFmt();

public:
	// NAME
	//	CurTimeStamp - get current timestamp which is accurate to second
	//
	// RETURN
	//	current timestamp in 32 bits integer
	static uint32_t CurTimeStamp();

	// NAME
	//	CurTime_asStr - get current timestamp which is accurate to second, and convert it to string format
	//
	// DESCRIPTION
	//	eOutFmt: format type of time string
	//
	// RETURN
	//	time in string format
	static string CurTime_asStr(const ETimeFmtType eOutFmt = _TIME_FMT_STD);

	// NAME
	//	TimeConv_Uint32ToStr - convert timestamp to string format
	// 
	// DESCRIPTION
	//	unTimeStamp: timestamp in 32 bits integer
	//	eOutFmt: format type of time string
	//
	// RETURN
	//	time in string format
	static string TimeConv_Uint32ToStr(const uint32_t unTimeStamp, const ETimeFmtType eOutFmt = _TIME_FMT_STD);
	
	// NAME
	//	TimeConv_StrToUint32 - convert time string to timestamp
	//
	// DESCRIPTION
	//	sTime: time string
	//	eInFmt: format type of time string
	//
	// RETURN
	//	timestamp in 32 bits integer
	static uint32_t TimeConv_StrToUint32(const char* sTime, const ETimeFmtType eInFmt = _TIME_FMT_STD);

	// NAME
	//	DateConv_Uint32ToStr - convert timestamp to date format
	// 
	// DESCRIPTION
	//	unTimeStamp: timestamp in 32 bits integer
	//	eOutFmt: format type of time string
	// 
	// RETURN 
	//	date string	
	static string DateConv_Uint32ToStr(const uint32_t unTimeStamp, const EDateFmtType eOutFmt = _DATE_FMT_STD);
	
	// NAME
	//	DateConv_StrToUint32 - convert date string to timestamp
	// 
	// DESCRIPTION
	//	sDate: date string
	//
	// RETURN 
	//	timestamp in 32 bits integer
	static uint32_t DateConv_StrToUint32(const char* sDate);
};

}


#endif /* _METIS_UTILITY_TIME_FMT_H */

