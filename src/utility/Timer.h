// Timer.h
//
// Definition of class Timer 
//
// AUTHOR
//	fengyoung (fengyoung82@sina.cn)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _METIS_UTILITY_TIMER_H 
#define _METIS_UTILITY_TIMER_H 

#include <string>
#include <vector>
using namespace std;
#include <stdint.h>
#include <sys/time.h>



namespace metis_uti
{

// CLASS
//	Timer - timekeeper
class Timer
{
public:
	// Construction & Destruction
	Timer();
	Timer(const Timer& timer);
	virtual ~Timer();

	// Reload assignment
	Timer& operator = (const Timer& timer);

	// NAME
	//	Clear - clear current timer
	void Clear(); 

	// NAME
	//  Start / Stop - Start or stop timer
	void Start();
	void Stop();

	// NAME
	//	SegCnt - get count of time segments
	//
	// RETURN
	//	time segments count
	uint32_t SegCnt(); 

	// NAME
	//	GetLast_asUSec / GetLast_asMSec / GetLast_asSec - get last time segment in micro-sec, milli-sec or sec
	//
	// RETURN
	//	last time segment
	int32_t GetLast_asUSec();
	float GetLast_asMSec();
	float GetLast_asSec();

	// NAME
	//  Get_asUSec / Get_asMSec / Get_asSec - get indicated time segment in micro-sec, milli-sec or sec
	//
	// DESCRIPTION
	//	seg: the index of time segment
	//
	// RETURN
	//  indicated time segment
	int32_t Get_asUSec(const uint32_t seg);
	float Get_asMSec(const uint32_t seg);
	float Get_asSec(const uint32_t seg);

	// NAME 
	//	GetTotal_asUSec / GetTotal_asMSec / GetTotal_asSec - get total time in micro-sec, milli-sec or sec
	//
	// RETURN
	//	total time
	int32_t GetTotal_asUSec(); 
	float GetTotal_asMSec(); 
	float GetTotal_asSec(); 

private: 
	timeval m_laststart;
	bool m_bPingPang;
	vector<pair<uint32_t, uint32_t> > m_vtrSeg;
};


}


#endif /* _METIS_UTILITY_TIMER_H */

