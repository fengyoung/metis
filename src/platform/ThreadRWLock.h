#ifndef _METIS_PLATFORM_THREAD_RW_LOCK_H 
#define _METIS_PLATFORM_THREAD_RW_LOCK_H 

#include <pthread.h>
#include <stdint.h>

namespace metis_plat
{

class ThreadRWLock
{ 
public:
	ThreadRWLock();
	virtual ~ThreadRWLock();

	void RdLock() const;
	void WrLock() const;

	bool TryRdLock() const;
	bool TryWrLock() const;

	void Unlock() const;

protected:
	mutable pthread_rwlock_t _rw_lock;	
};

}

#endif /* _METIS_PLATFORM_THREAD_RW_LOCK_H */  
 
