#include "ThreadRWLock.h"
using namespace metis_plat; 
#include <string>
#include <iostream>
using namespace std; 
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

ThreadRWLock::ThreadRWLock()
{
	int rc = -1;

	rc = pthread_rwlock_init(&_rw_lock, NULL); 
	assert(rc == 0);

	if(rc != 0)
		throw "pthread_rwlock_init error"; 
}


ThreadRWLock::~ThreadRWLock()
{
	int rc = -1;
	rc = pthread_rwlock_destroy(&_rw_lock); 
	if(rc != 0)
	{
		cerr << "pthread_mutex_destroy error:" << string(strerror(rc)) << endl;
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void ThreadRWLock::RdLock() const
{
	int rc = pthread_rwlock_rdlock(&_rw_lock); 
	if(rc != 0)
	{
		throw "pthread_rwlock_rdlock error!"; 
	} 
}


void ThreadRWLock::WrLock() const
{
	int rc = pthread_rwlock_wrlock(&_rw_lock); 
	if(rc != 0)
	{
		throw "pthread_rwlock_wrlock error!"; 
	} 
}


bool ThreadRWLock::TryRdLock() const
{
	int rc = pthread_rwlock_tryrdlock(&_rw_lock); 
	if(rc != 0)
	{
		throw "pthread_rwlock_tryrdlock error!"; 
	} 
	return (rc == 0);
}


bool ThreadRWLock::TryWrLock() const
{
	int rc = pthread_rwlock_trywrlock(&_rw_lock); 
	if(rc != 0)
	{
		throw "pthread_rwlock_trywrlock error!"; 
	} 
	return (rc == 0);
}


void ThreadRWLock::Unlock() const
{
	int rc = pthread_rwlock_unlock(&_rw_lock);
	if(rc != 0)
	{
		throw "pthread_rwlock_unlock error!";
	} 
}



