DEBUG = 0
CXX = g++
ifneq ($(DEBUG),1)
	CFLAGS = -Wall -W -fpic -O1
else
	CFLAGS = -DDEBUG -g -Wall -W -fpic -O1
endif
