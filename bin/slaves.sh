#! /bin/sh


if [ $# -lt 1 ]
then
	echo "Usage: slaves.sh [--start <serv_config> <slave_list>]"
	echo "                 [--kill <slave_list>]"
	exit -1
fi


if [ "$1" == "--start" ] 
then
	if [ $# -ne 3 ]
	then
		echo "Usage: slaves.sh --start <serv_config> <slave_list>"
		exit -2	
	fi
	SERV_CONFIG_FILE=$2
	SLAVE_LIST=$3
elif [ "$1" == "--kill" ] 
then
	if [ $# -ne 2 ]
	then
		echo "Usage: slaves.sh --kill <slave_list>"
		exit -2	
	fi
	SLAVE_LIST=$2
else
	echo "unsupported command, exit!"
	exit -2
fi


LOCAL_IPS=(`ifconfig | grep "inet" | grep -v "127.0.0.1" | grep -v "inet6" | awk '{print $2}' | tr -d "addr:"`) 
PORTS=()
while read LINE
do
	if [ "${LINE:0:1}" == "#" ]
	then
		continue
	else
		LEN=`echo ${LINE} | awk '{print length($0)}'`
		if [ "${LEN}" == "0" ]	
		then
			continue
		fi

		TARGET_IP=`echo ${LINE} | awk -F ":" '{print $1}'`
		TARGET_PORT=`echo ${LINE} | awk -F ":" '{print $2}'`

		for IP in ${LOCAL_IPS[@]}
		do
			if [ "${IP}" == "${TARGET_IP}" ]	
			then
				PORTS[${#PORTS[@]}]=${TARGET_PORT} 
			fi
		done
	fi
done < ${SLAVE_LIST}





if [ "$1" == "--start" ] 
then
	for PORT in ${PORTS[@]}
	do
		echo "./metis_slave ${PORT} ${SERV_CONFIG_FILE}"
		./metis_slave ${PORT} ${SERV_CONFIG_FILE} &
		usleep 500000
	done
elif [ "$1" == "--kill" ] 
then
	for PORT in ${PORTS[@]}
	do
		FLAG=`ps -ef | grep metis_slave | grep -v grep | grep -v vim | grep ${PORT} | wc -l`
		if ((FLAG==0))
		then
			continue	
		fi
		PID=`ps -ef | grep metis_slave | grep -v grep | grep -v vim | grep ${PORT} | awk '{print $2}'`
		echo "kill -9 ${PID}, in port ${PORT}"
		kill -9 ${PID}
	done
else
	echo "unsupported command, exit!"
	exit -2
fi



exit 0


