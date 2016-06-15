#! /bin/sh


DATA_PATH=../data/
PATT_TAGS=("01_linear" "01_nonlinear")
PERCEPTRON_CONFIG=../config/perceptron_learning.conf
MLP_CONFIG=../config/mlp_learning.conf
SLAVE_LIST=../config/slaves.list
SERV_CONF=../config/server.conf

sh slaves.sh --kill ${SLAVE_LIST}
sh slaves.sh --start ${SERV_CONF} ${SLAVE_LIST}
echo ""
echo ""


for PATT_TAG in ${PATT_TAGS[@]}
do
	PATT_FILE=${DATA_PATH}/${PATT_TAG}.train
	
	MODEL_FILE=./${PATT_TAG}.mlp
	echo "./mlp_example --train ${MLP_CONFIG} ${PATT_FILE} ${MODEL_FILE}"
	./mlp_example --train ${MLP_CONFIG} ${PATT_FILE} ${MODEL_FILE}
	echo ""
	echo ""


	MODEL_FILE=./${PATT_TAG}.mlp.parallel
	echo "./metis_master ${SLAVE_LIST} ${PATT_FILE} --new_mlp ${MLP_CONFIG} ${MODEL_FILE}"
	./metis_master ${SLAVE_LIST} ${PATT_FILE} --new_mlp ${MLP_CONFIG} ${MODEL_FILE}
	echo ""
	echo ""

	
	MODEL_FILE=./${PATT_TAG}.percep
	echo "./perceptron_example --train ${PERCEPTRON_CONFIG} ${PATT_FILE} ${MODEL_FILE}"
	./perceptron_example --train ${PERCEPTRON_CONFIG} ${PATT_FILE} ${MODEL_FILE}
	echo ""
	echo ""


	MODEL_FILE=./${PATT_TAG}.percep.parallel
	echo "./metis_master ${SLAVE_LIST} ${PATT_FILE} --new_percep ${PERCEPTRON_CONFIG} ${MODEL_FILE}"
	./metis_master ${SLAVE_LIST} ${PATT_FILE} --new_percep ${PERCEPTRON_CONFIG} ${MODEL_FILE}
	echo ""
	echo ""
done


