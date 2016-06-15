include make.inc

SRC_UTILITY_PATH = ./utility/
SRC_NN_PATH = ./neural_network/
SRC_PREDICTION_PATH = ./prediction/
SRC_PLATFORM_PATH = ./platform/

OBJS = $(SRC_UTILITY_PATH)/StringArray.o
OBJS += $(SRC_UTILITY_PATH)/Config.o
OBJS += $(SRC_UTILITY_PATH)/Timer.o
OBJS += $(SRC_UTILITY_PATH)/TimeFmt.o
OBJS += $(SRC_UTILITY_PATH)/Random.o
OBJS += $(SRC_UTILITY_PATH)/RocAnalyzer.o
OBJS += $(SRC_NN_PATH)/TypeDefs.o
OBJS += $(SRC_NN_PATH)/Pattern.o
OBJS += $(SRC_NN_PATH)/Matrix.o
OBJS += $(SRC_NN_PATH)/Activation.o
OBJS += $(SRC_NN_PATH)/Perceptron.o
OBJS += $(SRC_NN_PATH)/MLP.o
OBJS += $(SRC_PREDICTION_PATH)/Model.o
OBJS += $(SRC_PREDICTION_PATH)/Model_Perceptron.o
OBJS += $(SRC_PREDICTION_PATH)/Model_MLP.o

PLAT_OBJS = $(SRC_PLATFORM_PATH)/Ether.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/ServConf.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/JsonTransf.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/ThreadRWLock.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/SubMLP.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/SubPerceptron.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/Worker.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/SlaveWorker.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/SlaveTrainer.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/MasterTrainer.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/MasterTrainer_Perceptron.o
PLAT_OBJS += $(SRC_PLATFORM_PATH)/MasterTrainer_MLP.o


LIB_OUT_PATH = ../lib/
LIB_OUT = $(LIB_OUT_PATH)/libmetis.a
INC_OUT_PATH = ../include/

SRC_EXAMPLE_PATH = ./example/
BIN_PATH = ../bin/
EXAMPLES = $(BIN_PATH)/perceptron_example
EXAMPLES += $(BIN_PATH)/mlp_example
EXAMPLES += $(BIN_PATH)/predict_example 
#EXAMPLES += $(BIN_PATH)/predict_avg_example 

PLAT_BIN = $(BIN_PATH)/metis_slave
PLAT_BIN += $(BIN_PATH)/metis_master
PLAT_BIN += $(BIN_PATH)/metis_assistant


.PHONY: all

all: $(LIB_OUT) $(EXAMPLES) $(PLAT_BIN)

$(LIB_OUT): $(OBJS)
	mkdir -p $(LIB_OUT_PATH)
	ar -rv $@ $(OBJS)
	mkdir -p $(INC_OUT_PATH)/utility/
	cp $(SRC_UTILITY_PATH)/*.h $(INC_OUT_PATH)/utility/
	mkdir -p $(INC_OUT_PATH)/neural_network/
	cp $(SRC_NN_PATH)/*.h $(INC_OUT_PATH)/neural_network/
	mkdir -p $(INC_OUT_PATH)/prediction/
	cp $(SRC_PREDICTION_PATH)/*.h $(INC_OUT_PATH)/prediction/
	cp ./*.h $(INC_OUT_PATH)
	
$(BIN_PATH)/perceptron_example: $(SRC_EXAMPLE_PATH)/perceptron_example.o
	$(CXX) $(CFLAGS) -o $@ $< -L$(LIB_OUT_PATH) -lmetis

$(BIN_PATH)/mlp_example: $(SRC_EXAMPLE_PATH)/mlp_example.o
	$(CXX) $(CFLAGS) -o $@ $< -L$(LIB_OUT_PATH) -lmetis

$(BIN_PATH)/predict_example: $(SRC_EXAMPLE_PATH)/predict_example.o
	$(CXX) $(CFLAGS) -o $@ $< -L$(LIB_OUT_PATH) -lmetis

#$(BIN_PATH)/predict_avg_example: $(SRC_EXAMPLE_PATH)/predict_avg_example.o
#	$(CXX) $(CFLAGS) -o $@ $< -L$(LIB_OUT_PATH) -lmetis

$(BIN_PATH)/metis_slave: $(SRC_PLATFORM_PATH)/metis_slave.o $(PLAT_OBJS)
	$(CXX) $(CFLAGS) -o $@ $< $(PLAT_OBJS) -L$(LIB_OUT_PATH) -L../depend/woo/lib/ -L../depend/jsoncpp/lib/ -lmetis -lwoo -ljsoncpp -pthread

$(BIN_PATH)/metis_master: $(SRC_PLATFORM_PATH)/metis_master.o $(PLAT_OBJS)
	$(CXX) $(CFLAGS) -o $@ $< $(PLAT_OBJS) -L$(LIB_OUT_PATH) -L../depend/woo/lib/ -L../depend/jsoncpp/lib/ -lmetis -lwoo -ljsoncpp -pthread

$(BIN_PATH)/metis_assistant: $(SRC_PLATFORM_PATH)/metis_assistant.o $(PLAT_OBJS)
	$(CXX) $(CFLAGS) -o $@ $< $(PLAT_OBJS) -L$(LIB_OUT_PATH) -L../depend/woo/lib/ -L../depend/jsoncpp/lib/ -lmetis -lwoo -ljsoncpp -pthread
	
	
.SUFFIXES: .o .cpp .h

.cpp.o:
	$(CXX) $(CFLAGS) -o $@ -c $< -I$(SRC_UTILITY_PATH) -I$(SRC_NN_PATH) -I$(INC_OUT_PATH) -I$(SRC_PREDICTION_PATH) -I../depend/woo/include/ -I../depend/jsoncpp/include/

.PHONY: clean

clean: 
	rm -rf $(LIB_OUT_PATH)
	rm -rf $(INC_OUT_PATH)
	rm -rf $(OBJS)
	rm -rf $(PLAT_OBJS)
	rm -rf $(SRC_EXAMPLE_PATH)/*.o
	rm -rf $(SRC_PLATFORM_PATH)/*.o
	rm -rf *.o
	rm -rf $(EXAMPLES)
	rm -rf $(PLAT_BIN)


