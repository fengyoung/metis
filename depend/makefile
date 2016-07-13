SCONS_PATH = ./scons-2.3.0/
SCONS_LIB_DIR = $(SCONS_PATH)/engine/
JSONCPP_SRC_PATH = ./jsoncpp-src-0.6.0-rc2/
JSONCPP_OUT_PATH = ./jsoncpp/


all:
	tar -xvzf woo.tar.gz
	cd ./woo && make && cd ..
	tar -xvzf scons-2.3.0.tar.gz
	tar -xvzf jsoncpp-src-0.6.0-rc2.tar.gz 
	export SCONS_LIB_DIR=$(SCONS_PATH)/engine/
	cd $(JSONCPP_SRC_PATH) && python ../$(SCONS_PATH)/script/scons platform=linux-gcc && cd ..
	mkdir -p $(JSONCPP_OUT_PATH)/lib
	cp -r $(JSONCPP_SRC_PATH)/include $(JSONCPP_OUT_PATH)
	cp $(JSONCPP_SRC_PATH)/libs/linux-gcc*/*.a $(JSONCPP_OUT_PATH)/lib/libjsoncpp.a

clean:
	rm -rf ./woo
	rm -rf $(SCONS_PATH)
	rm -rf $(JSONCPP_SRC_PATH)
	rm -rf $(JSONCPP_OUT_PATH)
