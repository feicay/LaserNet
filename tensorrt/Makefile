OUTNAME=lidar_seg_int8

CC = g++
CUCC =$(CUDA_INSTALL_DIR)/bin/nvcc -m64

TENSORRT_DIR=/usr/local/TensorRT-5.0.2.6
CUDA_INSTALL_DIR=/usr/local/cuda
CUDA_LIBDIR=lib64

LIBPATHS=-L/usr/local/lib -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(TENSORRT_DIR)/lib"
TENSORRT_LIB=-lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime
INCPATHS=-I/usr/local/include -I"$(CUDA_INSTALL_DIR)/include"  -I"$(TENSORRT_DIR)/include" 

COMMON_FLAGS += -Wall -std=c++11  $(INCPATHS) -Wl,-rpath "$(TENSORRT_DIR)/lib"
COMMON_LD_FLAGS += $(LIBPATHS)
COMMON_LIBS = -lcudnn -lcublas -lnvToolsExt -lcudart $(TENSORRT_LIB)

OBJS=make_input.o lasernet.o 
BIN=demo.bin

all:$(OBJS)
	$(CC) $(OBJS) $(COMMON_FLAGS) $(COMMON_LIBS) $(COMMON_LD_FLAGS) -o $(BIN) 

%.o:%.cpp
	$(CC) $(COMMON_FLAGS) $(COMMON_LIBS) $(COMMON_LD_FLAGS) -c $< -o $@

clean:
	rm *.o $(BIN) 