APP=summation
CUFILES=summation.cu utils.cu
# Target Tesla K40 and Geforce GTX 980
CFLAGS=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52
LDFLAGS=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52
CUDA_PATH ?=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc
OBJS=$(patsubst %.cu,%.o,$(CUFILES))

all: $(APP)

%.o : %.cu
	$(NVCC) -c $< $(CFLAGS) -o $@

$(APP) : $(OBJS)
	$(NVCC) $^ $(LDFLAGS) -o $@

clean :
	rm -f $(OBJS) $(APP)

# Dependencies
summation.o : summation_kernel.cu utils.h
utils.o : utils.h
