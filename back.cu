
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < data_size) {
		data_out[i] += ((i % 2 == 0) ? 1 : - 1) / (i + 1.0f);
	}
}

__global__ void reduce(float data_size, float * data_out, float * data_block) {
	extern __shared__ float sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	sdata[tid] = (i < data_size) ? data_out[i] : 0;
	__syncthreads();

	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if(index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if(tid == 0) data_block[blockIdx.x] = sdata[0];

}


