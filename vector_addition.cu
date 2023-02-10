#include<stdio.h>
#include<stdlib.h>
#include"cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vecadd(int* c, const int* b, const int* a) {

	int id = blockIdx.x*blockDim.x+threadIdx.x;

	c[id] = a[id] + b[id];
}

int main() {
	const int s = 5;
	const int a[5] = { 1,2,3,4,5 };
	const int b[5] = { 4,5,6,7,8 };
	int c[5]={0};

	int* d_a=0;
	int* d_b=0;
	int* d_c=0;

	cudaMalloc((void**)&d_a, 5 * sizeof(int));
	cudaMalloc((void**)&d_b, 5 * sizeof(int));
	cudaMalloc((void**)&d_c, 5 * sizeof(int));

	cudaMemcpy(d_a, a, 5 * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, 5 * sizeof(int),cudaMemcpyHostToDevice);

	vecadd << <1, 5 >> > (d_c, d_b, d_a);

	cudaMemcpy(c, d_c, 5 * sizeof(int),cudaMemcpyDeviceToHost);

	for(int i=0;i<5;i++)
		printf("%d ", c[i]);
	

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

