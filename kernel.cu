#include<iostream>
#include<cstdlib>
#include<assert.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "time.h"

using namespace std;

#define BLOCK_SIZE 16

__global__ void matmul(int *a, int *b, int *c, int n) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	if (row < n && col < n) {
	int temp=0;
		for (int i = 0; i < n; i++) {
		// Matrices are stored in row-major order:
		// M(row, col) = *(M.elements + row * M.width + col)
			temp += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = temp;
	}
}

__host__ void verify(int* a, int* b, int* c, int n) {

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
		int temp=0;
			for (int k = 0; k < n; ++k)
			{
				temp += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = temp;
		}
	}
}

int main() {
	int N = 1 << 10; // left shift to 10 place.
	size_t bytes = N * N * sizeof(int);

	int *a, *b, *c,*h_c;
	int* d_a, * d_b, * d_c;

	cudaMallocHost(&a, bytes);
	cudaMallocHost(&b, bytes);
	cudaMallocHost(&c, bytes);
	cudaMallocHost(&h_c, bytes);

	for (int i=0; i < N * N; ++i) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	float gpu_elapsed_time_ms;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start to count execution time of GPU version
	cudaEventRecord(start, 0);


	cudaMallocManaged(&d_a, bytes);
	cudaMallocManaged(&d_b, bytes);
	cudaMallocManaged(&d_c, bytes);

	cudaMemcpy(d_a, a,bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b,bytes, cudaMemcpyHostToDevice);

	unsigned int block_rows = N + BLOCK_SIZE  / BLOCK_SIZE;
	unsigned int block_cols = N + BLOCK_SIZE  / BLOCK_SIZE;
	dim3 blockDim(block_rows, block_cols);
	dim3 threadDim(BLOCK_SIZE, BLOCK_SIZE);

	//dim3 numthreads(N / blocksize.x, N / blocksize.y);

	printf("Number of threads per block: %d \n\n", threadDim.x * threadDim.y);

	matmul << <blockDim,threadDim >> > (d_a, d_b, d_c, N);
	
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

	printf("Time for matrix multiplication of %d * %d on GPU: %f ms.\n\n", N, N, gpu_elapsed_time_ms);


	clock_t begin = clock();

	verify(a, b, c, N);

	clock_t end = clock();
	double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	printf("Time elapsed on matrix multiplication of %d x %d on CPU: %f ms.\n\n",N, N, time_spent);


	int all_ok = 1;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			//printf("[%d][%d]:%d == [%d][%d]:%d\n", i, j, c[i*N + j], i, j, h_c[i*N + j]);
			if (h_c[i * N + j] != c[i * N + j])
			{
				all_ok = 0;
			}
		}
		//printf("\n");
	}

	// roughly compute speedup
	if (all_ok)
	{
		printf("all results are correct!!!, speedup = %f\n", time_spent / gpu_elapsed_time_ms);
	}
	else
	{
		printf("incorrect results\n");
	}


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}