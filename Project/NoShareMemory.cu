#include "head.cuh"

__global__ void kernel(float* device_A, float* device_B, float* device_C, const int M, const int N, const int K)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < M & col < N)
	{
		float temp = 0.0f;
		for (int i = 0; i < K; i++) {
			temp += device_A[row*K + i] * device_B[N*i + col];
		}
		device_C[row*N + col] = temp;

	}
}

extern "C" void noShareMemKernel(float *device_A, float *device_B, float *device_C, int M, int N, int K)
{
	dim3 block(32, 4);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	kernel<<<grid, block >>> (device_A, device_B, device_C, M, N, K);
	CHECK(cudaGetLastError());
}