#include "head.cuh"

__global__ void kernel2_SharedMem(float* device_A, float* device_B, float* device_C, const int M, const int N, const int K)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = TILE_WIDTH * blockIdx.x + threadIdx.x; //col
	int row = TILE_WIDTH * blockIdx.y + threadIdx.y; //row

	__shared__ float S_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float S_B[TILE_WIDTH][TILE_WIDTH];

	float temp = 0.0f;
	for (int tile_id = 0; tile_id < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile_id) {
		//copy to share memory
		if (row < M && tile_id * TILE_WIDTH + tx < K) {
			S_A[tx][ty] = device_A[row*K + tile_id * TILE_WIDTH + tx];
		}

		if (col < N && tile_id * TILE_WIDTH + ty < K) {
			S_B[tx][ty] = device_B[(tile_id * TILE_WIDTH + ty)*N + col];
		}

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i) {
			temp += S_A[i][ty] * S_B[tx][i];
		}

		__syncthreads();

		if (row < M && col < N) {
			device_C[row*N + col] = temp;
		}
	}


}


extern "C" void withSharedMemKernel(float *device_A, float *device_B, float *device_C, int M, int N, int K)
{
	dim3 block_s(TILE_WIDTH, TILE_WIDTH);
	dim3 grid_s((M + block_s.x - 1) / block_s.x, (N + block_s.y - 1) / block_s.y);
	kernel2_SharedMem << <grid_s, block_s >> > (device_A, device_B, device_C, M, N, K);
	CHECK(cudaGetLastError());

}

