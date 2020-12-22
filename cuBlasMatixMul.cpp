#include "head.cuh"
#include <cublas_v2.h>
#include <helper_cuda.h>

extern "C" void cuBlasMatrixMul(float *h_A, float *h_B, float *h_CUBLAS, int M, int N, int K)
{
	int SIZE_A = M * K * sizeof(float);
	int SIZE_B = K * N * sizeof(float);
	int SIZE_C = M * N * sizeof(float);
	float *d_A, *d_B, *d_C;

	checkCudaErrors(cudaMalloc((void **)&d_A, SIZE_A));
	checkCudaErrors(cudaMalloc((void **)&d_B, SIZE_B));
	checkCudaErrors(cudaMalloc((void **)&d_C, SIZE_C));
	checkCudaErrors(cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(M/block.x, N/block.y);

	//CUBLAS
	printf("Computing with CUBLAS...");
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasHandle_t handle;
	cudaEvent_t start;
	cudaEvent_t stop;
	checkCudaErrors(cublasCreate(&handle));
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	int nIter = 30;
	for (int i = 0; i < nIter; i++)
	{
		checkCudaErrors(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
	}
	
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float elapsedTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	float time = elapsedTime / nIter;
	double flops = 2.0 * (double)M * (double)N* (double)K;
	double gigaFlops = (flops * 1.0e-9f) / (time / 1000.0f);
	printf("CUBLAS Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",	gigaFlops,time,flops);

	checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, SIZE_C, cudaMemcpyDeviceToHost));
	checkCudaErrors(cublasDestroy(handle));


	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
}