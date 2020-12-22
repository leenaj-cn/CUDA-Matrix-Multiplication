#include "head.cuh"


extern "C" void cudaMatrixMul(float *h_a, float *h_b, float *h_c, int M, int N, int K)
{
	int SIZE_A = M * K * sizeof(float);
	int SIZE_B = K * N * sizeof(float);
	int SIZE_C = M * N * sizeof(float);
	//device data
	float* device_A=0;
	float* device_B=0;
	float* device_C=0;
	cudaSetDevice(0);
	//device memory
	CHECK(cudaMalloc((void**)&device_A, SIZE_A));
	CHECK(cudaMalloc((void**)&device_B, SIZE_B));
	CHECK(cudaMalloc((void**)&device_C, SIZE_C));
	CHECK(cudaMemcpy(device_A, h_a, SIZE_A, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(device_B, h_b, SIZE_B, cudaMemcpyHostToDevice));

	//without share memory~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start); //can not call by CHECK() used for WDDM mode GPU

	//noShareMemKernel(device_A, device_B, device_C, M, N, K);

	CHECK(cudaDeviceSynchronize());
	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("no ShareMem Kernel time=%f ms\n\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	//double flops_cuda = 2.0 * (double)M * (double)N* (double)K;
	//double gigaFlops_cuda = (flops_cuda * 1.0e-9f) / (elapsedTime_cuda / 1000.0f);
	//printf("No Share Memory Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n", gigaFlops_cuda, elapsedTime_cuda, flops_cuda);


	//with shared memory~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cudaEvent_t start1, stop1;
	CHECK(cudaEventCreate(&start1));
	CHECK(cudaEventCreate(&stop1));
	CHECK(cudaEventRecord(start1, 0));
	cudaEventQuery(start1);    //need for WDDM mode GPU =
	
	withSharedMemKernel(device_A, device_B, device_C, M, N, K);


	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop1, 0));
	CHECK(cudaEventSynchronize(stop1));
	float elapsedTime_cuda_sharememory;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda_sharememory, start1, stop1));
	printf("with SharedMem Kernel time=%f ms\n", elapsedTime_cuda_sharememory);
	CHECK(cudaEventDestroy(start1));
	CHECK(cudaEventDestroy(stop1));

	CHECK(cudaMemcpy(h_c, device_C, SIZE_C, cudaMemcpyDeviceToHost));
	
	
	//double flops_cuda_share = 2.0 * (double)M * (double)N* (double)K;
	//double gigaFlops_share = (flops_cuda_share * 1.0e-9f) / (elapsedTime_cuda_sharememory / 1000.0f);
	//printf("CUDA Share Memory Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n", gigaFlops_share, elapsedTime_cuda_sharememory, flops_cuda_share);
	//
	

	//free gpu memory
	CHECK(cudaFree(device_A));
	CHECK(cudaFree(device_B));
	CHECK(cudaFree(device_C));

}

