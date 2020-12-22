#include "head.cuh"
#include <cuda.h>
#include <string>
#include <helper_functions.h>
#include <helper_cuda.h>

const int M= 256;
const int K = 256;
const int N = 256;

float* getRandom(float *a,int m,int n)
{
	for (int i = 0; i < m*n; ++i)
	{
		a[i] = (float)i;
	}

	//print host matrix
	//for (int i = 0; i < m; i++)
	//{
	//	for (int j = 0; j < n; j++)
	//	{
	//		printf("%f\t", a[i*n+j]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");

	return a;
}

void MatrixMulCPU(const float *A, const float *B, float *C,int M, int N, int K)
{
	//printf("Host matrix saved by row major method result:\n");
	for (int row = 0; row < M; ++row) {
		for (int col = 0; col < N; ++col) {

			float  temp = 0.0f;
			for (int n = 0; n < K; ++n) {
				temp += A[row*K + n] * B[n*N + col];
			}

			C[row*N + col] = temp;
			//printf("%f\t", C[row * N + col]);
		}
		//printf("\n");
	}

}

void MatrixMultiColMajor(float *A, float *B, float *C, int M, int N, int K)
{
	printf("Host matrix saved by col major method result:\n");
	for (int row = 0; row < M; row++) 
	{
		for (int col = 0; col < N; col++) 
		{
			float  temp = 0.0f;
			for (int i = 0; i < K; i++) {
				temp += A[i*M + row] * B[col*K + i];
			}
			C[col*M +row ] = temp;
			printf("%f\t", C[col*M + row]);
		}
		printf("\n");
	}
	
}

bool printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
	int i, j, k;
	int error_count = 0;

	for (int j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			k = j * width + i;
			float fDiff = fabs(data1[k] - data2[k]);

			if (fDiff > fListTol)
			{
				if (error_count < iListLength)
				{
					printf("Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
				}

				error_count++;
			}
		}
	}

	printf("\n Total Errors = %d\n", error_count);

	return error_count==0 ? true:false;
}


int main()
{
	size_t SIZE_A = M * K * sizeof(float);
	size_t SIZE_B = K * N * sizeof(float);
	size_t SIZE_C = M * N * sizeof(float);
	//host
	float *h_A = (float*)malloc(SIZE_A);
	float *h_B = (float*)malloc(SIZE_B);
	float *h_C = (float*)malloc(SIZE_C);
	//initialize
	memset(h_A, 0, SIZE_A);
	memset(h_B, 0, SIZE_B);
	memset(h_C, 0, SIZE_C);
	printf("get h_A:\n");
	getRandom(h_A, M, K);
	printf("get h_B:\n");
	getRandom(h_B,K,N);

	printf("CPU start:\n");
	float *reference = (float *)malloc(SIZE_C);
	MatrixMulCPU(h_A, h_B, reference, M, N, K);
	printf("done.\n");

	
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~CUDA time ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~CUDA start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start); //can not call by CHECK() used for WDDM mode GPU

	cudaMatrixMul(h_A, h_B, h_C, M, N, K);

	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("total cuda GPU time=%f ms\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	//check CUDA result
	bool resCUDA = printDiff(reference, h_C, N, M, 100, 1.0e-5f);
	printf("Comparing CUDA Matrix Multiply with CPU results: %s\n", (true == resCUDA) ? "PASS" : "FAIL");

	//PERFORMANCE
	double flops_cuda = 2.0 * (double)M * (double)N* (double)K;
	double gigaFlops_cuda = (flops_cuda * 1.0e-9f) / (elapsedTime_cuda / 1000.0f);
	printf("CUDA Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n\n\n", gigaFlops_cuda, elapsedTime_cuda, flops_cuda);

	//~~~~~~~~~~~~~~~~~~~~~~~~cuBLAS time~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	printf("~~~~~~~~~~~~~~~~~~~~~~~~CUBLAS Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	float *h_CUBLAS = (float *)malloc(SIZE_C);
	cuBlasMatrixMul(h_A, h_B, h_CUBLAS, M, N, K);

	//check result
	bool resCUBLAS = printDiff(reference, h_CUBLAS, N, M, 100, 1.0e-5f);
	printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");


	//free
	free(h_A);
	free(h_B);
	free(h_C);
	getchar();
	return 0;

}