#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "error.cuh"
//#include <iosteam> kernel 中不支持iosteam
#include <cstdbool>

#define TILE_WIDTH 32

extern "C" void cudaMatrixMul(float *h_a, float *h_b, float *h_c, int M, int N, int K);

extern "C" void cuBlasMatrixMul(float *h_A, float *h_B, float *h_CUBLAS, int M, int N, int K);

extern "C" void noShareMemKernel(float *device_A, float *device_B, float *device_C, int M, int N, int K);

extern "C" void withSharedMemKernel(float *device_A, float *device_B, float *device_C, int M, int N, int K);