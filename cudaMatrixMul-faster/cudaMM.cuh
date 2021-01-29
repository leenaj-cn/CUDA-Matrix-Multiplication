#pragma once
#include <stdio.h>
#include <string.h>
#include <cuda.h>
//#include <windows.h>

#define M 1024
#define N 1024
#define K 1024
#define TILE_WIDTH 32

typedef float DATA_TYPE;

#include <cstdlib>
#define CHECK(call)                                                     \
do{                                                                     \
    const cudaError_t error_code=call;                                  \
    if (error_code != cudaSuccess)                                       \
    {                                                                   \
        printf("CUDA Error:\n");                                         \
		printf("\tFile:\t%s\n", __FILE__);						\
		printf("\tLine:\t%d\n", __LINE__);						\
		printf("\tError code:%d\n", error_code);				\
		printf("\tError info:%s\n", cudaGetErrorString(error_code));			\
		exit(1);	                                                       \
    }                                                                   \
}while(0)                                                               


typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int stride; 
    DATA_TYPE* data;
} Matrix;
