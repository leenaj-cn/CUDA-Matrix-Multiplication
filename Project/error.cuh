//#pragma once
#ifndef ERROR_CUH_
#define ERROR_CUH_

#include <stdio.h>
#include <cstdlib>
#define CHECK(call)												\
do {															\
	const cudaError_t error_code = call;						\
	if (error_code != cudaSuccess)								\
	{															\
		printf("CUDA Error:\n");								\
		printf("\tFile:\t%s\n", __FILE__);						\
		printf("\tLine:\t%d\n", __LINE__);						\
		printf("\tError code:%d\n", error_code);				\
		printf("\tError info:%s\n", cudaGetErrorString(error_code));			\
		exit(1);																\
	}																			\
}while(0)																		

#endif