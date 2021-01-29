### Performance
```
-heightA=1024, widthA=1024, widthB=1024, heightB=1024

Allocate 4.000000e+00 MB for matrix h_A
Allocate 4.000000e+00 MB for matrix h_B

Performance= 16.59 GFlop/s, Time= 129.423782 msec, Size= 2147483648 Ops, WorkgroupSize= 1024 threads/block

Checking computed result for correctness:
 Result = PASS
```
### Define matrix
```
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int stride; 
    DATA_TYPE* data;
} Matrix;
```

### Typedef matrix data type
```
typedef float DATA_TYPE;
```

### Error Detector: 
```
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
```
### Fetch argc && argv
```
checkCmdLine()

```

### block thread 
```
    #define TILE_WIDTH 32

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_C.width + block.x - 1) / block.x, (d_C.height + block.y - 1) / block.y);

```
### Kernel Analysis

- Reduce memory access latency by  <b>Shared Memoroy</b>
- Avoid bank confilct by: 
```

    ...

    S_A[ty][tx]= d_A.data[ y * d_A.width + i * TILE_WIDTH + tx];

    ...

    S_B[ty][tx] = d_B.data[ (i * TILE_WIDTH + ty) * d_B.width + x];

    ...

    temp += S_A[ty][i] * S_B[i][tx];

    ....

```


