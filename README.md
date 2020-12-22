# CUDA-Matrix-Multiplication
Here are several methods to do matrix multiplication by GPU
# Overview
### Project Environment
  **Visual Studio 2017, CUDA 10.2, Windows 10**

  **GPU:**

        Device 0: "GeForce GTX 1060 6GB"
        CUDA Capability Major/Minor version number:    6.1
        Total amount of global memory:                 6144 MBytes (6442450944 bytes)
        Total amount of constant memory:               65536 bytes
        Total amount of shared memory per block:       49152 bytes
        Total number of registers available per block: 65536
        Warp size:                                     32
        Maximum number of threads per multiprocessor:  2048
        Maximum number of threads per block:           1024
        Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
        Max dimension size of a grid size (x,y,z):    (2147483647, 65535, 65535)
        Texture alignment:                             512 bytes
        Maximum memory pitch:                          2147483647 bytes
        Memory Bus Width:                              192-bit
        L2 Cache Size:                                 1572864 bytes
        Device has ECC support:                        Disabled
        CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)

# Function

```

const int M= 1024;
const int K = 2048;
const int N = 1024;
size_t SIZE_A = M * K * sizeof(float);
size_t SIZE_B = K * N * sizeof(float);
size_t SIZE_C = M * N * sizeof(float);

getRandom(h_A, M, K);
getRandom(h_B,K,N);

//compute matrix mulplication by CPU
MatrixMulCPU(h_A, h_B, reference, M, N, K);

//compute matrix mulplication by CUDA C
cudaMatrixMul(h_A, h_B, h_C, M, N, K);
    //compute matrix mulplication CUDA - without share memory 
    noShareMemKernel(device_A, device_B, device_C, M, N, K);
    
    //compute matrix mulplication CUDA - without share memory 
    withSharedMemKernel(device_A, device_B, device_C, M, N, K);
    
    //Check result
    printDiff(reference, h_CUBLAS, N, M, 100, 1.0e-5f);
    
//compute by cuBLAS
cuBlasMatrixMul(h_A, h_B, h_CUBLAS, M, N, K);
    checkCudaErrors(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    printDiff(reference, h_CUBLAS, N, M, 100, 1.0e-5f);

```

# How much Shared Memory I have used?

```
#define TILE_WIDTH 32
__shared__ float S_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float S_B[TILE_WIDTH][TILE_WIDTH];
```

```
dim3 block_s(TILE_WIDTH, TILE_WIDTH);
dim3 grid_s((M + block_s.x - 1) / block_s.x, (N + block_s.y - 1) / block_s.y);
 ```
So the block size is 32 * 32 = 1024.

According to the GPU properties above, we can know that the maximum threads per SM is 2048, so there are 2 block per SM. 

The total amount of my GPU's share memory is **48KB**,

So the amount of the share memory we can use should **NOT** bigger than 
**48KB / 2 = 24KB**. 

In the project allocated  **32 * 32 * sizeof ( float ) * 2 = 8KB**

that's ressonable. 
  
# Performance

  
  

 
 
