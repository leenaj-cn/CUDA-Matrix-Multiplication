#include "cudaMM.cuh"

unsigned int heightA = M;
unsigned int widthA = K;
unsigned int heightB = K;
unsigned int widthB = N;

int checkCmdLine(int argc, char *argv[], unsigned int *widthA, unsigned int *heightA, unsigned int *widthB, unsigned int *heightB);
void ConstInit(DATA_TYPE *matrix, unsigned int w, unsigned int h, DATA_TYPE value);

__global__ void kernel(Matrix d_A, Matrix d_B, Matrix d_C)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x = blockIdx.x * TILE_WIDTH + tx;  
    int y = blockIdx.y * TILE_WIDTH + ty; 

    __shared__ DATA_TYPE S_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ DATA_TYPE S_B[TILE_WIDTH][TILE_WIDTH];  

    DATA_TYPE temp = 0.0;

    int i;
    for(i = 0; i < (d_A.width + TILE_WIDTH - 1) / TILE_WIDTH; ++i)
    {
        if(y < d_A.height && i * TILE_WIDTH + tx < d_A.width)
            S_A[ty][tx]= d_A.data[ y * d_A.width + i * TILE_WIDTH + tx];

            
        if(i * TILE_WIDTH + ty < d_B.height && x < d_B.width)
            S_B[ty][tx] = d_B.data[ (i * TILE_WIDTH + ty) * d_B.width + x];
              
        __syncthreads();
        
        for(int j = 0; j < TILE_WIDTH; ++j)
            temp += S_A[ty][i] * S_B[i][tx];
           
        __syncthreads();

    }

    if(x < d_C.width && y < d_C.height)
        d_C.data[y * d_C.width + x] = temp;       

}


int main(int argc, char *argv[])
{
    if(argc>1) checkCmdLine(argc, argv, &widthA, &heightA, &widthB, &heightB);
    printf("-heightA=%d, widthA=%d, widthB=%d, heightB=%d\n",heightA, widthA, widthB, heightB);

    Matrix h_A, h_B, h_C;
    Matrix d_A, d_B, d_C;

    d_A.width = h_A.width = widthA;
    d_A.height = h_A.height = heightA;

    d_B.width = h_B.width = widthB;
    d_B.height = h_B.height = heightB;   

    d_C.width = h_C.width = widthB;
    d_C.height = h_C.height = heightA;  

    unsigned int data_size_A = h_A.width * h_A.height * sizeof(DATA_TYPE);
    unsigned int data_size_B = h_B.width * h_B.height * sizeof(DATA_TYPE);
    unsigned int data_size_C = h_A.height * h_B.width * sizeof(DATA_TYPE);

    printf("Allocate %e MB for matrix h_A\n", data_size_A / (1024.f*1024.f));
    printf("Allocate %e MB for matrix h_B\n", data_size_B / (1024.f*1024.f));

    h_A.data = (DATA_TYPE*)malloc(data_size_A);
    h_B.data = (DATA_TYPE*)malloc(data_size_B);
    h_C.data = (DATA_TYPE*)malloc(data_size_C);
    // CHECK(cudaHostAlloc((void**)&h_A.data, data_size_A,cudaHostAllocDefault));
    // CHECK(cudaHostAlloc((void**)&h_B.data, data_size_A,cudaHostAllocDefault));
    // CHECK(cudaHostAlloc((void**)&h_C.data, data_size_A,cudaHostAllocDefault));


    if(h_A.data == NULL || h_B.data == NULL || h_C.data == NULL)
        printf("Failed to allocate memory space on host");

    const DATA_TYPE valB = 2.0f;
    ConstInit(h_A.data, h_A.width, h_A.height, 1.0f);
    ConstInit(h_B.data, h_B.width, h_B.height, valB);

    unsigned int size_C = h_C.width * h_C.height;
    memset(h_C.data, 0, size_C);

    //GPU
    printf("Allocate memory on GPU....\n");

    CHECK(cudaMalloc(&d_A.data, data_size_A));
    CHECK(cudaMalloc(&d_B.data, data_size_B));
    CHECK(cudaMalloc(&d_C.data, data_size_C));

    //cudaStream_t stream;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
     
    //CHECK(cudaMemcpyAsync(d_A.data, h_A.data, data_size_A, cudaMemcpyHostToDevice, stream));
    //CHECK(cudaMemcpyAsync(d_B.data, h_B.data, data_size_B, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpy(d_A.data, h_A.data, data_size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B.data, h_B.data, data_size_B, cudaMemcpyHostToDevice));    
    
    //CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventRecord(start, 0));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_C.width + block.x - 1) / block.x, (d_C.height + block.y - 1) / block.y);

    kernel<<<grid, block>>>(d_A, d_B, d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //D -> H
    //CHECK(cudaMemcpyAsync(h_C.data, d_C.data, data_size_C, cudaMemcpyDeviceToHost, stream));    
    CHECK(cudaMemcpy(h_C.data, d_C.data, data_size_C, cudaMemcpyDeviceToHost));
    //CHECK(cudaStreamSynchronize(stream));

    float time=0.0f;
    CHECK(cudaEventElapsedTime(&time, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    //compute performance
    double flopsPerMatrixMul = 2.0 * static_cast<double>(d_A.width) * static_cast<double>(d_A.height) * static_cast<double>(d_B.width);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (time / 1000.0f);

    printf(
        "Performance= %.2f GFlop/s, Time= %f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        time,
        flopsPerMatrixMul,
        block.x * block.y);   


    printf("Checking computed result for correctness:\n ");
    bool correct = true;

    double eps = 1.e-6;
    printf("d_A.width=%d, valB=%f, ref = %.8f\n", d_A.width, valB, d_A.width * valB);
    //for(unsigned int i = 0; i< d_C.width * d_C.height; i++){
    for(unsigned int i = 0; i < 50; i++){    
        double abs_err = fabs(h_C.data[i] - (d_A.width * valB));
        double abs_val = fabs(h_C.data[i]);
        double rel_err = abs_err / abs_val / d_A.width;

        if( rel_err > eps){
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C.data[i], d_A.width * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");


    free(h_A.data);
    free(h_B.data);
    free(h_C.data);
    // cudaFreeHost(h_A.data);
    // cudaFreeHost(h_B.data);
    // cudaFreeHost(h_C.data);
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }

}

void ConstInit(DATA_TYPE *matrix, unsigned int w, unsigned int h, DATA_TYPE value)
{
    unsigned int i;
    unsigned int size = w * h;
    for(i=0; i < size; i++)
        matrix[i] = value;

}

int checkCmdLine(int argc, char *argv[], unsigned int *widthA,unsigned int *heightA, unsigned int *widthB,unsigned int *heightB)
{
    printf("argc=%d\n",argc);
    if(argc==7)
    {
        printf("Your Input: ");

        int i;
        for(i=0;i<argc;i++)
        {
            printf("%s ",argv[i]);
        }
        printf("\n");

        *widthA = atoi(argv[2]);
        *heightB = atoi(argv[4]);
        *heightA = *widthB = atoi(argv[6]);

        printf("widthA=%d, heightA=%d, widthB=%d, heightB=%d\n",*widthA, *heightA, *widthB, *heightB);
    }
    else// && strcmp(argv[1] ,"--help")
    {
            printf("Please set the M,N,K size of matrix A and B, ");
            printf("For example: -M WidthA -N HeightB -K HeightA/WidthB\n");
            exit(-1);
    }
 
    return 0;
}