#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void MatrixMultiplication(float* M, float* N, float* P, int height, int width, int depth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < depth) {
        float pvalue = 0;
        for (int k = 0; k < width; k++) {
            pvalue += M[row * width + k] * N[k * depth + col];
        }
        P[row * depth + col] = pvalue;
    }
}

// Function to print a matrix
void print_matrix(float* matrix, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", matrix[i * M + j]);
        }
        printf("\n");
    }
}

int main() {
    int N = 1024; // number of rows in M and P
    int M = 512; // number of columns in M and number of rows in N
    int K = 2048; // number of columns in N and P

    // Allocate memory for input and output matrices on host
    float *h_M = (float*)malloc(N * M * sizeof(float));
    float *h_N = (float*)malloc(M * K * sizeof(float));
    float *h_P = (float*)malloc(N * K * sizeof(float));

    // Initialize input matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * M; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < M * K; i++) {
        h_N[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for input and output matrices on device
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, N * M * sizeof(float));
    cudaMalloc((void**)&d_N, M * K * sizeof(float));
    cudaMalloc((void**)&d_P, N * K * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_M, h_M, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, M * K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel function with appropriate thread layout and block size
    int block_size = 16; // number of threads per block
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(ceil(K / (float)block_size), ceil(N / (float)block_size));
    
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_P, N, M, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy output matrix from device to host
    cudaMemcpy(h_P, d_P, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Print input and output matrices
   // printf("Matrix M:\n");
   // print_matrix(h_M, N, M);

   // printf("Matrix N:\n");
    //print_matrix(h_N, M, K);

  //  printf("Matrix P:\n");
   // print_matrix(h_P, N, K);

    printf("Elapsed time: %f ms\n", elapsed_time);

    // Free memory
    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
