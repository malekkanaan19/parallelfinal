#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int N = 1024; // number of rows in M and P
    int M = 512; // number of columns in M and number of rows in N
    int K = 2048; // number of columns in N and P

    float** MM = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        MM[i] = (float*)malloc(M * sizeof(float));
        for (int j = 0; j < M; j++) {
            MM[i][j] = rand() / (float)RAND_MAX;
        }
    }

    float** NN = (float**)malloc(M * sizeof(float*));
    for (int i = 0; i < M; i++) {
        NN[i] = (float*)malloc(K * sizeof(float));
        for (int j = 0; j < K; j++) {
            NN[i][j] = rand() / (float)RAND_MAX;
        }
    }

    float** R = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        R[i] = (float*)malloc(K * sizeof(float));
    }   
   
   clock_t start_time = clock();
    //MM
    for(int i = 0; i < N;i++){
        for (int j = 0; j < K; j++) {
            R[i][j]=0;
            for(int k =0; k < M; k++)
            R[i][j] += MM[i][k] * NN[k][j];
    }}   
    clock_t end_time = clock();

   // for(int i = 0 ; i < N ; i++){
        //for ( int j =0 ; j < K ; j++)
        //{
//  printf("%f ", R[i][j]);
    //  }
    //}
    double duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Execution time: %lf seconds.\n", duration);

     
    return 0;
}
