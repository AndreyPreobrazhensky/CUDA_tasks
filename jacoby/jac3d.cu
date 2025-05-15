#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define L 900

typedef double dtype;

#define SAFE_CALL(err) do \
{ if (err != 0) \
        { printf("ERROR [%s] in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
          exit(1); \
        }\
} while (0)

int i, j, k, it, ITMAX = 20;
double eps, MAXEPS = 0.5f;

void writeF(const char *file, dtype *B)
{
    FILE* f = fopen(file, "w");
    if (f == NULL) {
        printf("Could not open %s\n", file);
        return;
    }

    for (int i = 0; i < L; i++){
        if (i % 100 == 0){
            printf("%d/9 written\n", i / 100);
        }
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++){
                fprintf(f, "%lf\n", B[i * L * L + j * L + k]);
    }}

    fclose(f);
}


__global__ void kernel(dtype *A, dtype *B)
{

        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > 0 && i < L - 1)
                if (j > 0 && j < L - 1)
                        if (k > 0 && k < L - 1){
                                 B[(i * L + j) * L + k] = (A[((i - 1) * L + j) * L + k] + A[((i + 1) * L + j) * L + k] + A[(i * L + j - 1) * L + k]  + A[(i * L + j + 1) * L + k]  + A[(i * L + j) * L + k - 1]  + A[(i * L + j) * L + k + 1] ) / 6.0f;
                        }
}

__global__ void diff_swap(dtype *A, dtype *B)
{
        dtype co;

        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= 0 && i < L)
                if (j >= 0 && j < L)
                        if (k >= 0 && k < L){
                                co = fabs(A[(i * L + j) * L + k] - B[(i * L + j) * L + k]);
                                A[(i * L + j) * L + k] = B[(i * L + j) * L + k];
                                B[(i * L + j) * L + k] = co;

                        }
}


__global__ void reduce(dtype *A, dtype *b, int size){
        __shared__ dtype s[16 * 4 * 2];

        int x = threadIdx.x;
        int y = threadIdx.y;
        int z = threadIdx.z;
        int co = x + y * blockDim.x + z * blockDim.y * blockDim.x;

        int gtid = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + co;


        if (gtid + blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z < size)
              s[co] = (A[gtid] >= A[gtid + blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z] ? A[gtid] : A[gtid + blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z]);
        else
              s[co] = 0.0f;

        __syncthreads();

        for (int t = blockDim.x / 2; t > 0; t >>= 1){  // po osi x
                if (x < t)
                        s[co] = s[co + t] > s[co] ? s[co + t] : s[co];
                __syncthreads();
        }
        for (int t = blockDim.y / 2; t > 0; t >>= 1){  // po osi y
                if (y < t && x == 0)
                        s[co] = s[co + t * blockDim.x] > s[co] ? s[co + t * blockDim.x] : s[co];
                __syncthreads();
        }
        for (int t = blockDim.z / 2; t > 0; t >>= 1){  // po osi z
                if (z < t && x == 0 && y == 0)
                        s[co] = s[co + t * blockDim.x * blockDim.y] > s[co] ? s[co + t * blockDim.x * blockDim.y] : s[co];
                __syncthreads();
        }

        if (x == 0 && y == 0 && z == 0) b[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = s[0];
}



int main(){

        size_t task_sz = L * L * L * sizeof(dtype);
        size_t size_res = ((L + 15) / 16) * ((L + 3) / 4) * ((L + 3) / 4) * sizeof(dtype);
        //size_t size_res_2 = (((L + 15) / 16 + 15) / 16) * (((L + 3) / 4 + 3) / 4) * (((L + 3) / 4 + 3) / 4) * sizeof(dtype);
        //dtype *A = (dtype*)malloc(task_sz);
        dtype *B = (dtype*)malloc(task_sz);
        dtype *D;
        dtype *res_max = (dtype*)malloc(size_res);
        dtype *A_dev, *b_res;

        SAFE_CALL(cudaMalloc((void**)& A_dev, task_sz));
        //SAFE_CALL(cudaMalloc((void**)& B_dev, task_sz));
        SAFE_CALL(cudaMalloc((void**)& b_res, size_res));
        SAFE_CALL(cudaMalloc((void**)& D, task_sz));
        //SAFE_CALL(cudaMalloc((void**)& b_res_2, size_res_2));
                
	cudaEvent_t start, stop;

        float time;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (i = 0; i < L; i++)
            for (j = 0; j < L; j++)
                for (k = 0; k < L; k++)
                {
                    //A[(i * L + j) * L + k] = 0;
                    if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                        B[(i * L + j) * L + k]= 0;
                    else
                        B[(i * L + j) * L + k] = 4 + i + j + k;
                }

        cudaEventRecord(start, 0);

        SAFE_CALL(cudaMemcpy(D, B, task_sz, cudaMemcpyHostToDevice));
        dim3 thread(16, 4, 2), block((L + 15) / 16, (L + 3) / 4, (L + 3) / 4);
        //dim3 block_new(((L + 15) / 16 + 15) / 16, ((L + 3) / 4 + 3) / 4, ((L + 3) / 4 + 3) / 4);
        dim3 block_base((L + 15) / 16, (L + 3) / 4, (L + 1) / 2);
        //dim3 thread_r(128, 1, 1), block_r((L + 127) / 128, L, L);
        /* iteration loop */

        for (it = 1; it <= ITMAX; it++){
                eps = 0;
                diff_swap <<< block_base, thread>>> (A_dev, D);
                reduce <<<block, thread>>> (D, b_res, L * L * L);
                //reduce <<<block_new, thread>>> (b_res, b_res_2, size_res / sizeof(dtype));
                SAFE_CALL(cudaMemcpy(res_max, b_res, size_res, cudaMemcpyDeviceToHost));
                kernel <<< block_base, thread >>> (A_dev, D);
                for (int j = 0; j < size_res / sizeof(dtype); j++){
                        eps = eps >= res_max[j] ? eps : res_max[j];
                        //printf("res_max[j]: %f\n", res_max[j]);
                }

                printf(" IT = %4i   EPS = %14.7E\n", it, eps);
         if (eps < MAXEPS)
                    break;
        }
        SAFE_CALL(cudaMemcpy(B, D, task_sz, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A_dev);
        //cudaFree(B_dev);
        cudaFree(D);
        cudaFree(b_res);
        //writeF("gpu_result.txt", B);                
        //free(A);
        free(B);
        free(res_max);
        printf("EPS = %14.7E\n", eps);
        printf(" Jacobi3D Benchmark Completed.\n");
        printf(" Size            = %4d x %4d x %4d\n", L, L, L);
        printf(" Iterations      =       %12d\n", ITMAX);
        printf(" Time in seconds =       %12.2f\n", time / 1000);
        printf(" Operation type  =     double precision\n");
        printf(" Device used:     %s\n", prop.name);

        printf(" END OF Jacobi3D Benchmark\n");
        return 0;
}

                                                 
