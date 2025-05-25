#include <math.h>
#include <stdlib.h>
#include <stdio.h>

typedef double dtype;

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define nx 900
#define ny 900
#define nz 900


#define SAFE_CALL(err) do \
{ if (err != 0) \
        { printf("ERROR [%s] in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
          exit(1); \
        }\
} while (0)


void init(dtype *a) {
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[(i * ny + j) * nx + k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[(i * ny + j) * nx + k] = 0.0;
            }
}

__global__ void kernel_x(dtype* a){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
        for (int i = 1; i < nx - 1; i++){
            //int idx = i * ny * nz + j * nz + k;
            //int idx_prev = (i-1) * ny * nz + j * nz + k;
            //int idx_next = (i+1) * ny * nz + j * nz + k;

            a[(i * ny + j) * nz + k] = (a[((i - 1) * ny + j) * nz + k] + a[((i + 1) * ny + j) * nz + k]) / 2;
        }
    }
}

__global__ void kernel_y(dtype* a) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < nx - 1 && k > 0 && k < nz - 1) {
        for (int j = 1; j < ny - 1; j++) {
            //int idx = i * ny * nz + j * nz + k;
            //int idx_prev = i * ny * nz + (j-1) * nz + k;
            //int idx_next = i * ny * nz + (j+1) * nz + k;

            a[(i * ny + j) * nz + k] = (a[(i * ny + j - 1) * nz + k] + a[(i * ny + j + 1) * nz + k] ) / 2;
        }
    }
}

__global__ void reorder_to_z_major(dtype* a, dtype* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            //int idx_a = k * ny * nz + i * nz + j;         // [i][j][k]
            //int idx_b = j * nx * ny + k * ny + i;         // [k][i][j]
            b[(j * nx + k) * ny + i] = a[(k * ny + i) * nz + j];
        }
    }
}

__global__ void kernel_z_and_reduction(dtype* a, dtype* D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    dtype eps = 0;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        for (int k = 1; k < nz - 1; k++) {
            //int idx      = k * nx * ny + i * ny + j;     // b[k][i][j]
            //int idx_prev = (k - 1) * nx * ny + i * ny + j;
            //int idx_next = (k + 1) * nx * ny + i * ny + j;

            dtype tmp1 = (a[((k - 1) * nx + i) * ny + j] + a[((k + 1) * nx + i) * ny + j]) / 2;
            dtype tmp2 = fabs(a[(k * nx + i) * ny + j] - tmp1);
            eps = Max(eps, tmp2);
            a[(k * nx + i) * ny + j] = tmp1;
        }
    }

    __shared__ dtype s[16 * 16];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    s[tid] = eps;
    __syncthreads(); // загрузка в разделяемую память

    for (int t = 16 * 16 / 2; t > 0; t >>= 1) {
        if (tid < t) {
            s[tid] = Max(s[tid], s[tid + t]);
        }
        __syncthreads();
    } // редукция одномерного массива строк в каждом из блоков

    if (tid == 0) {
        D[blockDim.z * blockDim.y * blockIdx.z + blockDim.x * blockIdx.y + blockIdx.x] = s[0];
    }
}

__global__ void reorder_from_z_major(dtype* b, dtype* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            //int idx_b = i * nx * ny + k * ny + j;         // [k][i][j]
            //int idx_a = k * ny * nz + j * nz + i;         // [i][j][k]
            a[(k * ny + j) * nz + i] = b[(i * nx + k) * ny + j];
        }
    }
}

void printA(const char* filename, dtype* a, int size){
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%lf\n", a[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    dtype maxeps, eps;
    dtype *a, *dev_a, *D, *dev_b, *D_host;
    int it, itmax = 10;
    cudaEvent_t start, stop;
    float time;

    maxeps = 0.01;
    itmax = 10;

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    a = (dtype*)malloc(nx * ny * nz * sizeof(dtype));
    dim3 block(16, 16);
    dim3 grid((ny + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
    D_host = (dtype*)malloc(grid.x * grid.y * grid.z * sizeof(dtype));
    init(a);
    SAFE_CALL(cudaMalloc(&dev_a, nx * ny * nz * sizeof(dtype)));
    SAFE_CALL(cudaMalloc(&dev_b, nx * ny * nz * sizeof(dtype)));
    SAFE_CALL(cudaMalloc(&D, grid.x * grid.y * grid.z * sizeof(dtype)));

    SAFE_CALL(cudaMemcpy(dev_a, a, nx * ny * nz * sizeof(dtype), cudaMemcpyHostToDevice));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (it = 1; it <= itmax; it++) {
        eps = 0.0;

        kernel_x<<<grid, block>>>(dev_a);

        kernel_y<<<grid, block>>>(dev_a);

        reorder_to_z_major<<<grid, block>>>(dev_a, dev_b);
        kernel_z_and_reduction<<<grid, block>>>(dev_b, D);

        SAFE_CALL(cudaMemcpy(D_host, D, grid.x * grid.y * grid.z * sizeof(dtype), cudaMemcpyDeviceToHost));
        reorder_from_z_major<<<grid, block>>>(dev_b, dev_a);
        for (int h = 0; h < grid.x * grid.y * grid.z; h++){
                eps = Max(D_host[h], eps);
        }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps) break;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    SAFE_CALL(cudaMemcpy(a, dev_a, nx * ny * nz * sizeof(dtype), cudaMemcpyDeviceToHost));


    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2f\n", time / 1000);
    printf(" Operation type  =             %s\n", (sizeof(dtype) == sizeof(float)) ? "float" : "double");
    printf(" GPU Device: %s\n", prop.name);
    printf(" END OF ADI Benchmark\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(D);
    free(a);

    return 0;
}

