/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

typedef double dtype;
#define L 900
#define ITMAX 20

int i, j, k, it;
dtype eps;
dtype MAXEPS = 0.5f;

void writeF(const char *file, dtype (*B)[L][L])
{
    FILE* f = fopen(file, "w");
    if (f == NULL) {
        printf("Could not open %s\n", file);
        return;
    }

    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
	    for (int k = 0; k < L; k++){
                fprintf(f, "%lf\n", B[i][j][k]);
    }

    fclose(f);
}

int main(int an, char **as)
{
    dtype (*A)[L][L], (*B)[L][L];
    dtype startt, endt;
    A = (dtype (*)[L][L])malloc(L * L * L * sizeof(dtype));
    B = (dtype (*)[L][L])malloc(L * L * L * sizeof(dtype));

    for (i = 0; i < L; i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < L; k++)
            {
                A[i][j][k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[i][j][k] = 0;
                else
                    B[i][j][k] = 4 + i + j + k;
            }    
    

    startt = omp_get_wtime();
    /* iteration loop */
    for (it = 1; it <= ITMAX; it++)
    {
        eps = 0;
        
        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                {
                    dtype tmp = fabs(B[i][j][k] - A[i][j][k]);
                    eps = Max(tmp, eps);
                    A[i][j][k] = B[i][j][k];
                }

        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                    B[i][j][k] = (A[i - 1][j][k] + A[i][j - 1][k] + A[i][j][k - 1] + A[i][j][k + 1] + A[i][j + 1][k] + A[i + 1][j][k]) / 6.0f;
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }
    endt = omp_get_wtime();
    writeF("cpu_result.txt", B);
    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =     floating point\n");

    printf(" END OF Jacobi3D Benchmark\n");
    return 0;
}
