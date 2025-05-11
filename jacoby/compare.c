#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE 512

int main(int argc, char** argv) {

    char command1[256];
    char command2[256];
    snprintf(command1, sizeof(command1), "./jacc");
    snprintf(command2, sizeof(command2), "./jacg.o");

    printf("\nRun CPU version\n");
    system(command1);

    printf("\nRun GPU version\n");
    system(command2);


    const char * file1 = "cpu_result.txt";
    const char * file2 = "gpu_result.txt";
    FILE *f1 = fopen(file1, "r");
    FILE *f2 = fopen(file2, "r");

    if (f1 == NULL || f2 == NULL) {
        fprintf(stderr, "Could not open\n");
        fclose(f1);
        fclose(f2);
    }

    double num1, num2;
    int flag = 1;

    while (1) {
        int res1 = fscanf(f1, "%lf", &num1);
        int res2 = fscanf(f2, "%lf", &num2);

        if (res1 == EOF && res2 == EOF) {
            break;
        }

        if (res1 == EOF || res2 == EOF || num1 != num2) {
            flag = 0;
            break;
        }
    }

    fclose(f1);
    fclose(f2);
    if (flag == 1) {
        printf("Correct\n");
    } else {
        printf("Incorrect\n");
    }

    return 0;
}

