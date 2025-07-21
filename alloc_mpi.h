#ifndef ALLOC_MPI_H
#define ALLOC_MPI_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to initialize 2D array
static inline double ** init_2Darray(int m, int n) {
    double ** arr = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        arr[i] = (double *)malloc(n * sizeof(double));
    }
    return arr;
}

// Function to initialize 3D array
static inline double *** init_3Darray(int a, int b, int c) {
    double *** arr = (double ***)malloc(a * sizeof(double **));
    for (int i = 0; i < a; i++) {
        arr[i] = (double **)malloc(b * sizeof(double *));
        for (int j = 0; j < b; j++) {
            arr[i][j] = (double *)malloc(c * sizeof(double));
        }
    }
    return arr;
}
#endif // ALLOC_MPI_H
