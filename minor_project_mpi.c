#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <sys/resource.h>  // For getrusage

#include "alloc_mpi.h"
#include "ann_mpi.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void train_from_csv(char* filename, double** train_data, int* total_samples, int rank, int size);
void predict_from_csv(char *sourceFile, char* destFile, int rank, int size);
void save_image_as_png(const char *filename, double *pixels, int width, int height);

network* ann;

long get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024L;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ann = (network*)malloc(sizeof(network));
    double **train_data = init_2Darray(MAX_SIZE, MAX_SIZE);
    int *dim = (int*)malloc(3 * sizeof(int));
    dim[0] = 784;
    dim[1] = 32;
    dim[2] = 10;
    init_ann(ann, dim, 3);

    clock_t start_train_time = clock();
    long start_train_memory = get_memory_usage();

    int total_samples = 0;
    train_from_csv("train.csv", train_data, &total_samples, rank, size);

    clock_t end_train_time = clock();
    long end_train_memory = get_memory_usage();

    double train_time_taken = (double)(end_train_time - start_train_time) / CLOCKS_PER_SEC;
    long train_memory_used = end_train_memory - start_train_memory;

    if (rank == 0) {
        printf("Time taken for training: %.2f seconds\n", train_time_taken);
        printf("Memory used during training: %ld bytes (%.2f MB)\n", train_memory_used, (double)train_memory_used / (1024 * 1024));
    }

    clock_t start_test_time = clock();
    long start_test_memory = get_memory_usage();

    predict_from_csv("test.csv", "submission.csv", rank, size);

    clock_t end_test_time = clock();
    long end_test_memory = get_memory_usage();

    double test_time_taken = (double)(end_test_time - start_test_time) / CLOCKS_PER_SEC;
    long test_memory_used = end_test_memory - start_test_memory;

    if (rank == 0) {
        printf("Time taken for testing: %.2f seconds\n", test_time_taken);
        printf("Memory used during testing: %ld bytes (%.2f MB)\n", test_memory_used, (double)test_memory_used / (1024 * 1024));
    }

    free_ann(ann);
    free(train_data);
    free(dim);

    MPI_Finalize();
    return 0;
}

void train_from_csv(char* filename, double** train_data, int* total_samples, int rank, int size) {
    FILE *fptr = NULL;
    int total_count = 0;

    if (rank == 0) {
        char line[16000];
        if ((fptr = fopen(filename, "r")) == NULL) {
            printf("Unable to open file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fgets(line, sizeof(line), fptr);  // skip header

        while (fgets(line, sizeof(line), fptr)) {
            char *token = strtok(line, ",");
            train_data[total_count][784] = atof(token);
            for (int i = 0; i < 784; i++) {
                token = strtok(NULL, ",");
                train_data[total_count][i] = atof(token) / 255.0;
            }
            total_count++;
            if (total_count >= MAX_SIZE) break;
        }
        fclose(fptr);
    }

    // Broadcast sample count to all
    MPI_Bcast(&total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *total_samples = total_count;

    // Broadcast data to all processes
    MPI_Bcast(&(train_data[0][0]), MAX_SIZE * MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Train in parallel
    train(ann, train_data, total_count, 0.25, rank, size);

    if (rank == 0) printf("Done Reading and Training...\n");
}

void predict_from_csv(char *sourceFile, char* destFile, int rank, int size) {
    FILE *fptr = NULL;
    char line[16000];
    int total_lines = 0;
    int lines_per_proc = 0;

    if (rank == 0) {
        if ((fptr = fopen(sourceFile, "r")) == NULL) {
            printf("Unable to open file %s\n", sourceFile);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fgets(line, sizeof(line), fptr); // skip header

        while (fgets(line, sizeof(line), fptr)) {
            total_lines++;
        }
        fclose(fptr);
    }

    MPI_Bcast(&total_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    lines_per_proc = (total_lines + size - 1) / size;
    int my_start = rank * lines_per_proc;
    int my_end = (rank + 1) * lines_per_proc;
    if (my_end > total_lines) my_end = total_lines;

    double **local_data = init_2Darray(lines_per_proc, MAX_SIZE);
    int count = 0;

    if (rank == 0) {
        fptr = fopen(sourceFile, "r");
        fgets(line, sizeof(line), fptr); // skip header
        int idx = 0;
        while (fgets(line, sizeof(line), fptr)) {
            if (idx >= my_start && idx < my_end) {
                char *token = strtok(line, ",");
                for (int i = 0; i < 784; i++) {
                    if (i > 0) token = strtok(NULL, ",");
                    local_data[count][i] = atof(token) / 255.0;
                }
                count++;
            }
            idx++;
        }
        fclose(fptr);
    }

    // Broadcast local data for this rank
    MPI_Bcast(&(local_data[0][0]), lines_per_proc * MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prediction and gather results
    int *labels = (int*)malloc(lines_per_proc * sizeof(int));
    for (int i = 0; i < count; i++) {
        labels[i] = predict(ann, local_data[i]);
        if (rank == 0 && (i == 16)) {
            save_image_as_png("sample_17.png", local_data[i], 28, 28);
        }
    }

    int *all_labels = NULL;
    if (rank == 0) all_labels = (int*)malloc(total_lines * sizeof(int));

    MPI_Gather(labels, lines_per_proc, MPI_INT, all_labels, lines_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *out = fopen(destFile, "w");
        fprintf(out, "ImageId,Label\n");
        for (int i = 0; i < total_lines; i++) {
            fprintf(out, "%d,%d\n", i + 1, all_labels[i]);
        }
        fclose(out);
        free(all_labels);
    }

    free(labels);
    free(local_data);
}

void save_image_as_png(const char *filename, double *pixels, int width, int height) {
    unsigned char *image_data = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < width * height; i++) {
        image_data[i] = (unsigned char)(pixels[i] * 255);
    }

    stbi_write_png(filename, width, height, 1, image_data, width);
    free(image_data);
}
