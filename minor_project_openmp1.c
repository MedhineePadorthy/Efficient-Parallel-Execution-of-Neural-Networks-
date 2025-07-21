#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <psapi.h>
#include <omp.h>
#include "alloc.h"
#include "ann_openmp1.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BUFFER_SIZE 10000

void train_from_csv(char* filename, char* buffer, double** train_data);
void predict_from_csv(char *sourceFile, char* destFile, char* buffer);
network* ann;

void save_image_as_png(const char *filename, double *pixels, int width, int height);

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());
    setvbuf(stdout, NULL, _IONBF, 0);

    ann = (network*)malloc(sizeof(network));
    double **train_data = init_2Darray(MAX_SIZE, MAX_SIZE);

    int *dim = (int*)malloc(3 * sizeof(int));
    dim[0] = 784;
    dim[1] = 32;
    dim[2] = 10;
    init_ann(ann, dim, 3);

    char* buffer = (char*)malloc(BUFFER_SIZE * sizeof(char));

    clock_t start_train_time = clock();
    PROCESS_MEMORY_COUNTERS memCounter;
    SIZE_T start_train_memory = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter))) {
        start_train_memory = memCounter.WorkingSetSize;
    }

    train_from_csv("train.csv", buffer, train_data);

    clock_t end_train_time = clock();
    SIZE_T end_train_memory = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter))) {
        end_train_memory = memCounter.WorkingSetSize;
    }

    double train_time_taken = (double)(end_train_time - start_train_time) / CLOCKS_PER_SEC;
    SIZE_T train_memory_used = end_train_memory - start_train_memory;

    printf("Time taken for training: %.2f seconds\n", train_time_taken);
    printf("Memory used during training: %zu bytes (%.2f MB)\n", train_memory_used, (double)train_memory_used / (1024 * 1024));
    fflush(stdout);

    clock_t start_test_time = clock();
    SIZE_T start_test_memory = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter))) {
        start_test_memory = memCounter.WorkingSetSize;
    }

    predict_from_csv("test.csv", "submission_openmp1.csv", buffer);

    clock_t end_test_time = clock();
    SIZE_T end_test_memory = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter))) {
        end_test_memory = memCounter.WorkingSetSize;
    }

    double test_time_taken = (double)(end_test_time - start_test_time) / CLOCKS_PER_SEC;
    SIZE_T test_memory_used = end_test_memory - start_test_memory;

    printf("Time taken for testing: %.2f seconds\n", test_time_taken);
    printf("Memory used during testing: %zu bytes (%.2f MB)\n", test_memory_used, (double)test_memory_used / (1024 * 1024));
    fflush(stdout);

    free(ann);
    free(train_data);
    free(buffer);
    free(dim);

    return 0;
}

void train_from_csv(char* filename, char* buffer, double** train_data) {
    FILE *fptr;
    if ((fptr = fopen(filename, "r")) == NULL) {
        printf("Unable to open file %s\n", filename);
        exit(1);
    }

    int count = 0;
    fgets(buffer, BUFFER_SIZE, fptr); // skip header

    char* tokens[785];

    while (fgets(buffer, BUFFER_SIZE, fptr)) {
        tokens[0] = strtok(buffer, ",");
        for (int i = 1; i < 785 && tokens[i - 1] != NULL; i++) {
            tokens[i] = strtok(NULL, ",");
        }

        train_data[count][784] = atof(tokens[0]);

        for (int i = 0; i < 784; i++) {
            train_data[count][i] = atof(tokens[i + 1]) / 255.0;
        }

        count++;
        if (count == MAX_SIZE) {
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < omp_get_num_threads(); i++) {
                    int start = i * (MAX_SIZE / omp_get_num_threads());
                    int end = (i == omp_get_num_threads() - 1) ? MAX_SIZE : start + (MAX_SIZE / omp_get_num_threads());
                    train(ann, &train_data[start], end - start, 0.25);
                }
            }
            printf("Trained 1000 samples..\n");
            fflush(stdout);
            count = 0;
        }
    }

    if (count > 0) {
        train(ann, train_data, count, 0.25);
        printf("Trained remaining %d samples..\n", count);
        fflush(stdout);
    }

    fclose(fptr);
    printf("Done Reading..\n");
    fflush(stdout);
}

void predict_from_csv(char *sourceFile, char* destFile, char* buffer) {
    FILE *fptr;
    FILE *dest;
    if ((fptr = fopen(sourceFile, "r")) == NULL) {
        printf("Unable to open file %s\n", sourceFile);
        exit(1);
    }
    if ((dest = fopen(destFile, "w")) == NULL) {
        printf("Unable to open file %s\n", destFile);
        exit(1);
    }

    fprintf(dest, "ImageId,Label\n");

    fgets(buffer, BUFFER_SIZE, fptr); // skip header

    double** batch_data = init_2Darray(MAX_SIZE, 784);
    int* batch_results = (int*)malloc(MAX_SIZE * sizeof(int));
    int count = 0;
    char* tokens[784];

    while (fgets(buffer, BUFFER_SIZE, fptr)) {
        char* token = strtok(buffer, ",");
        for (int i = 0; i < 784 && token != NULL; i++) {
            tokens[i] = token;
            token = strtok(NULL, ",");
        }

        for (int i = 0; i < 784; i++) {
            batch_data[count][i] = atof(tokens[i]) / 255.0;
        }

        count++;

        if (count == MAX_SIZE) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < count; i++) {
                batch_results[i] = predict(ann, batch_data[i]);
            }

            for (int i = 0; i < count; i++) {
                fprintf(dest, "%d,%d\n", i + 1, batch_results[i]);
                if (i == 16) {
                    save_image_as_png("sample_17_openmp1.png", batch_data[i], 28, 28);
                }
            }

            count = 0;
        }
    }

    if (count > 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; i++) {
            batch_results[i] = predict(ann, batch_data[i]);
        }

        for (int i = 0; i < count; i++) {
            fprintf(dest, "%d,%d\n", i + 1, batch_results[i]);
        }
    }

    fclose(fptr);
    fclose(dest);
    free(batch_data);
    free(batch_results);
}

void save_image_as_png(const char *filename, double *pixels, int width, int height) {
    unsigned char *image_data = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < width * height; i++) {
        image_data[i] = (unsigned char)(pixels[i] * 255);
    }

    stbi_write_png(filename, width, height, 1, image_data, width);
    free(image_data);
}
