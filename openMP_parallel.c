#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // Include OpenMP header

#define NUM_POINTS 10000000   // Total number of data points
#define NUM_CLUSTERS 50       // Number of clusters
#define DIMENSIONS 3          // Number of dimensions (3D points)
#define ITERATIONS 16         // Number of iterations (adjustable)

// Function to calculate Euclidean distance
double calculate_distance(double *point1, double *point2) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

// Function to randomly select initial centroids
void initialize_centroids(double centroids[NUM_CLUSTERS][DIMENSIONS], double data[NUM_POINTS][DIMENSIONS]) {
    srand(time(NULL));  // Seed random number generator
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        int random_index = rand() % NUM_POINTS;
        #pragma omp simd
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = data[random_index][j];
        }
        #pragma omp critical
        {
            printf("Initial centroid %d: %.2lf %.2lf %.2lf\n", i + 1, centroids[i][0], centroids[i][1], centroids[i][2]);
        }
    }
}

int main() {
    double (*data)[DIMENSIONS] = malloc(NUM_POINTS * DIMENSIONS * sizeof(double));  // Array for data points
    double centroids[NUM_CLUSTERS][DIMENSIONS];                                    // Array for centroids
    int *cluster_assignment = malloc(NUM_POINTS * sizeof(int));                   // Array for cluster assignments
    omp_set_num_threads(20);

    if (!data || !cluster_assignment) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1; 
    }

    FILE *file = fopen("points.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        free(data);
        free(cluster_assignment);
        return 1;
    }

    printf("Reading data points...\n");
    double start_time = omp_get_wtime();  // Start timing initialization

    // Read data points sequentially (file I/O is not parallelized)
    for (int i = 0; i < NUM_POINTS; i++) {
        if (fscanf(file, "%lf %lf %lf", &data[i][0], &data[i][1], &data[i][2]) != DIMENSIONS) {
            fprintf(stderr, "Error reading data point %d\n", i);
            break;
        }
    }
    fclose(file);

    printf("Initializing centroids...\n");
    initialize_centroids(centroids, data);
    double init_end_time = omp_get_wtime();  // End timing initialization

    printf("Initialization completed in %.2f seconds.\n", init_end_time - start_time);

    printf("Starting clustering...\n");
    double exec_start_time = omp_get_wtime();  // Start timing execution

    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("Iteration %d\n", iter + 1);

        // Step 1: Assign each point to the nearest centroid
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NUM_POINTS; i++) {
            double min_distance = calculate_distance(data[i], centroids[0]);
            int best_cluster = 0;
            #pragma omp simd
            for (int j = 1; j < NUM_CLUSTERS; j++) {
                double distance = calculate_distance(data[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = j;
                }
            }
            cluster_assignment[i] = best_cluster;
        }

        // Step 2: Recompute centroids
        double sum[NUM_CLUSTERS][DIMENSIONS] = {0};
        int count[NUM_CLUSTERS] = {0};

        // Parallelize the accumulation of sums and counts
        #pragma omp parallel for schedule(static) reduction(+:sum[:NUM_CLUSTERS][:DIMENSIONS], count[:NUM_CLUSTERS])
        for (int i = 0; i < NUM_POINTS; i++) {
            int cluster = cluster_assignment[i];
            #pragma omp simd
            for (int j = 0; j < DIMENSIONS; j++) {
                sum[cluster][j] += data[i][j];
            }
            count[cluster]++;
        }

        // Update centroids
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            #pragma omp simd
            for (int j = 0; j < DIMENSIONS; j++) {
                if (count[i] > 0) {
                    centroids[i][j] = sum[i][j] / count[i];
                }
            }
            #pragma omp critical
            {
                printf("Updated centroid %d: %.2lf %.2lf %.2lf (Count: %d)\n", i + 1, centroids[i][0], centroids[i][1], centroids[i][2], count[i]);
            }
        }
    }

    double exec_end_time = omp_get_wtime();  // End timing execution

    // Print timing information
    printf("Execution completed in %.2f seconds.\n", exec_end_time - exec_start_time);
    printf("Initialization time: %.2f seconds.\n", init_end_time - start_time);
    printf("Total time: %.2f seconds.\n", exec_end_time - start_time);

    free(data);
    free(cluster_assignment);
    printf("Clustering completed.\n");
    return 0;
}
