#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 10000000   // Total number of data points
#define NUM_CLUSTERS 50      // Number of clusters
#define DIMENSIONS 3         // Number of dimensions (3D points)
#define ITERATIONS 16         // Number of iterations (adjustable)

// Function to calculate Euclidean distance
double calculate_distance(double *point1, double *point2) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

// Function to randomly select initial centroids
void initialize_centroids(double centroids[NUM_CLUSTERS][DIMENSIONS], double data[NUM_POINTS][DIMENSIONS]) {
    srand(time(NULL));  // Seed random number generator
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        int random_index = rand() % NUM_POINTS;
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = data[random_index][j];
        }
        printf("Initial centroid %d: %.2lf %.2lf %.2lf\n", i + 1, centroids[i][0], centroids[i][1], centroids[i][2]);  // Debugging line
    }
}

int main() {
    FILE *file;
    double (*data)[DIMENSIONS] = malloc(NUM_POINTS * DIMENSIONS * sizeof(double));  // Array for data points
    double centroids[NUM_CLUSTERS][DIMENSIONS];                                    // Array for centroids
    int *cluster_assignment = malloc(NUM_POINTS * sizeof(int));                   // Array for cluster assignments

    if (!data || !cluster_assignment) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Start timing for data initialization
    clock_t start_init = clock();

    // Open file and read data points
    file = fopen("points.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        free(data);
        free(cluster_assignment);
        return 1;
    }

    printf("Reading data points...\n");
    for (int i = 0; i < NUM_POINTS; i++) {
        if (fscanf(file, "%lf %lf %lf", &data[i][0], &data[i][1], &data[i][2]) != DIMENSIONS) {
            fprintf(stderr, "Error reading data point %d\n", i);
            break;
        }
    }
    fclose(file);

    // Initialize centroids
    printf("Initializing centroids...\n");
    initialize_centroids(centroids, data);

    // End timing for data initialization
    clock_t end_init = clock();
    double init_time = ((double)(end_init - start_init)) / CLOCKS_PER_SEC;
    printf("Data initialization time: %.2lf seconds\n", init_time);

    // Start timing for computation
    clock_t start_computation = clock();

    // Start clustering
    printf("Starting clustering...\n");
    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("Iteration %d\n", iter + 1);

        // Step 1: Assign each point to the nearest centroid
        for (int i = 0; i < NUM_POINTS; i++) {
            double min_distance = calculate_distance(data[i], centroids[0]);
            int best_cluster = 0;
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
        for (int i = 0; i < NUM_POINTS; i++) {
            int cluster = cluster_assignment[i];
            for (int j = 0; j < DIMENSIONS; j++) {
                sum[cluster][j] += data[i][j];
            }
            count[cluster]++;
        }
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            for (int j = 0; j < DIMENSIONS; j++) {
                if (count[i] > 0) {
                    centroids[i][j] = sum[i][j] / count[i];
                }
            }
        }
    }

    // End timing for computation
    clock_t end_computation = clock();
    double computation_time = ((double)(end_computation - start_computation)) / CLOCKS_PER_SEC;
    printf("Computation time: %.2lf seconds\n", computation_time);

    // Cleanup
    free(data);
    free(cluster_assignment);
    printf("Clustering completed.\n");
    return 0;
}
