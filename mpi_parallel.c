#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> // Include MPI header
#include <string.h>

#define NUM_POINTS 10000000   // Total number of data points
#define NUM_CLUSTERS 50       // Number of clusters
#define DIMENSIONS 3          // Number of dimensions (3D points)
#define ITERATIONS 16         // Number of iterations (adjustable)

// Function to calculate Euclidean distance between two points
double calculate_distance(double *point1, double *point2) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

// Function to initialize centroids randomly from data points
void initialize_centroids(double centroids[NUM_CLUSTERS][DIMENSIONS], double data[][DIMENSIONS], int total_points) {
    srand(0);  // Use fixed seed for reproducibility
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        int random_index = rand() % total_points;
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i][j] = data[random_index][j];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int points_per_process = NUM_POINTS / size;
    double (*local_data)[DIMENSIONS] = malloc(points_per_process * DIMENSIONS * sizeof(double));
    double centroids[NUM_CLUSTERS][DIMENSIONS];
    int *local_cluster_assignment = malloc(points_per_process * sizeof(int));

    // Check memory allocation success
    if (!local_data || !local_cluster_assignment) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Start timing initialization
    double init_start_time = MPI_Wtime();

    // Root process reads data and distributes it
    if (rank == 0) {
        FILE *file = fopen("points.txt", "r");
        if (!file) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double (*global_data)[DIMENSIONS] = malloc(NUM_POINTS * DIMENSIONS * sizeof(double));
        if (!global_data) {
            fprintf(stderr, "Root process: Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Reading data points...\n");
        for (int i = 0; i < NUM_POINTS; i++) {
            if (fscanf(file, "%lf %lf %lf", &global_data[i][0], &global_data[i][1], &global_data[i][2]) != DIMENSIONS) {
                fprintf(stderr, "Error reading data point %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(file);

        // Initialize centroids
        printf("Initializing centroids...\n");
        initialize_centroids(centroids, global_data, NUM_POINTS);

        // Scatter data to all processes
        MPI_Scatter(global_data, points_per_process * DIMENSIONS, MPI_DOUBLE,
                    local_data, points_per_process * DIMENSIONS, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        free(global_data);  // Free global data on the root process
    } else {
        // Receive scattered data
        MPI_Scatter(NULL, points_per_process * DIMENSIONS, MPI_DOUBLE,
                    local_data, points_per_process * DIMENSIONS, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Broadcast initial centroids to all processes
    MPI_Bcast(centroids, NUM_CLUSTERS * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // End timing initialization
    double init_end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Initialization time: %.2f seconds\n", init_end_time - init_start_time);
    }

    // Start timing execution
    double exec_start_time = MPI_Wtime();

    // Main loop: Clustering iterations
    for (int iter = 0; iter < ITERATIONS; iter++) {
        if (rank == 0) {
            printf("Iteration %d\n", iter + 1);
        }

        // Step 1: Assign points to the nearest cluster
        for (int i = 0; i < points_per_process; i++) {
            double min_distance = calculate_distance(local_data[i], centroids[0]);
            int best_cluster = 0;
            for (int j = 1; j < NUM_CLUSTERS; j++) {
                double distance = calculate_distance(local_data[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = j;
                }
            }
            local_cluster_assignment[i] = best_cluster;
        }

        // Step 2: Compute local sums and counts for each cluster
        double local_sum[NUM_CLUSTERS][DIMENSIONS] = {0};
        int local_count[NUM_CLUSTERS] = {0};

        for (int i = 0; i < points_per_process; i++) {
            int cluster = local_cluster_assignment[i];
            for (int j = 0; j < DIMENSIONS; j++) {
                local_sum[cluster][j] += local_data[i][j];
            }
            local_count[cluster]++;
        }

        // Step 3: Reduce local sums and counts to the root process
        double global_sum[NUM_CLUSTERS][DIMENSIONS] = {0};
        int global_count[NUM_CLUSTERS] = {0};

        MPI_Reduce(local_sum, global_sum, NUM_CLUSTERS * DIMENSIONS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_count, global_count, NUM_CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Step 4: Update centroids on the root process
        if (rank == 0) {
            for (int i = 0; i < NUM_CLUSTERS; i++) {
                for (int j = 0; j < DIMENSIONS; j++) {
                    if (global_count[i] > 0) {
                        centroids[i][j] = global_sum[i][j] / global_count[i];
                    }
                }
            }
        }

        // Broadcast updated centroids to all processes
        MPI_Bcast(centroids, NUM_CLUSTERS * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // End timing execution
    double exec_end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution time: %.2f seconds\n", exec_end_time - exec_start_time);
        printf("Total time: %.2f seconds\n", exec_end_time - init_start_time);
    }

    // Final message after clustering is completed
    if (rank == 0) {
        printf("Clustering completed.\n");
    }

    // Cleanup memory
    free(local_data);
    free(local_cluster_assignment);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
