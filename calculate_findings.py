import matplotlib.pyplot as plt
import numpy as np

# Data
points = [10_000_000, 1_000_000, 100_000, 10_000]
serial_times = [1674, 163.46, 17.14, 1.82]

# OpenMP times (threads 1/2/4/8/10/12/20)
openmp_times = {
    1: [1671, 177.36, 18.10, 2.06],
    2: [883.20, 98.50, 9.61, 1.24],
    4: [563.05, 58.81, 5.78, 0.71],
    8: [397.75, 32.60, 3.63, 0.48],
    10: [321.01, 32.84, 3.79, 0.56],
    12: [322.12, 30.31, 3.44, 0.53],
    20: [333.48, 29.77, 3.22, 0.49],
}

# MPI times (processes 1/2/4/8/10/12/20)
mpi_times = {
    1: [844, 84.64, 8.41, 0.87],
    2: [471.89, 46.26, 5.24, 0.51],
    4: [337, 33.88, 3.61, 0.44],
    8: [327.14, 32.05, 4.04, 1.03],
    10: [326.75, 36.82, 4.87, 1.52],
    12: [342.34, 34.46, 6.18, 1.98],
    20: [353.23, 35.47, 8.05, 4.22],
}

# Prepare points for log-scale plots
log_points = np.log10(points)

# Plot Serial Execution Time
plt.figure(figsize=(10, 6))
plt.plot(points, serial_times, marker='o', label='Serial Execution')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Execution Time (log scale)')
plt.title('Serial Execution Time vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Plot OpenMP Execution Time
plt.figure(figsize=(12, 8))
for threads, times in openmp_times.items():
    plt.plot(points, times, marker='o', label=f'{threads} Threads')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Execution Time (log scale)')
plt.title('OpenMP Execution Time vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Plot MPI Execution Time
plt.figure(figsize=(12, 8))
for processes, times in mpi_times.items():
    plt.plot(points, times, marker='o', label=f'{processes} Processes')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Execution Time (log scale)')
plt.title('MPI Execution Time vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Speed-Up and Efficiency Calculations
serial_base = np.array(serial_times)

# OpenMP Speed-Up and Efficiency
openmp_speedup = {threads: serial_base / np.array(times) for threads, times in openmp_times.items()}
openmp_efficiency = {threads: (serial_base / np.array(times)) / threads for threads, times in openmp_times.items()}

# MPI Speed-Up and Efficiency
mpi_speedup = {processes: serial_base / np.array(times) for processes, times in mpi_times.items()}
mpi_efficiency = {processes: (serial_base / np.array(times)) / processes for processes, times in mpi_times.items()}

# Plot OpenMP Speed-Up
plt.figure(figsize=(12, 8))
for threads, speedup in openmp_speedup.items():
    plt.plot(points, speedup, marker='o', label=f'{threads} Threads')
plt.xscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Speed-Up')
plt.title('OpenMP Speed-Up vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Plot MPI Speed-Up
plt.figure(figsize=(12, 8))
for processes, speedup in mpi_speedup.items():
    plt.plot(points, speedup, marker='o', label=f'{processes} Processes')
plt.xscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Speed-Up')
plt.title('MPI Speed-Up vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Plot OpenMP Efficiency
plt.figure(figsize=(12, 8))
for threads, efficiency in openmp_efficiency.items():
    plt.plot(points, efficiency, marker='o', label=f'{threads} Threads')
plt.xscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Efficiency')
plt.title('OpenMP Efficiency vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Plot MPI Efficiency
plt.figure(figsize=(12, 8))
for processes, efficiency in mpi_efficiency.items():
    plt.plot(points, efficiency, marker='o', label=f'{processes} Processes')
plt.xscale('log')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Efficiency')
plt.title('MPI Efficiency vs Number of Points')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
