# Parallel K-Means Benchmark (Serial, OpenMP, MPI)

This repo benchmarks a simple 3D k-means–style clustering workload in three execution modes: pure serial C, OpenMP for shared-memory parallelism, and MPI for distributed-memory parallelism. A Python helper generates synthetic points, and another script plots the captured timings.

## Repository Layout
- `serial.c` – single-threaded baseline implementation.
- `openMP_parallel.c` – OpenMP version with SIMD and threaded loops.
- `mpi_parallel.c` – MPI version that scatters the dataset across ranks.
- `makePoints.py` – generates the `points.txt` dataset (default: 10,000,000 points).
- `calculate_findings.py` – plots timing, speedup, and efficiency from `findings.txt`.
- `findings.txt` – recorded runtimes for various point counts and thread/process counts.
- `IMAGES/` – saved plot outputs produced by `calculate_findings.py`.
- `21K_3339_21K_3403_21K_3383_REPORT.docx` – project report (not needed to run code).

## Prerequisites
- C compiler with math library (e.g., `gcc`) and support for:
  - OpenMP (`-fopenmp`).
  - MPI toolchain (`mpicc`, `mpirun`/`mpiexec`) from an MPI distribution (OpenMPI/MPICH/MS-MPI).
- Python 3 with `matplotlib` and `numpy` (`pip install matplotlib numpy`).
- Sufficient RAM and disk: the default 10M-point dataset is ~240 MB of doubles plus cluster arrays.

## Generate Input Data
The C programs read `points.txt` from the working directory.
```
python makePoints.py         # writes points.txt with 10,000,000 random 3D points
```
To use a smaller dataset, edit the last line in `makePoints.py` (e.g., `generate_3d_points(100000, "points.txt")`) or change the `NUM_POINTS` macro in the C sources to match.

## Build
Use `-O3` for performance and link `-lm` for math. Examples:
```
# Serial
gcc -O3 -lm serial.c -o serial

# OpenMP
gcc -O3 -fopenmp -lm openMP_parallel.c -o openmp_parallel

# MPI
mpicc -O3 -lm mpi_parallel.c -o mpi_parallel
```
On Windows with MSYS2/MinGW, ensure `gcc`, `mpicc`, and `libgomp`/MPI runtime are on PATH; with MS-MPI, use `cl /openmp` or `mpicc` equivalents as available.

## Run
All executables expect `points.txt` in the current directory.
```
./serial

OMP_NUM_THREADS=8 ./openmp_parallel     # override thread count if desired

mpirun -np 4 ./mpi_parallel             # choose process count; NUM_POINTS divides by ranks
```
### Tuning Parameters
- `NUM_POINTS`, `NUM_CLUSTERS`, `DIMENSIONS`, `ITERATIONS` are set via macros near the top of each C file.
- For OpenMP, threads can be set with `OMP_NUM_THREADS` or by editing the `omp_set_num_threads` call.
- For MPI, `NUM_POINTS` should be divisible by `-np`; otherwise some points are ignored with the current logic.

### Performance Notes
Expected magnitudes (from `findings.txt`, initialization excluded):
- Serial: ~1674 s for 10M points.
- OpenMP: ~321–333 s for 10M points with 10–20 threads.
- MPI: ~327–353 s for 10M points with 8–20 processes (13 s init observed separately).

## Plot Collected Results
`calculate_findings.py` uses the hardcoded data in `findings.txt` to plot execution time, speedup, and efficiency. To regenerate plots (interactive windows):
```
python calculate_findings.py
```
Saved PNGs are in `IMAGES/` (`openmp_*`, `mpi_*`, `serial_obs.png`).

## Troubleshooting
- Large runs are memory- and IO-heavy; consider reducing `NUM_POINTS` during testing.
- Ensure the compiler flags match the toolchain: some MPI wrappers already link `-lm`.
- On headless systems, set `matplotlib` to a non-interactive backend (e.g., `Agg`) before importing pyplot if you need to save figures without a display.

## Notes and Potential Improvements
- Current implementations read all data into memory; chunked IO would improve scalability.
- MPI version assumes even division of `NUM_POINTS` by ranks; adding remainder handling would avoid dropped points.
- Convergence/termination is a fixed iteration count; a tolerance-based stop could reduce work.
