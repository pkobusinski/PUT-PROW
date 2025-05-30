#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "tsp.h"

#define MAX_CITIES 1024
#define TRIALS_PER_THREAD 5
#define MAX_THREADS_PER_BLOCK 1024
#define NO_IMPROVEMENT_LIMIT 200
#define MUTATIONS_BEFORE_EVAL 10

inline int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power <<= 1;
    return power;
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ int device_calculate_path_length(int* tsp_matrix, int* path, int n) {
    int length = 0;
    for (int i = 0; i < n; i++) {
        length += tsp_matrix[path[i] * n + path[i + 1]];
    }
    return length;
}

__device__ bool device_check_if_route_ok(int* tsp_matrix, int* gas_stations, int* path, int n, int fuel) {
    int current_fuel = fuel;
    for (int i = 0; i < n; i++) {
        int from = path[i];
        int to = path[i + 1];
        int distance = tsp_matrix[from * n + to];
        if (current_fuel < distance) return false;
        current_fuel -= distance;
        current_fuel += gas_stations[to];
    }
    return true;
}

__global__ void kernel_greedy_multistart(
    int* tsp_matrix, int* gas_stations, int* all_paths, int* all_lengths,
    int n, int fuel, int trials_per_thread, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(seed + tid, 0, 0, &state);

    int best_len = INT_MAX;
    int path[MAX_CITIES + 1];
    int best_path[MAX_CITIES + 1];

    for (int t = 0; t < trials_per_thread; t++) {
        int start_city = curand(&state) % n;
        int fuel_left = fuel;
        bool visited[MAX_CITIES] = {0};

        path[0] = start_city;
        visited[start_city] = true;

        int current = start_city;
        int idx = 1;

        while (idx < n) {
            int best = -1, min_cost = INT_MAX;
            for (int i = 0; i < n; i++) {
                if (!visited[i] && tsp_matrix[current * n + i] <= fuel_left) {
                    if (tsp_matrix[current * n + i] < min_cost) {
                        best = i;
                        min_cost = tsp_matrix[current * n + i];
                    }
                }
            }

            if (best == -1) break;
            path[idx++] = best;
            visited[best] = true;
            fuel_left -= min_cost;
            fuel_left += gas_stations[best];
            current = best;
        }

        if (idx == n && fuel_left >= tsp_matrix[current * n + start_city]) {
            path[n] = start_city;
            int length = device_calculate_path_length(tsp_matrix, path, n);
            if (length < best_len) {
                best_len = length;
                for (int i = 0; i <= n; i++) best_path[i] = path[i];
            }
        }
    }

    for (int i = 0; i <= n; i++) all_paths[tid * (n + 1) + i] = best_path[i];
    all_lengths[tid] = best_len;
}

int TSP_greedy_multistart_cuda(TSP* tsp, int* path, int fuel, int num_starts) {
    int N = tsp->N;

    int num_threads = num_starts;
    int block_size = 512;
    int grid_size = (num_threads + block_size - 1) / block_size;
    int trials_per_thread = (num_starts + num_threads - 1) / num_threads;

    int *d_tsp_matrix, *d_gas_stations, *d_all_paths, *d_all_lengths;

    int* flattened_tsp = (int*)malloc(N * N * sizeof(int));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            flattened_tsp[i * N + j] = tsp->tsp[i][j];

    cudaMalloc(&d_tsp_matrix, N * N * sizeof(int));
    cudaMalloc(&d_gas_stations, N * sizeof(int));
    cudaMalloc(&d_all_paths, num_threads * (N + 1) * sizeof(int));
    cudaMalloc(&d_all_lengths, num_threads * sizeof(int));

    cudaMemcpy(d_tsp_matrix, flattened_tsp, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gas_stations, tsp->gas_stations, N * sizeof(int), cudaMemcpyHostToDevice);

    unsigned long seed = time(NULL);
    kernel_greedy_multistart<<<grid_size, block_size>>>(d_tsp_matrix, d_gas_stations,
                                                        d_all_paths, d_all_lengths,
                                                        N, fuel, trials_per_thread, seed);

    int* h_all_paths = (int*)malloc(num_threads * (N + 1) * sizeof(int));
    int* h_all_lengths = (int*)malloc(num_threads * sizeof(int));
    cudaMemcpy(h_all_paths, d_all_paths, num_threads * (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_lengths, d_all_lengths, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    int best_len = INT_MAX;
    bool found_valid_path = false;
    for (int t = 0; t < num_threads; t++) {
        if (h_all_lengths[t] > 0 && (h_all_lengths[t] < best_len || !found_valid_path)) {
            best_len = h_all_lengths[t];
            for (int i = 0; i <= N; i++) {
                path[i] = h_all_paths[t * (N + 1) + i];
            }
            found_valid_path = true;
        }
    }

    free(flattened_tsp);
    free(h_all_paths);
    free(h_all_lengths);
    cudaFree(d_tsp_matrix);
    cudaFree(d_gas_stations);
    cudaFree(d_all_paths);
    cudaFree(d_all_lengths);

    return found_valid_path ? best_len : -1;
}

__global__ void kernel_local_search(
    int* tsp_matrix, int* gas_stations, int* initial_path,
    int* all_paths, int* all_lengths,
    int n, int fuel, int iterations_per_thread, unsigned long seed)
{
    extern __shared__ int shared_memory[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int* local_path = &shared_memory[threadIdx.x * (n + 1)];

    curandState_t state;
    curand_init(seed + tid, 0, 0, &state);

    for (int i = 0; i <= n; i++) {
        local_path[i] = initial_path[i];
    }

    int best_len = 0;
    for (int i = 0; i < n; i++) {
        int from = local_path[i];
        int to = local_path[i + 1];
        int edge_len = tsp_matrix[from * n + to];
        best_len += edge_len;
    }

    all_lengths[tid] = best_len;

    int no_improvement = 0;
    int current_iteration = 0;

    while (current_iteration < iterations_per_thread && no_improvement < NO_IMPROVEMENT_LIMIT) {
        int idx1 = 1 + (curand(&state) % (n - 1));
        int idx2 = 1 + (curand(&state) % (n - 1));
        if (idx1 == idx2) continue;

        int temp = local_path[idx1];
        local_path[idx1] = local_path[idx2];
        local_path[idx2] = temp;

        int new_len = 0;
        for (int i = 0; i < n; i++) {
            int from = local_path[i];
            int to = local_path[i + 1];
            new_len += tsp_matrix[from * n + to];
        }

        bool valid = true;
        int fuel_check = fuel;
        for (int i = 0; i < n; i++) {
            int from = local_path[i];
            int to = local_path[i + 1];
            int distance = tsp_matrix[from * n + to];
            if (fuel_check < distance) {
                valid = false;
                break;
            }
            fuel_check -= distance;
            fuel_check += gas_stations[to];
        }

        if (valid && new_len < best_len) {
            best_len = new_len;
            all_lengths[tid] = best_len;
            no_improvement = 0;
        } else {
            temp = local_path[idx1];
            local_path[idx1] = local_path[idx2];
            local_path[idx2] = temp;
            no_improvement++;
        }

        current_iteration++;
    }
    for (int i = 0; i <= n; i++) {
        all_paths[tid * (n + 1) + i] = local_path[i];
    }
    all_lengths[tid] = best_len;
}

int TSP_local_search_mutation_cuda(TSP* tsp, int* path, int fuel, int iterations) {
    int N = tsp->N;

    int initial_length = 0;
    for (int i = 0; i < N; i++) {
        initial_length += tsp->tsp[path[i]][path[i + 1]];
    }
    printf("Initial path length: %d\n", initial_length);

    bool valid_path = true;
    int current_fuel = fuel;
    for (int i = 0; i < N; i++) {
        int from = path[i];
        int to = path[i + 1];
        int distance = tsp->tsp[from][to];
        if (current_fuel < distance) {
            valid_path = false;
            break;
        }
        current_fuel -= distance;
        current_fuel += tsp->gas_stations[to];
    }

    if (!valid_path) {
        printf("Initial path is invalid\n");
        return -1;
    }

    printf("Initial path is valid\n");

    bool improved = false;
    int best_length = initial_length;

    for (int iter = 0; iter < min(iterations, 10000); iter++) {
        int idx1 = 1 + rand() % (N - 1);
        int idx2 = 1 + rand() % (N - 1);
        if (idx1 == idx2) continue;

        int temp = path[idx1];
        path[idx1] = path[idx2];
        path[idx2] = temp;


        int new_length = 0;
        for (int i = 0; i < N; i++) {
            new_length += tsp->tsp[path[i]][path[i + 1]];
        }

        bool path_ok = true;
        int test_fuel = fuel;
        for (int i = 0; i < N; i++) {
            int from = path[i];
            int to = path[i + 1];
            int dist = tsp->tsp[from][to];
            if (test_fuel < dist) {
                path_ok = false;
                break;
            }
            test_fuel -= dist;
            test_fuel += tsp->gas_stations[to];
        }

        if (path_ok && new_length < best_length) {
            best_length = new_length;
            improved = true;
        } else {
            temp = path[idx1];
            path[idx1] = path[idx2];
            path[idx2] = temp;
        }
    }
    if (!improved) {
        best_length = initial_length;
    }

    return best_length;
}

