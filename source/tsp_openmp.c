#include <limits.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "tsp.h"

int TSP_greedy_single_openmp(TSP* tsp, int* path, int fuel, int start_city) {
    int* visited = (int*)calloc(tsp->N, sizeof(int));
    if (!visited) {
        return -1;
    }

    int current = start_city, total = 0, fuel_left = fuel;

    for (int i = 0; i <= tsp->N; i++) {
        path[i] = 0;
    }

    path[0] = start_city;
    visited[start_city] = 1;

    int idx = 1;
    for (; idx < tsp->N; idx++) {
        int best = -1, min_cost = INT_MAX;

        for (int i = 0; i < tsp->N; i++) {
            if (!visited[i] && tsp->tsp[current][i] <= fuel_left) {
                if (tsp->tsp[current][i] < min_cost) {
                    best = i;
                    min_cost = tsp->tsp[current][i];
                }
            }
        }

        if (best == -1) {
            free(visited);
            return 0;
        }

        total += min_cost;
        fuel_left -= min_cost;
        fuel_left += tsp->gas_stations[best];
        current = best;
        path[idx] = current;
        visited[current] = 1;
    }

    if (fuel_left < tsp->tsp[current][start_city]) {
        free(visited);
        return 0;
    }

    path[tsp->N] = start_city;
    total += tsp->tsp[current][start_city];

    free(visited);
    return total;
}


int TSP_greedy_multistart_openmp(TSP* tsp, int* path, int fuel, int num_starts) {
    if (!tsp || !path || tsp->N <= 0 || num_starts <= 0) {
        return -1;
    }

    int best_length = INT_MAX;
    bool found_valid_path = false;

    int num_threads = omp_get_max_threads();
    int** thread_best_paths = (int**)malloc(num_threads * sizeof(int*));
    int* thread_best_lengths = (int*)malloc(num_threads * sizeof(int));
    bool* thread_found_valid = (bool*)calloc(num_threads, sizeof(bool));

    if (!thread_best_paths || !thread_best_lengths || !thread_found_valid) {
        if (thread_best_paths) free(thread_best_paths);
        if (thread_best_lengths) free(thread_best_lengths);
        if (thread_found_valid) free(thread_found_valid);
        return -1;
    }

    for (int i = 0; i < num_threads; i++) {
        thread_best_paths[i] = (int*)malloc((tsp->N + 1) * sizeof(int));
        if (!thread_best_paths[i]) {
            for (int j = 0; j < i; j++) {
                free(thread_best_paths[j]);
            }
            free(thread_best_paths);
            free(thread_best_lengths);
            free(thread_found_valid);
            return -1;
        }
        thread_best_lengths[i] = INT_MAX;
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int* temp_path = (int*)malloc((tsp->N + 1) * sizeof(int));
        if (temp_path) {
            unsigned int thread_seed = time(NULL) + thread_id;

            #pragma omp for schedule(dynamic, 1)
            for (int start_idx = 0; start_idx < num_starts; start_idx++) {
                int start_city;

                if (start_idx < tsp->N) {
                    start_city = start_idx;
                } else {
                    start_city = rand_r(&thread_seed) % tsp->N;
                }

                int length = TSP_greedy_single_openmp(tsp, temp_path, fuel, start_city);

                if (length > 0 && (length < thread_best_lengths[thread_id] || !thread_found_valid[thread_id])) {
                    thread_best_lengths[thread_id] = length;
                    TSP_copy_array(temp_path, thread_best_paths[thread_id], tsp->N + 1);
                    thread_found_valid[thread_id] = true;
                }
            }

            free(temp_path);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if (thread_found_valid[i] && (thread_best_lengths[i] < best_length || !found_valid_path)) {
            best_length = thread_best_lengths[i];
            TSP_copy_array(thread_best_paths[i], path, tsp->N + 1);
            found_valid_path = true;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        free(thread_best_paths[i]);
    }
    free(thread_best_paths);
    free(thread_best_lengths);
    free(thread_found_valid);

    if (found_valid_path) {
        return best_length;
    }

    return -1;
}

int TSP_local_search_mutation_openmp(TSP* tsp, int* path, int fuel, int iterations) {
    if (!tsp || !path || tsp->N <= 0 || iterations <= 0) {
        return -1;
    }

    unsigned int seed = time(NULL);
    int start_city = rand_r(&seed) % tsp->N;

    int initial_length = TSP_calculate_path_length(tsp, path, fuel);
    if (initial_length <= 0 || !TSP_check_if_route_ok(tsp, path, fuel)) {
        path[0] = start_city;

        int idx = 1;
        for (int i = 0; i < tsp->N; i++) {
            if (i != start_city) {
                path[idx++] = i;
            }
        }
        path[tsp->N] = start_city;

        TSP_shuffle(path + 1, tsp->N - 1, seed, 1);

        if (!TSP_check_if_route_ok(tsp, path, fuel)) {
            return -1;
        }
        initial_length = TSP_calculate_path_length(tsp, path, fuel);
    }

    int* best = (int*)malloc((tsp->N + 1) * sizeof(int));
    if (!best) return -1;

    int best_len = initial_length;
    TSP_copy_array(path, best, tsp->N + 1);

    int num_threads = omp_get_max_threads();
    int** thread_best_paths = (int**)malloc(num_threads * sizeof(int*));
    int* thread_best_lengths = (int*)malloc(num_threads * sizeof(int));

    if (!thread_best_paths || !thread_best_lengths) {
        if (best) free(best);
        if (thread_best_paths) free(thread_best_paths);
        if (thread_best_lengths) free(thread_best_lengths);
        return -1;
    }

    for (int i = 0; i < num_threads; i++) {
        thread_best_paths[i] = (int*)malloc((tsp->N + 1) * sizeof(int));
        if (!thread_best_paths[i]) {
            for (int j = 0; j < i; j++) {
                free(thread_best_paths[j]);
            }
            free(thread_best_paths);
            free(thread_best_lengths);
            free(best);
            return -1;
        }
        TSP_copy_array(best, thread_best_paths[i], tsp->N + 1);
        thread_best_lengths[i] = best_len;
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int* candidate = (int*)malloc((tsp->N + 1) * sizeof(int));
        unsigned int thread_seed = time(NULL) + thread_id;

        if (candidate) {
            #pragma omp for schedule(dynamic, 10)
            for (int i = 0; i < iterations; i++) {
                int op_type = rand_r(&thread_seed) % 3;

                TSP_copy_array(thread_best_paths[thread_id], candidate, tsp->N + 1);

                int start_end_city = candidate[0];

                switch (op_type) {
                    case 0:
                        {
                            int idx1 = 1 + rand_r(&thread_seed) % (tsp->N - 1);
                            int idx2 = 1 + rand_r(&thread_seed) % (tsp->N - 1);
                            if (idx1 != idx2) {
                                int temp = candidate[idx1];
                                candidate[idx1] = candidate[idx2];
                                candidate[idx2] = temp;
                            }
                        }
                        break;
                    case 1:
                        {
                            int idx1 = 1 + rand_r(&thread_seed) % (tsp->N - 2);
                            int idx2 = idx1 + 1 + rand_r(&thread_seed) % (tsp->N - idx1 - 1);
                            while (idx1 < idx2) {
                                int temp = candidate[idx1];
                                candidate[idx1] = candidate[idx2];
                                candidate[idx2] = temp;
                                idx1++;
                                idx2--;
                            }
                        }
                        break;
                    case 2:
                        {
                            int idx1 = 1 + rand_r(&thread_seed) % (tsp->N - 1);
                            int idx2 = 1 + rand_r(&thread_seed) % (tsp->N - 1);
                            int idx3 = 1 + rand_r(&thread_seed) % (tsp->N - 1);
                            if (idx1 != idx2 && idx2 != idx3 && idx1 != idx3) {
                                int temp = candidate[idx1];
                                candidate[idx1] = candidate[idx2];
                                candidate[idx2] = candidate[idx3];
                                candidate[idx3] = temp;
                            }
                        }
                        break;
                }

                candidate[tsp->N] = start_end_city;

                if (TSP_check_if_route_ok(tsp, candidate, fuel)) {
                    int candidate_len = TSP_calculate_path_length(tsp, candidate, fuel);
                    if (candidate_len > 0 && candidate_len < thread_best_lengths[thread_id]) {
                        thread_best_lengths[thread_id] = candidate_len;
                        TSP_copy_array(candidate, thread_best_paths[thread_id], tsp->N + 1);
                    }
                }
            }
            free(candidate);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if (thread_best_lengths[i] > 0 && thread_best_lengths[i] < best_len) {
            best_len = thread_best_lengths[i];
            TSP_copy_array(thread_best_paths[i], best, tsp->N + 1);
        }
    }

    TSP_copy_array(best, path, tsp->N + 1);

    for (int i = 0; i < num_threads; i++) {
        free(thread_best_paths[i]);
    }
    free(thread_best_paths);
    free(thread_best_lengths);
    free(best);

    return best_len > 0 ? best_len : -1;
}