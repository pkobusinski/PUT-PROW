#include <limits.h>
#include <time.h>
#include "tsp.h"

int TSP_greedy_single(TSP* tsp, int* path, int fuel, int start_city) {
    int* visited = (int*)calloc(tsp->N, sizeof(int));
    int current = start_city, total = 0, fuel_left = fuel;

    for (int i = 0; i <= tsp->N; i++) {
        path[i] = 0;
    }

    path[0] = start_city;
    visited[start_city] = 1;
    current = start_city;

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


int TSP_greedy_multistart(TSP* tsp, int* path, int fuel, int num_starts) {
    int* best_path = (int*)malloc((tsp->N + 1) * sizeof(int));
    int* temp_path = (int*)malloc((tsp->N + 1) * sizeof(int));
    int best_length = INT_MAX;
    unsigned int seed = time(NULL);
    bool found_valid_path = false;

    int start_city = rand_r(&seed) % tsp->N;

    int length = TSP_greedy_single(tsp, temp_path, fuel, start_city);
    if (length > 0) {
        best_length = length;
        TSP_copy_array(temp_path, best_path, tsp->N + 1);
        found_valid_path = true;
    }

    for (int i = 1; i < num_starts; i++) {
        start_city = rand_r(&seed) % tsp->N;

        length = TSP_greedy_single(tsp, temp_path, fuel, start_city);

        if (length > 0 && (length < best_length || !found_valid_path)) {
            best_length = length;
            TSP_copy_array(temp_path, best_path, tsp->N + 1);
            found_valid_path = true;
        }
    }

    if (found_valid_path) {
        TSP_copy_array(best_path, path, tsp->N + 1);
        free(best_path);
        free(temp_path);
        return best_length;
    }

    free(best_path);
    free(temp_path);
    return -1;
}


int TSP_local_search_mutation(TSP* tsp, int* path, int fuel, int iterations) {
    int initial_length = TSP_calculate_path_length(tsp, path, fuel);
    int* best = (int*)malloc((tsp->N + 1) * sizeof(int));
    int* candidate = (int*)malloc((tsp->N + 1) * sizeof(int));
    if (!best || !candidate) {
        if (best) free(best);
        if (candidate) free(candidate);
        return -1;
    }

    unsigned int seed = time(NULL);
    int best_len = initial_length;

    TSP_copy_array(path, best, tsp->N + 1);

    int start_city = path[0];

    for (int i = 0; i < iterations; i++) {
        TSP_copy_array(best, candidate, tsp->N + 1);

        int idx1 = 1 + rand_r(&seed) % (tsp->N - 1);
        int idx2 = 1 + rand_r(&seed) % (tsp->N - 1);
        if (idx1 != idx2) {
            int temp = candidate[idx1];
            candidate[idx1] = candidate[idx2];
            candidate[idx2] = temp;
        }

        candidate[0] = start_city;
        candidate[tsp->N] = start_city;

        if (TSP_check_if_route_ok(tsp, candidate, fuel)) {
            int candidate_len = TSP_calculate_path_length(tsp, candidate, fuel);
            if (candidate_len > 0 && candidate_len < best_len) {
                best_len = candidate_len;
                TSP_copy_array(candidate, best, tsp->N + 1);
            }
        }
    }

    TSP_copy_array(best, path, tsp->N + 1);
    free(best);
    free(candidate);

    return best_len > 0 ? best_len : -1;
}