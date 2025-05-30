#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tsp.h"
#include "generator.h"
#include "zen_timer.h"

void solveAndSave(TSP* tsp, int mode) {
    int* greedy_path = (int*)malloc((tsp->N + 1) * sizeof(int));
    int* path = (int*)malloc((tsp->N + 1) * sizeof(int));
    zen_timer_t timer;
    int64_t time;
    double path_length;
    FILE* results_file = fopen("results.txt", "a");
    FILE* json_file = fopen("results.json", "a");


    if (!results_file || !json_file) {
        printf("Error opening results file(s)\n");
        return;
    }

    // SEQUENTIAL
    if (mode == 1 || mode == 4 || mode == 5 || mode == 6) {
        timer = ZenTimer_Start(false);
        TSP_greedy_multistart(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Greedy Multi-Start (Sequential)", greedy_path,
                      TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Greedy Multi-Start (Sequential)", greedy_path,
                       TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);

        timer = ZenTimer_Start(false);
        path_length = TSP_local_search_mutation(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Local Search with Mutations (Sequential)", greedy_path,
                      path_length, tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Local Search with Mutations (Sequential)", greedy_path,
                       path_length, tsp->N + 1, time);
    }

    // OPENMP
    if (mode == 2 || mode == 4 || mode == 6 ) {
        timer = ZenTimer_Start(false);
        TSP_greedy_multistart_openmp(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Greedy Multi-Start (OpenMP)", greedy_path,
                      TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Greedy Multi-Start (OpenMP)", greedy_path,
                       TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);

        timer = ZenTimer_Start(false);
        path_length = TSP_local_search_mutation_openmp(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Local Search with Mutations (OpenMP)", greedy_path,
                      path_length, tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Local Search with Mutations (OpenMP)", greedy_path,
                       path_length, tsp->N + 1, time);
    }

    // CUDA
    if (mode == 3 || mode == 5 || mode == 6 ) {
        timer = ZenTimer_Start(true);
        TSP_greedy_multistart_cuda(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Greedy Multi-Start (CUDA)", greedy_path,
                      TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Greedy Multi-Start (CUDA)", greedy_path,
                       TSP_calculate_path_length(tsp, greedy_path, tsp->start_fuel), tsp->N + 1, time);

        timer = ZenTimer_Start(true);
        path_length = TSP_local_search_mutation_cuda(tsp, greedy_path, tsp->start_fuel, 1000);
        time = ZenTimer_End(&timer);
        TSP_saveToTXT(tsp, results_file, "Local Search with Mutations (CUDA)", greedy_path,
                      path_length, tsp->N + 1, time);
        TSP_saveToJSON(tsp, json_file, "Local Search with Mutations (CUDA)", greedy_path,
                       path_length, tsp->N + 1, time);
    }

    free(path);
    free(greedy_path);
    fclose(results_file);
    fclose(json_file);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    int min_cities = 100;
    int max_cities = 1000;
    int city_jump = 100;
    int num_trials = 10;
    int mode = 3;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-cities") == 0 && i + 1 < argc) {
            min_cities = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-cities") == 0 && i + 1 < argc) {
            max_cities = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--city-jump") == 0 && i + 1 < argc) {
            city_jump = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            num_trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --min-cities N   Set the minimum number of cities (default: 100)\n");
            printf("  --max-cities N   Set the maximum number of cities (default: 1000)\n");
            printf("  --city-jump N    Set the increment between city sizes (default: 100)\n");
            printf("  --trials N       Set the number of trials per city size (default: 10)\n");
            printf("  --mode N         Execution mode:\n");
            printf("                   1: Sequential\n");
            printf("                   2: OpenMP\n");
            printf("                   3: CUDA\n");
            printf("                   4: Sequential + OpenMP\n");
            printf("                   5: Sequential + CUDA\n");
            printf("                   6: All\n");
            return 0;
        }
    }

    printf("Running TSP algorithms\n");

    FILE* results_file = fopen("results.txt", "w");
    if (results_file) {
        fclose(results_file);
    }

    for (int cities = min_cities; cities <= max_cities; cities += city_jump) {
        printf("\nTesting with %d cities\n", cities);


        for (int trial = 0; trial < num_trials; trial++) {
            printf("  Trial %d/%d\n", trial + 1, num_trials);
            generator(cities);
            TSP tsp_instance;
            TSP_initialize(&tsp_instance, "data.json");

            solveAndSave(&tsp_instance, mode);

            TSP_cleanup(&tsp_instance);
        }
    }

    printf("\nAll tests completed.\n");
    return 0;
}