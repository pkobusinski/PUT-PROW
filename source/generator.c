#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cJSON.h"

int generator(int number) {
    int N = number;
    int max_distance = 0;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "N", N);
    cJSON_AddNumberToObject(root, "IT", 1000);
    cJSON* tsp = cJSON_AddArrayToObject(root, "tsp");

    // First, generate the distance matrix
    int** distances = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        distances[i] = (int*)malloc(N * sizeof(int));
        for (int j = 0; j < N; j++) {
            if (i == j) {
                distances[i][j] = 0;
            } else {
                distances[i][j] = rand() % 201 + 50; // Between 50 and 250
                if (distances[i][j] > max_distance) {
                    max_distance = distances[i][j];
                }
            }
        }
    }

    // Create JSON array for the distances
    for (int i = 0; i < N; i++) {
        cJSON* row = cJSON_CreateArray();
        cJSON_AddItemToArray(tsp, row);
        for (int j = 0; j < N; j++) {
            cJSON_AddItemToArray(row, cJSON_CreateNumber(distances[i][j]));
        }
    }

    // Calculate a reasonable starting fuel
    // Enough to travel at least between 2-3 cities
    int start_fuel = max_distance + rand() % max_distance;
    cJSON_AddNumberToObject(root, "start_fuel", start_fuel);

    // Generate gas stations with enough fuel to continue travel
    cJSON* gas_stations = cJSON_AddArrayToObject(root, "gas_stations");
    for (int i = 0; i < N; i++) {
        // First city has no fuel (it's the starting point)
        if (i == 0) {
            cJSON_AddItemToArray(gas_stations, cJSON_CreateNumber(0));
        } else {
            // Other cities have enough fuel to travel at least one more leg
            int gas = (rand() % 3 + 1) * max_distance / 2; // 0.5 to 1.5 times max_distance
            cJSON_AddItemToArray(gas_stations, cJSON_CreateNumber(gas));
        }
    }

    // Write to file and clean up
    char* json_str = cJSON_Print(root);
    FILE* file;
    file = fopen("data.json", "w");
    if (file != NULL) {
        fputs(json_str, file);
        fclose(file);
        printf("Plik data.json został wygenerowany.\n");
    } else {
        fprintf(stderr, "Błąd podczas tworzenia pliku data.json.\n");
    }

    // Free memory
    for (int i = 0; i < N; i++) {
        free(distances[i]);
    }
    free(distances);
    cJSON_Delete(root);
    free(json_str);

    return 0;
}