#include "tsp.h"
#include <string.h>

void TSP_initialize(TSP* tsp, const char* jsonFilename) {
    FILE* jsonFile;

    if ((jsonFile = fopen(jsonFilename, "r")) == NULL) {
        printf("Failed to open JSON file\n");
        exit(1);
    }

    fseek(jsonFile, 0, SEEK_END);
    long file_size = ftell(jsonFile);
    rewind(jsonFile);

    char* buffer = (char*)malloc(file_size + 1);
    if (buffer == NULL) {
        printf("Failed to allocate memory for JSON buffer\n");
        exit(1);
    }

    size_t length = fread(buffer, 1, file_size, jsonFile);
    buffer[length] = '\0';

    tsp->root = cJSON_Parse(buffer);
    if (!tsp->root) {
        printf("Error parsing JSON.\n");
        exit(1);
    }

    tsp->N = cJSON_GetObjectItem(tsp->root, "N")->valueint;
    tsp->IT = cJSON_GetObjectItem(tsp->root, "IT")->valueint;
    tsp->start_fuel = cJSON_GetObjectItem(tsp->root, "start_fuel")->valueint;

    tsp->tsp = (int**)malloc(tsp->N * sizeof(int*));
    tsp->gas_stations = (int*)malloc(tsp->N * sizeof(int));
    for (int i = 0; i < tsp->N; i++) {
        tsp->tsp[i] = (int*)malloc(tsp->N * sizeof(int));
    }

    cJSON* tspArray = cJSON_GetObjectItem(tsp->root, "tsp");
    cJSON* gasStationsArray = cJSON_GetObjectItem(tsp->root, "gas_stations");

    for (int i = 0; i < tsp->N; i++) {
        cJSON* row = cJSON_GetArrayItem(tspArray, i);
        for (int j = 0; j < tsp->N; j++) {
            tsp->tsp[i][j] = cJSON_GetArrayItem(row, j)->valueint;
        }
        tsp->gas_stations[i] = cJSON_GetArrayItem(gasStationsArray, i)->valueint;
    }

    fclose(jsonFile);
    free(buffer);
}

void TSP_cleanup(TSP* tsp) {
    for (int i = 0; i < tsp->N; i++) {
        free(tsp->tsp[i]);
    }
    free(tsp->tsp);
    free(tsp->gas_stations);
    cJSON_Delete(tsp->root);
}

void TSP_print_data(TSP* tsp) {
    printf("N = %d\n", tsp->N);
    printf("IT = %d\n", tsp->IT);
    printf("start_fuel = %d\n", tsp->start_fuel);

    printf("tsp = {\n");
    for (int i = 0; i < tsp->N; i++) {
        printf("    {");
        for (int j = 0; j < tsp->N; j++) {
            printf("%d", tsp->tsp[i][j]);
            if (j < tsp->N - 1) {
                printf(", ");
            }
        }
        printf("}");
        if (i < tsp->N - 1) {
            printf(",");
        }
        printf("\n");
    }
    printf("}\n");

    printf("gas_stations = {");
    for (int i = 0; i < tsp->N; i++) {
        printf("%d", tsp->gas_stations[i]);
        if (i < tsp->N - 1) {
            printf(", ");
        }
    }
    printf("}\n");
}

bool TSP_can_reach_gas_station(int current_fuel, int distance, int station_fuel) {
    return current_fuel >= distance;
}

bool TSP_check_if_route_ok(TSP* tsp, int* path, int fuel) {
    int current_fuel = fuel;
    int start_city = path[0];

    if (path[0] != path[tsp->N]) {
        return false;
    }

    bool visited[tsp->N];
    for (int i = 0; i < tsp->N; i++) {
        visited[i] = false;
    }

    visited[start_city] = true;

    for (int i = 1; i < tsp->N; i++) {
        if (path[i] < 0 || path[i] >= tsp->N || visited[path[i]]) {
            return false;
        }
        visited[path[i]] = true;
    }

    for (int i = 0; i < tsp->N; i++) {
        if (!visited[i]) {
            return false;
        }
    }

    for (int i = 0; i < tsp->N; i++) {
        int from = path[i];
        int to = path[i + 1];
        int distance = tsp->tsp[from][to];

        if (current_fuel < distance) {
            return false;
        }

        current_fuel -= distance;
        current_fuel += tsp->gas_stations[to];
    }

    return true;
}

void TSP_fill_array_with_indices(int array[], int length, int start_city) {
    int idx = 1;
    for (int i = 0; i < length - 1; i++) {
        if (i != start_city) {
            array[idx++] = i;
        }
    }
    array[0] = start_city;
    array[length - 1] = start_city;
}

void TSP_copy_array(int* source, int* destination, int length) {
    for (int i = 0; i < length; i++) {
        destination[i] = source[i];
    }
}

void TSP_shuffle(int* array, int size, int keep_from_start, int keep_from_end) {
    if (keep_from_start + keep_from_end >= size)
        return;
    for (int i = keep_from_start; i < size - keep_from_end - 1; i++) {
        int j = keep_from_start + rand() % (size - keep_from_end - keep_from_start - 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void TSP_print_array(int* array, int size) {
    if (array == NULL || size <= 0) {
        printf("Invalid array or size.\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        printf("%d", array[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

void TSP_change_path(int source[], int destination[], int length) {
    for (int i = 0; i < length; i++) {
        destination[i] = source[i];
    }
    int index1, index2;
    do {
        index1 = 1 + rand() % (length - 2);
        index2 = 1 + rand() % (length - 2);
    } while (index1 == index2);
    int temp = destination[index1];
    destination[index1] = destination[index2];
    destination[index2] = temp;
}

int TSP_calculate_path_length(TSP* tsp, int* path, int fuel) {
    int length = 0;

    if (!TSP_check_if_route_ok(tsp, path, fuel)) {
        return -1;
    }

    for (int i = 0; i < tsp->N; i++) {
        length += tsp->tsp[path[i]][path[i + 1]];
    }

    return length;
}

void TSP_saveToTXT(TSP* tsp, FILE* file, const char* algorithm, int* path, int path_length, int size, int64_t time) {
    fprintf(file, "%d;", tsp->N);
    fprintf(file, "%s;", algorithm);
    fprintf(file, "%d;", path_length);
    //for (int i = 0; i < size; i++) {
    //   fprintf(file, "%d,", path[i]);
    //}
    //fprintf(file, ";");
    fprintf(file, "%ld\n", time);
}

void TSP_saveToJSON(TSP* tsp, FILE* file, const char* algorithm, int* path, int path_length, int size, int64_t elapsed_time) {
    cJSON* root = cJSON_CreateObject();
    if (!root) {
        printf("Failed to create JSON object.\n");
        return;
    }

    cJSON_AddNumberToObject(root, "N", tsp->N);
    cJSON_AddStringToObject(root, "algorithm", algorithm);
    cJSON_AddNumberToObject(root, "path_length", path_length);

    cJSON* pathArray = cJSON_CreateIntArray(path, size);
    if (pathArray) {
        cJSON_AddItemToObject(root, "path", pathArray);
    }

    cJSON_AddNumberToObject(root, "time", elapsed_time);

    char* jsonString = cJSON_PrintUnformatted(root);
    if (jsonString) {
        fprintf(file, "%s\n", jsonString);
        free(jsonString);
    } else {
        printf("Failed to print JSON string.\n");
    }

    cJSON_Delete(root);
}

//int random_index(int range, unsigned int* seed) {
//    return rand_r(seed) % range;
//}