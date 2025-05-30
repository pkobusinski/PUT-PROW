#ifndef TSP_H
#define TSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "cJSON.h"
#include "zen_timer.h"

typedef struct {
    int N;              // liczba miast
    int IT;             // liczba iteracji dla metaheurystyki
    int start_fuel;     // początkowa ilość paliwa
    int** tsp;          // macierz odległości
    int* gas_stations;  // ilość paliwa dostępnego w każdym mieście
    cJSON* root;        // wczytane dane JSON
} TSP;

// Inicjalizacja problemu z pliku JSON
void TSP_initialize(TSP* tsp, const char* filename);

// Zwolnienie zasobów
void TSP_cleanup(TSP* tsp);

// Wypisanie danych problemu
void TSP_print_data(TSP* tsp);

// Sprawdzenie czy możemy dotrzeć do stacji benzynowej
bool TSP_can_reach_gas_station(int current_fuel, int distance, int station_fuel);

// Sprawdzenie czy trasa jest możliwa do przejechania
bool TSP_check_if_route_ok(TSP* tsp, int* path, int fuel);

// Wypełnienie tablicy indeksami miast
void TSP_fill_array_with_indices(int array[], int length, int start_city) ;

// Kopiowanie zawartości jednej tablicy do drugiej
void TSP_copy_array(int* source, int* destination, int length);

// Przetasowanie tablicy z zachowaniem początku i końca
void TSP_shuffle(int* array, int size, int keep_from_start, int keep_from_end);

// Wypisanie tablicy
void TSP_print_array(int* array, int size);

// Modyfikacja ścieżki przez zamianę dwóch losowych miast
void TSP_change_path(int source[], int destination[], int length);

// Obliczenie długości ścieżki dla problemu TSP
int TSP_calculate_path_length(TSP* tsp, int* path, int fuel);

// Zapisanie wyniku do pliku TXT
void TSP_saveToTXT(TSP* tsp, FILE* file, const char* algorithm, int* path, int path_length, int size, int64_t time);

int random_index(int range, unsigned int* seed);

// Implementacje algorytmów (deklaracje - implementacje będą w osobnych plikach)
int TSP_greedy_multistart(TSP* tsp, int* path, int fuel, int k);
int TSP_local_search_mutation(TSP* tsp, int* path, int fuel, int iterations);

int TSP_greedy_multistart_openmp(TSP* tsp, int* path, int fuel, int k);
int TSP_local_search_mutation_openmp(TSP* tsp, int* path, int fuel, int iterations);

// New CUDA implementations
void TSP_initialize_cuda(TSP* tsp);
int TSP_greedy_multistart_cuda(TSP* tsp, int* path, int fuel, int num_starts);
int TSP_local_search_mutation_cuda(TSP* tsp, int* path, int fuel, int iterations);


void TSP_saveToJSON(TSP* tsp, FILE* file, const char* algorithm, int* path, int path_length, int size, int64_t elapsed_time);

#ifdef __cplusplus
}
#endif

#endif // TSP_H