#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Struktura reprezentująca timer
typedef struct {
    struct timespec start_time_cpu;
    struct timespec end_time_cpu;
    cudaEvent_t start_event_gpu;
    cudaEvent_t end_event_gpu;
    bool is_gpu_timer;
} zen_timer_t;

static int64_t zen_ticks_per_microsecond = 1000; // 1 µs = 1000 ns

// Inicjalizacja timera – dla Linuxa nie wymaga nic więcej
static inline void ZenTimer_Init() {
    zen_ticks_per_microsecond = 1000;
}

// Funkcja do rozpoczęcia liczenia czasu
static inline zen_timer_t ZenTimer_Start(bool use_gpu) {
    zen_timer_t timer;
    timer.is_gpu_timer = use_gpu;

    if (use_gpu) {
        // Użycie GPU – CUDA Event
        cudaEventCreate(&timer.start_event_gpu);
        cudaEventCreate(&timer.end_event_gpu);
        cudaEventRecord(timer.start_event_gpu, 0);
    } else {
        // Użycie CPU – clock_gettime
        clock_gettime(CLOCK_MONOTONIC, &timer.start_time_cpu);
    }

    return timer;
}

// Funkcja do zatrzymania timera i zwrócenia czasu w mikrosekundach
static inline int64_t ZenTimer_End(zen_timer_t* timer) {
    int64_t microseconds = 0;

    if (timer->is_gpu_timer) {
        // Zatrzymanie dla GPU
        cudaEventRecord(timer->end_event_gpu, 0);
        cudaEventSynchronize(timer->end_event_gpu);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, timer->start_event_gpu, timer->end_event_gpu);
        microseconds = (int64_t)(time_ms * 1000); // Przeliczenie na mikrosekundy

        // Zwalnianie zasobów CUDA
        cudaEventDestroy(timer->start_event_gpu);
        cudaEventDestroy(timer->end_event_gpu);
    } else {
        // Zatrzymanie dla CPU
        clock_gettime(CLOCK_MONOTONIC, &timer->end_time_cpu);

        int64_t seconds = timer->end_time_cpu.tv_sec - timer->start_time_cpu.tv_sec;
        int64_t nanoseconds = timer->end_time_cpu.tv_nsec - timer->start_time_cpu.tv_nsec;
        microseconds = seconds * 1000000 + nanoseconds / 1000;
    }

    return microseconds;
}
