CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O2 -I../headers -fopenmp
NVCCFLAGS = -O2 -I../headers -Xcompiler -fPIC
LDFLAGS = -fopenmp -lm
CUDALDFLAGS = -lcudart

# Source files
SEQ_SRC = ../source/main.c ../source/cJSON.c ../source/generator.c ../source/tsp.c ../source/tsp_sequential.c
OMP_SRC = ../source/tsp_openmp.c
CUDA_SRC = ../source/tsp_cuda.cu

# Object files
SEQ_OBJ = $(SEQ_SRC:../source/%.c=./%.o)
OMP_OBJ = $(OMP_SRC:../source/%.c=./%.o)
CUDA_OBJ = $(CUDA_SRC:../source/%.cu=./%.o)

TARGET = ./tsp.app

BUILD_DIR = .

.PHONY: all seq omp cuda clean

# Ensure the current directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule for compiling everything
all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(SEQ_OBJ) $(OMP_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(CUDALDFLAGS)


seq: $(BUILD_DIR) $(SEQ_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(SEQ_OBJ) $(LDFLAGS)

# Rule for OpenMP version compilation
omp: $(BUILD_DIR) $(SEQ_OBJ) $(OMP_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(SEQ_OBJ) $(OMP_OBJ) $(LDFLAGS)

# Rule for CUDA version compilation
cuda: $(BUILD_DIR) $(SEQ_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ $(SEQ_OBJ) $(CUDA_OBJ) $(LDFLAGS) $(CUDALDFLAGS)

# Rules for compiling .c to .o files
./%.o: ../source/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for compiling .cu to .o files
./%.o: ../source/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf *.o *.txt $(BUILD_DIR)/*.o $(BUILD_DIR)/*.exe $(BUILD_DIR)/*.txt $(BUILD_DIR)/*.json $(TARGET)
