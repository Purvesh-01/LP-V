#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel
__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Initialize array with random values
void initialize(int* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;
    }
}

// Print array in required format
void printArray(const string& name, int* arr, int N) {
    cout << name << " => \n";
    for (int i = 0; i < N; i++) {
        cout << arr[i];
        if (i != N - 1) cout << ", ";
    }
    cout << "\n\n";
}

// CPU reference (for error checking)
void cpuAdd(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 16;  // Keep small for printing (can increase later)

    size_t size = N * sizeof(int);

    int *A, *B, *C, *C_cpu;
    int *d_A, *d_B, *d_C;

    A = new int[N];
    B = new int[N];
    C = new int[N];
    C_cpu = new int[N];

    // Initialize
    initialize(A, N);
    initialize(B, N);

    printArray("array A is", A, N);
    printArray("array B is", B, N);

    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Start timer
    cudaEventRecord(start);

    // Kernel launch
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU result for verification
    cpuAdd(A, B, C_cpu, N);

    // Print GPU result
    printArray("GPU sum is", C, N);

    // Error calculation
    int error = 0;
    for (int i = 0; i < N; i++) {
        error += abs(C[i] - C_cpu[i]);
    }

    cout << "Error : " << error << endl;

    cout << "Time Elapsed:  " << time_ms << " ms" << endl;

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_cpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}