#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel
__global__ void multiply(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Initialize matrix
void initialize(int* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 10;
    }
}

// Print matrix (limited output)
void printMatrix(const string& name, int* matrix, int N) {
    cout << name << " (first 4x4 block):\n";

    int limit = N;

    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < limit; j++) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "...\n\n";
}

// CPU reference (for correctness)
void cpuMultiply(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 8;  // 🔥 increase to 256, 512 later

    size_t bytes = N * N * sizeof(int);

    int *A, *B, *C, *C_cpu;
    int *d_A, *d_B, *d_C;

    A = new int[N * N];
    B = new int[N * N];
    C = new int[N * N];
    C_cpu = new int[N * N];

    initialize(A, N);
    initialize(B, N);

    printMatrix("Matrix A", A, N);
    printMatrix("Matrix B", B, N);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // 🔥 CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(16, 16);
    dim3 blocks(
        (N + 15) / 16,
        (N + 15) / 16
    );

    cudaEventRecord(start);

    multiply<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU check
    cpuMultiply(A, B, C_cpu, N);

    printMatrix("GPU Result", C, N);

    // Error check
    int error = 0;
    for (int i = 0; i < N * N; i++) {
        error += abs(C[i] - C_cpu[i]);
    }

    cout << "Error : " << error << endl;
    cout << "Time Elapsed: " << time_ms << " ms" << endl;

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