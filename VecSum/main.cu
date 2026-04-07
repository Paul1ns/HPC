#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

#define BLOCK_SIZE 256
#define RUNS 5

#ifndef TYPE
#define TYPE int
#endif

using DataType = TYPE;

std::string getTypeName() {
    if (typeid(DataType) == typeid(int)) return "int";
    if (typeid(DataType) == typeid(float)) return "float";
    if (typeid(DataType) == typeid(double)) return "double";
    return "unknown";
}

double sumCPU(const std::vector<DataType>& vec) {
    double sum = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}

__global__ void sumKernel(DataType* input, double* partialSums, int n) {
    __shared__ double cache[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    double temp = 0.0;

    while (tid < n) {
        temp += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        partialSums[blockIdx.x] = cache[0];
    }
}

struct GpuTimings {
    float h2d = 0;
    float kernel = 0;
    float d2h = 0;
};

double sumGPU(const std::vector<DataType>& vec, GpuTimings& t) {
    int n = vec.size();

    DataType* d_input;
    double* d_partial;

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input, n * sizeof(DataType));
    cudaMalloc(&d_partial, gridSize * sizeof(double));

    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);

    cudaEventRecord(e1);
    cudaMemcpy(d_input, vec.data(), n * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&t.h2d, e1, e2);

    cudaEventRecord(e1);
    sumKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_partial, n);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&t.kernel, e1, e2);

    cudaDeviceSynchronize();

    std::vector<double> h_partial(gridSize);

    cudaEventRecord(e1);
    cudaMemcpy(h_partial.data(), d_partial, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&t.d2h, e1, e2);

    double finalSum = 0.0;
    for (int i = 0; i < gridSize; i++) {
        finalSum += h_partial[i];
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);

    return finalSum;
}

int main() {
    std::vector<int> sizes = {1000, 10000, 100000, 500000, 1000000};

    std::ifstream check("results_vec.csv");
    bool isEmpty = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    check.close();

    std::ofstream file("results_vec.csv", std::ios::app);

    if (isEmpty) {
        file << "type,size,cpu_time_ms,gpu_total_ms,kernel_ms,h2d_ms,d2h_ms,speedup\n";
    }

    std::string typeName = getTypeName();

    for (int size : sizes) {
        std::vector<DataType> vec(size);

        for (int i = 0; i < size; i++) {
            vec[i] = rand() % 10;
        }

        double cpuTotal = 0;
        double gpuTotal = 0;
        double h2dTotal = 0;
        double kernelTotal = 0;
        double d2hTotal = 0;

        double cpuSum = 0;
        double gpuSum = 0;

        for (int r = 0; r < RUNS; r++) {

            auto startCPU = std::chrono::high_resolution_clock::now();
            cpuSum = sumCPU(vec);
            auto endCPU = std::chrono::high_resolution_clock::now();
            cpuTotal += std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

            GpuTimings t;
            gpuSum = sumGPU(vec, t);

            h2dTotal += t.h2d;
            kernelTotal += t.kernel;
            d2hTotal += t.d2h;
            gpuTotal += (t.h2d + t.kernel + t.d2h);
        }

        double cpuAvg = cpuTotal / RUNS;
        double gpuAvg = gpuTotal / RUNS;
        double speedup = cpuAvg / gpuAvg;

        std::cout << "Type: " << typeName << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "CPU avg: " << cpuAvg << " ms" << std::endl;
        std::cout << "GPU total avg: " << gpuAvg << " ms" << std::endl;
        std::cout << "Kernel: " << kernelTotal / RUNS << " ms" << std::endl;
        std::cout << "H2D: " << h2dTotal / RUNS << " ms" << std::endl;
        std::cout << "D2H: " << d2hTotal / RUNS << " ms" << std::endl;
        std::cout << "Speedup: " << speedup << std::endl;
        std::cout << "Sum_cpu: " << cpuSum << std::endl;
        std::cout << "Sum_gpu: " << gpuSum << std::endl;

        file << typeName << ","
             << size << ","
             << cpuAvg << ","
             << gpuAvg << ","
             << (kernelTotal / RUNS) << ","
             << (h2dTotal / RUNS) << ","
             << (d2hTotal / RUNS) << ","
             << speedup << "\n";
    }

    file.close();
    return 0;
}