#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#ifndef MATRIX_TYPE
#define MATRIX_TYPE float
#endif

#ifndef NUM_RUNS
#define NUM_RUNS 3
#endif

const std::vector<int> MATRIX_SIZES = {100, 500, 1000, 1500, 2000};

template <typename T>
__global__ void multiplyGPU(T* A, T* B, T* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        T sum = 0;
        for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

template <typename T>
void multiplyCUDA(const std::vector<T>& h_A,
                  const std::vector<T>& h_B,
                  std::vector<T>& h_C,
                  int N,
                  float& time_ms) 
{
    T *d_A, *d_B, *d_C;
    size_t size = N*N*sizeof(T);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    multiplyGPU<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time_ms, start, stop); 

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename T>
void multiplyCPU(const std::vector<T>& A,
                 const std::vector<T>& B,
                 std::vector<T>& C,
                 int N) 
{
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++){
            T sum = 0;
            for (int k=0;k<N;k++) sum += A[i*N+k]*B[k*N+j];
            C[i*N+j]=sum;
        }
}

template <typename T>
bool check(const std::vector<T>& A,const std::vector<T>& B){
    for(size_t i=0;i<A.size();i++)
        if(std::abs(A[i]-B[i])>1e-3) return false;
    return true;
}

template <typename T>
void fillRandom(std::vector<T>& matrix){
    std::mt19937 gen(42);
    if constexpr (std::is_integral<T>::value){
        std::uniform_int_distribution<int> dist(0,10);
        for(auto& x:matrix) x=dist(gen);
    } else {
        std::uniform_real_distribution<double> dist(0.0,1.0);
        for(auto& x:matrix) x=static_cast<T>(dist(gen));
    }
}

template <typename T>
std::string getTypeName(){return "unknown";}
template <>
std::string getTypeName<int>(){return "int";}
template <>
std::string getTypeName<float>(){return "float";}
template <>
std::string getTypeName<double>(){return "double";}

template <typename T>
void runTest(std::ofstream& file){
    for(int N : MATRIX_SIZES){
        std::cout<<"\nMatrix size: "<<N<<"x"<<N<<"\n";

        std::vector<T> A(N*N),B(N*N),C_cpu(N*N),C_gpu(N*N);
        fillRandom(A); fillRandom(B);

        double cpu_total_ms = 0.0;
        double gpu_total_ms = 0.0;

        for(int run=0; run<NUM_RUNS; run++){
            
            auto start = std::chrono::high_resolution_clock::now();
            multiplyCPU(A,B,C_cpu,N);
            auto end = std::chrono::high_resolution_clock::now();
            double cpu_ms = std::chrono::duration<double>(end-start).count() * 1000.0; 
            cpu_total_ms += cpu_ms;

            
            float gpu_ms = 0.0f;
            multiplyCUDA(A,B,C_gpu,N,gpu_ms);
            gpu_total_ms += gpu_ms;
        }

        double cpu_avg = cpu_total_ms / NUM_RUNS;
        double gpu_avg = gpu_total_ms / NUM_RUNS;
        double speedup = cpu_avg / gpu_avg;
        bool ok = check(C_cpu,C_gpu);

        std::cout<<"Type: "<<getTypeName<T>()
                 <<"\nCPU avg: "<<cpu_avg<<" ms"
                 <<"\nGPU avg: "<<gpu_avg<<" ms"
                 <<"\nSpeedup: "<<speedup
                 <<"\nCorrect: "<<(ok?"YES":"NO")<<"\n";

        file<<getTypeName<T>()<<","<<N<<","<<cpu_avg<<","<<gpu_avg<<","<<speedup<<"\n";
    }
}

int main(){
    std::ofstream file("results.csv", std::ios::app);
    std::ifstream check("results.csv");
    if(!check.good() || check.peek() == std::ifstream::traits_type::eof()){
        file << "type,size,cpu_time_ms,gpu_time_ms,speedup\n";
    }
    check.close();

    runTest<MATRIX_TYPE>(file);

    file.close();
    std::cout<<"\nResults saved to results.csv\n";
    return 0;
}