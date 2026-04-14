#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 16

#pragma pack(push,1)
struct BMPHeader {
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
};

struct BMPInfoHeader {
    int size;
    int width;
    int height;
    short planes;
    short bits;
    int compression;
    int imagesize;
    int xresolution;
    int yresolution;
    int ncolors;
    int importantcolors;
};
#pragma pack(pop)

bool loadBMP(const char* filename, std::vector<unsigned char>& data, int& width, int& height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    BMPHeader header;
    BMPInfoHeader info;

    fread(&header, sizeof(header), 1, f);
    fread(&info, sizeof(info), 1, f);

    width = info.width;
    height = info.height;

    int row_padded = (width + 3) & (~3);
    data.resize(width * height);

    fseek(f, header.offset, SEEK_SET);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            fread(&data[y * width + x], 1, 1, f);
        }
        fseek(f, row_padded - width, SEEK_CUR);
    }

    fclose(f);
    return true;
}

bool saveBMP(const char* filename, const std::vector<unsigned char>& data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    int row_padded = (width * 3 + 3) & (~3);
    int filesize = 54 + row_padded * height;

    unsigned char fileHeader[14] = {
        'B','M',
        (unsigned char)(filesize), (unsigned char)(filesize >> 8),
        (unsigned char)(filesize >> 16), (unsigned char)(filesize >> 24),
        0,0,0,0,
        54,0,0,0
    };

    unsigned char infoHeader[40] = {
        40,0,0,0,
        (unsigned char)(width), (unsigned char)(width >> 8),
        (unsigned char)(width >> 16), (unsigned char)(width >> 24),
        (unsigned char)(height), (unsigned char)(height >> 8),
        (unsigned char)(height >> 16), (unsigned char)(height >> 24),
        1,0,24,0
    };

    fwrite(fileHeader, 1, 14, f);
    fwrite(infoHeader, 1, 40, f);

    unsigned char padding[3] = {0,0,0};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char val = data[y * width + x];
            unsigned char color[3] = {val, val, val};
            fwrite(color, 1, 3, f);
        }
        fwrite(padding, 1, row_padded - width * 3, f);
    }

    fclose(f);
    return true;
}

double computeVariance(const std::vector<unsigned char>& img) {
    double mean = 0.0;
    for (auto v : img) mean += v;
    mean /= img.size();

    double var = 0.0;
    for (auto v : img) var += (v - mean) * (v - mean);

    return var / img.size();
}

double computeMSE(const std::vector<unsigned char>& a,
                  const std::vector<unsigned char>& b) {
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / a.size();
}

void bilateralCPU(const std::vector<unsigned char>& input,
                  std::vector<unsigned char>& output,
                  int width, int height,
                  float sigma_d, float sigma_r) {

    int dx[9] = {-1,0,1,-1,0,1,-1,0,1};
    int dy[9] = {-1,-1,-1,0,0,0,1,1,1};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float sum = 0.0f, norm = 0.0f;
            float center = input[y * width + x];

            for (int k = 0; k < 9; k++) {
                int nx = min(max(x + dx[k], 0), width - 1);
                int ny = min(max(y + dy[k], 0), height - 1);

                float neighbor = input[ny * width + nx];

                float gd = exp(-(dx[k]*dx[k] + dy[k]*dy[k]) / (2 * sigma_d * sigma_d));
                float gr = exp(-(neighbor - center)*(neighbor - center) / (2 * sigma_r * sigma_r));

                float w = gd * gr;
                sum += neighbor * w;
                norm += w;
            }

            output[y * width + x] = (unsigned char)(sum / norm);
        }
    }
}


__global__ void bilateralGPU(cudaTextureObject_t tex,
                             unsigned char* output,
                             int width, int height,
                             float sigma_d, float sigma_r) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int dx[9] = {-1,0,1,-1,0,1,-1,0,1};
    int dy[9] = {-1,-1,-1,0,0,0,1,1,1};

    float center = tex2D<unsigned char>(tex, x, y);

    float sum = 0.0f, norm = 0.0f;

    for (int k = 0; k < 9; k++) {
        int nx = min(max(x + dx[k], 0), width - 1);
        int ny = min(max(y + dy[k], 0), height - 1);

        float neighbor = tex2D<unsigned char>(tex, nx, ny);

        float gd = exp(-(dx[k]*dx[k] + dy[k]*dy[k]) / (2 * sigma_d * sigma_d));
        float gr = exp(-(neighbor - center)*(neighbor - center) / (2 * sigma_r * sigma_r));

        float w = gd * gr;
        sum += neighbor * w;
        norm += w;
    }

    output[y * width + x] = (unsigned char)(sum / norm);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./bilateral input.bmp sigma_d sigma_r\n";
        return -1;
    }

    const char* filename = argv[1];
    float sigma_d = atof(argv[2]);
    float sigma_r = atof(argv[3]);

    int width, height;
    std::vector<unsigned char> input;

    if (!loadBMP(filename, input, width, height)) {
        std::cout << "Error loading image\n";
        return -1;
    }

    std::vector<unsigned char> cpu_out(width * height);
    std::vector<unsigned char> gpu_out(width * height);

    auto t1 = std::chrono::high_resolution_clock::now();
    bilateralCPU(input, cpu_out, width, height, sigma_d, sigma_r);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t2 - t1).count();

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, input.data(), width * height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_input;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bilateralGPU<<<grid, block>>>(tex, d_output, width, height, sigma_d, sigma_r);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_time = gpu_ms / 1000.0;

    cudaMemcpy(gpu_out.data(), d_output, width * height, cudaMemcpyDeviceToHost);

    double var_cpu = computeVariance(cpu_out);
    double var_gpu = computeVariance(gpu_out);
    double mse_cpu = computeMSE(input, cpu_out);
    double mse_gpu = computeMSE(input, gpu_out);
    double mse_cpu_gpu = computeMSE(cpu_out, gpu_out);

    std::cout << "CPU time: " << cpu_time << "\n";
    std::cout << "GPU time: " << gpu_time << "\n";
    std::cout << "MSE CPU-GPU: " << mse_cpu_gpu << "\n";

    FILE* csv = fopen("results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "sigma_d,sigma_r,cpu_time,gpu_time,var_cpu,var_gpu,mse_cpu,mse_gpu,mse_cpu_gpu\n");

        fprintf(csv, "%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.6f\n",
                sigma_d, sigma_r,
                cpu_time, gpu_time,
                var_cpu, var_gpu,
                mse_cpu, mse_gpu,
                mse_cpu_gpu);

        fclose(csv);
    }

    saveBMP("cpu.bmp", cpu_out, width, height);
    saveBMP("gpu.bmp", gpu_out, width, height);

    cudaDestroyTextureObject(tex);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}