#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

const float ALPHA = 0.04f;
const int GAUSSIAN_KERNEL_SIZE = 5;
const float GAUSSIAN_SIGMA = 1.0f;

void harris_cpu(const cv::Mat& src_float, const cv::Mat& src_gray, cv::Mat& dst, float threshold)
{
    CV_Assert(src_float.type() == CV_32FC1);
    int rows = src_float.rows;
    int cols = src_float.cols;

    cv::Mat dx, dy;
    cv::Sobel(src_float, dx, CV_32F, 1, 0, 3);
    cv::Sobel(src_float, dy, CV_32F, 0, 1, 3);

    cv::Mat Ix2 = dx.mul(dx);
    cv::Mat Iy2 = dy.mul(dy);
    cv::Mat Ixy = dx.mul(dy);

    cv::GaussianBlur(Ix2, Ix2, cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), GAUSSIAN_SIGMA);
    cv::GaussianBlur(Iy2, Iy2, cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), GAUSSIAN_SIGMA);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), GAUSSIAN_SIGMA);

    cv::Mat R(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; ++i) {
        float* r_ptr = R.ptr<float>(i);
        float* i2x_ptr = Ix2.ptr<float>(i);
        float* i2y_ptr = Iy2.ptr<float>(i);
        float* ixy_ptr = Ixy.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            float a = i2x_ptr[j];
            float b = ixy_ptr[j];
            float c = i2y_ptr[j];
            float det = a * c - b * b;
            float trace = a + c;
            r_ptr[j] = det - ALPHA * trace * trace;
        }
    }

    cv::cvtColor(src_gray, dst, cv::COLOR_GRAY2BGR);

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float val = R.at<float>(i, j);
            if (val > threshold) {
                bool is_max = true;
                for (int di = -1; di <= 1 && is_max; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        if (R.at<float>(i + di, j + dj) >= val) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (is_max) {
                    dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                }
            }
        }
    }
}

__global__ void compute_derivatives_kernel(cudaTextureObject_t texSrc,
                                           float* d_Ix, float* d_Iy,
                                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float left  = tex2D<float>(texSrc, x - 0.5f, y + 0.5f);
    float right = tex2D<float>(texSrc, x + 1.5f, y + 0.5f);
    float top   = tex2D<float>(texSrc, x + 0.5f, y - 0.5f);
    float bottom= tex2D<float>(texSrc, x + 0.5f, y + 1.5f);

    d_Ix[y * width + x] = (right - left) * 0.5f;
    d_Iy[y * width + x] = (bottom - top) * 0.5f;
}

__global__ void compute_components_kernel(const float* d_Ix, const float* d_Iy,
                                          float* d_Ix2, float* d_Iy2, float* d_Ixy,
                                          int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float ix = d_Ix[idx];
    float iy = d_Iy[idx];
    d_Ix2[idx] = ix * ix;
    d_Iy2[idx] = iy * iy;
    d_Ixy[idx] = ix * iy;
}

__global__ void gaussian_blur_channel_kernel(cudaTextureObject_t texChannel,
                                             float* d_out,
                                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float kernel[5][5] = {
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
        {0.015019f, 0.059915f, 0.094907f, 0.059915f, 0.015019f},
        {0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
        {0.015019f, 0.059915f, 0.094907f, 0.059915f, 0.015019f},
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
    };
    const int offset = GAUSSIAN_KERNEL_SIZE / 2;

    float sum = 0.0f;
    for (int ky = -offset; ky <= offset; ++ky) {
        for (int kx = -offset; kx <= offset; ++kx) {
            float val = tex2D<float>(texChannel, x + kx + 0.5f, y + ky + 0.5f);
            sum += val * kernel[ky + offset][kx + offset];
        }
    }
    d_out[y * width + x] = sum;
}

__global__ void compute_R_and_maxima_kernel(const float* d_Ix2, const float* d_Iy2, const float* d_Ixy,
                                            unsigned char* d_corners, float threshold,
                                            int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;

    int idx = y * width + x;
    float A = d_Ix2[idx];
    float B = d_Ixy[idx];
    float C = d_Iy2[idx];
    float det = A * C - B * B;
    float trace = A + C;
    float R_val = det - ALPHA * trace * trace;

    if (R_val > threshold) {
        bool is_max = true;
        for (int dy = -1; dy <= 1 && is_max; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                int nidx = (y + dy) * width + (x + dx);
                float A_n = d_Ix2[nidx];
                float B_n = d_Ixy[nidx];
                float C_n = d_Iy2[nidx];
                float det_n = A_n * C_n - B_n * B_n;
                float trace_n = A_n + C_n;
                float R_n = det_n - ALPHA * trace_n * trace_n;
                if (R_n >= R_val) {
                    is_max = false;
                    break;
                }
            }
        }
        if (is_max) {
            d_corners[y * width + x] = 1;
        }
    }
}

void harris_gpu(const cv::Mat& src_float, const cv::Mat& src_gray, cv::Mat& dst, float threshold)
{
    CV_Assert(src_float.type() == CV_32FC1);
    int width = src_float.cols;
    int height = src_float.rows;
    size_t img_size = width * height * sizeof(float);

    float *d_Ix, *d_Iy, *d_Ix2, *d_Iy2, *d_Ixy;
    cudaMalloc(&d_Ix, img_size);
    cudaMalloc(&d_Iy, img_size);
    cudaMalloc(&d_Ix2, img_size);
    cudaMalloc(&d_Iy2, img_size);
    cudaMalloc(&d_Ixy, img_size);
    unsigned char *d_corners;
    cudaMalloc(&d_corners, width * height * sizeof(unsigned char));
    cudaMemset(d_corners, 0, width * height * sizeof(unsigned char));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, src_float.ptr<float>(), width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyHostToDevice);

    cudaTextureObject_t texSrc = 0;
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    cudaCreateTextureObject(&texSrc, &resDesc, &texDesc, nullptr);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    compute_derivatives_kernel<<<grid, block>>>(texSrc, d_Ix, d_Iy, width, height);
    cudaDeviceSynchronize();

    compute_components_kernel<<<grid, block>>>(d_Ix, d_Iy, d_Ix2, d_Iy2, d_Ixy, width, height);
    cudaDeviceSynchronize();

    cudaArray *cuIx2, *cuIy2, *cuIxy;
    cudaMallocArray(&cuIx2, &channelDesc, width, height);
    cudaMallocArray(&cuIy2, &channelDesc, width, height);
    cudaMallocArray(&cuIxy, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuIx2, 0, 0, d_Ix2, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(cuIy2, 0, 0, d_Iy2, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(cuIxy, 0, 0, d_Ixy, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyDeviceToDevice);

    cudaTextureObject_t texIx2, texIy2, texIxy;
    resDesc.res.array.array = cuIx2;
    cudaCreateTextureObject(&texIx2, &resDesc, &texDesc, nullptr);
    resDesc.res.array.array = cuIy2;
    cudaCreateTextureObject(&texIy2, &resDesc, &texDesc, nullptr);
    resDesc.res.array.array = cuIxy;
    cudaCreateTextureObject(&texIxy, &resDesc, &texDesc, nullptr);

    float *d_Ix2_blur, *d_Iy2_blur, *d_Ixy_blur;
    cudaMalloc(&d_Ix2_blur, img_size);
    cudaMalloc(&d_Iy2_blur, img_size);
    cudaMalloc(&d_Ixy_blur, img_size);

    gaussian_blur_channel_kernel<<<grid, block>>>(texIx2, d_Ix2_blur, width, height);
    gaussian_blur_channel_kernel<<<grid, block>>>(texIy2, d_Iy2_blur, width, height);
    gaussian_blur_channel_kernel<<<grid, block>>>(texIxy, d_Ixy_blur, width, height);
    cudaDeviceSynchronize();

    compute_R_and_maxima_kernel<<<grid, block>>>(d_Ix2_blur, d_Iy2_blur, d_Ixy_blur,
                                                  d_corners, threshold, width, height);
    cudaDeviceSynchronize();

    std::vector<unsigned char> corners_host(width * height);
    cudaMemcpy(corners_host.data(), d_corners, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::cvtColor(src_gray, dst, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (corners_host[y * width + x]) {
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    cudaDestroyTextureObject(texSrc);
    cudaDestroyTextureObject(texIx2);
    cudaDestroyTextureObject(texIy2);
    cudaDestroyTextureObject(texIxy);
    cudaFreeArray(cuArray);
    cudaFreeArray(cuIx2);
    cudaFreeArray(cuIy2);
    cudaFreeArray(cuIxy);
    cudaFree(d_Ix); cudaFree(d_Iy);
    cudaFree(d_Ix2); cudaFree(d_Iy2); cudaFree(d_Ixy);
    cudaFree(d_Ix2_blur); cudaFree(d_Iy2_blur); cudaFree(d_Ixy_blur);
    cudaFree(d_corners);
}

bool compare_results(const cv::Mat& img_cpu, const cv::Mat& img_gpu)
{
    auto count_corners = [](const cv::Mat& img) -> int {
        int cnt = 0;
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                if (p[0] == 0 && p[1] == 0 && p[2] == 255) cnt++;
            }
        }
        return cnt;
    };
    int cnt_cpu = count_corners(img_cpu);
    int cnt_gpu = count_corners(img_gpu);
    std::cout << "Углы: CPU = " << cnt_cpu << ", GPU = " << cnt_gpu << std::endl;
    return cnt_cpu == cnt_gpu;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <threshold>" << std::endl;
        std::cerr << "Example: " << argv[0] << " square.png 0.00001" << std::endl;
        return -1;
    }
    std::string image_path = argv[1];
    float threshold = std::atof(argv[2]);

    cv::Mat src_gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::Mat src_float;
    src_gray.convertTo(src_float, CV_32FC1, 1.0 / 255.0);

    cv::Mat result_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    harris_cpu(src_float, src_gray, result_cpu, threshold);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << elapsed_cpu.count() << " s" << std::endl;

    cv::Mat result_gpu;
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    harris_gpu(src_float, src_gray, result_gpu, threshold);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float elapsed_gpu_ms;
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    std::cout << "GPU time: " << elapsed_gpu_ms / 1000.0 << " s" << std::endl;
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cv::imwrite("output_cpu.png", result_cpu);
    cv::imwrite("output_gpu.png", result_gpu);

    bool match = compare_results(result_cpu, result_gpu);
    std::cout << "Совпадение результатов: " << (match ? "Да" : "Нет") << std::endl;

    return 0;
}