#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

constexpr int Threads=128;
constexpr bool single_block = false;

// dot product kernel
template <int THREADS>
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[THREADS];

    int i = threadIdx.x;

    buf[i] = i<n? x[i]*y[i]: 0.;

    int m = THREADS/2;

    while (m) {
        __syncthreads();
        if (i<m) {
            buf[i] += buf[i+m];
        }
        m /= 2;
    }

    if (i==0) {
        *result = buf[0];
    }
}

// dot product kernel for arbitrary n
template <int THREADS>
__global__
void dot_gpu_kernel_full(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[THREADS];

    int i = threadIdx.x;
    int gi = i + blockIdx.x*blockDim.x;

    buf[i] = gi<n? x[gi]*y[gi]: 0.;

    int m = THREADS/2;

    while (m) {
        __syncthreads();
        if (i<m) {
            buf[i] += buf[i+m];
        }
        m /= 2;
    }

    if (i==0) {
        atomicAdd(result, buf[0]);
    }
}

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
    if (single_block) {
        dot_gpu_kernel<Threads><<<1, Threads>>>(x, y, result, n);
    }
    else {
        *result = 0.;
        dot_gpu_kernel_full<Threads><<<(n-1)/Threads+1, Threads>>>(x, y, result, n);
    }
    cudaDeviceSynchronize();
    return *result;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

