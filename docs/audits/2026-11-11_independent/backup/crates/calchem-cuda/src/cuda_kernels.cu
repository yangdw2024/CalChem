#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
typedef int (__stdcall *PFN_cuInit)(unsigned int Flags);
typedef int (__stdcall *PFN_cuDriverGetVersion)(int *driverVersion);
typedef int (__stdcall *PFN_cuDeviceGetCount)(int *count);

static int check_cuda_driver_safe() {
    HMODULE hMod = LoadLibraryA("nvcuda.dll");
    if (!hMod) return 0;
    PFN_cuInit pCuInit = (PFN_cuInit)GetProcAddress(hMod, "cuInit");
    PFN_cuDriverGetVersion pCuDriverGetVersion = (PFN_cuDriverGetVersion)GetProcAddress(hMod, "cuDriverGetVersion");
    PFN_cuDeviceGetCount pCuDeviceGetCount = (PFN_cuDeviceGetCount)GetProcAddress(hMod, "cuDeviceGetCount");
    if (!pCuInit || !pCuDriverGetVersion || !pCuDeviceGetCount) {
        FreeLibrary(hMod);
        return 0;
    }
    if (pCuInit(0) != 0) {
        FreeLibrary(hMod);
        return 0;
    }
    int driverVersion = 0;
    if (pCuDriverGetVersion(&driverVersion) != 0 || driverVersion < CUDART_VERSION) {
        FreeLibrary(hMod);
        return 0;
    }
    int count = 0;
    if (pCuDeviceGetCount(&count) != 0 || count <= 0) {
        FreeLibrary(hMod);
        return 0;
    }
    FreeLibrary(hMod);
    return 1;
}
#endif

extern "C" {

int calchem_cuda_is_available() {
#ifdef _WIN32
    if (!check_cuda_driver_safe()) {
        return 0;
    }
#endif
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

int calchem_cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes) return -1;
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return (err == cudaSuccess) ? 0 : (int)err;
}

// Double-precision CUDA kernel for J += sum_p x_p * B_p
__global__ void rij_accumulate_kernel(
    const double* __restrict__ b_block,
    const double* __restrict__ x_block,
    double* __restrict__ j_matrix,
    int n_p,
    int n_ao_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ao_sq) {
        double sum = 0.0;
        #pragma unroll 4
        for (int p = 0; p < n_p; ++p) {
            sum += x_block[p] * b_block[p * n_ao_sq + idx];
        }
        j_matrix[idx] += sum;
    }
}

int calchem_cuda_rij_contract_coulomb(
    const double* h_b_block,
    const double* h_x_block,
    double* h_j,
    int n_p,
    int n_ao_sq,
    size_t memory_limit_bytes
) {
    if (!h_b_block || !h_x_block || !h_j || n_p <= 0 || n_ao_sq <= 0) {
        return -1;
    }

    size_t req_bytes = ((size_t)n_p * (size_t)n_ao_sq + (size_t)n_p + (size_t)n_ao_sq) * sizeof(double);
    if (memory_limit_bytes > 0 && req_bytes > memory_limit_bytes) {
        return -2; // Memory budget exceeded -> CPU fallback
    }

    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess || free_mem < req_bytes + 64 * 1024 * 1024) {
        return -3; // Insufficient VRAM -> CPU fallback
    }

    double *d_b = NULL, *d_x = NULL, *d_j = NULL;
    cudaError_t err = cudaMalloc((void**)&d_b, (size_t)n_p * n_ao_sq * sizeof(double));
    if (err != cudaSuccess) return -4;

    err = cudaMalloc((void**)&d_x, (size_t)n_p * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_b); return -4; }

    err = cudaMalloc((void**)&d_j, (size_t)n_ao_sq * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_b); cudaFree(d_x); return -4; }

    cudaMemcpy(d_b, h_b_block, (size_t)n_p * n_ao_sq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_block, (size_t)n_p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, h_j, (size_t)n_ao_sq * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n_ao_sq + block_size - 1) / block_size;
    rij_accumulate_kernel<<<grid_size, block_size>>>(d_b, d_x, d_j, n_p, n_ao_sq);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_b); cudaFree(d_x); cudaFree(d_j);
        return -5;
    }

    cudaMemcpy(h_j, d_j, (size_t)n_ao_sq * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_j);

    return 0;
}
}
