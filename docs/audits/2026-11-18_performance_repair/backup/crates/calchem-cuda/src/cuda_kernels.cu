#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mutex>

#ifdef HAVE_CALCHEM_CUBIN_H
#include "calchem_cubin.h"
#endif

// Serialize allocation scopes so the per-call ceiling also bounds this module's
// aggregate live allocations. Destructors cover every error/early return.
static std::mutex rij_mutex;

extern "C" __global__ void rij_accumulate_kernel(
    const double* __restrict__ b_block,
    const double* __restrict__ x_block,
    double* __restrict__ j_matrix,
    int n_p,
    int n_ao_sq
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)n_ao_sq) {
        double sum = 0.0;
        #pragma unroll 4
        for (int p = 0; p < n_p; ++p) {
            sum += x_block[p] * b_block[(size_t)p * (size_t)n_ao_sq + idx];
        }
        j_matrix[idx] += sum;
    }
}

extern "C" __global__ void rij_contract_v_kernel(
    const double* __restrict__ b_block,
    const double* __restrict__ p_mat,
    double* __restrict__ v_out,
    int n_p,
    int n_ao_sq
) {
    int p = (int)blockIdx.x;
    if (p >= n_p) return;
    int tid = (int)threadIdx.x;
    const double* row = b_block + (size_t)p * (size_t)n_ao_sq;
    double sum = 0.0;
    for (size_t idx = (size_t)tid; idx < (size_t)n_ao_sq; idx += (size_t)blockDim.x) {
        sum += row[idx] * p_mat[idx];
    }
    __shared__ double sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32) {
        double val = sdata[tid] + sdata[tid + 32];
        unsigned mask = 0xffffffffu;
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if (tid == 0) {
            v_out[p] = val;
        }
    }
}

// Fallback PTX (ISA 8.0, sm_75) for pre-Blackwell architectures if needed
static const char CALCHEM_PTX_FALLBACK[] =
".version 8.0\n"
".target sm_75\n"
".address_size 64\n"
".visible .entry rij_accumulate_kernel(\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_x,\n"
"    .param .u64 param_j,\n"
"    .param .u32 param_np,\n"
"    .param .u32 param_naosq\n"
")\n"
"{\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<14>;\n"
"    .reg .f64 %fd<5>;\n"
"    ld.param.u64 %rd1, [param_b];\n"
"    ld.param.u64 %rd2, [param_x];\n"
"    ld.param.u64 %rd3, [param_j];\n"
"    ld.param.u32 %r1, [param_np];\n"
"    ld.param.u32 %r2, [param_naosq];\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %tid.x;\n"
"    mul.wide.u32 %rd4, %r3, %r4;\n"
"    cvt.u64.u32 %rd5, %r5;\n"
"    add.u64 %rd6, %rd4, %rd5;\n"
"    cvt.u64.u32 %rd7, %r2;\n"
"    setp.ge.u64 %p1, %rd6, %rd7;\n"
"    @%p1 bra DONE;\n"
"    cvta.to.global.u64 %rd8, %rd1;\n"
"    cvta.to.global.u64 %rd9, %rd2;\n"
"    cvta.to.global.u64 %rd10, %rd3;\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    mov.u32 %r3, 0;\n"
"    setp.le.s32 %p2, %r1, 0;\n"
"    @%p2 bra STORE;\n"
"LOOP:\n"
"    cvt.u64.u32 %rd11, %r3;\n"
"    shl.b64 %rd12, %rd11, 3;\n"
"    add.u64 %rd12, %rd9, %rd12;\n"
"    ld.global.f64 %fd2, [%rd12];\n"
"    mul.lo.u64 %rd13, %rd11, %rd7;\n"
"    add.u64 %rd13, %rd13, %rd6;\n"
"    shl.b64 %rd13, %rd13, 3;\n"
"    add.u64 %rd13, %rd8, %rd13;\n"
"    ld.global.f64 %fd3, [%rd13];\n"
"    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;\n"
"    add.s32 %r3, %r3, 1;\n"
"    setp.lt.s32 %p2, %r3, %r1;\n"
"    @%p2 bra LOOP;\n"
"STORE:\n"
"    shl.b64 %rd12, %rd6, 3;\n"
"    add.u64 %rd12, %rd10, %rd12;\n"
"    ld.global.f64 %fd4, [%rd12];\n"
"    add.f64 %fd4, %fd4, %fd1;\n"
"    st.global.f64 [%rd12], %fd4;\n"
"DONE:\n"
"    ret;\n"
"}\n"
".visible .entry rij_contract_v_kernel(\n"
"    .param .u64 rij_contract_v_kernel_param_0,\n"
"    .param .u64 rij_contract_v_kernel_param_1,\n"
"    .param .u64 rij_contract_v_kernel_param_2,\n"
"    .param .u32 rij_contract_v_kernel_param_3,\n"
"    .param .u32 rij_contract_v_kernel_param_4\n"
")\n"
"{\n"
"    .reg .pred %p<18>;\n"
"    .reg .b32 %r<36>;\n"
"    .reg .f64 %fd<29>;\n"
"    .reg .b64 %rd<22>;\n"
"    .shared .align 8 .b8 _ZZ21rij_contract_v_kernelE5sdata[2048];\n"
"    ld.param.u64 %rd10, [rij_contract_v_kernel_param_0];\n"
"    ld.param.u64 %rd11, [rij_contract_v_kernel_param_1];\n"
"    ld.param.u64 %rd12, [rij_contract_v_kernel_param_2];\n"
"    ld.param.u32 %r5, [rij_contract_v_kernel_param_3];\n"
"    ld.param.u32 %r4, [rij_contract_v_kernel_param_4];\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    setp.ge.s32 %p1, %r1, %r5;\n"
"    @%p1 bra $L__BB1_11;\n"
"    mov.u32 %r2, %tid.x;\n"
"    cvt.s64.s32 %rd1, %r4;\n"
"    cvt.s64.s32 %rd2, %r1;\n"
"    setp.ge.u32 %p2, %r2, %r4;\n"
"    mov.f64 %fd28, 0d0000000000000000;\n"
"    @%p2 bra $L__BB1_4;\n"
"    cvt.s64.s32 %rd21, %r2;\n"
"    mov.u32 %r6, %ntid.x;\n"
"    cvt.u64.u32 %rd4, %r6;\n"
"    cvta.to.global.u64 %rd5, %rd11;\n"
"    cvta.to.global.u64 %rd6, %rd10;\n"
"    mul.lo.s64 %rd7, %rd1, %rd2;\n"
"    mov.f64 %fd28, 0d0000000000000000;\n"
"$L__BB1_3:\n"
"    add.s64 %rd13, %rd21, %rd7;\n"
"    shl.b64 %rd14, %rd13, 3;\n"
"    add.s64 %rd15, %rd6, %rd14;\n"
"    shl.b64 %rd16, %rd21, 3;\n"
"    add.s64 %rd17, %rd5, %rd16;\n"
"    ld.global.nc.f64 %fd7, [%rd17];\n"
"    ld.global.nc.f64 %fd8, [%rd15];\n"
"    fma.rn.f64 %fd28, %fd8, %fd7, %fd28;\n"
"    add.s64 %rd21, %rd21, %rd4;\n"
"    setp.lt.u64 %p3, %rd21, %rd1;\n"
"    @%p3 bra $L__BB1_3;\n"
"$L__BB1_4:\n"
"    shl.b32 %r7, %r2, 3;\n"
"    mov.u32 %r8, _ZZ21rij_contract_v_kernelE5sdata;\n"
"    add.s32 %r3, %r8, %r7;\n"
"    st.shared.f64 [%r3], %fd28;\n"
"    bar.sync 0;\n"
"    setp.gt.s32 %p4, %r2, 127;\n"
"    @%p4 bra $L__BB1_6;\n"
"    ld.shared.f64 %fd9, [%r3];\n"
"    ld.shared.f64 %fd10, [%r3+1024];\n"
"    add.f64 %fd11, %fd10, %fd9;\n"
"    st.shared.f64 [%r3], %fd11;\n"
"$L__BB1_6:\n"
"    bar.sync 0;\n"
"    setp.gt.s32 %p5, %r2, 63;\n"
"    @%p5 bra $L__BB1_8;\n"
"    ld.shared.f64 %fd12, [%r3];\n"
"    ld.shared.f64 %fd13, [%r3+512];\n"
"    add.f64 %fd14, %fd13, %fd12;\n"
"    st.shared.f64 [%r3], %fd14;\n"
"$L__BB1_8:\n"
"    bar.sync 0;\n"
"    setp.gt.s32 %p6, %r2, 31;\n"
"    @%p6 bra $L__BB1_11;\n"
"    ld.shared.f64 %fd25, [%r3];\n"
"    ld.shared.f64 %fd26, [%r3+256];\n"
"    add.f64 %fd15, %fd25, %fd26;\n"
"    mov.b64 {%r9,%r10}, %fd15;\n"
"    mov.u32 %r29, 2;\n"
"    mov.u32 %r30, 31;\n"
"    mov.u32 %r31, 16;\n"
"    mov.u32 %r32, -1;\n"
"    shfl.sync.down.b32 %r12|%p7, %r10, %r31, %r30, %r32;\n"
"    shfl.sync.down.b32 %r11|%p8, %r9, %r31, %r30, %r32;\n"
"    mov.b64 %fd16, {%r11,%r12};\n"
"    add.f64 %fd17, %fd15, %fd16;\n"
"    mov.b64 {%r13,%r14}, %fd17;\n"
"    mov.u32 %r33, 8;\n"
"    shfl.sync.down.b32 %r16|%p9, %r14, %r33, %r30, %r32;\n"
"    shfl.sync.down.b32 %r15|%p10, %r13, %r33, %r30, %r32;\n"
"    mov.b64 %fd18, {%r15,%r16};\n"
"    add.f64 %fd19, %fd17, %fd18;\n"
"    mov.b64 {%r17,%r18}, %fd19;\n"
"    mov.u32 %r34, 4;\n"
"    shfl.sync.down.b32 %r20|%p11, %r18, %r34, %r30, %r32;\n"
"    shfl.sync.down.b32 %r19|%p12, %r17, %r34, %r30, %r32;\n"
"    mov.b64 %fd20, {%r19,%r20};\n"
"    add.f64 %fd21, %fd19, %fd20;\n"
"    mov.b64 {%r21,%r22}, %fd21;\n"
"    shfl.sync.down.b32 %r24|%p13, %r22, %r29, %r30, %r32;\n"
"    shfl.sync.down.b32 %r23|%p14, %r21, %r29, %r30, %r32;\n"
"    mov.b64 %fd22, {%r23,%r24};\n"
"    add.f64 %fd23, %fd21, %fd22;\n"
"    mov.b64 {%r25,%r26}, %fd23;\n"
"    mov.u32 %r35, 1;\n"
"    shfl.sync.down.b32 %r28|%p15, %r26, %r35, %r30, %r32;\n"
"    shfl.sync.down.b32 %r27|%p16, %r25, %r35, %r30, %r32;\n"
"    mov.b64 %fd24, {%r27,%r28};\n"
"    add.f64 %fd4, %fd23, %fd24;\n"
"    setp.ne.s32 %p17, %r2, 0;\n"
"    @%p17 bra $L__BB1_11;\n"
"    cvta.to.global.u64 %rd18, %rd12;\n"
"    shl.b64 %rd19, %rd2, 3;\n"
"    add.s64 %rd20, %rd18, %rd19;\n"
"    st.global.f64 [%rd20], %fd4;\n"
"$L__BB1_11:\n"
"    ret;\n"
"}\n";

#ifdef _WIN32
#include <windows.h>
typedef int (__stdcall *PFN_cuInit)(unsigned int);
typedef int (__stdcall *PFN_cuDriverGetVersion)(int*);
typedef int (__stdcall *PFN_cuDeviceGetCount)(int*);
typedef int (__stdcall *PFN_cuDeviceGet)(int*, int);
typedef int (__stdcall *PFN_cuDevicePrimaryCtxRetain)(void**, int);
typedef int (__stdcall *PFN_cuDevicePrimaryCtxRelease)(int);
typedef int (__stdcall *PFN_cuCtxSetCurrent)(void*);
typedef int (__stdcall *PFN_cuMemGetInfo_v2)(size_t*, size_t*);
typedef int (__stdcall *PFN_cuMemAlloc_v2)(uint64_t*, size_t);
typedef int (__stdcall *PFN_cuMemFree_v2)(uint64_t);
typedef int (__stdcall *PFN_cuMemcpyHtoD_v2)(uint64_t, const void*, size_t);
typedef int (__stdcall *PFN_cuMemcpyDtoH_v2)(void*, uint64_t, size_t);
typedef int (__stdcall *PFN_cuModuleLoadData)(void**, const void*);
typedef int (__stdcall *PFN_cuModuleUnload)(void*);
typedef int (__stdcall *PFN_cuModuleGetFunction)(void**, void*, const char*);
typedef int (__stdcall *PFN_cuLaunchKernel)(
    void*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, void*, void**, void**
);
typedef int (__stdcall *PFN_cuCtxSynchronize)(void);

struct CudaDriverApi {
    HMODULE hMod = nullptr;
    PFN_cuInit cuInit = nullptr;
    PFN_cuDriverGetVersion cuDriverGetVersion = nullptr;
    PFN_cuDeviceGetCount cuDeviceGetCount = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain = nullptr;
    PFN_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease = nullptr;
    PFN_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    PFN_cuMemGetInfo_v2 cuMemGetInfo = nullptr;
    PFN_cuMemAlloc_v2 cuMemAlloc = nullptr;
    PFN_cuMemFree_v2 cuMemFree = nullptr;
    PFN_cuMemcpyHtoD_v2 cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cuCtxSynchronize cuCtxSynchronize = nullptr;
    void* primaryCtx = nullptr;
    int primaryDev = 0;
    bool attempted = false;
    bool initialized = false;
    void* rij_mod = nullptr;
    void* fn_rij_accum = nullptr;
    void* fn_rij_v = nullptr;
    uint64_t cached_b_ptr = 0;
    const double* cached_h_b = nullptr;
    int cached_n_p = 0;
    int cached_n_ao_sq = 0;
    double cached_first_val = 0.0;
    double cached_last_val = 0.0;
    uint64_t scratch_p_ptr = 0;
    size_t scratch_p_cap = 0;
    uint64_t scratch_ao_ptr = 0;
    size_t scratch_ao_cap = 0;

    void release_cached_resources() {
        if (cached_b_ptr && cuMemFree) {
            cuMemFree(cached_b_ptr);
        }
        cached_b_ptr = 0;
        cached_h_b = nullptr;
        cached_n_p = 0;
        cached_n_ao_sq = 0;
        if (scratch_p_ptr && cuMemFree) {
            cuMemFree(scratch_p_ptr);
        }
        scratch_p_ptr = 0;
        scratch_p_cap = 0;
        if (scratch_ao_ptr && cuMemFree) {
            cuMemFree(scratch_ao_ptr);
        }
        scratch_ao_ptr = 0;
        scratch_ao_cap = 0;
        if (rij_mod && cuModuleUnload) {
            cuModuleUnload(rij_mod);
        }
        rij_mod = nullptr;
        fn_rij_accum = nullptr;
        fn_rij_v = nullptr;
    }

    ~CudaDriverApi() {
        release_cached_resources();
        if (primaryCtx && cuDevicePrimaryCtxRelease) {
            cuDevicePrimaryCtxRelease(primaryDev);
            primaryCtx = nullptr;
        }
        if (hMod) {
            FreeLibrary(hMod);
            hMod = nullptr;
        }
    }

    bool fail_cleanup() {
        release_cached_resources();
        if (primaryCtx && cuDevicePrimaryCtxRelease) {
            cuDevicePrimaryCtxRelease(primaryDev);
            primaryCtx = nullptr;
        }
        if (hMod) {
            FreeLibrary(hMod);
            hMod = nullptr;
        }
        return false;
    }

    bool init() {
        if (initialized) return true;
        if (attempted) return false;
        attempted = true;
        hMod = LoadLibraryA("nvcuda.dll");
        if (!hMod) return false;
        cuInit = (PFN_cuInit)GetProcAddress(hMod, "cuInit");
        cuDriverGetVersion = (PFN_cuDriverGetVersion)GetProcAddress(hMod, "cuDriverGetVersion");
        cuDeviceGetCount = (PFN_cuDeviceGetCount)GetProcAddress(hMod, "cuDeviceGetCount");
        cuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(hMod, "cuDeviceGet");
        cuDevicePrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(hMod, "cuDevicePrimaryCtxRetain");
        cuDevicePrimaryCtxRelease = (PFN_cuDevicePrimaryCtxRelease)GetProcAddress(hMod, "cuDevicePrimaryCtxRelease");
        if (!cuDevicePrimaryCtxRelease) {
            cuDevicePrimaryCtxRelease = (PFN_cuDevicePrimaryCtxRelease)GetProcAddress(hMod, "cuDevicePrimaryCtxRelease_v2");
        }
        cuCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(hMod, "cuCtxSetCurrent");
        cuMemGetInfo = (PFN_cuMemGetInfo_v2)GetProcAddress(hMod, "cuMemGetInfo_v2");
        if (!cuMemGetInfo) cuMemGetInfo = (PFN_cuMemGetInfo_v2)GetProcAddress(hMod, "cuMemGetInfo");
        cuMemAlloc = (PFN_cuMemAlloc_v2)GetProcAddress(hMod, "cuMemAlloc_v2");
        if (!cuMemAlloc) cuMemAlloc = (PFN_cuMemAlloc_v2)GetProcAddress(hMod, "cuMemAlloc");
        cuMemFree = (PFN_cuMemFree_v2)GetProcAddress(hMod, "cuMemFree_v2");
        if (!cuMemFree) cuMemFree = (PFN_cuMemFree_v2)GetProcAddress(hMod, "cuMemFree");
        cuMemcpyHtoD = (PFN_cuMemcpyHtoD_v2)GetProcAddress(hMod, "cuMemcpyHtoD_v2");
        if (!cuMemcpyHtoD) cuMemcpyHtoD = (PFN_cuMemcpyHtoD_v2)GetProcAddress(hMod, "cuMemcpyHtoD");
        cuMemcpyDtoH = (PFN_cuMemcpyDtoH_v2)GetProcAddress(hMod, "cuMemcpyDtoH_v2");
        if (!cuMemcpyDtoH) cuMemcpyDtoH = (PFN_cuMemcpyDtoH_v2)GetProcAddress(hMod, "cuMemcpyDtoH");
        cuModuleLoadData = (PFN_cuModuleLoadData)GetProcAddress(hMod, "cuModuleLoadData");
        cuModuleUnload = (PFN_cuModuleUnload)GetProcAddress(hMod, "cuModuleUnload");
        cuModuleGetFunction = (PFN_cuModuleGetFunction)GetProcAddress(hMod, "cuModuleGetFunction");
        cuLaunchKernel = (PFN_cuLaunchKernel)GetProcAddress(hMod, "cuLaunchKernel");
        cuCtxSynchronize = (PFN_cuCtxSynchronize)GetProcAddress(hMod, "cuCtxSynchronize");

        if (!cuInit || !cuDriverGetVersion || !cuDeviceGetCount || !cuDeviceGet ||
            !cuDevicePrimaryCtxRetain || !cuDevicePrimaryCtxRelease || !cuCtxSetCurrent || !cuMemGetInfo ||
            !cuMemAlloc || !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH ||
            !cuModuleLoadData || !cuModuleUnload || !cuModuleGetFunction ||
            !cuLaunchKernel || !cuCtxSynchronize) {
            return fail_cleanup();
        }
        if (cuInit(0) != 0) return fail_cleanup();
        int driverVersion = 0;
        if (cuDriverGetVersion(&driverVersion) != 0 || driverVersion < 11000) return fail_cleanup();
        int count = 0;
        if (cuDeviceGetCount(&count) != 0 || count <= 0) return fail_cleanup();
        int dev = 0;
        if (cuDeviceGet(&dev, 0) != 0) return fail_cleanup();
        primaryDev = dev;
        if (cuDevicePrimaryCtxRetain(&primaryCtx, dev) != 0 || !primaryCtx) return fail_cleanup();
        if (cuCtxSetCurrent(primaryCtx) != 0) return fail_cleanup();
        initialized = true;
        return true;
    }

    bool bind_context() {
        if (!init()) return false;
        return cuCtxSetCurrent(primaryCtx) == 0;
    }

    bool ensure_kernels() {
        if (rij_mod && fn_rij_accum && fn_rij_v) return true;
        if (rij_mod) {
            cuModuleUnload(rij_mod);
            rij_mod = nullptr;
            fn_rij_accum = nullptr;
            fn_rij_v = nullptr;
        }
#ifdef HAVE_CALCHEM_CUBIN_H
        if (cuModuleLoadData(&rij_mod, CALCHEM_SM120_CUBIN) != 0) {
            rij_mod = nullptr;
        }
#endif
        if (!rij_mod) {
            if (cuModuleLoadData(&rij_mod, CALCHEM_PTX_FALLBACK) != 0 || !rij_mod) {
                rij_mod = nullptr;
                return false;
            }
        }
        if (cuModuleGetFunction(&fn_rij_accum, rij_mod, "rij_accumulate_kernel") != 0
            || !fn_rij_accum
            || cuModuleGetFunction(&fn_rij_v, rij_mod, "rij_contract_v_kernel") != 0
            || !fn_rij_v) {
            cuModuleUnload(rij_mod);
            rij_mod = nullptr;
            fn_rij_accum = nullptr;
            fn_rij_v = nullptr;
            return false;
        }
        return true;
    }

    int ensure_b_and_scratch(const double* h_b_block, int n_p, int n_ao_sq) {
        size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
        double first_val = h_b_block[0];
        double last_val = h_b_block[b_count - 1];
        bool b_hit = (cached_b_ptr != 0
            && cached_h_b == h_b_block
            && cached_n_p == n_p
            && cached_n_ao_sq == n_ao_sq
            && cached_first_val == first_val
            && cached_last_val == last_val);

        size_t alloc_needed = 0;
        if (!b_hit) {
            if (cached_b_ptr) {
                cuMemFree(cached_b_ptr);
                cached_b_ptr = 0;
                cached_h_b = nullptr;
                cached_n_p = 0;
                cached_n_ao_sq = 0;
            }
            alloc_needed += b_count * sizeof(double);
        }
        if ((size_t)n_p > scratch_p_cap) {
            if (scratch_p_ptr) {
                cuMemFree(scratch_p_ptr);
                scratch_p_ptr = 0;
                scratch_p_cap = 0;
            }
            alloc_needed += (size_t)n_p * sizeof(double);
        }
        if ((size_t)n_ao_sq > scratch_ao_cap) {
            if (scratch_ao_ptr) {
                cuMemFree(scratch_ao_ptr);
                scratch_ao_ptr = 0;
                scratch_ao_cap = 0;
            }
            alloc_needed += (size_t)n_ao_sq * sizeof(double);
        }

        if (alloc_needed > 0) {
            size_t free_mem = 0, total_mem = 0;
            if (cuMemGetInfo(&free_mem, &total_mem) != 0
                || free_mem < alloc_needed
                || free_mem - alloc_needed < 64 * 1024 * 1024) {
                return -3;
            }
        }

        if (!b_hit) {
            size_t b_bytes = b_count * sizeof(double);
            if (cuMemAlloc(&cached_b_ptr, b_bytes) != 0) {
                cached_b_ptr = 0;
                return -4;
            }
            if (cuMemcpyHtoD(cached_b_ptr, h_b_block, b_bytes) != 0) {
                cuMemFree(cached_b_ptr);
                cached_b_ptr = 0;
                return -5;
            }
            cached_h_b = h_b_block;
            cached_n_p = n_p;
            cached_n_ao_sq = n_ao_sq;
            cached_first_val = first_val;
            cached_last_val = last_val;
        }
        if ((size_t)n_p > scratch_p_cap) {
            if (cuMemAlloc(&scratch_p_ptr, (size_t)n_p * sizeof(double)) != 0) {
                scratch_p_ptr = 0;
                return -4;
            }
            scratch_p_cap = (size_t)n_p;
        }
        if ((size_t)n_ao_sq > scratch_ao_cap) {
            if (cuMemAlloc(&scratch_ao_ptr, (size_t)n_ao_sq * sizeof(double)) != 0) {
                scratch_ao_ptr = 0;
                return -4;
            }
            scratch_ao_cap = (size_t)n_ao_sq;
        }
        return 0;
    }
};

static CudaDriverApi g_drv;
#else
struct HostRijCache {
    double* cached_b = nullptr;
    const double* cached_h_b = nullptr;
    int cached_n_p = 0;
    int cached_n_ao_sq = 0;
    double cached_first_val = 0.0;
    double cached_last_val = 0.0;
    double* scratch_p = nullptr;
    size_t scratch_p_cap = 0;
    double* scratch_ao = nullptr;
    size_t scratch_ao_cap = 0;

    ~HostRijCache() {
        if (cached_b) cudaFree(cached_b);
        if (scratch_p) cudaFree(scratch_p);
        if (scratch_ao) cudaFree(scratch_ao);
    }

    int ensure_b_and_scratch(const double* h_b_block, int n_p, int n_ao_sq) {
        size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
        double first_val = h_b_block[0];
        double last_val = h_b_block[b_count - 1];
        bool b_hit = (cached_b != nullptr
            && cached_h_b == h_b_block
            && cached_n_p == n_p
            && cached_n_ao_sq == n_ao_sq
            && cached_first_val == first_val
            && cached_last_val == last_val);

        size_t alloc_needed = 0;
        if (!b_hit) {
            if (cached_b) {
                cudaFree(cached_b);
                cached_b = nullptr;
                cached_h_b = nullptr;
                cached_n_p = 0;
                cached_n_ao_sq = 0;
            }
            alloc_needed += b_count * sizeof(double);
        }
        if ((size_t)n_p > scratch_p_cap) {
            if (scratch_p) {
                cudaFree(scratch_p);
                scratch_p = nullptr;
                scratch_p_cap = 0;
            }
            alloc_needed += (size_t)n_p * sizeof(double);
        }
        if ((size_t)n_ao_sq > scratch_ao_cap) {
            if (scratch_ao) {
                cudaFree(scratch_ao);
                scratch_ao = nullptr;
                scratch_ao_cap = 0;
            }
            alloc_needed += (size_t)n_ao_sq * sizeof(double);
        }
        if (alloc_needed > 0) {
            size_t free_mem = 0, total_mem = 0;
            if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess
                || free_mem < alloc_needed
                || free_mem - alloc_needed < 64 * 1024 * 1024) {
                return -3;
            }
        }
        if (!b_hit) {
            size_t b_bytes = b_count * sizeof(double);
            if (cudaMalloc((void**)&cached_b, b_bytes) != cudaSuccess) {
                cached_b = nullptr;
                return -4;
            }
            if (cudaMemcpy(cached_b, h_b_block, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(cached_b);
                cached_b = nullptr;
                return -5;
            }
            cached_h_b = h_b_block;
            cached_n_p = n_p;
            cached_n_ao_sq = n_ao_sq;
            cached_first_val = first_val;
            cached_last_val = last_val;
        }
        if ((size_t)n_p > scratch_p_cap) {
            if (cudaMalloc((void**)&scratch_p, (size_t)n_p * sizeof(double)) != cudaSuccess) {
                scratch_p = nullptr;
                return -4;
            }
            scratch_p_cap = (size_t)n_p;
        }
        if ((size_t)n_ao_sq > scratch_ao_cap) {
            if (cudaMalloc((void**)&scratch_ao, (size_t)n_ao_sq * sizeof(double)) != cudaSuccess) {
                scratch_ao = nullptr;
                return -4;
            }
            scratch_ao_cap = (size_t)n_ao_sq;
        }
        return 0;
    }
};

static HostRijCache g_rt_cache;
#endif

extern "C" {

int calchem_cuda_is_available() {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    return g_drv.bind_context() ? 1 : 0;
#else
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
#endif
}

int calchem_cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!free_bytes || !total_bytes) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -1;
    return g_drv.cuMemGetInfo(free_bytes, total_bytes);
#else
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return (err == cudaSuccess) ? 0 : (int)err;
#endif
}

static int check_rij_budget(int n_p, int n_ao_sq, size_t memory_limit_bytes) {
    size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
    if (n_ao_sq > 0 && b_count / (size_t)n_ao_sq != (size_t)n_p) return -2;
    if (b_count > SIZE_MAX / sizeof(double) - (size_t)n_p - (size_t)n_ao_sq) return -2;
    size_t req_bytes = (b_count + (size_t)n_p + (size_t)n_ao_sq) * sizeof(double);
    if (req_bytes > memory_limit_bytes) {
        return -2; // Memory budget exceeded -> CPU fallback
    }
    return 0;
}

int calchem_cuda_rij_contract_coulomb(
    const double* h_b_block,
    const double* h_x_block,
    double* h_j,
    int n_p,
    int n_ao_sq,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_b_block || !h_x_block || !h_j || n_p <= 0 || n_ao_sq <= 0) {
        return -1;
    }
    int budget_rc = check_rij_budget(n_p, n_ao_sq, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels()) return -5;
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq);
    if (prep != 0) return prep;

    size_t j_bytes = (size_t)n_ao_sq * sizeof(double);
    if (g_drv.cuMemcpyHtoD(g_drv.scratch_p_ptr, h_x_block, (size_t)n_p * sizeof(double)) != 0
        || g_drv.cuMemcpyHtoD(g_drv.scratch_ao_ptr, h_j, j_bytes) != 0) {
        return -5;
    }

    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)(((size_t)n_ao_sq + block_size - 1) / block_size);
    void* params[5] = {
        &g_drv.cached_b_ptr,
        &g_drv.scratch_p_ptr,
        &g_drv.scratch_ao_ptr,
        &n_p,
        &n_ao_sq
    };
    if (g_drv.cuLaunchKernel(
            g_drv.fn_rij_accum, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, params, nullptr) != 0) {
        return -5;
    }
    if (g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_j, g_drv.scratch_ao_ptr, j_bytes) == 0) ? 0 : -5;
#else
    int prep = g_rt_cache.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq);
    if (prep != 0) return prep;

    size_t j_bytes = (size_t)n_ao_sq * sizeof(double);
    if (cudaMemcpy(g_rt_cache.scratch_p, h_x_block, (size_t)n_p * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(g_rt_cache.scratch_ao, h_j, j_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -5;
    }
    int block_size = 256;
    int grid_size = (int)(((size_t)n_ao_sq + block_size - 1) / block_size);
    rij_accumulate_kernel<<<grid_size, block_size>>>(
        g_rt_cache.cached_b, g_rt_cache.scratch_p, g_rt_cache.scratch_ao, n_p, n_ao_sq);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return -5;
    }
    return (cudaMemcpy(h_j, g_rt_cache.scratch_ao, j_bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -5;
#endif
}

int calchem_cuda_rij_contract_v(
    const double* h_b_block,
    const double* h_p_block,
    double* h_v,
    int n_p,
    int n_ao_sq,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_b_block || !h_p_block || !h_v || n_p <= 0 || n_ao_sq <= 0) {
        return -1;
    }
    int budget_rc = check_rij_budget(n_p, n_ao_sq, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels()) return -5;
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq);
    if (prep != 0) return prep;

    size_t p_bytes = (size_t)n_ao_sq * sizeof(double);
    size_t v_bytes = (size_t)n_p * sizeof(double);
    if (g_drv.cuMemcpyHtoD(g_drv.scratch_ao_ptr, h_p_block, p_bytes) != 0) {
        return -5;
    }

    const unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)n_p;
    void* params[5] = {
        &g_drv.cached_b_ptr,
        &g_drv.scratch_ao_ptr,
        &g_drv.scratch_p_ptr,
        &n_p,
        &n_ao_sq
    };
    if (g_drv.cuLaunchKernel(
            g_drv.fn_rij_v, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, params, nullptr) != 0) {
        return -5;
    }
    if (g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_v, g_drv.scratch_p_ptr, v_bytes) == 0) ? 0 : -5;
#else
    int prep = g_rt_cache.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq);
    if (prep != 0) return prep;

    size_t p_bytes = (size_t)n_ao_sq * sizeof(double);
    size_t v_bytes = (size_t)n_p * sizeof(double);
    if (cudaMemcpy(g_rt_cache.scratch_ao, h_p_block, p_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -5;
    }
    const int block_size = 256;
    int grid_size = n_p;
    rij_contract_v_kernel<<<grid_size, block_size>>>(
        g_rt_cache.cached_b, g_rt_cache.scratch_ao, g_rt_cache.scratch_p, n_p, n_ao_sq);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return -5;
    }
    return (cudaMemcpy(h_v, g_rt_cache.scratch_p, v_bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -5;
#endif
}

}
