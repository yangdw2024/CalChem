#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mutex>

struct AuxShellGpu {
    double center[3];
    double bound;
    int lp;
    int ao_offset;
    int prim_start;
    int n_prims;
};

struct AuxPrimGpu {
    double gamma_p;
    double cp_norm[6];
};

struct ShellPair3CGpu {
    double q_ij;
    int li;
    int lj;
    int same_shell;
    int col_offset;
    int prim_start;
    int n_prims;
};

struct PrimPair3CGpu {
    double q;
    double p_center[3];
    double qc[3];
    double qd[3];
    double two_pi25_kab_over_q;
    double ci_norm[6];
    double cj_norm[6];
};

struct AoShellGpu {
    double center[3];
    double r2_cutoff;
    int ao_offset;
    int n_funcs;
    int prim_start;
    int n_prims;
    int comp_start;
    int _pad;
};

struct AoPrimGpu {
    double exponent;
    double weighted_norm;
};

struct AoCompGpu {
    double scale;
    int lx;
    int ly;
    int lz;
    int _pad;
};

struct GridPtGpu {
    double x;
    double y;
    double z;
    int batch_id;
    int _pad;
};

__device__ __forceinline__ void pow_d0_d1_dev(double x, int l, double& p, double& dp) {
    if (l == 0) {
        p = 1.0; dp = 0.0;
    } else if (l == 1) {
        p = x; dp = 1.0;
    } else if (l == 2) {
        p = x * x; dp = 2.0 * x;
    } else if (l == 3) {
        double x2 = x * x;
        p = x2 * x; dp = 3.0 * x2;
    } else {
        double xm1 = pow(x, (double)(l - 1));
        p = xm1 * x; dp = (double)l * xm1;
    }
}

extern "C" __global__ void zero_doubles_kernel(double* __restrict__ buf, size_t len) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < len) buf[idx] = 0.0;
}

#define VXC_MAX_ACTIVE_AO 512

extern "C" __global__ void vxc_rho_grad_kernel(
    const AoShellGpu* __restrict__ shells,
    const AoPrimGpu* __restrict__ prims,
    const AoCompGpu* __restrict__ comps,
    const GridPtGpu* __restrict__ pts,
    const int* __restrict__ cand_offsets,
    const int* __restrict__ cand_shells,
    const double* __restrict__ p_mat,
    double* __restrict__ rho_grad_out,
    int* __restrict__ overflow_flag,
    int n_pts,
    int n_ao
) {
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= n_pts) return;

    const GridPtGpu pt = pts[g];
    const int c_start = cand_offsets[pt.batch_id];
    const int c_end = cand_offsets[pt.batch_id + 1];

    int act_mu[VXC_MAX_ACTIVE_AO];
    double act_phi[VXC_MAX_ACTIVE_AO];
    int n_act = 0;

    // Pass 1: collect active AOs (mu, phi)
    for (int ci = c_start; ci < c_end; ++ci) {
        const AoShellGpu sh = shells[cand_shells[ci]];
        const double dx = pt.x - sh.center[0];
        const double dy = pt.y - sh.center[1];
        const double dz = pt.z - sh.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > sh.r2_cutoff) continue;

        double r0 = 0.0, r1 = 0.0;
        for (int pi = 0; pi < sh.n_prims; ++pi) {
            const AoPrimGpu pr = prims[sh.prim_start + pi];
            const double e = pr.weighted_norm * exp(-pr.exponent * r2);
            r0 += e;
            r1 -= 2.0 * pr.exponent * e;
        }
        if (fabs(r0) < 1e-18 && fabs(r1) < 1e-18) continue;

        for (int ic = 0; ic < sh.n_funcs; ++ic) {
            const AoCompGpu cp = comps[sh.comp_start + ic];
            double px, dpx, py, dpy, pz, dpz;
            pow_d0_d1_dev(dx, cp.lx, px, dpx);
            pow_d0_d1_dev(dy, cp.ly, py, dpy);
            pow_d0_d1_dev(dz, cp.lz, pz, dpz);
            const double m = px * py * pz;
            const double val = cp.scale * m * r0;
            const double gx = cp.scale * (dpx * py * pz * r0 + m * dx * r1);
            const double gy = cp.scale * (px * dpy * pz * r0 + m * dy * r1);
            const double gz = cp.scale * (px * py * dpz * r0 + m * dz * r1);
            if (fabs(val) > 1e-16 || fabs(gx) > 1e-16 || fabs(gy) > 1e-16 || fabs(gz) > 1e-16) {
                if (n_act >= VXC_MAX_ACTIVE_AO) {
                    atomicExch(overflow_flag, 1);
                    return;
                }
                act_mu[n_act] = sh.ao_offset + ic;
                act_phi[n_act] = val;
                n_act++;
            }
        }
    }

    if (n_act == 0) {
        size_t o = (size_t)g * 4;
        rho_grad_out[o] = 0.0;
        rho_grad_out[o + 1] = 0.0;
        rho_grad_out[o + 2] = 0.0;
        rho_grad_out[o + 3] = 0.0;
        return;
    }

    // Pass 2: evaluate gradients in registers and contract with P * phi
    double rho_t = 0.0, grx = 0.0, gry = 0.0, grz = 0.0;
    for (int ci = c_start; ci < c_end; ++ci) {
        const AoShellGpu sh = shells[cand_shells[ci]];
        const double dx = pt.x - sh.center[0];
        const double dy = pt.y - sh.center[1];
        const double dz = pt.z - sh.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > sh.r2_cutoff) continue;

        double r0 = 0.0, r1 = 0.0;
        for (int pi = 0; pi < sh.n_prims; ++pi) {
            const AoPrimGpu pr = prims[sh.prim_start + pi];
            const double e = pr.weighted_norm * exp(-pr.exponent * r2);
            r0 += e;
            r1 -= 2.0 * pr.exponent * e;
        }
        if (fabs(r0) < 1e-18 && fabs(r1) < 1e-18) continue;

        for (int ic = 0; ic < sh.n_funcs; ++ic) {
            const AoCompGpu cp = comps[sh.comp_start + ic];
            double px, dpx, py, dpy, pz, dpz;
            pow_d0_d1_dev(dx, cp.lx, px, dpx);
            pow_d0_d1_dev(dy, cp.ly, py, dpy);
            pow_d0_d1_dev(dz, cp.lz, pz, dpz);
            const double m = px * py * pz;
            const double val = cp.scale * m * r0;
            const double gx = cp.scale * (dpx * py * pz * r0 + m * dx * r1);
            const double gy = cp.scale * (px * dpy * pz * r0 + m * dy * r1);
            const double gz = cp.scale * (px * py * dpz * r0 + m * dz * r1);
            if (fabs(val) > 1e-16 || fabs(gx) > 1e-16 || fabs(gy) > 1e-16 || fabs(gz) > 1e-16) {
                const int mu = sh.ao_offset + ic;
                const double* p_row = p_mat + (size_t)mu * (size_t)n_ao;
                double s = 0.0;
                for (int j = 0; j < n_act; ++j) {
                    s += p_row[act_mu[j]] * act_phi[j];
                }
                rho_t += val * s;
                grx += 2.0 * gx * s;
                gry += 2.0 * gy * s;
                grz += 2.0 * gz * s;
            }
        }
    }

    size_t o = (size_t)g * 4;
    rho_grad_out[o] = rho_t;
    rho_grad_out[o + 1] = grx;
    rho_grad_out[o + 2] = gry;
    rho_grad_out[o + 3] = grz;
}

extern "C" __global__ void vxc_accum_kernel(
    const AoShellGpu* __restrict__ shells,
    const AoPrimGpu* __restrict__ prims,
    const AoCompGpu* __restrict__ comps,
    const GridPtGpu* __restrict__ pts,
    const int* __restrict__ cand_offsets,
    const int* __restrict__ cand_shells,
    const double* __restrict__ wv_in,
    double* __restrict__ v_xc_out,
    int* __restrict__ overflow_flag,
    int n_pts,
    int n_ao
) {
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= n_pts) return;

    size_t o = (size_t)g * 4;
    const double w_vrho = wv_in[o];
    const double wu_x = wv_in[o + 1];
    const double wu_y = wv_in[o + 2];
    const double wu_z = wv_in[o + 3];
    if (w_vrho == 0.0 && wu_x == 0.0 && wu_y == 0.0 && wu_z == 0.0) return;

    const GridPtGpu pt = pts[g];
    const int c_start = cand_offsets[pt.batch_id];
    const int c_end = cand_offsets[pt.batch_id + 1];

    int act_mu[VXC_MAX_ACTIVE_AO];
    double act_phi[VXC_MAX_ACTIVE_AO];
    double act_z[VXC_MAX_ACTIVE_AO];
    int n_act = 0;

    for (int ci = c_start; ci < c_end; ++ci) {
        const AoShellGpu sh = shells[cand_shells[ci]];
        const double dx = pt.x - sh.center[0];
        const double dy = pt.y - sh.center[1];
        const double dz = pt.z - sh.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > sh.r2_cutoff) continue;

        double r0 = 0.0, r1 = 0.0;
        for (int pi = 0; pi < sh.n_prims; ++pi) {
            const AoPrimGpu pr = prims[sh.prim_start + pi];
            const double e = pr.weighted_norm * exp(-pr.exponent * r2);
            r0 += e;
            r1 -= 2.0 * pr.exponent * e;
        }
        if (fabs(r0) < 1e-18 && fabs(r1) < 1e-18) continue;

        for (int ic = 0; ic < sh.n_funcs; ++ic) {
            const AoCompGpu cp = comps[sh.comp_start + ic];
            double px, dpx, py, dpy, pz, dpz;
            pow_d0_d1_dev(dx, cp.lx, px, dpx);
            pow_d0_d1_dev(dy, cp.ly, py, dpy);
            pow_d0_d1_dev(dz, cp.lz, pz, dpz);
            const double m = px * py * pz;
            const double val = cp.scale * m * r0;
            const double gx = cp.scale * (dpx * py * pz * r0 + m * dx * r1);
            const double gy = cp.scale * (px * dpy * pz * r0 + m * dy * r1);
            const double gz = cp.scale * (px * py * dpz * r0 + m * dz * r1);
            if (fabs(val) > 1e-16 || fabs(gx) > 1e-16 || fabs(gy) > 1e-16 || fabs(gz) > 1e-16) {
                if (n_act >= VXC_MAX_ACTIVE_AO) {
                    atomicExch(overflow_flag, 1);
                    return;
                }
                act_mu[n_act] = sh.ao_offset + ic;
                act_phi[n_act] = val;
                act_z[n_act] = w_vrho * val + wu_x * gx + wu_y * gy + wu_z * gz;
                n_act++;
            }
        }
    }

    for (int i = 0; i < n_act; ++i) {
        const int mu_i = act_mu[i];
        const double phi_i = act_phi[i];
        const double z_i = act_z[i];
        for (int j = 0; j <= i; ++j) {
            const int mu_j = act_mu[j];
            const double contrib = z_i * act_phi[j] + phi_i * act_z[j];
            const int r = (mu_i >= mu_j) ? mu_i : mu_j;
            const int c = (mu_i >= mu_j) ? mu_j : mu_i;
            atomicAdd(&v_xc_out[(size_t)r * (size_t)n_ao + (size_t)c], contrib);
        }
    }
}

__device__ __forceinline__ void boys_batch_dev(int max_m, double t, double* out) {
    if (t < 1e-10) {
        for (int m = 0; m <= max_m; ++m) {
            double dm = (double)(2 * m);
            out[m] = 1.0 / (dm + 1.0) - t / (dm + 3.0) + 0.5 * t * t / (dm + 5.0);
        }
        return;
    }
    if (t >= 35.0) {
        const double inv_2t = 0.5 / t;
        out[0] = 0.5 * sqrt(3.14159265358979323846 / t);
        for (int m = 1; m <= max_m; ++m) {
            out[m] = (double)(2 * m - 1) * out[m - 1] * inv_2t;
        }
        return;
    }
    if (t >= 4.0) {
        const double inv_2t = 0.5 / t;
        const double exp_mt = exp(-t);
        out[0] = 0.5 * sqrt(3.14159265358979323846 / t) * erf(sqrt(t));
        for (int m = 1; m <= max_m; ++m) {
            out[m] = ((double)(2 * m - 1) * out[m - 1] - exp_mt) * inv_2t;
        }
        return;
    }
    // Downward recurrence for 1e-10 <= t < 4.0: full FP64 precision (< 2e-16)
    const double exp_mt = exp(-t);
    const double two_t = 2.0 * t;
    const int m_top = max_m + 34;
    double fm = 0.0;
    for (int m = m_top; m > max_m; --m) {
        fm = (two_t * fm + exp_mt) / (double)(2 * m - 1);
    }
    out[max_m] = fm;
    for (int m = max_m; m >= 1; --m) {
        fm = (two_t * fm + exp_mt) / (double)(2 * m - 1);
        out[m - 1] = fm;
    }
}

// Static table of 35 components for L <= 4
struct Comp3C {
    signed char x, y, z, L;
    signed char axis, n_k, idx1, idx2;
    signed char sub[3];
};

__constant__ const Comp3C COMPS_3C[35] = {
    {0, 0, 0, 0, 0, 0, 0, -1, {-1, -1, -1}}, // 0: (0,0,0)
    {1, 0, 0, 1, 0, 0, 0, -1, {0, -1, -1}}, // 1: (1,0,0)
    {0, 1, 0, 1, 1, 0, 0, -1, {-1, 0, -1}}, // 2: (0,1,0)
    {0, 0, 1, 1, 2, 0, 0, -1, {-1, -1, 0}}, // 3: (0,0,1)
    {2, 0, 0, 2, 0, 1, 1, 0, {1, -1, -1}}, // 4: (2,0,0)
    {1, 1, 0, 2, 0, 0, 2, -1, {2, 1, -1}}, // 5: (1,1,0)
    {1, 0, 1, 2, 0, 0, 3, -1, {3, -1, 1}}, // 6: (1,0,1)
    {0, 2, 0, 2, 1, 1, 2, 0, {-1, 2, -1}}, // 7: (0,2,0)
    {0, 1, 1, 2, 1, 0, 3, -1, {-1, 3, 2}}, // 8: (0,1,1)
    {0, 0, 2, 2, 2, 1, 3, 0, {-1, -1, 3}}, // 9: (0,0,2)
    {3, 0, 0, 3, 0, 2, 4, 1, {4, -1, -1}}, // 10: (3,0,0)
    {2, 1, 0, 3, 0, 1, 5, 2, {5, 4, -1}}, // 11: (2,1,0)
    {2, 0, 1, 3, 0, 1, 6, 3, {6, -1, 4}}, // 12: (2,0,1)
    {1, 2, 0, 3, 0, 0, 7, -1, {7, 5, -1}}, // 13: (1,2,0)
    {1, 1, 1, 3, 0, 0, 8, -1, {8, 6, 5}}, // 14: (1,1,1)
    {1, 0, 2, 3, 0, 0, 9, -1, {9, -1, 6}}, // 15: (1,0,2)
    {0, 3, 0, 3, 1, 2, 7, 2, {-1, 7, -1}}, // 16: (0,3,0)
    {0, 2, 1, 3, 1, 1, 8, 3, {-1, 8, 7}}, // 17: (0,2,1)
    {0, 1, 2, 3, 1, 0, 9, -1, {-1, 9, 8}}, // 18: (0,1,2)
    {0, 0, 3, 3, 2, 2, 9, 3, {-1, -1, 9}}, // 19: (0,0,3)
    {4, 0, 0, 4, 0, 3, 10, 4, {10, -1, -1}}, // 20: (4,0,0)
    {3, 1, 0, 4, 0, 2, 11, 5, {11, 10, -1}}, // 21: (3,1,0)
    {3, 0, 1, 4, 0, 2, 12, 6, {12, -1, 10}}, // 22: (3,0,1)
    {2, 2, 0, 4, 0, 1, 13, 7, {13, 11, -1}}, // 23: (2,2,0)
    {2, 1, 1, 4, 0, 1, 14, 8, {14, 12, 11}}, // 24: (2,1,1)
    {2, 0, 2, 4, 0, 1, 15, 9, {15, -1, 12}}, // 25: (2,0,2)
    {1, 3, 0, 4, 0, 0, 16, -1, {16, 13, -1}}, // 26: (1,3,0)
    {1, 2, 1, 4, 0, 0, 17, -1, {17, 14, 13}}, // 27: (1,2,1)
    {1, 1, 2, 4, 0, 0, 18, -1, {18, 15, 14}}, // 28: (1,1,2)
    {1, 0, 3, 4, 0, 0, 19, -1, {19, -1, 15}}, // 29: (1,0,3)
    {0, 4, 0, 4, 1, 3, 16, 7, {-1, 16, -1}}, // 30: (0,4,0)
    {0, 3, 1, 4, 1, 2, 17, 8, {-1, 17, 16}}, // 31: (0,3,1)
    {0, 2, 2, 4, 1, 1, 18, 9, {-1, 18, 17}}, // 32: (0,2,2)
    {0, 1, 3, 4, 1, 0, 19, -1, {-1, 19, 18}}, // 33: (0,1,3)
    {0, 0, 4, 4, 2, 3, 19, 9, {-1, -1, 19}}, // 34: (0,0,4)
};

__constant__ const signed char ADD_AXIS_3C[35][3] = {
    {1, 2, 3}, // 0: (0,0,0)
    {4, 5, 6}, // 1: (1,0,0)
    {5, 7, 8}, // 2: (0,1,0)
    {6, 8, 9}, // 3: (0,0,1)
    {10, 11, 12}, // 4: (2,0,0)
    {11, 13, 14}, // 5: (1,1,0)
    {12, 14, 15}, // 6: (1,0,1)
    {13, 16, 17}, // 7: (0,2,0)
    {14, 17, 18}, // 8: (0,1,1)
    {15, 18, 19}, // 9: (0,0,2)
    {20, 21, 22}, // 10: (3,0,0)
    {21, 23, 24}, // 11: (2,1,0)
    {22, 24, 25}, // 12: (2,0,1)
    {23, 26, 27}, // 13: (1,2,0)
    {24, 27, 28}, // 14: (1,1,1)
    {25, 28, 29}, // 15: (1,0,2)
    {26, 30, 31}, // 16: (0,3,0)
    {27, 31, 32}, // 17: (0,2,1)
    {28, 32, 33}, // 18: (0,1,2)
    {29, 33, 34}, // 19: (0,0,3)
    {-1, -1, -1}, // 20: (4,0,0)
    {-1, -1, -1}, // 21: (3,1,0)
    {-1, -1, -1}, // 22: (3,0,1)
    {-1, -1, -1}, // 23: (2,2,0)
    {-1, -1, -1}, // 24: (2,1,1)
    {-1, -1, -1}, // 25: (2,0,2)
    {-1, -1, -1}, // 26: (1,3,0)
    {-1, -1, -1}, // 27: (1,2,1)
    {-1, -1, -1}, // 28: (1,1,2)
    {-1, -1, -1}, // 29: (1,0,3)
    {-1, -1, -1}, // 30: (0,4,0)
    {-1, -1, -1}, // 31: (0,3,1)
    {-1, -1, -1}, // 32: (0,2,2)
    {-1, -1, -1}, // 33: (0,1,3)
    {-1, -1, -1}, // 34: (0,0,4)
};

extern "C" __global__ void eri3c_build_packed_kernel(
    const AuxShellGpu* __restrict__ aux_shells,
    const AuxPrimGpu* __restrict__ aux_prims,
    const ShellPair3CGpu* __restrict__ shell_pairs,
    const PrimPair3CGpu* __restrict__ prim_pairs,
    double* __restrict__ b_out,
    int n_aux_shells,
    int n_shell_pairs,
    int width
) {
    int sp_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int p_idx = (int)blockIdx.y;
    if (sp_idx >= n_shell_pairs || p_idx >= n_aux_shells) return;

    const AuxShellGpu aux = aux_shells[p_idx];
    const ShellPair3CGpu sp = shell_pairs[sp_idx];
    const int lp = aux.lp;
    const int li = sp.li;
    const int lj = sp.lj;
    const int np = (lp == 0) ? 1 : ((lp == 1) ? 3 : 6);
    const int ni = (li == 0) ? 1 : ((li == 1) ? 3 : 6);
    const int nj = (lj == 0) ? 1 : ((lj == 1) ? 3 : 6);
    const int n_cols = sp.same_shell ? (ni * (ni + 1) / 2) : (ni * nj);

    if (aux.bound * sp.q_ij < 1e-12) {
        for (int pf = 0; pf < np; ++pf) {
            size_t base = (size_t)(aux.ao_offset + pf) * (size_t)width + (size_t)sp.col_offset;
            for (int c = 0; c < n_cols; ++c) {
                b_out[base + (size_t)c] = 0.0;
            }
        }
        return;
    }

    double acc[216]; // max 6 * 6 * 6 = 216
    const int blk_len = np * ni * nj;
    for (int k = 0; k < blk_len; ++k) acc[k] = 0.0;

    const int max_m = lp + li + lj;
    const int l_ket = li + lj;
    const double rp0 = aux.center[0];
    const double rp1 = aux.center[1];
    const double rp2 = aux.center[2];
    const int D_K[6] = {0, 0, 0, 1, 1, 2};
    const int D_L[6] = {0, 1, 2, 1, 2, 2};

    const int comp_off_i = (li == 0) ? 0 : ((li == 1) ? 1 : 4);

    for (int pp_i = 0; pp_i < aux.n_prims; ++pp_i) {
        const AuxPrimGpu ap = aux_prims[aux.prim_start + pp_i];
        const double p = ap.gamma_p;
        const double inv_2p = 0.5 / p;

        for (int ij_i = 0; ij_i < sp.n_prims; ++ij_i) {
            const PrimPair3CGpu pp = prim_pairs[sp.prim_start + ij_i];
            const double q = pp.q;
            const double zeta = p + q;
            const double inv_zeta = 1.0 / zeta;
            const double inv_2zeta = 0.5 * inv_zeta;
            const double rho = p * q * inv_zeta;
            const double dx = rp0 - pp.p_center[0];
            const double dy = rp1 - pp.p_center[1];
            const double dz = rp2 - pp.p_center[2];
            const double pq2 = dx * dx + dy * dy + dz * dz;

            double boys[7];
            boys_batch_dev(max_m, rho * pq2, boys);
            const double pref = pp.two_pi25_kab_over_q / (p * sqrt(zeta));

            const double wx = (p * rp0 + q * pp.p_center[0]) * inv_zeta;
            const double wy = (p * rp1 + q * pp.p_center[1]) * inv_zeta;
            const double wz = (p * rp2 + q * pp.p_center[2]) * inv_zeta;
            const double wp[3] = {wx - rp0, wy - rp1, wz - rp2};
            const double wq[3] = {wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]};
            const double rab[3] = {pp.qd[0] - pp.qc[0], pp.qd[1] - pp.qc[1], pp.qd[2] - pp.qc[2]};
            const double inv_2q = 0.5 / q;
            const double rho_over_q = rho / q;
            const double rho_over_p = rho / p;

            // Step 1: V0[35][7]
            double V0[35][7];
            for (int m = 0; m <= max_m; ++m) {
                V0[0][m] = pref * boys[m];
            }

            for (int idx = 1; idx < 35; ++idx) {
                const Comp3C& c = COMPS_3C[idx];
                if (c.L > l_ket) continue;
                int axis = c.axis;
                int idx1 = c.idx1;
                int idx2 = c.idx2;
                int n_k = c.n_k;
                for (int m = 0; m <= max_m - c.L; ++m) {
                    double val = pp.qc[axis] * V0[idx1][m] + wq[axis] * V0[idx1][m + 1];
                    if (n_k > 0) {
                        val += (double)n_k * inv_2q * (V0[idx2][m] - rho_over_q * V0[idx2][m + 1]);
                    }
                    V0[idx][m] = val;
                }
            }

            // Step 2: Aux VRR
            double V_P[6][35];
            if (lp == 0) {
                for (int idx = 0; idx < 35; ++idx) {
                    if (COMPS_3C[idx].L > l_ket) continue;
                    V_P[0][idx] = V0[idx][0];
                }
            } else if (lp == 1) {
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < 35; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        if (c.L > l_ket) continue;
                        double val = wp[u] * V0[idx][1];
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V0[sub_idx][1];
                        }
                        V_P[u][idx] = val;
                    }
                }
            } else { // lp == 2
                double V1[3][35][2];
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < 35; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        if (c.L > l_ket) continue;
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        for (int m_sub = 0; m_sub < 2; ++m_sub) {
                            double val = wp[u] * V0[idx][m_sub + 1];
                            if (sub_idx >= 0) {
                                val += (double)cu * inv_2zeta * V0[sub_idx][m_sub + 1];
                            }
                            V1[u][idx][m_sub] = val;
                        }
                    }
                }
                for (int kl = 0; kl < 6; ++kl) {
                    int u = D_K[kl];
                    int v = D_L[kl];
                    for (int idx = 0; idx < 35; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        if (c.L > l_ket) continue;
                        double val = wp[u] * V1[v][idx][1];
                        if (u == v) {
                            val += inv_2p * (V0[idx][0] - rho_over_p * V0[idx][1]);
                        }
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[v][sub_idx][1];
                        }
                        V_P[kl][idx] = val;
                    }
                }
            }

            // Step 3: HRR to (P | a, b) and accumulate into acc
            for (int pf = 0; pf < np; ++pf) {
                double cp_pf = ap.cp_norm[pf];
                int p_blk = pf * ni * nj;
                for (int a = 0; a < ni; ++a) {
                    int a_idx = comp_off_i + a;
                    double cp_ci = cp_pf * pp.ci_norm[a];
                    for (int b = 0; b < nj; ++b) {
                        double val;
                        if (lj == 0) {
                            val = V_P[pf][a_idx];
                        } else if (lj == 1) {
                            int k = b; // 0, 1, 2 for x, y, z
                            int a_k = ADD_AXIS_3C[a_idx][k];
                            val = V_P[pf][a_k] + rab[k] * V_P[pf][a_idx];
                        } else { // lj == 2
                            int k = D_K[b];
                            int l = D_L[b];
                            int a_k = ADD_AXIS_3C[a_idx][k];
                            int a_kl = ADD_AXIS_3C[a_k][l];
                            int a_l = ADD_AXIS_3C[a_idx][l];
                            double term1 = V_P[pf][a_kl] + rab[l] * V_P[pf][a_k];
                            double term2 = V_P[pf][a_l] + rab[l] * V_P[pf][a_idx];
                            val = term1 + rab[k] * term2;
                        }
                        acc[p_blk + a * nj + b] += cp_ci * pp.cj_norm[b] * val;
                    }
                }
            }
        }
    }

    // Write packed block to B
    for (int pf = 0; pf < np; ++pf) {
        size_t base = (size_t)(aux.ao_offset + pf) * (size_t)width + (size_t)sp.col_offset;
        int p_blk = pf * ni * nj;
        if (sp.same_shell) {
            int pos = 0;
            for (int mu = 0; mu < ni; ++mu) {
                for (int nu = 0; nu <= mu; ++nu) {
                    b_out[base + (size_t)pos] = acc[p_blk + mu * nj + nu];
                    pos++;
                }
            }
        } else {
            for (int c = 0; c < ni * nj; ++c) {
                b_out[base + (size_t)c] = acc[p_blk + c];
            }
        }
    }
}

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
    void* fn_eri3c = nullptr;
    void* fn_zero_doubles = nullptr;
    void* fn_vxc_rho = nullptr;
    void* fn_vxc_accum = nullptr;
    uint64_t vxc_grid_id = 0;
    int vxc_n_pts = 0;
    int vxc_n_ao = 0;
    uint64_t d_vxc_shells = 0;
    uint64_t d_vxc_prims = 0;
    uint64_t d_vxc_comps = 0;
    uint64_t d_vxc_pts = 0;
    uint64_t d_vxc_cand_off = 0;
    uint64_t d_vxc_cand_sh = 0;
    uint64_t d_vxc_p = 0;
    uint64_t d_vxc_v = 0;
    uint64_t d_vxc_rg = 0;
    uint64_t d_vxc_wv = 0;
    uint64_t d_vxc_ovf = 0;

    void release_vxc_resources() {
        if (cuMemFree) {
            if (d_vxc_shells) cuMemFree(d_vxc_shells);
            if (d_vxc_prims) cuMemFree(d_vxc_prims);
            if (d_vxc_comps) cuMemFree(d_vxc_comps);
            if (d_vxc_pts) cuMemFree(d_vxc_pts);
            if (d_vxc_cand_off) cuMemFree(d_vxc_cand_off);
            if (d_vxc_cand_sh) cuMemFree(d_vxc_cand_sh);
            if (d_vxc_p) cuMemFree(d_vxc_p);
            if (d_vxc_v) cuMemFree(d_vxc_v);
            if (d_vxc_rg) cuMemFree(d_vxc_rg);
            if (d_vxc_wv) cuMemFree(d_vxc_wv);
            if (d_vxc_ovf) cuMemFree(d_vxc_ovf);
        }
        d_vxc_shells = d_vxc_prims = d_vxc_comps = d_vxc_pts = 0;
        d_vxc_cand_off = d_vxc_cand_sh = 0;
        d_vxc_p = d_vxc_v = d_vxc_rg = d_vxc_wv = d_vxc_ovf = 0;
        vxc_grid_id = 0;
        vxc_n_pts = 0;
        vxc_n_ao = 0;
    }
    uint64_t cached_b_ptr = 0;
    const double* cached_h_b = nullptr;
    int cached_n_p = 0;
    int cached_n_ao_sq = 0;
    uint64_t cached_tensor_id = 0;
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
        release_vxc_resources();
        if (rij_mod && cuModuleUnload) {
            cuModuleUnload(rij_mod);
        }
        rij_mod = nullptr;
        fn_rij_accum = nullptr;
        fn_rij_v = nullptr;
        fn_eri3c = nullptr;
        fn_zero_doubles = nullptr;
        fn_vxc_rho = nullptr;
        fn_vxc_accum = nullptr;
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
            fn_eri3c = nullptr;
            return false;
        }
        // Optional native sm_120 kernels (not in PTX fallback)
        if (cuModuleGetFunction(&fn_eri3c, rij_mod, "eri3c_build_packed_kernel") != 0) {
            fn_eri3c = nullptr;
        }
        if (cuModuleGetFunction(&fn_zero_doubles, rij_mod, "zero_doubles_kernel") != 0) {
            fn_zero_doubles = nullptr;
        }
        if (cuModuleGetFunction(&fn_vxc_rho, rij_mod, "vxc_rho_grad_kernel") != 0) {
            fn_vxc_rho = nullptr;
        }
        if (cuModuleGetFunction(&fn_vxc_accum, rij_mod, "vxc_accum_kernel") != 0) {
            fn_vxc_accum = nullptr;
        }
        return true;
    }

    int ensure_b_and_scratch(const double* h_b_block, int n_p, int n_ao_sq, uint64_t tensor_id) {
        size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
        bool same_shape = cached_b_ptr && cached_n_p == n_p && cached_n_ao_sq == n_ao_sq;
        bool b_hit = same_shape && tensor_id != 0 && cached_tensor_id == tensor_id
            && cached_h_b == h_b_block;

        size_t alloc_needed = 0;
        if (!same_shape) {
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
            if (!same_shape && cuMemAlloc(&cached_b_ptr, b_bytes) != 0) {
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
            cached_tensor_id = tensor_id;
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
    uint64_t cached_tensor_id = 0;
    double* scratch_p = nullptr;
    size_t scratch_p_cap = 0;
    double* scratch_ao = nullptr;
    size_t scratch_ao_cap = 0;

    ~HostRijCache() {
        if (cached_b) cudaFree(cached_b);
        if (scratch_p) cudaFree(scratch_p);
        if (scratch_ao) cudaFree(scratch_ao);
    }

    int ensure_b_and_scratch(const double* h_b_block, int n_p, int n_ao_sq, uint64_t tensor_id) {
        size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
        bool same_shape = cached_b && cached_n_p == n_p && cached_n_ao_sq == n_ao_sq;
        bool b_hit = same_shape && tensor_id != 0 && cached_tensor_id == tensor_id
            && cached_h_b == h_b_block;

        size_t alloc_needed = 0;
        if (!same_shape) {
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
            if (!same_shape && cudaMalloc((void**)&cached_b, b_bytes) != cudaSuccess) {
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
            cached_tensor_id = tensor_id;
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

int calchem_cuda_rij_contract_coulomb_tagged(
    const double* h_b_block,
    const double* h_x_block,
    double* h_j,
    int n_p,
    int n_ao_sq,
    size_t memory_limit_bytes,
    uint64_t tensor_id
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
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, tensor_id);
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
    int prep = g_rt_cache.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, tensor_id);
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

int calchem_cuda_rij_contract_v_tagged(
    const double* h_b_block,
    const double* h_p_block,
    double* h_v,
    int n_p,
    int n_ao_sq,
    size_t memory_limit_bytes,
    uint64_t tensor_id
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
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, tensor_id);
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
    int prep = g_rt_cache.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, tensor_id);
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

// Raw callers may change a tensor between calls, including at the same address.
int calchem_cuda_rij_contract_coulomb(const double* b, const double* x, double* j,
                                      int n, int width, size_t limit) {
    return calchem_cuda_rij_contract_coulomb_tagged(b, x, j, n, width, limit, 0);
}

int calchem_cuda_rij_contract_v(const double* b, const double* p, double* v,
                                int n, int width, size_t limit) {
    return calchem_cuda_rij_contract_v_tagged(b, p, v, n, width, limit, 0);
}

int calchem_cuda_build_eri3c_packed(
    const AuxShellGpu* h_aux_shells,
    int n_aux_shells,
    const AuxPrimGpu* h_aux_prims,
    int n_aux_prims,
    const ShellPair3CGpu* h_shell_pairs,
    int n_shell_pairs,
    const PrimPair3CGpu* h_prim_pairs,
    int n_prim_pairs,
    double* h_b_out,
    int n_aux,
    int width,
    size_t memory_limit_bytes,
    uint64_t tensor_id
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_aux_shells || !h_aux_prims || !h_shell_pairs || !h_prim_pairs || !h_b_out
        || n_aux_shells <= 0 || n_aux_prims <= 0 || n_shell_pairs <= 0 || n_prim_pairs <= 0
        || n_aux <= 0 || width <= 0) {
        return -1;
    }
    int budget_rc = check_rij_budget(n_aux, width, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

    size_t b_bytes = (size_t)n_aux * (size_t)width * sizeof(double);
    size_t aux_sh_bytes = (size_t)n_aux_shells * sizeof(AuxShellGpu);
    size_t aux_pr_bytes = (size_t)n_aux_prims * sizeof(AuxPrimGpu);
    size_t sp_bytes = (size_t)n_shell_pairs * sizeof(ShellPair3CGpu);
    size_t pp_bytes = (size_t)n_prim_pairs * sizeof(PrimPair3CGpu);
    size_t meta_bytes = aux_sh_bytes + aux_pr_bytes + sp_bytes + pp_bytes;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_eri3c) return -5;

    bool same_shape = g_drv.cached_b_ptr && g_drv.cached_n_p == n_aux && g_drv.cached_n_ao_sq == width;
    size_t alloc_needed = meta_bytes + (same_shape ? 0 : b_bytes);
    if (!same_shape && g_drv.cached_b_ptr) {
        g_drv.cuMemFree(g_drv.cached_b_ptr);
        g_drv.cached_b_ptr = 0;
        g_drv.cached_h_b = nullptr;
        g_drv.cached_n_p = 0;
        g_drv.cached_n_ao_sq = 0;
        g_drv.cached_tensor_id = 0;
    }
    size_t free_mem = 0, total_mem = 0;
    if (g_drv.cuMemGetInfo(&free_mem, &total_mem) != 0
        || free_mem < alloc_needed
        || free_mem - alloc_needed < 64 * 1024 * 1024) {
        return -3;
    }
    if (!same_shape) {
        if (g_drv.cuMemAlloc(&g_drv.cached_b_ptr, b_bytes) != 0) {
            g_drv.cached_b_ptr = 0;
            return -4;
        }
    }

    uint64_t d_aux_sh = 0, d_aux_pr = 0, d_sp = 0, d_pp = 0;
    auto cleanup_meta = [&]() {
        if (d_aux_sh) g_drv.cuMemFree(d_aux_sh);
        if (d_aux_pr) g_drv.cuMemFree(d_aux_pr);
        if (d_sp) g_drv.cuMemFree(d_sp);
        if (d_pp) g_drv.cuMemFree(d_pp);
    };
    if (g_drv.cuMemAlloc(&d_aux_sh, aux_sh_bytes) != 0
        || g_drv.cuMemAlloc(&d_aux_pr, aux_pr_bytes) != 0
        || g_drv.cuMemAlloc(&d_sp, sp_bytes) != 0
        || g_drv.cuMemAlloc(&d_pp, pp_bytes) != 0) {
        cleanup_meta();
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(d_aux_sh, h_aux_shells, aux_sh_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_aux_pr, h_aux_prims, aux_pr_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_sp, h_shell_pairs, sp_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_pp, h_prim_pairs, pp_bytes) != 0) {
        cleanup_meta();
        return -5;
    }

    const unsigned int block_x = 128;
    unsigned int grid_x = ((unsigned int)n_shell_pairs + block_x - 1) / block_x;
    unsigned int grid_y = (unsigned int)n_aux_shells;
    void* params[8] = {
        &d_aux_sh,
        &d_aux_pr,
        &d_sp,
        &d_pp,
        &g_drv.cached_b_ptr,
        &n_aux_shells,
        &n_shell_pairs,
        &width
    };
    int rc = g_drv.cuLaunchKernel(
        g_drv.fn_eri3c, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, params, nullptr);
    if (rc == 0) rc = g_drv.cuCtxSynchronize();
    cleanup_meta();
    if (rc != 0) {
        g_drv.cached_tensor_id = 0;
        return -5;
    }

    if (g_drv.cuMemcpyDtoH(h_b_out, g_drv.cached_b_ptr, b_bytes) != 0) {
        g_drv.cached_tensor_id = 0;
        return -5;
    }
    g_drv.cached_h_b = h_b_out;
    g_drv.cached_n_p = n_aux;
    g_drv.cached_n_ao_sq = width;
    g_drv.cached_tensor_id = tensor_id;
    return 0;
#else
    (void)meta_bytes;
    return -5;
#endif
}

int calchem_cuda_vxc_upload_grid(
    const AoShellGpu* h_shells,
    int n_shells,
    const AoPrimGpu* h_prims,
    int n_prims,
    const AoCompGpu* h_comps,
    int n_comps,
    const GridPtGpu* h_pts,
    int n_pts,
    const int* h_cand_offsets,
    int n_cand_offsets,
    const int* h_cand_shells,
    int n_cand_shells,
    int n_ao,
    uint64_t grid_id
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_shells || !h_prims || !h_comps || !h_pts || !h_cand_offsets || !h_cand_shells
        || n_shells <= 0 || n_prims <= 0 || n_comps <= 0 || n_pts <= 0
        || n_cand_offsets <= 1 || n_cand_shells <= 0 || n_ao <= 0 || grid_id == 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_vxc_rho || !g_drv.fn_vxc_accum || !g_drv.fn_zero_doubles) {
        return -5;
    }
    if (g_drv.vxc_grid_id == grid_id && g_drv.vxc_n_pts == n_pts && g_drv.vxc_n_ao == n_ao) {
        return 0;
    }
    g_drv.release_vxc_resources();

    size_t sh_b = (size_t)n_shells * sizeof(AoShellGpu);
    size_t pr_b = (size_t)n_prims * sizeof(AoPrimGpu);
    size_t cp_b = (size_t)n_comps * sizeof(AoCompGpu);
    size_t pt_b = (size_t)n_pts * sizeof(GridPtGpu);
    size_t co_b = (size_t)n_cand_offsets * sizeof(int);
    size_t cs_b = (size_t)n_cand_shells * sizeof(int);
    size_t mat_b = (size_t)n_ao * (size_t)n_ao * sizeof(double);
    size_t rg_b = (size_t)n_pts * 4 * sizeof(double);
    size_t total_req = sh_b + pr_b + cp_b + pt_b + co_b + cs_b + 2 * mat_b + 2 * rg_b + sizeof(int);

    size_t free_mem = 0, total_mem = 0;
    if (g_drv.cuMemGetInfo(&free_mem, &total_mem) != 0
        || free_mem < total_req
        || free_mem - total_req < 64 * 1024 * 1024) {
        return -3;
    }
    if (g_drv.cuMemAlloc(&g_drv.d_vxc_shells, sh_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_prims, pr_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_comps, cp_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_pts, pt_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_cand_off, co_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_cand_sh, cs_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_p, mat_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_v, mat_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_rg, rg_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_wv, rg_b) != 0
        || g_drv.cuMemAlloc(&g_drv.d_vxc_ovf, sizeof(int)) != 0) {
        g_drv.release_vxc_resources();
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(g_drv.d_vxc_shells, h_shells, sh_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_prims, h_prims, pr_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_comps, h_comps, cp_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_pts, h_pts, pt_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_cand_off, h_cand_offsets, co_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_cand_sh, h_cand_shells, cs_b) != 0) {
        g_drv.release_vxc_resources();
        return -5;
    }
    g_drv.vxc_grid_id = grid_id;
    g_drv.vxc_n_pts = n_pts;
    g_drv.vxc_n_ao = n_ao;
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_vxc_eval_rho(
    uint64_t grid_id,
    const double* h_p_mat,
    double* h_rho_grad_out,
    int n_pts,
    int n_ao
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_p_mat || !h_rho_grad_out || n_pts <= 0 || n_ao <= 0 || grid_id == 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (g_drv.vxc_grid_id != grid_id || g_drv.vxc_n_pts != n_pts || g_drv.vxc_n_ao != n_ao) return -2;
    size_t mat_b = (size_t)n_ao * (size_t)n_ao * sizeof(double);
    size_t rg_b = (size_t)n_pts * 4 * sizeof(double);
    int zero_ovf = 0;
    if (g_drv.cuMemcpyHtoD(g_drv.d_vxc_p, h_p_mat, mat_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_ovf, &zero_ovf, sizeof(int)) != 0) {
        return -5;
    }
    const unsigned int block = 128;
    unsigned int grid = ((unsigned int)n_pts + block - 1) / block;
    void* params[11] = {
        &g_drv.d_vxc_shells, &g_drv.d_vxc_prims, &g_drv.d_vxc_comps,
        &g_drv.d_vxc_pts, &g_drv.d_vxc_cand_off, &g_drv.d_vxc_cand_sh,
        &g_drv.d_vxc_p, &g_drv.d_vxc_rg, &g_drv.d_vxc_ovf,
        &n_pts, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_vxc_rho, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    int ovf = 0;
    if (g_drv.cuMemcpyDtoH(&ovf, g_drv.d_vxc_ovf, sizeof(int)) != 0 || ovf != 0) {
        return -6;
    }
    return (g_drv.cuMemcpyDtoH(h_rho_grad_out, g_drv.d_vxc_rg, rg_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_vxc_accum(
    uint64_t grid_id,
    const double* h_wv_in,
    double* h_vxc_out,
    int n_pts,
    int n_ao
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_wv_in || !h_vxc_out || n_pts <= 0 || n_ao <= 0 || grid_id == 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (g_drv.vxc_grid_id != grid_id || g_drv.vxc_n_pts != n_pts || g_drv.vxc_n_ao != n_ao) return -2;
    size_t mat_len = (size_t)n_ao * (size_t)n_ao;
    size_t mat_b = mat_len * sizeof(double);
    size_t wv_b = (size_t)n_pts * 4 * sizeof(double);
    int zero_ovf = 0;
    if (g_drv.cuMemcpyHtoD(g_drv.d_vxc_wv, h_wv_in, wv_b) != 0
        || g_drv.cuMemcpyHtoD(g_drv.d_vxc_ovf, &zero_ovf, sizeof(int)) != 0) {
        return -5;
    }
    // Zero d_vxc_v on GPU
    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((mat_len + z_block - 1) / z_block);
    void* z_params[2] = { &g_drv.d_vxc_v, &mat_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_params, nullptr) != 0) {
        return -5;
    }
    const unsigned int block = 128;
    unsigned int grid = ((unsigned int)n_pts + block - 1) / block;
    void* params[11] = {
        &g_drv.d_vxc_shells, &g_drv.d_vxc_prims, &g_drv.d_vxc_comps,
        &g_drv.d_vxc_pts, &g_drv.d_vxc_cand_off, &g_drv.d_vxc_cand_sh,
        &g_drv.d_vxc_wv, &g_drv.d_vxc_v, &g_drv.d_vxc_ovf,
        &n_pts, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_vxc_accum, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    int ovf = 0;
    if (g_drv.cuMemcpyDtoH(&ovf, g_drv.d_vxc_ovf, sizeof(int)) != 0 || ovf != 0) {
        return -6;
    }
    return (g_drv.cuMemcpyDtoH(h_vxc_out, g_drv.d_vxc_v, mat_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}
}
