#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    double cp_norm[10];
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
    int atom_index;
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
    int host_atom;
};

struct Grad3cAuxShellGpu {
    double center[3];
    double bound;
    double comp_scales[10];
    int lp;
    int ao_offset;
    int prim_start;
    int n_prims;
    int atom_index;
    int _pad;
};

struct Grad3cAuxPrimGpu {
    double gamma_p;
    double cp_lead;
};

struct Grad3cShellPairGpu {
    double q_ij;
    double ab_vec[3];
    double comp_scales_i[6];
    double comp_scales_j[6];
    int li;
    int lj;
    int same_shell;
    int atom_i;
    int atom_j;
    int ao_off_i;
    int ao_off_j;
    int prim_start;
    int n_prims;
    int _pad;
};

struct Grad3cPrimPairGpu {
    double q;
    double alpha;
    double beta;
    double two_pi25_kab_w_over_q;
    double p_center[3];
    double qc[3];
};

struct Grad4cShellGpu {
    double center[3];
    double comp_scales[10];
    int l;
    int ao_offset;
    int n_funcs;
    int atom_index;
};

struct Grad4cShellPairGpu {
    int shell_i;
    int shell_j;
    double bound;
    int prim_pair_start;
    int n_prim_pairs;
};

struct Grad4cPrimPairGpu {
    double gamma;
    double kab_w;
    double alpha;
    double beta;
    double p[3];
    double pa[3];
};

struct DirectShellGpu {
    double center[3];
    double comp_scales[10];
    int l;
    int ao_offset;
    int n_funcs;
    int _pad;
};

struct DirectShellPairGpu {
    int shell_i;
    int shell_j;
    double bound;
    int prim_pair_start;
    int n_prim_pairs;
    int cached_kets;
    int cache_valid_offset;
    uint64_t cache_data_offset;
};

struct DirectPrimPairGpu {
    double gamma;
    double kab_w;
    double p[3];
    double pa[3];
};

struct DirectStatsGpu {
    uint64_t computed_quartets;
    uint64_t skipped_quartets;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t fp64_quartets;
    uint64_t fp32_quartets;
};

struct Comp6 {
    signed char x, y, z, L;
    signed char axis, n_k, idx1, idx2;
    signed char sub[3];
    signed char add[3];
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

__device__ __forceinline__ void pow_d0_d1_d2_dev(double x, int l, double& p, double& dp, double& d2p) {
    if (l == 0) {
        p = 1.0; dp = 0.0; d2p = 0.0;
    } else if (l == 1) {
        p = x; dp = 1.0; d2p = 0.0;
    } else if (l == 2) {
        p = x * x; dp = 2.0 * x; d2p = 2.0;
    } else if (l == 3) {
        double x2 = x * x;
        p = x2 * x; dp = 3.0 * x2; d2p = 6.0 * x;
    } else {
        double xm2 = pow(x, (double)(l - 2));
        double xm1 = xm2 * x;
        p = xm1 * x;
        dp = (double)l * xm1;
        d2p = (double)(l * (l - 1)) * xm2;
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
    (void)overflow_flag;
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= n_pts) return;

    const GridPtGpu pt = pts[g];
    const int c_start = cand_offsets[pt.batch_id];
    const int c_end = cand_offsets[pt.batch_id + 1];

    int act_mu[VXC_MAX_ACTIVE_AO];
    double act_phi[VXC_MAX_ACTIVE_AO];
    int n_act = 0;
    bool overflow = false;

    // Pass 1: collect active AOs (mu, phi) up to fast-path cap
    for (int ci = c_start; ci < c_end && !overflow; ++ci) {
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
                    overflow = true;
                    break;
                }
                act_mu[n_act] = sh.ao_offset + ic;
                act_phi[n_act] = val;
                n_act++;
            }
        }
    }

    if (n_act == 0 && !overflow) {
        size_t o = (size_t)g * 4;
        rho_grad_out[o] = 0.0;
        rho_grad_out[o + 1] = 0.0;
        rho_grad_out[o + 2] = 0.0;
        rho_grad_out[o + 3] = 0.0;
        return;
    }

    // Pass 2: evaluate gradients and contract with P * phi
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
                if (!overflow) {
                    for (int j = 0; j < n_act; ++j) {
                        s += p_row[act_mu[j]] * act_phi[j];
                    }
                } else {
                    for (int cj = c_start; cj < c_end; ++cj) {
                        const AoShellGpu sh_j = shells[cand_shells[cj]];
                        const double jdx = pt.x - sh_j.center[0];
                        const double jdy = pt.y - sh_j.center[1];
                        const double jdz = pt.z - sh_j.center[2];
                        const double jr2 = jdx * jdx + jdy * jdy + jdz * jdz;
                        if (jr2 > sh_j.r2_cutoff) continue;
                        double jr0 = 0.0, jr1 = 0.0;
                        for (int pi = 0; pi < sh_j.n_prims; ++pi) {
                            const AoPrimGpu pr = prims[sh_j.prim_start + pi];
                            const double e = pr.weighted_norm * exp(-pr.exponent * jr2);
                            jr0 += e;
                            jr1 -= 2.0 * pr.exponent * e;
                        }
                        if (fabs(jr0) < 1e-18 && fabs(jr1) < 1e-18) continue;
                        for (int jc = 0; jc < sh_j.n_funcs; ++jc) {
                            const AoCompGpu jcp = comps[sh_j.comp_start + jc];
                            double jpx, jdpx, jpy, jdpy, jpz, jdpz;
                            pow_d0_d1_dev(jdx, jcp.lx, jpx, jdpx);
                            pow_d0_d1_dev(jdy, jcp.ly, jpy, jdpy);
                            pow_d0_d1_dev(jdz, jcp.lz, jpz, jdpz);
                            const double jm = jpx * jpy * jpz;
                            const double jval = jcp.scale * jm * jr0;
                            const double jgx = jcp.scale * (jdpx * jpy * jpz * jr0 + jm * jdx * jr1);
                            const double jgy = jcp.scale * (jpx * jdpy * jpz * jr0 + jm * jdy * jr1);
                            const double jgz = jcp.scale * (jpx * jpy * jdpz * jr0 + jm * jdz * jr1);
                            if (fabs(jval) > 1e-16 || fabs(jgx) > 1e-16 || fabs(jgy) > 1e-16 || fabs(jgz) > 1e-16) {
                                s += p_row[sh_j.ao_offset + jc] * jval;
                            }
                        }
                    }
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
    (void)overflow_flag;
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
    bool overflow = false;

    for (int ci = c_start; ci < c_end && !overflow; ++ci) {
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
                    overflow = true;
                    break;
                }
                act_mu[n_act] = sh.ao_offset + ic;
                act_phi[n_act] = val;
                act_z[n_act] = w_vrho * val + wu_x * gx + wu_y * gy + wu_z * gz;
                n_act++;
            }
        }
    }

    if (!overflow) {
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
    } else {
        for (int ci_i = c_start; ci_i < c_end; ++ci_i) {
            const AoShellGpu sh_i = shells[cand_shells[ci_i]];
            const double idx = pt.x - sh_i.center[0];
            const double idy = pt.y - sh_i.center[1];
            const double idz = pt.z - sh_i.center[2];
            const double ir2 = idx * idx + idy * idy + idz * idz;
            if (ir2 > sh_i.r2_cutoff) continue;
            double ir0 = 0.0, ir1 = 0.0;
            for (int pi = 0; pi < sh_i.n_prims; ++pi) {
                const AoPrimGpu pr = prims[sh_i.prim_start + pi];
                const double e = pr.weighted_norm * exp(-pr.exponent * ir2);
                ir0 += e;
                ir1 -= 2.0 * pr.exponent * e;
            }
            if (fabs(ir0) < 1e-18 && fabs(ir1) < 1e-18) continue;

            for (int ic_i = 0; ic_i < sh_i.n_funcs; ++ic_i) {
                const AoCompGpu cp_i = comps[sh_i.comp_start + ic_i];
                double ipx, idpx, ipy, idpy, ipz, idpz;
                pow_d0_d1_dev(idx, cp_i.lx, ipx, idpx);
                pow_d0_d1_dev(idy, cp_i.ly, ipy, idpy);
                pow_d0_d1_dev(idz, cp_i.lz, ipz, idpz);
                const double im = ipx * ipy * ipz;
                const double phi_i = cp_i.scale * im * ir0;
                const double igx = cp_i.scale * (idpx * ipy * ipz * ir0 + im * idx * ir1);
                const double igy = cp_i.scale * (ipx * idpy * ipz * ir0 + im * idy * ir1);
                const double igz = cp_i.scale * (ipx * ipy * idpz * ir0 + im * idz * ir1);
                if (fabs(phi_i) <= 1e-16 && fabs(igx) <= 1e-16 && fabs(igy) <= 1e-16 && fabs(igz) <= 1e-16) {
                    continue;
                }
                const int mu_i = sh_i.ao_offset + ic_i;
                const double z_i = w_vrho * phi_i + wu_x * igx + wu_y * igy + wu_z * igz;

                for (int ci_j = c_start; ci_j <= ci_i; ++ci_j) {
                    const AoShellGpu sh_j = shells[cand_shells[ci_j]];
                    const double jdx = pt.x - sh_j.center[0];
                    const double jdy = pt.y - sh_j.center[1];
                    const double jdz = pt.z - sh_j.center[2];
                    const double jr2 = jdx * jdx + jdy * jdy + jdz * jdz;
                    if (jr2 > sh_j.r2_cutoff) continue;
                    double jr0 = 0.0, jr1 = 0.0;
                    for (int pi = 0; pi < sh_j.n_prims; ++pi) {
                        const AoPrimGpu pr = prims[sh_j.prim_start + pi];
                        const double e = pr.weighted_norm * exp(-pr.exponent * jr2);
                        jr0 += e;
                        jr1 -= 2.0 * pr.exponent * e;
                    }
                    if (fabs(jr0) < 1e-18 && fabs(jr1) < 1e-18) continue;

                    const int max_jc = (ci_j == ci_i) ? ic_i : (sh_j.n_funcs - 1);
                    for (int ic_j = 0; ic_j <= max_jc; ++ic_j) {
                        const AoCompGpu cp_j = comps[sh_j.comp_start + ic_j];
                        double jpx, jdpx, jpy, jdpy, jpz, jdpz;
                        pow_d0_d1_dev(jdx, cp_j.lx, jpx, jdpx);
                        pow_d0_d1_dev(jdy, cp_j.ly, jpy, jdpy);
                        pow_d0_d1_dev(jdz, cp_j.lz, jpz, jdpz);
                        const double jm = jpx * jpy * jpz;
                        const double phi_j = cp_j.scale * jm * jr0;
                        const double jgx = cp_j.scale * (jdpx * jpy * jpz * jr0 + jm * jdx * jr1);
                        const double jgy = cp_j.scale * (jpx * jdpy * jpz * jr0 + jm * jdy * jr1);
                        const double jgz = cp_j.scale * (jpx * jpy * jdpz * jr0 + jm * jdz * jr1);
                        if (fabs(phi_j) <= 1e-16 && fabs(jgx) <= 1e-16 && fabs(jgy) <= 1e-16 && fabs(jgz) <= 1e-16) {
                            continue;
                        }
                        const int mu_j = sh_j.ao_offset + ic_j;
                        const double z_j = w_vrho * phi_j + wu_x * jgx + wu_y * jgy + wu_z * jgz;
                        const double contrib = z_i * phi_j + phi_i * z_j;
                        const int r = (mu_i >= mu_j) ? mu_i : mu_j;
                        const int c = (mu_i >= mu_j) ? mu_j : mu_i;
                        atomicAdd(&v_xc_out[(size_t)r * (size_t)n_ao + (size_t)c], contrib);
                    }
                }
            }
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

__device__ __forceinline__ void boys_batch_table_f64(
    int max_m,
    double rho,
    double pq2,
    double omega,
    int has_omega,
    const double* __restrict__ boys_table_f64,
    double* out
) {
    double t;
    double theta = 1.0;
    if (has_omega) {
        theta = omega / hypot(omega, sqrt(rho));
        t = rho * theta * theta * pq2;
    } else {
        t = rho * pq2;
    }

    if (boys_table_f64 != nullptr && max_m <= 16 && t >= 1e-14 && t <= 50.0) {
        int i = (int)llrint(8.0 * t);
        double h = (double)i * 0.125 - t;
        const double* row = boys_table_f64 + i * 26;
        double weights[10];
        weights[0] = 1.0;
        for (int k = 1; k < 10; ++k) {
            weights[k] = weights[k - 1] * h / (double)k;
        }
        for (int m = 0; m <= max_m; ++m) {
            double v = 0.0;
            for (int k = 9; k >= 0; --k) {
                v += weights[k] * __ldg(&row[m + k]);
            }
            out[m] = v;
        }
    } else {
        boys_batch_dev(max_m, t, out);
    }

    if (has_omega) {
        double curr = theta;
        double theta2 = theta * theta;
        for (int m = 0; m <= max_m; ++m) {
            out[m] *= curr;
            curr *= theta2;
        }
    }
}

__device__ __forceinline__ void boys_batch_table_f32(
    int max_m,
    float rho,
    float pq2,
    float omega,
    int has_omega,
    const float* __restrict__ boys_table_f32,
    float* out
) {
    float t;
    float theta = 1.0f;
    if (has_omega) {
        theta = omega / hypotf(omega, sqrtf(rho));
        t = rho * theta * theta * pq2;
    } else {
        t = rho * pq2;
    }

    if (boys_table_f32 != nullptr && max_m <= 16 && t >= 1e-7f && t <= 50.0f) {
        int i = __float2int_rn(8.0f * t);
        float h = (float)i * 0.125f - t;
        const float* row = boys_table_f32 + i * 26;
        float weights[7];
        weights[0] = 1.0f;
        for (int k = 1; k < 7; ++k) {
            weights[k] = weights[k - 1] * h / (float)k;
        }
        for (int m = 0; m <= max_m; ++m) {
            float v = 0.0f;
            for (int k = 6; k >= 0; --k) {
                v += weights[k] * __ldg(&row[m + k]);
            }
            out[m] = v;
        }
    } else if (t < 1e-6f) {
        for (int m = 0; m <= max_m; ++m) {
            float dm = (float)(2 * m);
            out[m] = 1.0f / (dm + 1.0f) - t / (dm + 3.0f);
        }
    } else if (t >= 35.0f) {
        const float inv_2t = 0.5f / t;
        out[0] = 0.5f * sqrtf(3.14159265358979323846f / t);
        for (int m = 1; m <= max_m; ++m) {
            out[m] = (float)(2 * m - 1) * out[m - 1] * inv_2t;
        }
    } else {
        const float inv_2t = 0.5f / t;
        const float exp_mt = expf(-t);
        out[0] = 0.5f * sqrtf(3.14159265358979323846f / t) * erff(sqrtf(t));
        for (int m = 1; m <= max_m; ++m) {
            out[m] = ((float)(2 * m - 1) * out[m - 1] - exp_mt) * inv_2t;
        }
    }

    if (has_omega) {
        float curr = theta;
        float theta2 = theta * theta;
        for (int m = 0; m <= max_m; ++m) {
            out[m] *= curr;
            curr *= theta2;
        }
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
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
    {-1, -1, -1},
};

extern "C" __global__ void eri3c_build_packed_kernel(
    const AuxShellGpu* __restrict__ aux_shells,
    const AuxPrimGpu* __restrict__ aux_prims,
    const ShellPair3CGpu* __restrict__ shell_pairs,
    const PrimPair3CGpu* __restrict__ prim_pairs,
    double* __restrict__ b_out,
    int n_aux_shells,
    int n_shell_pairs,
    int width,
    int aux_ao_base,
    int aux_ao_limit
) {
    int sp_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int p_idx = (int)blockIdx.y;
    if (sp_idx >= n_shell_pairs || p_idx >= n_aux_shells) return;

    const AuxShellGpu aux = aux_shells[p_idx];
    const ShellPair3CGpu sp = shell_pairs[sp_idx];
    const int lp = aux.lp;
    const int li = sp.li;
    const int lj = sp.lj;
    if (lp < 0 || lp > 3 || li < 0 || li > 2 || lj < 0 || lj > 2) return;
    const int np = (lp == 0) ? 1 : ((lp == 1) ? 3 : ((lp == 2) ? 6 : 10));
    const int ni = (li == 0) ? 1 : ((li == 1) ? 3 : 6);
    const int nj = (lj == 0) ? 1 : ((lj == 1) ? 3 : 6);
    const int n_cols = sp.same_shell ? (ni * (ni + 1) / 2) : (ni * nj);
    const int ao_rel = aux.ao_offset - aux_ao_base;
    if (aux.ao_offset < aux_ao_base || ao_rel < 0 || ao_rel > aux_ao_limit - np) return;
    if (sp.col_offset < 0 || sp.col_offset > width - n_cols) return;

    if (aux.bound * sp.q_ij < 1e-12) {
        for (int pf = 0; pf < np; ++pf) {
            size_t base = (size_t)(ao_rel + pf) * (size_t)width + (size_t)sp.col_offset;
            for (int c = 0; c < n_cols; ++c) {
                b_out[base + (size_t)c] = 0.0;
            }
        }
        return;
    }

    double acc[360]; // max 10 * 6 * 6 = 360
    const int blk_len = np * ni * nj;
    for (int k = 0; k < blk_len; ++k) acc[k] = 0.0;

    const int max_m = lp + li + lj;
    const int l_ket = li + lj;
    const int n_ket = (l_ket + 1) * (l_ket + 2) * (l_ket + 3) / 6;
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

            double boys[8];
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

            // Step 1: V0[35][8]
            double V0[35][8];
            for (int m = 0; m <= max_m; ++m) {
                V0[0][m] = pref * boys[m];
            }

            for (int idx = 1; idx < n_ket; ++idx) {
                const Comp3C& c = COMPS_3C[idx];
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
            double V_P[10][35];
            if (lp == 0) {
                for (int idx = 0; idx < n_ket; ++idx) {
                    V_P[0][idx] = V0[idx][0];
                }
            } else if (lp == 1) {
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V0[idx][1];
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V0[sub_idx][1];
                        }
                        V_P[u][idx] = val;
                    }
                }
            } else if (lp == 2) {
                double V1[3][35][2];
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
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
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
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
            } else { // lp == 3
                double V1[3][35][3];
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        for (int m_sub = 0; m_sub < 3; ++m_sub) {
                            double val = wp[u] * V0[idx][m_sub + 1];
                            if (sub_idx >= 0) {
                                val += (double)cu * inv_2zeta * V0[sub_idx][m_sub + 1];
                            }
                            V1[u][idx][m_sub] = val;
                        }
                    }
                }
                double V_D[6][35]; // m = 1 only
                for (int kl = 0; kl < 6; ++kl) {
                    int u = D_K[kl];
                    int v = D_L[kl];
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V1[v][idx][2];
                        if (u == v) {
                            val += inv_2p * (V0[idx][1] - rho_over_p * V0[idx][2]);
                        }
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[v][sub_idx][2];
                        }
                        V_D[kl][idx] = val;
                    }
                }
                for (int pf = 0; pf < 10; ++pf) {
                    const Comp3C& cp_f = COMPS_3C[10 + pf];
                    int u = cp_f.axis;
                    int d_idx = cp_f.idx1 - 4;
                    int p_idx_v1 = cp_f.idx2 - 1;
                    int n_k = cp_f.n_k;
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V_D[d_idx][idx];
                        if (n_k > 0) {
                            val += (double)n_k * inv_2p * (V1[p_idx_v1][idx][0] - rho_over_p * V1[p_idx_v1][idx][1]);
                        }
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V_D[d_idx][sub_idx];
                        }
                        V_P[pf][idx] = val;
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
                            int k = b;
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
        size_t base = (size_t)(ao_rel + pf) * (size_t)width + (size_t)sp.col_offset;
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

// ============================================================================
// Direct-JK / Direct-K CUDA Kernels (Worklist, Cache, SSSS/SP/D_WARP/F_BLOCK)
// ============================================================================

struct WorkItem {
    uint32_t pi;
    uint32_t qi;
};

// Bucket indices:
// 0: cached hit
// 1: SSSS FP64, 2: SSSS FP32
// 3: SP_LOW FP64, 4: SP_LOW FP32
// 5: D_WARP FP64, 6: D_WARP FP32
// 7: F_BLOCK FP64
#define WL_HIT 0
#define WL_SSSS_64 1
#define WL_SSSS_32 2
#define WL_SP_64 3
#define WL_SP_32 4
#define WL_D_64 5
#define WL_D_32 6
#define WL_F_64 7
#define NUM_WL_BUCKETS 8

extern "C" __global__ void direct_filter_worklist_kernel(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const int* __restrict__ ends,
    const double* __restrict__ d_t,
    const double* __restrict__ d_a,
    const double* __restrict__ d_b,
    uint8_t* __restrict__ cache_valid,
    WorkItem* __restrict__ worklists,
    uint32_t wl_stride,
    int* __restrict__ wl_counts,
    unsigned long long* __restrict__ stats_out, // [computed, skipped, hits, misses, fp64, fp32]
    int pi_start,
    int n_active,
    int n_shells,
    int mode, // 0=RKS_JK, 1=K_ONLY, 2=K_PAIR, 3=UKS_JK, 4=TD_BATCH
    double exx,
    int has_omega,
    double thresh,
    double fp32_thresh,
    int qi_start,
    int qi_end_override,
    double k_weight
) {
    int rel_pi = (int)blockIdx.x;
    if (rel_pi >= n_active) return;
    int pi = pi_start + rel_pi;
    const DirectShellPairGpu bra = pairs[pi];
    const int end_raw = ends[pi];
    const int end = (qi_end_override >= 0 && qi_end_override < end_raw) ? qi_end_override : end_raw;
    const int i = bra.shell_i;
    const int j = bra.shell_j;
    const int la = shells[i].l;
    const int lb = shells[j].l;
    const double b_ij = bra.bound;
    const double exx_scale = fabs(exx);
    const double jw_scale = fabs(exx);
    const double kw_scale = fabs(k_weight);
    const int do_j_filt = (mode == 4 && jw_scale > 0.0) ? 1 : 0;
    const int do_k_filt = (mode == 4 && kw_scale > 0.0) ? 1 : 0;
    const int has_exch = (mode == 1 || mode == 2) ? 1 : ((mode == 4) ? do_k_filt : (exx_scale > 1e-15 ? 1 : 0));
    const double dt_ij = (mode == 0 || mode == 3 || do_j_filt) ? d_t[i * n_shells + j] : 0.0;

    unsigned long long loc_comp = 0, loc_skip = 0, loc_hit = 0, loc_miss = 0, loc_64 = 0, loc_32 = 0;

    for (int qi = qi_start + (int)threadIdx.x; qi < end; qi += (int)blockDim.x) {
        const DirectShellPairGpu ket = pairs[qi];
        const int k = ket.shell_i;
        const int l = ket.shell_j;
        const double b_ijkl = b_ij * ket.bound;

        double bound = 0.0;
        if (mode == 0) {
            double d_kl = d_t[k * n_shells + l];
            double bound_j = 4.0 * b_ijkl * fmax(dt_ij, d_kl);
            if (has_exch) {
                double d_ik = d_t[i * n_shells + k];
                double d_il = d_t[i * n_shells + l];
                double d_jk = d_t[j * n_shells + k];
                double d_jl = d_t[j * n_shells + l];
                double d_exch = fmax(fmax(d_ik, d_il), fmax(d_jk, d_jl));
                bound = bound_j + 4.0 * exx_scale * b_ijkl * d_exch;
            } else {
                bound = bound_j;
            }
        } else if (mode == 1) {
            double d_ik = d_a[i * n_shells + k];
            double d_il = d_a[i * n_shells + l];
            double d_jk = d_a[j * n_shells + k];
            double d_jl = d_a[j * n_shells + l];
            double d_exch = fmax(fmax(d_ik, d_il), fmax(d_jk, d_jl));
            bound = 4.0 * b_ijkl * d_exch;
        } else if (mode == 2) {
            double da_ex = fmax(fmax(d_a[i * n_shells + k], d_a[i * n_shells + l]),
                                fmax(d_a[j * n_shells + k], d_a[j * n_shells + l]));
            double db_ex = fmax(fmax(d_b[i * n_shells + k], d_b[i * n_shells + l]),
                                fmax(d_b[j * n_shells + k], d_b[j * n_shells + l]));
            bound = 4.0 * b_ijkl * fmax(da_ex, db_ex);
        } else if (mode == 3) { // mode == 3 (UKS_JK)
            double dt_kl = d_t[k * n_shells + l];
            double bound_j = 4.0 * b_ijkl * fmax(dt_ij, dt_kl);
            double da_ex = fmax(fmax(d_a[i * n_shells + k], d_a[i * n_shells + l]),
                                fmax(d_a[j * n_shells + k], d_a[j * n_shells + l]));
            double db_ex = fmax(fmax(d_b[i * n_shells + k], d_b[i * n_shells + l]),
                                fmax(d_b[j * n_shells + k], d_b[j * n_shells + l]));
            bound = bound_j + 4.0 * exx_scale * b_ijkl * fmax(da_ex, db_ex);
        } else { // mode == 4 (TD_BATCH)
            double bound_j = 0.0;
            if (do_j_filt) {
                double d_kl = d_t[k * n_shells + l];
                bound_j = 4.0 * jw_scale * b_ijkl * fmax(dt_ij, d_kl);
            }
            double bound_k = 0.0;
            if (do_k_filt) {
                double da_ex = fmax(fmax(d_a[i * n_shells + k], d_a[i * n_shells + l]),
                                    fmax(d_a[j * n_shells + k], d_a[j * n_shells + l]));
                bound_k = 4.0 * kw_scale * b_ijkl * da_ex;
            }
            bound = bound_j + bound_k;
        }

        if (thresh > 0.0 && bound < thresh) {
            loc_skip++;
            continue;
        }

        loc_comp++;
        const int cache_eligible = (has_omega == 0 && qi < bra.cached_kets) ? 1 : 0;
        if (cache_eligible) {
            const uint8_t valid = cache_valid[(size_t)bra.cache_valid_offset + (size_t)qi];
            if (valid != 0) {
                loc_hit++;
                loc_64++;
                int pos = atomicAdd(&wl_counts[WL_HIT], 1);
                worklists[(size_t)WL_HIT * (size_t)wl_stride + (size_t)pos] = {(uint32_t)pi, (uint32_t)qi};
                continue;
            }
            loc_miss++;
        }

        const int lc = shells[k].l;
        const int ld = shells[l].l;
        const int max_l = max(max(la, lb), max(lc, ld));
        const int force_fp64 = (cache_eligible || max_l >= 3 || fp32_thresh <= 0.0 || bound >= fp32_thresh) ? 1 : 0;
        if (force_fp64) loc_64++; else loc_32++;

        int bucket;
        if (max_l == 0) {
            bucket = force_fp64 ? WL_SSSS_64 : WL_SSSS_32;
        } else if (max_l <= 1 && ((la + lb) <= 1 || (lc + ld) <= 1)) {
            bucket = force_fp64 ? WL_SP_64 : WL_SP_32;
        } else if (max_l <= 2) {
            bucket = force_fp64 ? WL_D_64 : WL_D_32;
        } else {
            bucket = WL_F_64;
        }
        int pos = atomicAdd(&wl_counts[bucket], 1);
        worklists[(size_t)bucket * (size_t)wl_stride + (size_t)pos] = {(uint32_t)pi, (uint32_t)qi};
    }

    if (loc_comp) atomicAdd(&stats_out[0], loc_comp);
    if (loc_skip) atomicAdd(&stats_out[1], loc_skip);
    if (loc_hit)  atomicAdd(&stats_out[2], loc_hit);
    if (loc_miss) atomicAdd(&stats_out[3], loc_miss);
    if (loc_64)   atomicAdd(&stats_out[4], loc_64);
    if (loc_32)   atomicAdd(&stats_out[5], loc_32);
}

__device__ __forceinline__ void contract_quartet_element_canonical(
    double val,
    int mu0, int nu0, int lam0, int sig0,
    int n,
    int mode,
    double exx,
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    if (val == 0.0) return;
    int mu = (mu0 <= nu0) ? mu0 : nu0;
    int nu = (mu0 <= nu0) ? nu0 : mu0;
    int lam = (lam0 <= sig0) ? lam0 : sig0;
    int sig = (lam0 <= sig0) ? sig0 : lam0;
    double s1 = (mu == nu) ? 2.0 : 1.0;
    double s2 = (lam == sig) ? 2.0 : 1.0;
    double s3 = (mu == lam && nu == sig) ? 2.0 : 1.0;
    double v = val / (s1 * s2 * s3);

    if (mode == 0) {
        double p_ls = 2.0 * v * p_a[lam * n + sig];
        double p_mn = 2.0 * v * p_a[mu * n + nu];
        atomicAdd(&out_a[mu * n + nu], p_ls);
        atomicAdd(&out_a[lam * n + sig], p_mn);
        if (fabs(exx) > 1e-15) {
            double vc = 0.5 * exx * v;
            atomicAdd(&out_a[mu * n + lam], -vc * p_a[nu * n + sig]);
            atomicAdd(&out_a[nu * n + lam], -vc * p_a[mu * n + sig]);
            atomicAdd(&out_a[mu * n + sig], -vc * p_a[nu * n + lam]);
            atomicAdd(&out_a[nu * n + sig], -vc * p_a[mu * n + lam]);
        }
    } else if (mode == 1) {
        atomicAdd(&out_a[mu * n + lam], v * p_a[nu * n + sig]);
        atomicAdd(&out_a[nu * n + lam], v * p_a[mu * n + sig]);
        atomicAdd(&out_a[mu * n + sig], v * p_a[nu * n + lam]);
        atomicAdd(&out_a[nu * n + sig], v * p_a[mu * n + lam]);
    } else if (mode == 2) {
        atomicAdd(&out_a[mu * n + lam], v * p_a[nu * n + sig]);
        atomicAdd(&out_a[nu * n + lam], v * p_a[mu * n + sig]);
        atomicAdd(&out_a[mu * n + sig], v * p_a[nu * n + lam]);
        atomicAdd(&out_a[nu * n + sig], v * p_a[mu * n + lam]);

        atomicAdd(&out_b[mu * n + lam], v * p_b[nu * n + sig]);
        atomicAdd(&out_b[nu * n + lam], v * p_b[mu * n + sig]);
        atomicAdd(&out_b[mu * n + sig], v * p_b[nu * n + lam]);
        atomicAdd(&out_b[nu * n + sig], v * p_b[mu * n + lam]);
    } else { // mode == 3 (UKS_JK)
        double pt_ls = p_a[lam * n + sig] + p_b[lam * n + sig];
        double pt_mn = p_a[mu * n + nu] + p_b[mu * n + nu];
        double j_ls = 2.0 * v * pt_ls;
        double j_mn = 2.0 * v * pt_mn;
        atomicAdd(&out_a[mu * n + nu], j_ls);
        atomicAdd(&out_a[lam * n + sig], j_mn);
        atomicAdd(&out_b[mu * n + nu], j_ls);
        atomicAdd(&out_b[lam * n + sig], j_mn);
        if (fabs(exx) > 1e-15) {
            double vc = exx * v;
            atomicAdd(&out_a[mu * n + lam], -vc * p_a[nu * n + sig]);
            atomicAdd(&out_a[nu * n + lam], -vc * p_a[mu * n + sig]);
            atomicAdd(&out_a[mu * n + sig], -vc * p_a[nu * n + lam]);
            atomicAdd(&out_a[nu * n + sig], -vc * p_a[mu * n + lam]);

            atomicAdd(&out_b[mu * n + lam], -vc * p_b[nu * n + sig]);
            atomicAdd(&out_b[nu * n + lam], -vc * p_b[mu * n + sig]);
            atomicAdd(&out_b[mu * n + sig], -vc * p_b[nu * n + lam]);
            atomicAdd(&out_b[nu * n + sig], -vc * p_b[mu * n + lam]);
        }
    }
}

// Warp-parallel cached quartet contraction (1 warp = 32 threads per quartet)
extern "C" __global__ void direct_contract_cached_kernel(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const uint64_t* __restrict__ pair_prefix,
    const double* __restrict__ eri_cache,
    const WorkItem* __restrict__ wl_hit,
    int n_hit,
    int n,
    int mode,
    double exx,
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (warp_id >= n_hit) return;

    const WorkItem item = wl_hit[warp_id];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs;
    const int nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int bra_w = na * nb;
    const int ket_w = nc * nd;
    const double* blk = eri_cache + bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const int has_exch = (fabs(exx) > 1e-15) ? 1 : 0;
    const double half_c = 0.5 * exx;

    if (distinct) {
        for (int ab = lane; ab < bra_w; ab += 32) {
            int ia = ab / nb;
            int ib = ab - ia * nb;
            int mu = sh_a.ao_offset + ia;
            int nu = sh_b.ao_offset + ib;
            int mu_n = mu * n, nu_n = nu * n;
            const double* row = blk + ab * ket_w;
            if (mode == 0) {
                double p_mn2 = 2.0 * p_a[mu_n + nu];
                double acc_mn = 0.0;
                for (int ic = 0; ic < nc; ++ic) {
                    int lam = sh_c.ao_offset + ic;
                    int lam_n = lam * n;
                    if (!has_exch) {
                        for (int id = 0; id < nd; ++id) {
                            double v = row[ic * nd + id];
                            if (v != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                acc_mn += v * p_a[lam_n + sig];
                                atomicAdd(&out_a[lam_n + sig], v * p_mn2);
                            }
                        }
                    } else {
                        double c_nl = half_c * p_a[nu_n + lam];
                        double c_ml = half_c * p_a[mu_n + lam];
                        double acc_ml = 0.0, acc_nl = 0.0;
                        for (int id = 0; id < nd; ++id) {
                            double v = row[ic * nd + id];
                            if (v != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                acc_mn += v * p_a[lam_n + sig];
                                atomicAdd(&out_a[lam_n + sig], v * p_mn2);
                                acc_ml += v * p_a[nu_n + sig];
                                acc_nl += v * p_a[mu_n + sig];
                                atomicAdd(&out_a[mu_n + sig], -v * c_nl);
                                atomicAdd(&out_a[nu_n + sig], -v * c_ml);
                            }
                        }
                        if (acc_ml != 0.0) atomicAdd(&out_a[mu_n + lam], -half_c * acc_ml);
                        if (acc_nl != 0.0) atomicAdd(&out_a[nu_n + lam], -half_c * acc_nl);
                    }
                }
                if (acc_mn != 0.0) atomicAdd(&out_a[mu_n + nu], 2.0 * acc_mn);
            } else if (mode == 1) {
                for (int ic = 0; ic < nc; ++ic) {
                    int lam = sh_c.ao_offset + ic;
                    double p_nl = p_a[nu_n + lam];
                    double p_ml = p_a[mu_n + lam];
                    double acc_ml = 0.0, acc_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double v = row[ic * nd + id];
                        if (v != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_ml += v * p_a[nu_n + sig];
                            acc_nl += v * p_a[mu_n + sig];
                            atomicAdd(&out_a[mu_n + sig], v * p_nl);
                            atomicAdd(&out_a[nu_n + sig], v * p_ml);
                        }
                    }
                    if (acc_ml != 0.0) atomicAdd(&out_a[mu_n + lam], acc_ml);
                    if (acc_nl != 0.0) atomicAdd(&out_a[nu_n + lam], acc_nl);
                }
            } else if (mode == 2) {
                for (int ic = 0; ic < nc; ++ic) {
                    int lam = sh_c.ao_offset + ic;
                    double pa_nl = p_a[nu_n + lam], pa_ml = p_a[mu_n + lam];
                    double pb_nl = p_b[nu_n + lam], pb_ml = p_b[mu_n + lam];
                    double acc_a_ml = 0.0, acc_a_nl = 0.0;
                    double acc_b_ml = 0.0, acc_b_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double v = row[ic * nd + id];
                        if (v != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_a_ml += v * p_a[nu_n + sig];
                            acc_a_nl += v * p_a[mu_n + sig];
                            atomicAdd(&out_a[mu_n + sig], v * pa_nl);
                            atomicAdd(&out_a[nu_n + sig], v * pa_ml);

                            acc_b_ml += v * p_b[nu_n + sig];
                            acc_b_nl += v * p_b[mu_n + sig];
                            atomicAdd(&out_b[mu_n + sig], v * pb_nl);
                            atomicAdd(&out_b[nu_n + sig], v * pb_ml);
                        }
                    }
                    if (acc_a_ml != 0.0) atomicAdd(&out_a[mu_n + lam], acc_a_ml);
                    if (acc_a_nl != 0.0) atomicAdd(&out_a[nu_n + lam], acc_a_nl);
                    if (acc_b_ml != 0.0) atomicAdd(&out_b[mu_n + lam], acc_b_ml);
                    if (acc_b_nl != 0.0) atomicAdd(&out_b[nu_n + lam], acc_b_nl);
                }
            } else { // mode == 3 (UKS_JK)
                double pt_mn2 = 2.0 * (p_a[mu_n + nu] + p_b[mu_n + nu]);
                double acc_mn = 0.0;
                for (int ic = 0; ic < nc; ++ic) {
                    int lam = sh_c.ao_offset + ic;
                    int lam_n = lam * n;
                    double ca_nl = exx * p_a[nu_n + lam], ca_ml = exx * p_a[mu_n + lam];
                    double cb_nl = exx * p_b[nu_n + lam], cb_ml = exx * p_b[mu_n + lam];
                    double acc_a_ml = 0.0, acc_a_nl = 0.0;
                    double acc_b_ml = 0.0, acc_b_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double v = row[ic * nd + id];
                        if (v != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_mn += v * (p_a[lam_n + sig] + p_b[lam_n + sig]);
                            double j_ls = v * pt_mn2;
                            atomicAdd(&out_a[lam_n + sig], j_ls);
                            atomicAdd(&out_b[lam_n + sig], j_ls);
                            if (has_exch) {
                                acc_a_ml += v * p_a[nu_n + sig];
                                acc_a_nl += v * p_a[mu_n + sig];
                                atomicAdd(&out_a[mu_n + sig], -v * ca_nl);
                                atomicAdd(&out_a[nu_n + sig], -v * ca_ml);

                                acc_b_ml += v * p_b[nu_n + sig];
                                acc_b_nl += v * p_b[mu_n + sig];
                                atomicAdd(&out_b[mu_n + sig], -v * cb_nl);
                                atomicAdd(&out_b[nu_n + sig], -v * cb_ml);
                            }
                        }
                    }
                    if (has_exch) {
                        if (acc_a_ml != 0.0) atomicAdd(&out_a[mu_n + lam], -exx * acc_a_ml);
                        if (acc_a_nl != 0.0) atomicAdd(&out_a[nu_n + lam], -exx * acc_a_nl);
                        if (acc_b_ml != 0.0) atomicAdd(&out_b[mu_n + lam], -exx * acc_b_ml);
                        if (acc_b_nl != 0.0) atomicAdd(&out_b[nu_n + lam], -exx * acc_b_nl);
                    }
                }
                if (acc_mn != 0.0) {
                    double j_mn = 2.0 * acc_mn;
                    atomicAdd(&out_a[mu_n + nu], j_mn);
                    atomicAdd(&out_b[mu_n + nu], j_mn);
                }
            }
        }
    } else {
        for (int ab = lane; ab < bra_w; ab += 32) {
            int ia = ab / nb;
            int ib = ab - ia * nb;
            if (i == j && ib > ia) continue;
            int mu0 = sh_a.ao_offset + ia;
            int nu0 = sh_b.ao_offset + ib;
            const double* row = blk + ab * ket_w;
            for (int ic = 0; ic < nc; ++ic) {
                int id_end = (k == l) ? (ic + 1) : nd;
                int lam0 = sh_c.ao_offset + ic;
                for (int id = 0; id < id_end; ++id) {
                    if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                    int sig0 = sh_d.ao_offset + id;
                    double val = row[ic * nd + id];
                    contract_quartet_element_canonical(val, mu0, nu0, lam0, sig0, n, mode, exx, p_a, p_b, out_a, out_b);
                }
            }
        }
    }
}

// Kernel: SSSS (la=lb=lc=ld=0), 1 thread per quartet
template <typename T>
__device__ __forceinline__ void eval_ssss_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int mode,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qid >= count) return;
    const WorkItem item = wl[qid];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];

    const T two_pi25 = (T)34.986836655249725693;
    const T s_norm = (T)(sh_a.comp_scales[0] * sh_b.comp_scales[0] * sh_c.comp_scales[0] * sh_d.comp_scales[0]);
    T sum = (T)0.0;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T zeta = p + q;
            const T rho = p * q / zeta;
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;
            T b0[1];
            if constexpr (sizeof(T) == 8) {
                boys_batch_table_f64(0, rho, pq2, omega, has_omega, d_boys_f64, b0);
            } else {
                boys_batch_table_f32(0, rho, pq2, (float)omega, has_omega, d_boys_f32, b0);
            }
            const T pref = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;
            sum += pref * b0[0];
        }
    }
    const double val = (double)(s_norm * sum);
    if (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) {
        eri_cache[bra.cache_data_offset + pair_prefix[item.qi]] = val;
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }

    const int mu0 = sh_a.ao_offset, nu0 = sh_b.ao_offset;
    const int lam0 = sh_c.ao_offset, sig0 = sh_d.ao_offset;
    if (i != j && k != l && (i != k || j != l)) {
        // Distinct half-write
        if (mode == 0) {
            atomicAdd(&out_a[mu0 * n + nu0], 2.0 * val * p_a[lam0 * n + sig0]);
            atomicAdd(&out_a[lam0 * n + sig0], 2.0 * val * p_a[mu0 * n + nu0]);
            if (fabs(exx) > 1e-15) {
                double vc = 0.5 * exx * val;
                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
            }
        } else if (mode == 1) {
            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
        } else if (mode == 2) {
            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
            atomicAdd(&out_b[mu0 * n + lam0], val * p_b[nu0 * n + sig0]);
            atomicAdd(&out_b[nu0 * n + lam0], val * p_b[mu0 * n + sig0]);
            atomicAdd(&out_b[mu0 * n + sig0], val * p_b[nu0 * n + lam0]);
            atomicAdd(&out_b[nu0 * n + sig0], val * p_b[mu0 * n + lam0]);
        } else {
            double j_ls = 2.0 * val * (p_a[mu0 * n + nu0] + p_b[mu0 * n + nu0]);
            double j_mn = 2.0 * val * (p_a[lam0 * n + sig0] + p_b[lam0 * n + sig0]);
            atomicAdd(&out_a[mu0 * n + nu0], j_mn);
            atomicAdd(&out_a[lam0 * n + sig0], j_ls);
            atomicAdd(&out_b[mu0 * n + nu0], j_mn);
            atomicAdd(&out_b[lam0 * n + sig0], j_ls);
            if (fabs(exx) > 1e-15) {
                double vc = exx * val;
                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
                atomicAdd(&out_b[mu0 * n + lam0], -vc * p_b[nu0 * n + sig0]);
                atomicAdd(&out_b[nu0 * n + lam0], -vc * p_b[mu0 * n + sig0]);
                atomicAdd(&out_b[mu0 * n + sig0], -vc * p_b[nu0 * n + lam0]);
                atomicAdd(&out_b[nu0 * n + sig0], -vc * p_b[mu0 * n + lam0]);
            }
        }
    } else {
        contract_quartet_element_canonical(val, mu0, nu0, lam0, sig0, n, mode, exx, p_a, p_b, out_a, out_b);
    }
}

extern "C" __global__ void direct_eval_ssss_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_ssss_impl<double>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

extern "C" __global__ void direct_eval_ssss_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_ssss_impl<float>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

// Helper for HRR on L <= 2 from a 35x35 vrr_cont table (row stride 35)
template <typename T>
__device__ __forceinline__ T hrr_ket_L2(
    const T* __restrict__ row,
    int c_idx,
    int id,
    int ld,
    const T* __restrict__ cd_vec
) {
    if (ld == 0) {
        return row[c_idx];
    } else if (ld == 1) {
        int k = id;
        int c_k = ADD_AXIS_3C[c_idx][k];
        return row[c_k] + cd_vec[k] * row[c_idx];
    } else { // ld == 2
        const int D_K[6] = {0, 0, 0, 1, 1, 2};
        const int D_L[6] = {0, 1, 2, 1, 2, 2};
        int k = D_K[id];
        int l = D_L[id];
        int c_k = ADD_AXIS_3C[c_idx][k];
        int c_kl = ADD_AXIS_3C[c_k][l];
        int c_l = ADD_AXIS_3C[c_idx][l];
        T t1 = row[c_kl] + cd_vec[l] * row[c_k];
        T t2 = row[c_l] + cd_vec[l] * row[c_idx];
        return t1 + cd_vec[k] * t2;
    }
}

template <typename T>
__device__ __forceinline__ T hrr_bra_ket_L2(
    const T* __restrict__ vrr_cont,
    int row_stride,
    int a_idx,
    int ib,
    int lb,
    int c_idx,
    int id,
    int ld,
    const T* __restrict__ ab_vec,
    const T* __restrict__ cd_vec
) {
    if (lb == 0) {
        return hrr_ket_L2(vrr_cont + a_idx * row_stride, c_idx, id, ld, cd_vec);
    } else if (lb == 1) {
        int k = ib;
        int a_k = ADD_AXIS_3C[a_idx][k];
        T v1 = hrr_ket_L2(vrr_cont + a_k * row_stride, c_idx, id, ld, cd_vec);
        T v2 = hrr_ket_L2(vrr_cont + a_idx * row_stride, c_idx, id, ld, cd_vec);
        return v1 + ab_vec[k] * v2;
    } else { // lb == 2
        const int D_K[6] = {0, 0, 0, 1, 1, 2};
        const int D_L[6] = {0, 1, 2, 1, 2, 2};
        int k = D_K[ib];
        int l = D_L[ib];
        int a_k = ADD_AXIS_3C[a_idx][k];
        int a_kl = ADD_AXIS_3C[a_k][l];
        int a_l = ADD_AXIS_3C[a_idx][l];
        T t1 = hrr_ket_L2(vrr_cont + a_kl * row_stride, c_idx, id, ld, cd_vec)
             + ab_vec[l] * hrr_ket_L2(vrr_cont + a_k * row_stride, c_idx, id, ld, cd_vec);
        T t2 = hrr_ket_L2(vrr_cont + a_l * row_stride, c_idx, id, ld, cd_vec)
             + ab_vec[l] * hrr_ket_L2(vrr_cont + a_idx * row_stride, c_idx, id, ld, cd_vec);
        return t1 + ab_vec[k] * t2;
    }
}

// Kernel: SP_LOW (la,lb,lc,ld <= 1, and max_ab <= 1 || max_cd <= 1), 1 thread per quartet
template <typename T>
__device__ __forceinline__ void eval_sp_low_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int mode,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qid >= count) return;
    const WorkItem item = wl[qid];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int max_ab = la + lb;
    const int max_cd = lc + ld;
    const int max_m = max_ab + max_cd; // <= 3
    const int na_comp = (max_ab == 0) ? 1 : ((max_ab == 1) ? 4 : 10);
    const int nc_comp = (max_cd == 0) ? 1 : ((max_cd == 1) ? 4 : 10);

    // Compact [10][10] zero-init (at least one of na_comp, nc_comp <= 4, max 40 active)
    T comp_cont[100];
    for (int a = 0; a < na_comp; ++a) {
        for (int c = 0; c < nc_comp; ++c) {
            comp_cont[a * 10 + c] = (T)0.0;
        }
    }

    const int PAIRS2_K[6] = {0, 0, 0, 1, 1, 2};
    const int PAIRS2_L[6] = {0, 1, 2, 1, 2, 2};
    const T two_pi25 = (T)34.986836655249725693;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        const T inv_2p = (T)0.5 / p;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T inv_2q = (T)0.5 / q;
            const T zeta = p + q;
            const T inv_zeta = (T)1.0 / zeta;
            const T inv_2zeta = (T)0.5 * inv_zeta;
            const T rho = p * q * inv_zeta;
            const T w0 = (p * (T)pab.p[0] + q * (T)pcd.p[0]) * inv_zeta;
            const T w1 = (p * (T)pab.p[1] + q * (T)pcd.p[1]) * inv_zeta;
            const T w2 = (p * (T)pab.p[2] + q * (T)pcd.p[2]) * inv_zeta;
            const T wp[3] = {w0 - (T)pab.p[0], w1 - (T)pab.p[1], w2 - (T)pab.p[2]};
            const T wq[3] = {w0 - (T)pcd.p[0], w1 - (T)pcd.p[1], w2 - (T)pcd.p[2]};
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;

            T boys[4];
            if constexpr (sizeof(T) == 8) {
                boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
            } else {
                boys_batch_table_f32(max_m, rho, pq2, (float)omega, has_omega, d_boys_f32, boys);
            }
            const T scale = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;
            const T rho_over_p = rho / p;
            const T rho_over_q = rho / q;

            T g00[4];
            for (int m = 0; m <= max_m; ++m) g00[m] = scale * boys[m];

            T g10[3][3];
            if (max_ab >= 1) {
                for (int u = 0; u < 3; ++u) {
                    for (int m = 0; m <= max_m - 1; ++m) {
                        g10[u][m] = (T)pab.pa[u] * g00[m] + wp[u] * g00[m + 1];
                    }
                }
            }
            T g20[6][2];
            if (max_ab >= 2) {
                T diag_a[2];
                for (int m = 0; m <= max_m - 2; ++m) {
                    diag_a[m] = inv_2p * (g00[m] - rho_over_p * g00[m + 1]);
                }
                for (int idx = 0; idx < 6; ++idx) {
                    int u = PAIRS2_K[idx], v = PAIRS2_L[idx];
                    for (int m = 0; m <= max_m - 2; ++m) {
                        T val = (T)pab.pa[u] * g10[v][m] + wp[u] * g10[v][m + 1];
                        if (u == v) val += diag_a[m];
                        g20[idx][m] = val;
                    }
                }
            }

            if (lc == 0) {
                comp_cont[0] += g00[0];
                if (max_ab >= 1) {
                    comp_cont[10] += g10[0][0];
                    comp_cont[20] += g10[1][0];
                    comp_cont[30] += g10[2][0];
                }
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx) comp_cont[(4 + idx) * 10] += g20[idx][0];
                }
            }

            if (max_cd >= 1) {
                int m_lim_c1 = max_cd - 1;
                T g01[3][2];
                for (int u = 0; u < 3; ++u) {
                    for (int m = 0; m <= m_lim_c1; ++m) {
                        g01[u][m] = (T)pcd.pa[u] * g00[m] + wq[u] * g00[m + 1];
                    }
                }
                T g11[3][3][2];
                if (max_ab >= 1) {
                    for (int u = 0; u < 3; ++u) {
                        for (int v = 0; v < 3; ++v) {
                            for (int m = 0; m <= m_lim_c1; ++m) {
                                T val = (T)pcd.pa[v] * g10[u][m] + wq[v] * g10[u][m + 1];
                                if (u == v) val += inv_2zeta * g00[m + 1];
                                g11[u][v][m] = val;
                            }
                        }
                    }
                }
                T g21[6][3][2];
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx) {
                        int u = PAIRS2_K[idx], v = PAIRS2_L[idx];
                        for (int w = 0; w < 3; ++w) {
                            for (int m = 0; m <= m_lim_c1; ++m) {
                                T val = (T)pcd.pa[w] * g20[idx][m] + wq[w] * g20[idx][m + 1];
                                if (u == w) val += inv_2zeta * g10[v][m + 1];
                                if (v == w) val += inv_2zeta * g10[u][m + 1];
                                g21[idx][w][m] = val;
                            }
                        }
                    }
                }
                for (int w = 0; w < 3; ++w) comp_cont[1 + w] += g01[w][0];
                if (max_ab >= 1) {
                    for (int u = 0; u < 3; ++u)
                        for (int w = 0; w < 3; ++w)
                            comp_cont[(1 + u) * 10 + (1 + w)] += g11[u][w][0];
                }
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx)
                        for (int w = 0; w < 3; ++w)
                            comp_cont[(4 + idx) * 10 + (1 + w)] += g21[idx][w][0];
                }
                if (max_cd >= 2) {
                    T diag_c0 = inv_2q * (g00[0] - rho_over_q * g00[1]);
                    for (int cd_idx = 0; cd_idx < 6; ++cd_idx) {
                        int w = PAIRS2_K[cd_idx], z = PAIRS2_L[cd_idx];
                        T v0 = (T)pcd.pa[w] * g01[z][0] + wq[w] * g01[z][1];
                        if (w == z) v0 += diag_c0;
                        comp_cont[4 + cd_idx] += v0;
                    }
                    if (max_ab >= 1) {
                        for (int u = 0; u < 3; ++u) {
                            T diag_c1 = inv_2q * (g10[u][0] - rho_over_q * g10[u][1]);
                            for (int cd_idx = 0; cd_idx < 6; ++cd_idx) {
                                int w = PAIRS2_K[cd_idx], z = PAIRS2_L[cd_idx];
                                T v = (T)pcd.pa[w] * g11[u][z][0] + wq[w] * g11[u][z][1];
                                if (w == z) v += diag_c1;
                                if (u == w) v += inv_2zeta * g01[z][1];
                                comp_cont[(1 + u) * 10 + 4 + cd_idx] += v;
                            }
                        }
                    }
                }
            }
        }
    }

    const T ab_vec[3] = {
        (T)(sh_a.center[0] - sh_b.center[0]),
        (T)(sh_a.center[1] - sh_b.center[1]),
        (T)(sh_a.center[2] - sh_b.center[2])
    };
    const T cd_vec[3] = {
        (T)(sh_c.center[0] - sh_d.center[0]),
        (T)(sh_c.center[1] - sh_d.center[1]),
        (T)(sh_c.center[2] - sh_d.center[2])
    };
    const int a_off = (la == 0) ? 0 : 1;
    const int c_off = (lc == 0) ? 0 : 1;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const int write_cache = (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
    const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)(na * nb)) : 0;
    const int ket_w = nc * nd;

    for (int ia = 0; ia < na; ++ia) {
        int a_idx = a_off + ia;
        double sa = sh_a.comp_scales[ia];
        for (int ib = 0; ib < nb; ++ib) {
            double sab = sa * sh_b.comp_scales[ib];
            int mu0 = sh_a.ao_offset + ia;
            int nu0 = sh_b.ao_offset + ib;
            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                int lam0 = sh_c.ao_offset + ic;
                for (int id = 0; id < nd; ++id) {
                    int sig0 = sh_d.ao_offset + id;
                    T hval = hrr_bra_ket_L2(comp_cont, 10, a_idx, ib, lb, c_idx, id, ld, ab_vec, cd_vec);
                    double val = sabc * sh_d.comp_scales[id] * (double)hval;
                    if (write_cache) {
                        eri_cache[cache_base + (uint64_t)((ia * nb + ib) * ket_w + ic * nd + id)] = val;
                    }
                    if (!distinct) {
                        if (i == j && ib > ia) continue;
                        if (k == l && id > ic) continue;
                        if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                        contract_quartet_element_canonical(val, mu0, nu0, lam0, sig0, n, mode, exx, p_a, p_b, out_a, out_b);
                    } else if (val != 0.0) {
                        if (mode == 0) {
                            atomicAdd(&out_a[mu0 * n + nu0], 2.0 * val * p_a[lam0 * n + sig0]);
                            atomicAdd(&out_a[lam0 * n + sig0], 2.0 * val * p_a[mu0 * n + nu0]);
                            if (fabs(exx) > 1e-15) {
                                double vc = 0.5 * exx * val;
                                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
                            }
                        } else if (mode == 1) {
                            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
                            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
                            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
                            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
                        } else if (mode == 2) {
                            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
                            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
                            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
                            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
                            atomicAdd(&out_b[mu0 * n + lam0], val * p_b[nu0 * n + sig0]);
                            atomicAdd(&out_b[nu0 * n + lam0], val * p_b[mu0 * n + sig0]);
                            atomicAdd(&out_b[mu0 * n + sig0], val * p_b[nu0 * n + lam0]);
                            atomicAdd(&out_b[nu0 * n + sig0], val * p_b[mu0 * n + lam0]);
                        } else {
                            double pt_ls = p_a[lam0 * n + sig0] + p_b[lam0 * n + sig0];
                            double pt_mn = p_a[mu0 * n + nu0] + p_b[mu0 * n + nu0];
                            atomicAdd(&out_a[mu0 * n + nu0], 2.0 * val * pt_ls);
                            atomicAdd(&out_a[lam0 * n + sig0], 2.0 * val * pt_mn);
                            atomicAdd(&out_b[mu0 * n + nu0], 2.0 * val * pt_ls);
                            atomicAdd(&out_b[lam0 * n + sig0], 2.0 * val * pt_mn);
                            if (fabs(exx) > 1e-15) {
                                double vc = exx * val;
                                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
                                atomicAdd(&out_b[mu0 * n + lam0], -vc * p_b[nu0 * n + sig0]);
                                atomicAdd(&out_b[nu0 * n + lam0], -vc * p_b[mu0 * n + sig0]);
                                atomicAdd(&out_b[mu0 * n + sig0], -vc * p_b[nu0 * n + lam0]);
                                atomicAdd(&out_b[nu0 * n + sig0], -vc * p_b[mu0 * n + lam0]);
                            }
                        }
                    }
                }
            }
        }
    }
    if (write_cache) {
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }
}

extern "C" __global__ void direct_eval_sp_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_sp_low_impl<double>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

extern "C" __global__ void direct_eval_sp_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_sp_low_impl<float>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

// D_WARP: max(la, lb, lc, ld) <= 2, 1 warp (32 threads) per quartet, 2 warps (64 threads) per block.
// Shared memory layout per warp (in T elements):
//   vrr_cont: 35 * 35 = 1225
//   Va:       35 * 9  = 315
//   Vc1:   3 * 35 * 4 = 420
//   Vc2:   6 * 35 * 3 = 630
//   Vc3:  10 * 35 * 2 = 700
// Total per warp = 3290 T elements (26,320 B in double; 13,160 B in float).
// With 2 warps/block = 52,640 B in double (< 64 KB via cuFuncSetAttribute).
#define DWARP_VRR_OFF 0
#define DWARP_VA_OFF 1225
#define DWARP_VC1_OFF (1225 + 315)
#define DWARP_VC2_OFF (1225 + 315 + 420)
#define DWARP_VC3_OFF (1225 + 315 + 420 + 630)
#define DWARP_STRIDE_T 3290

template <typename T>
__device__ __forceinline__ void eval_d_warp_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int mode,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    extern __shared__ unsigned char dyn_smem[];
    const int warp_in_blk = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp_id >= count) return;

    T* wmem = ((T*)dyn_smem) + warp_in_blk * DWARP_STRIDE_T;
    T* vrr_cont = wmem + DWARP_VRR_OFF;
    T* Va = wmem + DWARP_VA_OFF;   // [35 * 9]
    T* Vc1 = wmem + DWARP_VC1_OFF; // [3 * 35 * 4]
    T* Vc2 = wmem + DWARP_VC2_OFF; // [6 * 35 * 3]
    T* Vc3 = wmem + DWARP_VC3_OFF; // [10 * 35 * 2]

    const WorkItem item = wl[warp_id];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int max_ab = la + lb;
    const int max_cd = lc + ld;
    const int max_m = max_ab + max_cd;
    const int n_a = (max_ab + 1) * (max_ab + 2) * (max_ab + 3) / 6;
    const int n_c = (max_cd + 1) * (max_cd + 2) * (max_cd + 3) / 6;

    for (int idx = lane; idx < n_a * 35; idx += 32) {
        vrr_cont[idx] = (T)0.0;
    }
    __syncwarp();

    const int D_K[6] = {0, 0, 0, 1, 1, 2};
    const int D_L[6] = {0, 1, 2, 1, 2, 2};
    const T two_pi25 = (T)34.986836655249725693;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        const T inv_2p = (T)0.5 / p;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T inv_2q = (T)0.5 / q;
            const T zeta = p + q;
            const T inv_zeta = (T)1.0 / zeta;
            const T inv_2zeta = (T)0.5 * inv_zeta;
            const T rho = p * q * inv_zeta;
            const T w0 = (p * (T)pab.p[0] + q * (T)pcd.p[0]) * inv_zeta;
            const T w1 = (p * (T)pab.p[1] + q * (T)pcd.p[1]) * inv_zeta;
            const T w2 = (p * (T)pab.p[2] + q * (T)pcd.p[2]) * inv_zeta;
            const T wp[3] = {w0 - (T)pab.p[0], w1 - (T)pab.p[1], w2 - (T)pab.p[2]};
            const T wq[3] = {w0 - (T)pcd.p[0], w1 - (T)pcd.p[1], w2 - (T)pcd.p[2]};
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;
            const T rho_over_p = rho / p;
            const T rho_over_q = rho / q;
            const T scale = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;

            // Compute Boys across lane 0 (or broadcast) -> lane 0..max_m
            if (lane == 0) {
                T boys[9];
                if constexpr (sizeof(T) == 8) {
                    boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
                } else {
                    boys_batch_table_f32(max_m, rho, pq2, (float)omega, has_omega, d_boys_f32, boys);
                }
                for (int m = 0; m <= max_m; ++m) {
                    Va[m] = scale * boys[m];
                }
            }
            __syncwarp();

            // Bra VRR by level |a| = 1..max_ab (each level parallelized across warp lanes)
            if (max_ab >= 1) {
                if (lane >= 1 && lane <= 3) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis;
                    for (int m = 0; m <= max_m - 1; ++m) {
                        Va[lane * 9 + m] = (T)pab.pa[u] * Va[m] + wp[u] * Va[m + 1];
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 2) {
                if (lane >= 4 && lane <= 9) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 2; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[lane * 9 + m] = val;
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 3) {
                if (lane >= 10 && lane <= 19) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 3; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[lane * 9 + m] = val;
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 4) {
                int a = (lane >= 20) ? lane : ((lane < 3) ? (32 + lane) : -1);
                if (a >= 20 && a < 35) {
                    const Comp3C& c = COMPS_3C[a];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 4; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[a * 9 + m] = val;
                    }
                }
                __syncwarp();
            }

            // Accumulate c = 0
            if (lc == 0) {
                for (int a = lane; a < n_a; a += 32) {
                    vrr_cont[a * 35 + 0] += Va[a * 9 + 0];
                }
            }

            // Ket |c| = 1
            if (max_cd >= 1) {
                int m_lim = max_cd - 1;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int w = 0; w < 3; ++w) {
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Va[a * 9 + m] + wq[w] * Va[a * 9 + m + 1];
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Va[sub_a * 9 + m + 1];
                            Vc1[(w * 35 + a) * 4 + m] = val;
                        }
                        if (lc <= 1) vrr_cont[a * 35 + 1 + w] += Vc1[(w * 35 + a) * 4 + 0];
                    }
                }
                __syncwarp();
            }

            // Ket |c| = 2
            if (max_cd >= 2) {
                int m_lim = max_cd - 2;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int kl = 0; kl < 6; ++kl) {
                        int w = D_K[kl], z = D_L[kl];
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Vc1[(z * 35 + a) * 4 + m]
                                  + wq[w] * Vc1[(z * 35 + a) * 4 + m + 1];
                            if (w == z) val += inv_2q * (Va[a * 9 + m] - rho_over_q * Va[a * 9 + m + 1]);
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc1[(z * 35 + sub_a) * 4 + m + 1];
                            Vc2[(kl * 35 + a) * 3 + m] = val;
                        }
                        if (lc <= 2) vrr_cont[a * 35 + 4 + kl] += Vc2[(kl * 35 + a) * 3 + 0];
                    }
                }
                __syncwarp();
            }

            // Ket |c| = 3
            if (max_cd >= 3) {
                int m_lim = max_cd - 3;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int cf = 0; cf < 10; ++cf) {
                        const Comp3C& cc = COMPS_3C[10 + cf];
                        int w = cc.axis;
                        int d_idx = cc.idx1 - 4;
                        int p_idx_c = cc.idx2 - 1;
                        int nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Vc2[(d_idx * 35 + a) * 3 + m]
                                  + wq[w] * Vc2[(d_idx * 35 + a) * 3 + m + 1];
                            if (nk > 0) {
                                val += (T)nk * inv_2q * (Vc1[(p_idx_c * 35 + a) * 4 + m]
                                    - rho_over_q * Vc1[(p_idx_c * 35 + a) * 4 + m + 1]);
                            }
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc2[(d_idx * 35 + sub_a) * 3 + m + 1];
                            Vc3[(cf * 35 + a) * 2 + m] = val;
                        }
                        if (lc <= 3) vrr_cont[a * 35 + 10 + cf] += Vc3[(cf * 35 + a) * 2 + 0];
                    }
                }
                __syncwarp();
            }

            // Ket |c| = 4 (m = 0 only!)
            if (max_cd >= 4) {
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int cg = 0; cg < 15; ++cg) {
                        const Comp3C& cc = COMPS_3C[20 + cg];
                        int w = cc.axis;
                        int f_idx = cc.idx1 - 10;
                        int d_idx = cc.idx2 - 4;
                        int nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        T val = (T)pcd.pa[w] * Vc3[(f_idx * 35 + a) * 2 + 0]
                              + wq[w] * Vc3[(f_idx * 35 + a) * 2 + 1];
                        if (nk > 0) {
                            val += (T)nk * inv_2q * (Vc2[(d_idx * 35 + a) * 3 + 0]
                                - rho_over_q * Vc2[(d_idx * 35 + a) * 3 + 1]);
                        }
                        if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc3[(f_idx * 35 + sub_a) * 2 + 1];
                        vrr_cont[a * 35 + 20 + cg] += val;
                    }
                }
            }
            __syncwarp();
        }
    }

    // HRR + Contraction: warp-parallel across ab = 0..na*nb
    const T ab_vec[3] = {
        (T)(sh_a.center[0] - sh_b.center[0]),
        (T)(sh_a.center[1] - sh_b.center[1]),
        (T)(sh_a.center[2] - sh_b.center[2])
    };
    const T cd_vec[3] = {
        (T)(sh_c.center[0] - sh_d.center[0]),
        (T)(sh_c.center[1] - sh_d.center[1]),
        (T)(sh_c.center[2] - sh_d.center[2])
    };
    const int a_off = (la == 0) ? 0 : ((la == 1) ? 1 : 4);
    const int c_off = (lc == 0) ? 0 : ((lc == 1) ? 1 : 4);
    const int bra_w = na * nb;
    const int ket_w = nc * nd;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const int has_exch = (fabs(exx) > 1e-15) ? 1 : 0;
    const double half_c = 0.5 * exx;
    const int write_cache = (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
    const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w) : 0;

    for (int ab = lane; ab < bra_w; ab += 32) {
        int ia = ab / nb;
        int ib = ab - ia * nb;
        int a_idx = a_off + ia;
        double sab = sh_a.comp_scales[ia] * sh_b.comp_scales[ib];

        // Compute bra-HRR once for this (ia, ib) into register vector bra_c[0..n_c]
        T bra_c[35];
        if (lb == 0) {
            const T* row = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) bra_c[c] = row[c];
        } else if (lb == 1) {
            int ax = ib;
            int a_k = ADD_AXIS_3C[a_idx][ax];
            const T* r1 = vrr_cont + a_k * 35;
            const T* r2 = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) bra_c[c] = r1[c] + ab_vec[ax] * r2[c];
        } else { // lb == 2
            int ax1 = D_K[ib], ax2 = D_L[ib];
            int a_k = ADD_AXIS_3C[a_idx][ax1];
            int a_kl = ADD_AXIS_3C[a_k][ax2];
            int a_l = ADD_AXIS_3C[a_idx][ax2];
            const T* r_kl = vrr_cont + a_kl * 35;
            const T* r_k  = vrr_cont + a_k  * 35;
            const T* r_l  = vrr_cont + a_l  * 35;
            const T* r_0  = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) {
                T t1 = r_kl[c] + ab_vec[ax2] * r_k[c];
                T t2 = r_l[c]  + ab_vec[ax2] * r_0[c];
                bra_c[c] = t1 + ab_vec[ax1] * t2;
            }
        }

        if (write_cache) {
            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                for (int id = 0; id < nd; ++id) {
                    T hval = hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                    double val = sabc * sh_d.comp_scales[id] * (double)hval;
                    eri_cache[cache_base + (uint64_t)(ab * ket_w + ic * nd + id)] = val;
                }
            }
        }

        int mu0 = sh_a.ao_offset + ia;
        int nu0 = sh_b.ao_offset + ib;
        if (distinct) {
            int mu_n = mu0 * n, nu_n = nu0 * n;
            if (mode == 0) {
                double p_mn2 = 2.0 * p_a[mu_n + nu0];
                double acc_mn = 0.0;
                for (int ic = 0; ic < nc; ++ic) {
                    int c_idx = c_off + ic;
                    int lam = sh_c.ao_offset + ic;
                    int lam_n = lam * n;
                    double sabc = sab * sh_c.comp_scales[ic];
                    if (!has_exch) {
                        for (int id = 0; id < nd; ++id) {
                            double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                            if (val != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                acc_mn += val * p_a[lam_n + sig];
                                atomicAdd(&out_a[lam_n + sig], val * p_mn2);
                            }
                        }
                    } else {
                        double c_nl = half_c * p_a[nu_n + lam];
                        double c_ml = half_c * p_a[mu_n + lam];
                        double acc_ml = 0.0, acc_nl = 0.0;
                        for (int id = 0; id < nd; ++id) {
                            double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                            if (val != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                acc_mn += val * p_a[lam_n + sig];
                                atomicAdd(&out_a[lam_n + sig], val * p_mn2);
                                acc_ml += val * p_a[nu_n + sig];
                                acc_nl += val * p_a[mu_n + sig];
                                atomicAdd(&out_a[mu_n + sig], -val * c_nl);
                                atomicAdd(&out_a[nu_n + sig], -val * c_ml);
                            }
                        }
                        if (acc_ml != 0.0) atomicAdd(&out_a[mu_n + lam], -half_c * acc_ml);
                        if (acc_nl != 0.0) atomicAdd(&out_a[nu_n + lam], -half_c * acc_nl);
                    }
                }
                if (acc_mn != 0.0) atomicAdd(&out_a[mu_n + nu0], 2.0 * acc_mn);
            } else if (mode == 1) {
                for (int ic = 0; ic < nc; ++ic) {
                    int c_idx = c_off + ic;
                    int lam = sh_c.ao_offset + ic;
                    double sabc = sab * sh_c.comp_scales[ic];
                    double p_nl = p_a[nu_n + lam], p_ml = p_a[mu_n + lam];
                    double acc_ml = 0.0, acc_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                        if (val != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_ml += val * p_a[nu_n + sig];
                            acc_nl += val * p_a[mu_n + sig];
                            atomicAdd(&out_a[mu_n + sig], val * p_nl);
                            atomicAdd(&out_a[nu_n + sig], val * p_ml);
                        }
                    }
                    if (acc_ml != 0.0) atomicAdd(&out_a[mu_n + lam], acc_ml);
                    if (acc_nl != 0.0) atomicAdd(&out_a[nu_n + lam], acc_nl);
                }
            } else if (mode == 2) {
                for (int ic = 0; ic < nc; ++ic) {
                    int c_idx = c_off + ic;
                    int lam = sh_c.ao_offset + ic;
                    double sabc = sab * sh_c.comp_scales[ic];
                    double pa_nl = p_a[nu_n + lam], pa_ml = p_a[mu_n + lam];
                    double pb_nl = p_b[nu_n + lam], pb_ml = p_b[mu_n + lam];
                    double acc_a_ml = 0.0, acc_a_nl = 0.0, acc_b_ml = 0.0, acc_b_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                        if (val != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_a_ml += val * p_a[nu_n + sig];
                            acc_a_nl += val * p_a[mu_n + sig];
                            atomicAdd(&out_a[mu_n + sig], val * pa_nl);
                            atomicAdd(&out_a[nu_n + sig], val * pa_ml);

                            acc_b_ml += val * p_b[nu_n + sig];
                            acc_b_nl += val * p_b[mu_n + sig];
                            atomicAdd(&out_b[mu_n + sig], val * pb_nl);
                            atomicAdd(&out_b[nu_n + sig], val * pb_ml);
                        }
                    }
                    if (acc_a_ml != 0.0) atomicAdd(&out_a[mu_n + lam], acc_a_ml);
                    if (acc_a_nl != 0.0) atomicAdd(&out_a[nu_n + lam], acc_a_nl);
                    if (acc_b_ml != 0.0) atomicAdd(&out_b[mu_n + lam], acc_b_ml);
                    if (acc_b_nl != 0.0) atomicAdd(&out_b[nu_n + lam], acc_b_nl);
                }
            } else { // mode == 3 (UKS_JK)
                double pt_mn2 = 2.0 * (p_a[mu_n + nu0] + p_b[mu_n + nu0]);
                double acc_mn = 0.0;
                for (int ic = 0; ic < nc; ++ic) {
                    int c_idx = c_off + ic;
                    int lam = sh_c.ao_offset + ic;
                    int lam_n = lam * n;
                    double sabc = sab * sh_c.comp_scales[ic];
                    double ca_nl = exx * p_a[nu_n + lam], ca_ml = exx * p_a[mu_n + lam];
                    double cb_nl = exx * p_b[nu_n + lam], cb_ml = exx * p_b[mu_n + lam];
                    double acc_a_ml = 0.0, acc_a_nl = 0.0, acc_b_ml = 0.0, acc_b_nl = 0.0;
                    for (int id = 0; id < nd; ++id) {
                        double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                        if (val != 0.0) {
                            int sig = sh_d.ao_offset + id;
                            acc_mn += val * (p_a[lam_n + sig] + p_b[lam_n + sig]);
                            double j_ls = val * pt_mn2;
                            atomicAdd(&out_a[lam_n + sig], j_ls);
                            atomicAdd(&out_b[lam_n + sig], j_ls);
                            if (has_exch) {
                                acc_a_ml += val * p_a[nu_n + sig];
                                acc_a_nl += val * p_a[mu_n + sig];
                                atomicAdd(&out_a[mu_n + sig], -val * ca_nl);
                                atomicAdd(&out_a[nu_n + sig], -val * ca_ml);

                                acc_b_ml += val * p_b[nu_n + sig];
                                acc_b_nl += val * p_b[mu_n + sig];
                                atomicAdd(&out_b[mu_n + sig], -val * cb_nl);
                                atomicAdd(&out_b[nu_n + sig], -val * cb_ml);
                            }
                        }
                    }
                    if (has_exch) {
                        if (acc_a_ml != 0.0) atomicAdd(&out_a[mu_n + lam], -exx * acc_a_ml);
                        if (acc_a_nl != 0.0) atomicAdd(&out_a[nu_n + lam], -exx * acc_a_nl);
                        if (acc_b_ml != 0.0) atomicAdd(&out_b[mu_n + lam], -exx * acc_b_ml);
                        if (acc_b_nl != 0.0) atomicAdd(&out_b[nu_n + lam], -exx * acc_b_nl);
                    }
                }
                if (acc_mn != 0.0) {
                    double j_mn = 2.0 * acc_mn;
                    atomicAdd(&out_a[mu_n + nu0], j_mn);
                    atomicAdd(&out_b[mu_n + nu0], j_mn);
                }
            }
        } else {
            if (i == j && ib > ia) continue;
            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                int id_end = (k == l) ? (ic + 1) : nd;
                int lam0 = sh_c.ao_offset + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                for (int id = 0; id < id_end; ++id) {
                    if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                    int sig0 = sh_d.ao_offset + id;
                    double val = sabc * sh_d.comp_scales[id] * (double)hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                    contract_quartet_element_canonical(val, mu0, nu0, lam0, sig0, n, mode, exx, p_a, p_b, out_a, out_b);
                }
            }
        }
    }
    __syncwarp();
    if (write_cache && lane == 0) {
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }
}

extern "C" __global__ void direct_eval_d_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_d_warp_impl<double>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

extern "C" __global__ void direct_eval_d_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n, int mode, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ p_a, const double* __restrict__ p_b,
    double* __restrict__ out_a, double* __restrict__ out_b
) {
    eval_d_warp_impl<float>(shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n, mode, exx, omega, has_omega, d_boys_f64, d_boys_f32, p_a, p_b, out_a, out_b);
}

// F_BLOCK: max(la, lb, lc, ld) == 3 (L <= 6, 84 comps), 1 block (64 threads) per quartet,
// grid-stride loop over wl_f using global V_scratch[blockIdx.x * 84 * 84 * 13] and shared vrr_cont[84 * 84].
__device__ double hrr_ket_6_dev(
    const Comp6* __restrict__ comps6,
    const double* __restrict__ row,
    int c_idx,
    int dx, int dy, int dz,
    const double* __restrict__ cd_vec
) {
    if (dx > 0) {
        int cp = comps6[c_idx].add[0];
        return hrr_ket_6_dev(comps6, row, cp, dx - 1, dy, dz, cd_vec)
             + cd_vec[0] * hrr_ket_6_dev(comps6, row, c_idx, dx - 1, dy, dz, cd_vec);
    }
    if (dy > 0) {
        int cp = comps6[c_idx].add[1];
        return hrr_ket_6_dev(comps6, row, cp, 0, dy - 1, dz, cd_vec)
             + cd_vec[1] * hrr_ket_6_dev(comps6, row, c_idx, 0, dy - 1, dz, cd_vec);
    }
    if (dz > 0) {
        int cp = comps6[c_idx].add[2];
        return hrr_ket_6_dev(comps6, row, cp, 0, 0, dz - 1, cd_vec)
             + cd_vec[2] * hrr_ket_6_dev(comps6, row, c_idx, 0, 0, dz - 1, cd_vec);
    }
    return row[c_idx];
}

__device__ double hrr_bra_6_dev(
    const Comp6* __restrict__ comps6,
    const double* __restrict__ vrr_cont,
    int a_idx,
    int bx, int by, int bz,
    int c_idx,
    int dx, int dy, int dz,
    const double* __restrict__ ab_vec,
    const double* __restrict__ cd_vec
) {
    if (bx > 0) {
        int ap = comps6[a_idx].add[0];
        return hrr_bra_6_dev(comps6, vrr_cont, ap, bx - 1, by, bz, c_idx, dx, dy, dz, ab_vec, cd_vec)
             + ab_vec[0] * hrr_bra_6_dev(comps6, vrr_cont, a_idx, bx - 1, by, bz, c_idx, dx, dy, dz, ab_vec, cd_vec);
    }
    if (by > 0) {
        int ap = comps6[a_idx].add[1];
        return hrr_bra_6_dev(comps6, vrr_cont, ap, 0, by - 1, bz, c_idx, dx, dy, dz, ab_vec, cd_vec)
             + ab_vec[1] * hrr_bra_6_dev(comps6, vrr_cont, a_idx, 0, by - 1, bz, c_idx, dx, dy, dz, ab_vec, cd_vec);
    }
    if (bz > 0) {
        int ap = comps6[a_idx].add[2];
        return hrr_bra_6_dev(comps6, vrr_cont, ap, 0, 0, bz - 1, c_idx, dx, dy, dz, ab_vec, cd_vec)
             + ab_vec[2] * hrr_bra_6_dev(comps6, vrr_cont, a_idx, 0, 0, bz - 1, c_idx, dx, dy, dz, ab_vec, cd_vec);
    }
    return hrr_ket_6_dev(comps6, vrr_cont + a_idx * 84, c_idx, dx, dy, dz, cd_vec);
}

extern "C" __global__ void direct_eval_f_f64_kernel(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int mode,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const Comp6* __restrict__ comps6,
    double* __restrict__ f_vrr_scratch, // gridDim.x * 84 * 84 * 13
    const double* __restrict__ p_a,
    const double* __restrict__ p_b,
    double* __restrict__ out_a,
    double* __restrict__ out_b
) {
    extern __shared__ double vrr_cont[];
    const int tid = (int)threadIdx.x;
    double* V = f_vrr_scratch + (size_t)blockIdx.x * (size_t)(84 * 84 * 13);

    const int L_START[8] = {0, 1, 4, 10, 20, 35, 56, 84};

    for (int qid = (int)blockIdx.x; qid < count; qid += (int)gridDim.x) {
        const WorkItem item = wl[qid];
        const DirectShellPairGpu bra = pairs[item.pi];
        const DirectShellPairGpu ket = pairs[item.qi];
        const int i = bra.shell_i, j = bra.shell_j;
        const int k = ket.shell_i, l = ket.shell_j;
        const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
        const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
        const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
        const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
        const int max_ab = la + lb;
        const int max_cd = lc + ld;
        const int max_m = max_ab + max_cd;
        const int n_a = L_START[max_ab + 1];
        const int n_c = L_START[max_cd + 1];

        for (int idx = tid; idx < n_a * 84; idx += (int)blockDim.x) {
            vrr_cont[idx] = 0.0;
        }
        __syncthreads();

        const double two_pi25 = 34.986836655249725693;
        for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
            const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
            const double p = pab.gamma;
            const double inv_2p = 0.5 / p;
            for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
                const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
                const double q = pcd.gamma;
                const double inv_2q = 0.5 / q;
                const double zeta = p + q;
                const double inv_zeta = 1.0 / zeta;
                const double inv_2zeta = 0.5 * inv_zeta;
                const double rho = p * q * inv_zeta;
                const double w0 = (p * pab.p[0] + q * pcd.p[0]) * inv_zeta;
                const double w1 = (p * pab.p[1] + q * pcd.p[1]) * inv_zeta;
                const double w2 = (p * pab.p[2] + q * pcd.p[2]) * inv_zeta;
                const double wp[3] = {w0 - pab.p[0], w1 - pab.p[1], w2 - pab.p[2]};
                const double wq[3] = {w0 - pcd.p[0], w1 - pcd.p[1], w2 - pcd.p[2]};
                const double dx = pab.p[0] - pcd.p[0];
                const double dy = pab.p[1] - pcd.p[1];
                const double dz = pab.p[2] - pcd.p[2];
                const double pq2 = dx * dx + dy * dy + dz * dz;
                const double rho_over_p = rho / p;
                const double rho_over_q = rho / q;
                const double scale = two_pi25 / (p * q * sqrt(zeta)) * pab.kab_w * pcd.kab_w;

                if (tid == 0) {
                    double boys[13];
                    boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
                    for (int m = 0; m <= max_m; ++m) V[m] = scale * boys[m];
                }
                __syncthreads();

                // Bra VRR: c = 0, |a| = 1..max_ab
                for (int la_cur = 1; la_cur <= max_ab; ++la_cur) {
                    int a_beg = L_START[la_cur], a_end = L_START[la_cur + 1];
                    for (int a = a_beg + tid; a < a_end; a += (int)blockDim.x) {
                        const Comp6 ca = comps6[a];
                        int u = ca.axis, i1 = ca.idx1, i2 = ca.idx2, nk = ca.n_k;
                        for (int m = 0; m <= max_m - la_cur; ++m) {
                            double val = pab.pa[u] * V[(size_t)i1 * 84 * 13 + m]
                                       + wp[u]     * V[(size_t)i1 * 84 * 13 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2p * (V[(size_t)i2 * 84 * 13 + m]
                                    - rho_over_p * V[(size_t)i2 * 84 * 13 + m + 1]);
                            }
                            V[(size_t)a * 84 * 13 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                // Ket VRR: |c| = 1..max_cd across all (a, c)
                for (int lc_cur = 1; lc_cur <= max_cd; ++lc_cur) {
                    int c_beg = L_START[lc_cur], c_end = L_START[lc_cur + 1];
                    int n_c_lvl = c_end - c_beg;
                    int total_ac = n_a * n_c_lvl;
                    for (int ac = tid; ac < total_ac; ac += (int)blockDim.x) {
                        int a = ac / n_c_lvl;
                        int c = c_beg + (ac - a * n_c_lvl);
                        const Comp6 ca = comps6[a];
                        const Comp6 cc = comps6[c];
                        int w = cc.axis, c1 = cc.idx1, c2 = cc.idx2, nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        int m_top = max_m - ca.L - lc_cur;
                        for (int m = 0; m <= m_top; ++m) {
                            double val = pcd.pa[w] * V[((size_t)a * 84 + c1) * 13 + m]
                                       + wq[w]     * V[((size_t)a * 84 + c1) * 13 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2q * (V[((size_t)a * 84 + c2) * 13 + m]
                                    - rho_over_q * V[((size_t)a * 84 + c2) * 13 + m + 1]);
                            }
                            if (sub_a >= 0) {
                                val += (double)aw * inv_2zeta * V[((size_t)sub_a * 84 + c1) * 13 + m + 1];
                            }
                            V[((size_t)a * 84 + c) * 13 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                // Accumulate m = 0 into vrr_cont
                for (int idx = tid; idx < n_a * n_c; idx += (int)blockDim.x) {
                    int a = idx / n_c;
                    int c = idx - a * n_c;
                    vrr_cont[a * 84 + c] += V[((size_t)a * 84 + c) * 13 + 0];
                }
                __syncthreads();
            }
        }

        const double ab_vec[3] = {
            sh_a.center[0] - sh_b.center[0],
            sh_a.center[1] - sh_b.center[1],
            sh_a.center[2] - sh_b.center[2]
        };
        const double cd_vec[3] = {
            sh_c.center[0] - sh_d.center[0],
            sh_c.center[1] - sh_d.center[1],
            sh_c.center[2] - sh_d.center[2]
        };
        const int a_off = L_START[la], b_off = L_START[lb];
        const int c_off = L_START[lc], d_off = L_START[ld];
        const int bra_w = na * nb, ket_w = nc * nd;
        const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
        const int write_cache = (has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
        const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w) : 0;

        for (int ab = tid; ab < bra_w; ab += (int)blockDim.x) {
            int ia = ab / nb, ib = ab - ia * nb;
            int a_idx = a_off + ia;
            const Comp6 cb = comps6[b_off + ib];
            double sab = sh_a.comp_scales[ia] * sh_b.comp_scales[ib];
            int mu0 = sh_a.ao_offset + ia, nu0 = sh_b.ao_offset + ib;

            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                int lam0 = sh_c.ao_offset + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                for (int id = 0; id < nd; ++id) {
                    const Comp6 cd_c = comps6[d_off + id];
                    double val = sabc * sh_d.comp_scales[id] * hrr_bra_6_dev(
                        comps6, vrr_cont, a_idx, cb.x, cb.y, cb.z, c_idx, cd_c.x, cd_c.y, cd_c.z, ab_vec, cd_vec);
                    if (write_cache) {
                        eri_cache[cache_base + (uint64_t)(ab * ket_w + ic * nd + id)] = val;
                    }
                    if (!distinct) {
                        if (i == j && ib > ia) continue;
                        if (k == l && id > ic) continue;
                        if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                        contract_quartet_element_canonical(val, mu0, nu0, lam0, sh_d.ao_offset + id, n, mode, exx, p_a, p_b, out_a, out_b);
                    } else if (val != 0.0) {
                        int sig0 = sh_d.ao_offset + id;
                        if (mode == 0) {
                            atomicAdd(&out_a[mu0 * n + nu0], 2.0 * val * p_a[lam0 * n + sig0]);
                            atomicAdd(&out_a[lam0 * n + sig0], 2.0 * val * p_a[mu0 * n + nu0]);
                            if (fabs(exx) > 1e-15) {
                                double vc = 0.5 * exx * val;
                                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
                            }
                        } else if (mode == 1) {
                            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
                            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
                            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
                            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
                        } else if (mode == 2) {
                            atomicAdd(&out_a[mu0 * n + lam0], val * p_a[nu0 * n + sig0]);
                            atomicAdd(&out_a[nu0 * n + lam0], val * p_a[mu0 * n + sig0]);
                            atomicAdd(&out_a[mu0 * n + sig0], val * p_a[nu0 * n + lam0]);
                            atomicAdd(&out_a[nu0 * n + sig0], val * p_a[mu0 * n + lam0]);
                            atomicAdd(&out_b[mu0 * n + lam0], val * p_b[nu0 * n + sig0]);
                            atomicAdd(&out_b[nu0 * n + lam0], val * p_b[mu0 * n + sig0]);
                            atomicAdd(&out_b[mu0 * n + sig0], val * p_b[nu0 * n + lam0]);
                            atomicAdd(&out_b[nu0 * n + sig0], val * p_b[mu0 * n + lam0]);
                        } else {
                            double pt_ls = p_a[lam0 * n + sig0] + p_b[lam0 * n + sig0];
                            double pt_mn = p_a[mu0 * n + nu0] + p_b[mu0 * n + nu0];
                            atomicAdd(&out_a[mu0 * n + nu0], 2.0 * val * pt_ls);
                            atomicAdd(&out_a[lam0 * n + sig0], 2.0 * val * pt_mn);
                            atomicAdd(&out_b[mu0 * n + nu0], 2.0 * val * pt_ls);
                            atomicAdd(&out_b[lam0 * n + sig0], 2.0 * val * pt_mn);
                            if (fabs(exx) > 1e-15) {
                                double vc = exx * val;
                                atomicAdd(&out_a[mu0 * n + lam0], -vc * p_a[nu0 * n + sig0]);
                                atomicAdd(&out_a[nu0 * n + lam0], -vc * p_a[mu0 * n + sig0]);
                                atomicAdd(&out_a[mu0 * n + sig0], -vc * p_a[nu0 * n + lam0]);
                                atomicAdd(&out_a[nu0 * n + sig0], -vc * p_a[mu0 * n + lam0]);
                                atomicAdd(&out_b[mu0 * n + lam0], -vc * p_b[nu0 * n + sig0]);
                                atomicAdd(&out_b[nu0 * n + lam0], -vc * p_b[mu0 * n + sig0]);
                                atomicAdd(&out_b[mu0 * n + sig0], -vc * p_b[nu0 * n + lam0]);
                                atomicAdd(&out_b[nu0 * n + sig0], -vc * p_b[mu0 * n + lam0]);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (write_cache && tid == 0) {
            cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
        }
        __syncthreads();
    }
}

// ============================================================================
// Batched TD-DFT Response Contraction Kernels (mode == 4 / TD_BATCH)
// ============================================================================

__device__ __forceinline__ void contract_quartet_element_td_canonical(
    double val,
    int mu0, int nu0, int lam0, int sig0,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    if (val == 0.0) return;
    int mu = (mu0 <= nu0) ? mu0 : nu0;
    int nu = (mu0 <= nu0) ? nu0 : mu0;
    int lam = (lam0 <= sig0) ? lam0 : sig0;
    int sig = (lam0 <= sig0) ? sig0 : lam0;
    double s1 = (mu == nu) ? 2.0 : 1.0;
    double s2 = (lam == sig) ? 2.0 : 1.0;
    double s3 = (mu == lam && nu == sig) ? 2.0 : 1.0;
    double v = val / (s1 * s2 * s3);

    size_t nn = (size_t)n * (size_t)n;
    size_t idx_mn = (size_t)mu * (size_t)n + (size_t)nu;
    size_t idx_ls = (size_t)lam * (size_t)n + (size_t)sig;
    size_t idx_ml = (size_t)mu * (size_t)n + (size_t)lam;
    size_t idx_nl = (size_t)nu * (size_t)n + (size_t)lam;
    size_t idx_ms = (size_t)mu * (size_t)n + (size_t)sig;
    size_t idx_ns = (size_t)nu * (size_t)n + (size_t)sig;

    if (do_j || do_k_sym) {
        for (int b = 0; b < b_sym; ++b) {
            const double* ds = d_sym + (size_t)b * nn;
            if (do_j) {
                double* oj = out_j + (size_t)b * nn;
                atomicAdd(&oj[idx_mn], 2.0 * v * ds[idx_ls]);
                atomicAdd(&oj[idx_ls], 2.0 * v * ds[idx_mn]);
            }
            if (do_k_sym) {
                double* oks = out_ks + (size_t)b * nn;
                atomicAdd(&oks[idx_ml], v * ds[idx_ns]);
                atomicAdd(&oks[idx_nl], v * ds[idx_ms]);
                atomicAdd(&oks[idx_ms], v * ds[idx_nl]);
                atomicAdd(&oks[idx_ns], v * ds[idx_ml]);
            }
        }
    }
    if (do_k_anti) {
        for (int b = 0; b < b_anti; ++b) {
            const double* da = d_anti + (size_t)b * nn;
            double* oka = out_ka + (size_t)b * nn;
            atomicAdd(&oka[idx_ml], v * da[idx_ns]);
            atomicAdd(&oka[idx_nl], v * da[idx_ms]);
            atomicAdd(&oka[idx_ms], v * da[idx_nl]);
            atomicAdd(&oka[idx_ns], v * da[idx_ml]);
        }
    }
}

extern "C" __global__ void direct_contract_cached_td_kernel(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const uint64_t* __restrict__ pair_prefix,
    const double* __restrict__ eri_cache,
    const WorkItem* __restrict__ wl_hit,
    int n_hit,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (warp_id >= n_hit) return;

    const WorkItem item = wl_hit[warp_id];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs;
    const int nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int bra_w = na * nb;
    const int ket_w = nc * nd;
    const double* blk = eri_cache + bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const size_t nn = (size_t)n * (size_t)n;

    if (distinct) {
        for (int ab = lane; ab < bra_w; ab += 32) {
            int ia = ab / nb;
            int ib = ab - ia * nb;
            int mu = sh_a.ao_offset + ia;
            int nu = sh_b.ao_offset + ib;
            size_t mu_n = (size_t)mu * (size_t)n;
            size_t nu_n = (size_t)nu * (size_t)n;
            const double* row = blk + ab * ket_w;

            if (do_j || do_k_sym) {
                for (int b0 = 0; b0 < b_sym; b0 += 4) {
                    int bt = (b_sym - b0 < 4) ? (b_sym - b0) : 4;
                    double d_mn2[4];
                    double acc_mn[4];
                    for (int t = 0; t < bt; ++t) {
                        const double* ds = d_sym + (size_t)(b0 + t) * nn;
                        d_mn2[t] = do_j ? (2.0 * ds[mu_n + nu]) : 0.0;
                        acc_mn[t] = 0.0;
                    }
                    for (int ic = 0; ic < nc; ++ic) {
                        int lam = sh_c.ao_offset + ic;
                        size_t lam_n = (size_t)lam * (size_t)n;
                        double d_nl[4], d_ml[4], acc_ml[4], acc_nl[4];
                        for (int t = 0; t < bt; ++t) {
                            const double* ds = d_sym + (size_t)(b0 + t) * nn;
                            d_nl[t] = do_k_sym ? ds[nu_n + lam] : 0.0;
                            d_ml[t] = do_k_sym ? ds[mu_n + lam] : 0.0;
                            acc_ml[t] = 0.0;
                            acc_nl[t] = 0.0;
                        }
                        for (int id = 0; id < nd; ++id) {
                            double v = row[ic * nd + id];
                            if (v == 0.0) continue;
                            int sig = sh_d.ao_offset + id;
                            for (int t = 0; t < bt; ++t) {
                                const double* ds = d_sym + (size_t)(b0 + t) * nn;
                                if (do_j) {
                                    double* oj = out_j + (size_t)(b0 + t) * nn;
                                    acc_mn[t] += v * ds[lam_n + sig];
                                    atomicAdd(&oj[lam_n + sig], v * d_mn2[t]);
                                }
                                if (do_k_sym) {
                                    double* oks = out_ks + (size_t)(b0 + t) * nn;
                                    acc_ml[t] += v * ds[nu_n + sig];
                                    acc_nl[t] += v * ds[mu_n + sig];
                                    atomicAdd(&oks[mu_n + sig], v * d_nl[t]);
                                    atomicAdd(&oks[nu_n + sig], v * d_ml[t]);
                                }
                            }
                        }
                        if (do_k_sym) {
                            for (int t = 0; t < bt; ++t) {
                                double* oks = out_ks + (size_t)(b0 + t) * nn;
                                if (acc_ml[t] != 0.0) atomicAdd(&oks[mu_n + lam], acc_ml[t]);
                                if (acc_nl[t] != 0.0) atomicAdd(&oks[nu_n + lam], acc_nl[t]);
                            }
                        }
                    }
                    if (do_j) {
                        for (int t = 0; t < bt; ++t) {
                            if (acc_mn[t] != 0.0) {
                                double* oj = out_j + (size_t)(b0 + t) * nn;
                                atomicAdd(&oj[mu_n + nu], 2.0 * acc_mn[t]);
                            }
                        }
                    }
                }
            }

            if (do_k_anti) {
                for (int b0 = 0; b0 < b_anti; b0 += 4) {
                    int bt = (b_anti - b0 < 4) ? (b_anti - b0) : 4;
                    for (int ic = 0; ic < nc; ++ic) {
                        int lam = sh_c.ao_offset + ic;
                        double d_nl[4], d_ml[4], acc_ml[4], acc_nl[4];
                        for (int t = 0; t < bt; ++t) {
                            const double* da = d_anti + (size_t)(b0 + t) * nn;
                            d_nl[t] = da[nu_n + lam];
                            d_ml[t] = da[mu_n + lam];
                            acc_ml[t] = 0.0;
                            acc_nl[t] = 0.0;
                        }
                        for (int id = 0; id < nd; ++id) {
                            double v = row[ic * nd + id];
                            if (v == 0.0) continue;
                            int sig = sh_d.ao_offset + id;
                            for (int t = 0; t < bt; ++t) {
                                const double* da = d_anti + (size_t)(b0 + t) * nn;
                                double* oka = out_ka + (size_t)(b0 + t) * nn;
                                acc_ml[t] += v * da[nu_n + sig];
                                acc_nl[t] += v * da[mu_n + sig];
                                atomicAdd(&oka[mu_n + sig], v * d_nl[t]);
                                atomicAdd(&oka[nu_n + sig], v * d_ml[t]);
                            }
                        }
                        for (int t = 0; t < bt; ++t) {
                            double* oka = out_ka + (size_t)(b0 + t) * nn;
                            if (acc_ml[t] != 0.0) atomicAdd(&oka[mu_n + lam], acc_ml[t]);
                            if (acc_nl[t] != 0.0) atomicAdd(&oka[nu_n + lam], acc_nl[t]);
                        }
                    }
                }
            }
        }
    } else {
        for (int ab = lane; ab < bra_w; ab += 32) {
            int ia = ab / nb;
            int ib = ab - ia * nb;
            if (i == j && ib > ia) continue;
            int mu0 = sh_a.ao_offset + ia;
            int nu0 = sh_b.ao_offset + ib;
            const double* row = blk + ab * ket_w;
            for (int ic = 0; ic < nc; ++ic) {
                int id_end = (k == l) ? (ic + 1) : nd;
                int lam0 = sh_c.ao_offset + ic;
                for (int id = 0; id < id_end; ++id) {
                    if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                    int sig0 = sh_d.ao_offset + id;
                    double val = row[ic * nd + id];
                    contract_quartet_element_td_canonical(
                        val, mu0, nu0, lam0, sig0, n,
                        b_sym, b_anti, do_j, do_k_sym, do_k_anti,
                        d_sym, d_anti, out_j, out_ks, out_ka
                    );
                }
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ void eval_ssss_td_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qid >= count) return;
    const WorkItem item = wl[qid];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];

    const T two_pi25 = (T)34.986836655249725693;
    const T s_norm = (T)(sh_a.comp_scales[0] * sh_b.comp_scales[0] * sh_c.comp_scales[0] * sh_d.comp_scales[0]);
    T sum = (T)0.0;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T zeta = p + q;
            const T rho = p * q / zeta;
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;
            T b0[1];
            if constexpr (sizeof(T) == 8) {
                boys_batch_table_f64(0, rho, pq2, omega, has_omega, d_boys_f64, b0);
            } else {
                boys_batch_table_f32(0, rho, pq2, (float)omega, has_omega, d_boys_f32, b0);
            }
            const T pref = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;
            sum += pref * b0[0];
        }
    }
    const double val = (double)(s_norm * sum);
    if (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) {
        eri_cache[bra.cache_data_offset + pair_prefix[item.qi]] = val;
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }
    if (val == 0.0) return;

    const int mu0 = sh_a.ao_offset, nu0 = sh_b.ao_offset;
    const int lam0 = sh_c.ao_offset, sig0 = sh_d.ao_offset;
    if (i != j && k != l && (i != k || j != l)) {
        const size_t nn = (size_t)n * (size_t)n;
        const size_t idx_mn = (size_t)mu0 * (size_t)n + (size_t)nu0;
        const size_t idx_ls = (size_t)lam0 * (size_t)n + (size_t)sig0;
        const size_t idx_ml = (size_t)mu0 * (size_t)n + (size_t)lam0;
        const size_t idx_nl = (size_t)nu0 * (size_t)n + (size_t)lam0;
        const size_t idx_ms = (size_t)mu0 * (size_t)n + (size_t)sig0;
        const size_t idx_ns = (size_t)nu0 * (size_t)n + (size_t)sig0;

        if (do_j || do_k_sym) {
            for (int b = 0; b < b_sym; ++b) {
                const double* ds = d_sym + (size_t)b * nn;
                if (do_j) {
                    double* oj = out_j + (size_t)b * nn;
                    atomicAdd(&oj[idx_mn], 2.0 * val * ds[idx_ls]);
                    atomicAdd(&oj[idx_ls], 2.0 * val * ds[idx_mn]);
                }
                if (do_k_sym) {
                    double* oks = out_ks + (size_t)b * nn;
                    atomicAdd(&oks[idx_ml], val * ds[idx_ns]);
                    atomicAdd(&oks[idx_nl], val * ds[idx_ms]);
                    atomicAdd(&oks[idx_ms], val * ds[idx_nl]);
                    atomicAdd(&oks[idx_ns], val * ds[idx_ml]);
                }
            }
        }
        if (do_k_anti) {
            for (int b = 0; b < b_anti; ++b) {
                const double* da = d_anti + (size_t)b * nn;
                double* oka = out_ka + (size_t)b * nn;
                atomicAdd(&oka[idx_ml], val * da[idx_ns]);
                atomicAdd(&oka[idx_nl], val * da[idx_ms]);
                atomicAdd(&oka[idx_ms], val * da[idx_nl]);
                atomicAdd(&oka[idx_ns], val * da[idx_ml]);
            }
        }
    } else {
        contract_quartet_element_td_canonical(
            val, mu0, nu0, lam0, sig0, n,
            b_sym, b_anti, do_j, do_k_sym, do_k_anti,
            d_sym, d_anti, out_j, out_ks, out_ka
        );
    }
}

extern "C" __global__ void direct_eval_ssss_td_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_ssss_td_impl<double>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

extern "C" __global__ void direct_eval_ssss_td_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_ssss_td_impl<float>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

template <typename T>
__device__ __forceinline__ void eval_sp_low_td_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qid >= count) return;
    const WorkItem item = wl[qid];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int max_ab = la + lb;
    const int max_cd = lc + ld;
    const int max_m = max_ab + max_cd;
    const int na_comp = (max_ab == 0) ? 1 : ((max_ab == 1) ? 4 : 10);
    const int nc_comp = (max_cd == 0) ? 1 : ((max_cd == 1) ? 4 : 10);

    T comp_cont[100];
    for (int a = 0; a < na_comp; ++a) {
        for (int c = 0; c < nc_comp; ++c) {
            comp_cont[a * 10 + c] = (T)0.0;
        }
    }

    const int PAIRS2_K[6] = {0, 0, 0, 1, 1, 2};
    const int PAIRS2_L[6] = {0, 1, 2, 1, 2, 2};
    const T two_pi25 = (T)34.986836655249725693;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        const T inv_2p = (T)0.5 / p;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T inv_2q = (T)0.5 / q;
            const T zeta = p + q;
            const T inv_zeta = (T)1.0 / zeta;
            const T inv_2zeta = (T)0.5 * inv_zeta;
            const T rho = p * q * inv_zeta;
            const T w0 = (p * (T)pab.p[0] + q * (T)pcd.p[0]) * inv_zeta;
            const T w1 = (p * (T)pab.p[1] + q * (T)pcd.p[1]) * inv_zeta;
            const T w2 = (p * (T)pab.p[2] + q * (T)pcd.p[2]) * inv_zeta;
            const T wp[3] = {w0 - (T)pab.p[0], w1 - (T)pab.p[1], w2 - (T)pab.p[2]};
            const T wq[3] = {w0 - (T)pcd.p[0], w1 - (T)pcd.p[1], w2 - (T)pcd.p[2]};
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;

            T boys[4];
            if constexpr (sizeof(T) == 8) {
                boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
            } else {
                boys_batch_table_f32(max_m, rho, pq2, (float)omega, has_omega, d_boys_f32, boys);
            }
            const T scale = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;
            const T rho_over_p = rho / p;
            const T rho_over_q = rho / q;

            T g00[4];
            for (int m = 0; m <= max_m; ++m) g00[m] = scale * boys[m];

            T g10[3][3];
            if (max_ab >= 1) {
                for (int u = 0; u < 3; ++u) {
                    for (int m = 0; m <= max_m - 1; ++m) {
                        g10[u][m] = (T)pab.pa[u] * g00[m] + wp[u] * g00[m + 1];
                    }
                }
            }
            T g20[6][2];
            if (max_ab >= 2) {
                T diag_a[2];
                for (int m = 0; m <= max_m - 2; ++m) {
                    diag_a[m] = inv_2p * (g00[m] - rho_over_p * g00[m + 1]);
                }
                for (int idx = 0; idx < 6; ++idx) {
                    int u = PAIRS2_K[idx], v = PAIRS2_L[idx];
                    for (int m = 0; m <= max_m - 2; ++m) {
                        T val = (T)pab.pa[u] * g10[v][m] + wp[u] * g10[v][m + 1];
                        if (u == v) val += diag_a[m];
                        g20[idx][m] = val;
                    }
                }
            }

            if (lc == 0) {
                comp_cont[0] += g00[0];
                if (max_ab >= 1) {
                    comp_cont[10] += g10[0][0];
                    comp_cont[20] += g10[1][0];
                    comp_cont[30] += g10[2][0];
                }
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx) comp_cont[(4 + idx) * 10] += g20[idx][0];
                }
            }

            if (max_cd >= 1) {
                int m_lim_c1 = max_cd - 1;
                T g01[3][2];
                for (int u = 0; u < 3; ++u) {
                    for (int m = 0; m <= m_lim_c1; ++m) {
                        g01[u][m] = (T)pcd.pa[u] * g00[m] + wq[u] * g00[m + 1];
                    }
                }
                T g11[3][3][2];
                if (max_ab >= 1) {
                    for (int u = 0; u < 3; ++u) {
                        for (int v = 0; v < 3; ++v) {
                            for (int m = 0; m <= m_lim_c1; ++m) {
                                T val = (T)pcd.pa[v] * g10[u][m] + wq[v] * g10[u][m + 1];
                                if (u == v) val += inv_2zeta * g00[m + 1];
                                g11[u][v][m] = val;
                            }
                        }
                    }
                }
                T g21[6][3][2];
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx) {
                        int u = PAIRS2_K[idx], v = PAIRS2_L[idx];
                        for (int w = 0; w < 3; ++w) {
                            for (int m = 0; m <= m_lim_c1; ++m) {
                                T val = (T)pcd.pa[w] * g20[idx][m] + wq[w] * g20[idx][m + 1];
                                if (u == w) val += inv_2zeta * g10[v][m + 1];
                                if (v == w) val += inv_2zeta * g10[u][m + 1];
                                g21[idx][w][m] = val;
                            }
                        }
                    }
                }
                for (int w = 0; w < 3; ++w) comp_cont[1 + w] += g01[w][0];
                if (max_ab >= 1) {
                    for (int u = 0; u < 3; ++u)
                        for (int w = 0; w < 3; ++w)
                            comp_cont[(1 + u) * 10 + (1 + w)] += g11[u][w][0];
                }
                if (max_ab >= 2) {
                    for (int idx = 0; idx < 6; ++idx)
                        for (int w = 0; w < 3; ++w)
                            comp_cont[(4 + idx) * 10 + (1 + w)] += g21[idx][w][0];
                }
                if (max_cd >= 2) {
                    T diag_c0 = inv_2q * (g00[0] - rho_over_q * g00[1]);
                    for (int cd_idx = 0; cd_idx < 6; ++cd_idx) {
                        int w = PAIRS2_K[cd_idx], z = PAIRS2_L[cd_idx];
                        T v0 = (T)pcd.pa[w] * g01[z][0] + wq[w] * g01[z][1];
                        if (w == z) v0 += diag_c0;
                        comp_cont[4 + cd_idx] += v0;
                    }
                    if (max_ab >= 1) {
                        for (int u = 0; u < 3; ++u) {
                            T diag_c1 = inv_2q * (g10[u][0] - rho_over_q * g10[u][1]);
                            for (int cd_idx = 0; cd_idx < 6; ++cd_idx) {
                                int w = PAIRS2_K[cd_idx], z = PAIRS2_L[cd_idx];
                                T v = (T)pcd.pa[w] * g11[u][z][0] + wq[w] * g11[u][z][1];
                                if (w == z) v += diag_c1;
                                if (u == w) v += inv_2zeta * g01[z][1];
                                comp_cont[(1 + u) * 10 + 4 + cd_idx] += v;
                            }
                        }
                    }
                }
            }
        }
    }

    const T ab_vec[3] = {
        (T)(sh_a.center[0] - sh_b.center[0]),
        (T)(sh_a.center[1] - sh_b.center[1]),
        (T)(sh_a.center[2] - sh_b.center[2])
    };
    const T cd_vec[3] = {
        (T)(sh_c.center[0] - sh_d.center[0]),
        (T)(sh_c.center[1] - sh_d.center[1]),
        (T)(sh_c.center[2] - sh_d.center[2])
    };
    const int a_off = (la == 0) ? 0 : 1;
    const int c_off = (lc == 0) ? 0 : 1;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const int write_cache = (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
    const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)(na * nb)) : 0;
    const int ket_w = nc * nd;

    // Max block size for SP_LOW is 3 * 9 = 27 integrals
    double eri_blk[27];
    for (int ia = 0; ia < na; ++ia) {
        int a_idx = a_off + ia;
        double sa = sh_a.comp_scales[ia];
        for (int ib = 0; ib < nb; ++ib) {
            double sab = sa * sh_b.comp_scales[ib];
            int ab = ia * nb + ib;
            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                for (int id = 0; id < nd; ++id) {
                    T hval = hrr_bra_ket_L2(comp_cont, 10, a_idx, ib, lb, c_idx, id, ld, ab_vec, cd_vec);
                    double val = sabc * sh_d.comp_scales[id] * (double)hval;
                    int flat = ab * ket_w + ic * nd + id;
                    eri_blk[flat] = val;
                    if (write_cache) {
                        eri_cache[cache_base + (uint64_t)flat] = val;
                    }
                }
            }
        }
    }
    if (write_cache) {
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }

    const size_t nn = (size_t)n * (size_t)n;
    if (distinct) {
        for (int ia = 0; ia < na; ++ia) {
            int mu0 = sh_a.ao_offset + ia;
            size_t mu_n = (size_t)mu0 * (size_t)n;
            for (int ib = 0; ib < nb; ++ib) {
                int nu0 = sh_b.ao_offset + ib;
                size_t nu_n = (size_t)nu0 * (size_t)n;
                int ab = ia * nb + ib;
                if (do_j || do_k_sym) {
                    for (int b = 0; b < b_sym; ++b) {
                        const double* ds = d_sym + (size_t)b * nn;
                        double* oj = do_j ? (out_j + (size_t)b * nn) : nullptr;
                        double* oks = do_k_sym ? (out_ks + (size_t)b * nn) : nullptr;
                        double d_mn2 = do_j ? (2.0 * ds[mu_n + nu0]) : 0.0;
                        double acc_mn = 0.0;
                        for (int ic = 0; ic < nc; ++ic) {
                            int lam0 = sh_c.ao_offset + ic;
                            size_t lam_n = (size_t)lam0 * (size_t)n;
                            double d_nl = do_k_sym ? ds[nu_n + lam0] : 0.0;
                            double d_ml = do_k_sym ? ds[mu_n + lam0] : 0.0;
                            double acc_ml = 0.0, acc_nl = 0.0;
                            for (int id = 0; id < nd; ++id) {
                                double val = eri_blk[ab * ket_w + ic * nd + id];
                                if (val != 0.0) {
                                    int sig0 = sh_d.ao_offset + id;
                                    if (do_j) {
                                        acc_mn += val * ds[lam_n + sig0];
                                        atomicAdd(&oj[lam_n + sig0], val * d_mn2);
                                    }
                                    if (do_k_sym) {
                                        acc_ml += val * ds[nu_n + sig0];
                                        acc_nl += val * ds[mu_n + sig0];
                                        atomicAdd(&oks[mu_n + sig0], val * d_nl);
                                        atomicAdd(&oks[nu_n + sig0], val * d_ml);
                                    }
                                }
                            }
                            if (do_k_sym) {
                                if (acc_ml != 0.0) atomicAdd(&oks[mu_n + lam0], acc_ml);
                                if (acc_nl != 0.0) atomicAdd(&oks[nu_n + lam0], acc_nl);
                            }
                        }
                        if (do_j && acc_mn != 0.0) {
                            atomicAdd(&oj[mu_n + nu0], 2.0 * acc_mn);
                        }
                    }
                }
                if (do_k_anti) {
                    for (int b = 0; b < b_anti; ++b) {
                        const double* da = d_anti + (size_t)b * nn;
                        double* oka = out_ka + (size_t)b * nn;
                        for (int ic = 0; ic < nc; ++ic) {
                            int lam0 = sh_c.ao_offset + ic;
                            double d_nl = da[nu_n + lam0];
                            double d_ml = da[mu_n + lam0];
                            double acc_ml = 0.0, acc_nl = 0.0;
                            for (int id = 0; id < nd; ++id) {
                                double val = eri_blk[ab * ket_w + ic * nd + id];
                                if (val != 0.0) {
                                    int sig0 = sh_d.ao_offset + id;
                                    acc_ml += val * da[nu_n + sig0];
                                    acc_nl += val * da[mu_n + sig0];
                                    atomicAdd(&oka[mu_n + sig0], val * d_nl);
                                    atomicAdd(&oka[nu_n + sig0], val * d_ml);
                                }
                            }
                            if (acc_ml != 0.0) atomicAdd(&oka[mu_n + lam0], acc_ml);
                            if (acc_nl != 0.0) atomicAdd(&oka[nu_n + lam0], acc_nl);
                        }
                    }
                }
            }
        }
    } else {
        for (int ia = 0; ia < na; ++ia) {
            int mu0 = sh_a.ao_offset + ia;
            for (int ib = 0; ib < nb; ++ib) {
                if (i == j && ib > ia) continue;
                int nu0 = sh_b.ao_offset + ib;
                int ab = ia * nb + ib;
                for (int ic = 0; ic < nc; ++ic) {
                    int lam0 = sh_c.ao_offset + ic;
                    int id_end = (k == l) ? (ic + 1) : nd;
                    for (int id = 0; id < id_end; ++id) {
                        if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                        int sig0 = sh_d.ao_offset + id;
                        double val = eri_blk[ab * ket_w + ic * nd + id];
                        contract_quartet_element_td_canonical(
                            val, mu0, nu0, lam0, sig0, n,
                            b_sym, b_anti, do_j, do_k_sym, do_k_anti,
                            d_sym, d_anti, out_j, out_ks, out_ka
                        );
                    }
                }
            }
        }
    }
}

extern "C" __global__ void direct_eval_sp_td_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_sp_low_td_impl<double>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

extern "C" __global__ void direct_eval_sp_td_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_sp_low_td_impl<float>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

template <typename T>
__device__ __forceinline__ void eval_d_warp_td_impl(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    extern __shared__ unsigned char dyn_smem[];
    const int warp_in_blk = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp_id >= count) return;

    T* wmem = ((T*)dyn_smem) + warp_in_blk * DWARP_STRIDE_T;
    T* vrr_cont = wmem + DWARP_VRR_OFF;
    T* Va = wmem + DWARP_VA_OFF;
    T* Vc1 = wmem + DWARP_VC1_OFF;
    T* Vc2 = wmem + DWARP_VC2_OFF;
    T* Vc3 = wmem + DWARP_VC3_OFF;

    const WorkItem item = wl[warp_id];
    const DirectShellPairGpu bra = pairs[item.pi];
    const DirectShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
    const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
    const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
    const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
    const int max_ab = la + lb;
    const int max_cd = lc + ld;
    const int max_m = max_ab + max_cd;
    const int n_a = (max_ab + 1) * (max_ab + 2) * (max_ab + 3) / 6;
    const int n_c = (max_cd + 1) * (max_cd + 2) * (max_cd + 3) / 6;

    for (int idx = lane; idx < n_a * 35; idx += 32) {
        vrr_cont[idx] = (T)0.0;
    }
    __syncwarp();

    const int D_K[6] = {0, 0, 0, 1, 1, 2};
    const int D_L[6] = {0, 1, 2, 1, 2, 2};
    const T two_pi25 = (T)34.986836655249725693;

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const T p = (T)pab.gamma;
        const T inv_2p = (T)0.5 / p;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const T q = (T)pcd.gamma;
            const T inv_2q = (T)0.5 / q;
            const T zeta = p + q;
            const T inv_zeta = (T)1.0 / zeta;
            const T inv_2zeta = (T)0.5 * inv_zeta;
            const T rho = p * q * inv_zeta;
            const T w0 = (p * (T)pab.p[0] + q * (T)pcd.p[0]) * inv_zeta;
            const T w1 = (p * (T)pab.p[1] + q * (T)pcd.p[1]) * inv_zeta;
            const T w2 = (p * (T)pab.p[2] + q * (T)pcd.p[2]) * inv_zeta;
            const T wp[3] = {w0 - (T)pab.p[0], w1 - (T)pab.p[1], w2 - (T)pab.p[2]};
            const T wq[3] = {w0 - (T)pcd.p[0], w1 - (T)pcd.p[1], w2 - (T)pcd.p[2]};
            const T dx = (T)pab.p[0] - (T)pcd.p[0];
            const T dy = (T)pab.p[1] - (T)pcd.p[1];
            const T dz = (T)pab.p[2] - (T)pcd.p[2];
            const T pq2 = dx * dx + dy * dy + dz * dz;
            const T rho_over_p = rho / p;
            const T rho_over_q = rho / q;
            const T scale = two_pi25 / (p * q * sqrt(zeta)) * (T)pab.kab_w * (T)pcd.kab_w;

            if (lane == 0) {
                T boys[9];
                if constexpr (sizeof(T) == 8) {
                    boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
                } else {
                    boys_batch_table_f32(max_m, rho, pq2, (float)omega, has_omega, d_boys_f32, boys);
                }
                for (int m = 0; m <= max_m; ++m) {
                    Va[m] = scale * boys[m];
                }
            }
            __syncwarp();

            if (max_ab >= 1) {
                if (lane >= 1 && lane <= 3) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis;
                    for (int m = 0; m <= max_m - 1; ++m) {
                        Va[lane * 9 + m] = (T)pab.pa[u] * Va[m] + wp[u] * Va[m + 1];
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 2) {
                if (lane >= 4 && lane <= 9) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 2; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[lane * 9 + m] = val;
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 3) {
                if (lane >= 10 && lane <= 19) {
                    const Comp3C& c = COMPS_3C[lane];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 3; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[lane * 9 + m] = val;
                    }
                }
                __syncwarp();
            }
            if (max_ab >= 4) {
                int a = (lane >= 20) ? lane : ((lane < 3) ? (32 + lane) : -1);
                if (a >= 20 && a < 35) {
                    const Comp3C& c = COMPS_3C[a];
                    int u = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - 4; ++m) {
                        T val = (T)pab.pa[u] * Va[i1 * 9 + m] + wp[u] * Va[i1 * 9 + m + 1];
                        if (nk > 0) val += (T)nk * inv_2p * (Va[i2 * 9 + m] - rho_over_p * Va[i2 * 9 + m + 1]);
                        Va[a * 9 + m] = val;
                    }
                }
                __syncwarp();
            }

            if (lc == 0) {
                for (int a = lane; a < n_a; a += 32) {
                    vrr_cont[a * 35 + 0] += Va[a * 9 + 0];
                }
            }

            if (max_cd >= 1) {
                int m_lim = max_cd - 1;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int w = 0; w < 3; ++w) {
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Va[a * 9 + m] + wq[w] * Va[a * 9 + m + 1];
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Va[sub_a * 9 + m + 1];
                            Vc1[(w * 35 + a) * 4 + m] = val;
                        }
                        if (lc <= 1) vrr_cont[a * 35 + 1 + w] += Vc1[(w * 35 + a) * 4 + 0];
                    }
                }
                __syncwarp();
            }

            if (max_cd >= 2) {
                int m_lim = max_cd - 2;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int kl = 0; kl < 6; ++kl) {
                        int w = D_K[kl], z = D_L[kl];
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Vc1[(z * 35 + a) * 4 + m]
                                  + wq[w] * Vc1[(z * 35 + a) * 4 + m + 1];
                            if (w == z) val += inv_2q * (Va[a * 9 + m] - rho_over_q * Va[a * 9 + m + 1]);
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc1[(z * 35 + sub_a) * 4 + m + 1];
                            Vc2[(kl * 35 + a) * 3 + m] = val;
                        }
                        if (lc <= 2) vrr_cont[a * 35 + 4 + kl] += Vc2[(kl * 35 + a) * 3 + 0];
                    }
                }
                __syncwarp();
            }

            if (max_cd >= 3) {
                int m_lim = max_cd - 3;
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int cf = 0; cf < 10; ++cf) {
                        const Comp3C& cc = COMPS_3C[10 + cf];
                        int w = cc.axis;
                        int d_idx = cc.idx1 - 4;
                        int p_idx_c = cc.idx2 - 1;
                        int nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        for (int m = 0; m <= m_lim; ++m) {
                            T val = (T)pcd.pa[w] * Vc2[(d_idx * 35 + a) * 3 + m]
                                  + wq[w] * Vc2[(d_idx * 35 + a) * 3 + m + 1];
                            if (nk > 0) {
                                val += (T)nk * inv_2q * (Vc1[(p_idx_c * 35 + a) * 4 + m]
                                    - rho_over_q * Vc1[(p_idx_c * 35 + a) * 4 + m + 1]);
                            }
                            if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc2[(d_idx * 35 + sub_a) * 3 + m + 1];
                            Vc3[(cf * 35 + a) * 2 + m] = val;
                        }
                        if (lc <= 3) vrr_cont[a * 35 + 10 + cf] += Vc3[(cf * 35 + a) * 2 + 0];
                    }
                }
                __syncwarp();
            }

            if (max_cd >= 4) {
                for (int a = lane; a < n_a; a += 32) {
                    const Comp3C& ca = COMPS_3C[a];
                    for (int cg = 0; cg < 15; ++cg) {
                        const Comp3C& cc = COMPS_3C[20 + cg];
                        int w = cc.axis;
                        int f_idx = cc.idx1 - 10;
                        int d_idx = cc.idx2 - 4;
                        int nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        T val = (T)pcd.pa[w] * Vc3[(f_idx * 35 + a) * 2 + 0]
                              + wq[w] * Vc3[(f_idx * 35 + a) * 2 + 1];
                        if (nk > 0) {
                            val += (T)nk * inv_2q * (Vc2[(d_idx * 35 + a) * 3 + 0]
                                - rho_over_q * Vc2[(d_idx * 35 + a) * 3 + 1]);
                        }
                        if (sub_a >= 0) val += (T)aw * inv_2zeta * Vc3[(f_idx * 35 + sub_a) * 2 + 1];
                        vrr_cont[a * 35 + 20 + cg] += val;
                    }
                }
            }
            __syncwarp();
        }
    }

    // Va + Vc1 + Vc2 + Vc3 (2065 T elements) is now dead; reuse it as shared eri_blk (max 6^4 = 1296 <= 2065)
    T* eri_blk = wmem + DWARP_VA_OFF;
    const T ab_vec[3] = {
        (T)(sh_a.center[0] - sh_b.center[0]),
        (T)(sh_a.center[1] - sh_b.center[1]),
        (T)(sh_a.center[2] - sh_b.center[2])
    };
    const T cd_vec[3] = {
        (T)(sh_c.center[0] - sh_d.center[0]),
        (T)(sh_c.center[1] - sh_d.center[1]),
        (T)(sh_c.center[2] - sh_d.center[2])
    };
    const int a_off = (la == 0) ? 0 : ((la == 1) ? 1 : 4);
    const int c_off = (lc == 0) ? 0 : ((lc == 1) ? 1 : 4);
    const int bra_w = na * nb;
    const int ket_w = nc * nd;
    const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
    const int write_cache = (sizeof(T) == 8 && has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
    const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w) : 0;

    for (int ab = lane; ab < bra_w; ab += 32) {
        int ia = ab / nb;
        int ib = ab - ia * nb;
        int a_idx = a_off + ia;
        double sab = sh_a.comp_scales[ia] * sh_b.comp_scales[ib];

        T bra_c[35];
        if (lb == 0) {
            const T* row = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) bra_c[c] = row[c];
        } else if (lb == 1) {
            int ax = ib;
            int a_k = ADD_AXIS_3C[a_idx][ax];
            const T* r1 = vrr_cont + a_k * 35;
            const T* r2 = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) bra_c[c] = r1[c] + ab_vec[ax] * r2[c];
        } else {
            int ax1 = D_K[ib], ax2 = D_L[ib];
            int a_k = ADD_AXIS_3C[a_idx][ax1];
            int a_kl = ADD_AXIS_3C[a_k][ax2];
            int a_l = ADD_AXIS_3C[a_idx][ax2];
            const T* r_kl = vrr_cont + a_kl * 35;
            const T* r_k  = vrr_cont + a_k  * 35;
            const T* r_l  = vrr_cont + a_l  * 35;
            const T* r_0  = vrr_cont + a_idx * 35;
            for (int c = 0; c < n_c; ++c) {
                T t1 = r_kl[c] + ab_vec[ax2] * r_k[c];
                T t2 = r_l[c]  + ab_vec[ax2] * r_0[c];
                bra_c[c] = t1 + ab_vec[ax1] * t2;
            }
        }

        for (int ic = 0; ic < nc; ++ic) {
            int c_idx = c_off + ic;
            double sabc = sab * sh_c.comp_scales[ic];
            for (int id = 0; id < nd; ++id) {
                T hval = hrr_ket_L2(bra_c, c_idx, id, ld, cd_vec);
                double val = sabc * sh_d.comp_scales[id] * (double)hval;
                int flat = ab * ket_w + ic * nd + id;
                eri_blk[flat] = (T)val;
                if (write_cache) {
                    eri_cache[cache_base + (uint64_t)flat] = val;
                }
            }
        }
    }
    __syncwarp();
    if (write_cache && lane == 0) {
        cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
    }

    const size_t nn = (size_t)n * (size_t)n;
    for (int ab = lane; ab < bra_w; ab += 32) {
        int ia = ab / nb;
        int ib = ab - ia * nb;
        int mu0 = sh_a.ao_offset + ia;
        int nu0 = sh_b.ao_offset + ib;
        const T* row = eri_blk + ab * ket_w;

        if (distinct) {
            size_t mu_n = (size_t)mu0 * (size_t)n;
            size_t nu_n = (size_t)nu0 * (size_t)n;
            if (do_j || do_k_sym) {
                for (int b = 0; b < b_sym; ++b) {
                    const double* ds = d_sym + (size_t)b * nn;
                    double* oj = do_j ? (out_j + (size_t)b * nn) : nullptr;
                    double* oks = do_k_sym ? (out_ks + (size_t)b * nn) : nullptr;
                    double d_mn2 = do_j ? (2.0 * ds[mu_n + nu0]) : 0.0;
                    double acc_mn = 0.0;
                    for (int ic = 0; ic < nc; ++ic) {
                        int lam = sh_c.ao_offset + ic;
                        size_t lam_n = (size_t)lam * (size_t)n;
                        double d_nl = do_k_sym ? ds[nu_n + lam] : 0.0;
                        double d_ml = do_k_sym ? ds[mu_n + lam] : 0.0;
                        double acc_ml = 0.0, acc_nl = 0.0;
                        for (int id = 0; id < nd; ++id) {
                            double val = (double)row[ic * nd + id];
                            if (val != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                if (do_j) {
                                    acc_mn += val * ds[lam_n + sig];
                                    atomicAdd(&oj[lam_n + sig], val * d_mn2);
                                }
                                if (do_k_sym) {
                                    acc_ml += val * ds[nu_n + sig];
                                    acc_nl += val * ds[mu_n + sig];
                                    atomicAdd(&oks[mu_n + sig], val * d_nl);
                                    atomicAdd(&oks[nu_n + sig], val * d_ml);
                                }
                            }
                        }
                        if (do_k_sym) {
                            if (acc_ml != 0.0) atomicAdd(&oks[mu_n + lam], acc_ml);
                            if (acc_nl != 0.0) atomicAdd(&oks[nu_n + lam], acc_nl);
                        }
                    }
                    if (do_j && acc_mn != 0.0) {
                        atomicAdd(&oj[mu_n + nu0], 2.0 * acc_mn);
                    }
                }
            }
            if (do_k_anti) {
                for (int b = 0; b < b_anti; ++b) {
                    const double* da = d_anti + (size_t)b * nn;
                    double* oka = out_ka + (size_t)b * nn;
                    for (int ic = 0; ic < nc; ++ic) {
                        int lam = sh_c.ao_offset + ic;
                        double d_nl = da[nu_n + lam];
                        double d_ml = da[mu_n + lam];
                        double acc_ml = 0.0, acc_nl = 0.0;
                        for (int id = 0; id < nd; ++id) {
                            double val = (double)row[ic * nd + id];
                            if (val != 0.0) {
                                int sig = sh_d.ao_offset + id;
                                acc_ml += val * da[nu_n + sig];
                                acc_nl += val * da[mu_n + sig];
                                atomicAdd(&oka[mu_n + sig], val * d_nl);
                                atomicAdd(&oka[nu_n + sig], val * d_ml);
                            }
                        }
                        if (acc_ml != 0.0) atomicAdd(&oka[mu_n + lam], acc_ml);
                        if (acc_nl != 0.0) atomicAdd(&oka[nu_n + lam], acc_nl);
                    }
                }
            }
        } else {
            if (i == j && ib > ia) continue;
            for (int ic = 0; ic < nc; ++ic) {
                int id_end = (k == l) ? (ic + 1) : nd;
                int lam0 = sh_c.ao_offset + ic;
                for (int id = 0; id < id_end; ++id) {
                    if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                    int sig0 = sh_d.ao_offset + id;
                    double val = (double)row[ic * nd + id];
                    contract_quartet_element_td_canonical(
                        val, mu0, nu0, lam0, sig0, n,
                        b_sym, b_anti, do_j, do_k_sym, do_k_anti,
                        d_sym, d_anti, out_j, out_ks, out_ka
                    );
                }
            }
        }
    }
    __syncwarp();
}

extern "C" __global__ void direct_eval_d_td_f64_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_d_warp_td_impl<double>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

extern "C" __global__ void direct_eval_d_td_f32_kernel(
    const DirectShellGpu* __restrict__ shells, const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims, const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache, uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl, int count, int n,
    int b_sym, int b_anti, int do_j, int do_k_sym, int do_k_anti,
    double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const float* __restrict__ d_boys_f32,
    const double* __restrict__ d_sym, const double* __restrict__ d_anti,
    double* __restrict__ out_j, double* __restrict__ out_ks, double* __restrict__ out_ka
) {
    eval_d_warp_td_impl<float>(
        shells, pairs, prims, pair_prefix, eri_cache, cache_valid, wl, count, n,
        b_sym, b_anti, do_j, do_k_sym, do_k_anti, omega, has_omega, d_boys_f64, d_boys_f32,
        d_sym, d_anti, out_j, out_ks, out_ka
    );
}

extern "C" __global__ void direct_eval_f_td_f64_kernel(
    const DirectShellGpu* __restrict__ shells,
    const DirectShellPairGpu* __restrict__ pairs,
    const DirectPrimPairGpu* __restrict__ prims,
    const uint64_t* __restrict__ pair_prefix,
    double* __restrict__ eri_cache,
    uint8_t* __restrict__ cache_valid,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int b_sym, int b_anti,
    int do_j, int do_k_sym, int do_k_anti,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const Comp6* __restrict__ comps6,
    double* __restrict__ f_vrr_scratch, // gridDim.x * 84 * 84 * 13
    const double* __restrict__ d_sym,
    const double* __restrict__ d_anti,
    double* __restrict__ out_j,
    double* __restrict__ out_ks,
    double* __restrict__ out_ka
) {
    extern __shared__ double vrr_cont[];
    const int tid = (int)threadIdx.x;
    double* V = f_vrr_scratch + (size_t)blockIdx.x * (size_t)(84 * 84 * 13);
    const int L_START[8] = {0, 1, 4, 10, 20, 35, 56, 84};
    const size_t nn = (size_t)n * (size_t)n;

    for (int qid = (int)blockIdx.x; qid < count; qid += (int)gridDim.x) {
        const WorkItem item = wl[qid];
        const DirectShellPairGpu bra = pairs[item.pi];
        const DirectShellPairGpu ket = pairs[item.qi];
        const int i = bra.shell_i, j = bra.shell_j;
        const int k = ket.shell_i, l = ket.shell_j;
        const DirectShellGpu sh_a = shells[i], sh_b = shells[j];
        const DirectShellGpu sh_c = shells[k], sh_d = shells[l];
        const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
        const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
        const int max_ab = la + lb;
        const int max_cd = lc + ld;
        const int max_m = max_ab + max_cd;
        const int n_a = L_START[max_ab + 1];
        const int n_c = L_START[max_cd + 1];

        for (int idx = tid; idx < n_a * 84; idx += (int)blockDim.x) {
            vrr_cont[idx] = 0.0;
        }
        __syncthreads();

        const double two_pi25 = 34.986836655249725693;
        for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
            const DirectPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
            const double p = pab.gamma;
            const double inv_2p = 0.5 / p;
            for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
                const DirectPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
                const double q = pcd.gamma;
                const double inv_2q = 0.5 / q;
                const double zeta = p + q;
                const double inv_zeta = 1.0 / zeta;
                const double inv_2zeta = 0.5 * inv_zeta;
                const double rho = p * q * inv_zeta;
                const double w0 = (p * pab.p[0] + q * pcd.p[0]) * inv_zeta;
                const double w1 = (p * pab.p[1] + q * pcd.p[1]) * inv_zeta;
                const double w2 = (p * pab.p[2] + q * pcd.p[2]) * inv_zeta;
                const double wp[3] = {w0 - pab.p[0], w1 - pab.p[1], w2 - pab.p[2]};
                const double wq[3] = {w0 - pcd.p[0], w1 - pcd.p[1], w2 - pcd.p[2]};
                const double dx = pab.p[0] - pcd.p[0];
                const double dy = pab.p[1] - pcd.p[1];
                const double dz = pab.p[2] - pcd.p[2];
                const double pq2 = dx * dx + dy * dy + dz * dz;
                const double rho_over_p = rho / p;
                const double rho_over_q = rho / q;
                const double scale = two_pi25 / (p * q * sqrt(zeta)) * pab.kab_w * pcd.kab_w;

                if (tid == 0) {
                    double boys[13];
                    boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
                    for (int m = 0; m <= max_m; ++m) V[m] = scale * boys[m];
                }
                __syncthreads();

                for (int la_cur = 1; la_cur <= max_ab; ++la_cur) {
                    int a_beg = L_START[la_cur], a_end = L_START[la_cur + 1];
                    for (int a = a_beg + tid; a < a_end; a += (int)blockDim.x) {
                        const Comp6 ca = comps6[a];
                        int u = ca.axis, i1 = ca.idx1, i2 = ca.idx2, nk = ca.n_k;
                        for (int m = 0; m <= max_m - la_cur; ++m) {
                            double val = pab.pa[u] * V[(size_t)i1 * 84 * 13 + m]
                                       + wp[u]     * V[(size_t)i1 * 84 * 13 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2p * (V[(size_t)i2 * 84 * 13 + m]
                                    - rho_over_p * V[(size_t)i2 * 84 * 13 + m + 1]);
                            }
                            V[(size_t)a * 84 * 13 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                for (int lc_cur = 1; lc_cur <= max_cd; ++lc_cur) {
                    int c_beg = L_START[lc_cur], c_end = L_START[lc_cur + 1];
                    int n_c_lvl = c_end - c_beg;
                    int total_ac = n_a * n_c_lvl;
                    for (int ac = tid; ac < total_ac; ac += (int)blockDim.x) {
                        int a = ac / n_c_lvl;
                        int c = c_beg + (ac - a * n_c_lvl);
                        const Comp6 ca = comps6[a];
                        const Comp6 cc = comps6[c];
                        int w = cc.axis, c1 = cc.idx1, c2 = cc.idx2, nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        int m_top = max_m - ca.L - lc_cur;
                        for (int m = 0; m <= m_top; ++m) {
                            double val = pcd.pa[w] * V[((size_t)a * 84 + c1) * 13 + m]
                                       + wq[w]     * V[((size_t)a * 84 + c1) * 13 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2q * (V[((size_t)a * 84 + c2) * 13 + m]
                                    - rho_over_q * V[((size_t)a * 84 + c2) * 13 + m + 1]);
                            }
                            if (sub_a >= 0) {
                                val += (double)aw * inv_2zeta * V[((size_t)sub_a * 84 + c1) * 13 + m + 1];
                            }
                            V[((size_t)a * 84 + c) * 13 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                for (int idx = tid; idx < n_a * n_c; idx += (int)blockDim.x) {
                    int a = idx / n_c;
                    int c = idx - a * n_c;
                    vrr_cont[a * 84 + c] += V[((size_t)a * 84 + c) * 13 + 0];
                }
                __syncthreads();
            }
        }

        const double ab_vec[3] = {
            sh_a.center[0] - sh_b.center[0],
            sh_a.center[1] - sh_b.center[1],
            sh_a.center[2] - sh_b.center[2]
        };
        const double cd_vec[3] = {
            sh_c.center[0] - sh_d.center[0],
            sh_c.center[1] - sh_d.center[1],
            sh_c.center[2] - sh_d.center[2]
        };
        const int a_off = L_START[la], b_off = L_START[lb];
        const int c_off = L_START[lc], d_off = L_START[ld];
        const int bra_w = na * nb, ket_w = nc * nd;
        const int distinct = (i != j && k != l && (i != k || j != l)) ? 1 : 0;
        const int write_cache = (has_omega == 0 && (int)item.qi < bra.cached_kets && eri_cache != nullptr) ? 1 : 0;
        const uint64_t cache_base = write_cache ? (bra.cache_data_offset + pair_prefix[item.qi] * (uint64_t)bra_w) : 0;

        for (int ab = tid; ab < bra_w; ab += (int)blockDim.x) {
            int ia = ab / nb, ib = ab - ia * nb;
            int mu0 = sh_a.ao_offset + ia, nu0 = sh_b.ao_offset + ib;
            const Comp6 cb = comps6[b_off + ib];
            double sab = sh_a.comp_scales[ia] * sh_b.comp_scales[ib];
            double row[100];
            for (int ic = 0; ic < nc; ++ic) {
                int c_idx = c_off + ic;
                double sabc = sab * sh_c.comp_scales[ic];
                for (int id = 0; id < nd; ++id) {
                    const Comp6 cd_c = comps6[d_off + id];
                    double val = sabc * sh_d.comp_scales[id] * hrr_bra_6_dev(
                        comps6, vrr_cont, a_off + ia, cb.x, cb.y, cb.z,
                        c_idx, cd_c.x, cd_c.y, cd_c.z, ab_vec, cd_vec
                    );
                    int cd = ic * nd + id;
                    row[cd] = val;
                    if (write_cache) {
                        eri_cache[cache_base + (uint64_t)(ab * ket_w + cd)] = val;
                    }
                }
            }

            if (distinct) {
                size_t mu_n = (size_t)mu0 * (size_t)n;
                size_t nu_n = (size_t)nu0 * (size_t)n;
                if (do_j || do_k_sym) {
                    for (int b = 0; b < b_sym; ++b) {
                        const double* ds = d_sym + (size_t)b * nn;
                        double* oj = do_j ? (out_j + (size_t)b * nn) : nullptr;
                        double* oks = do_k_sym ? (out_ks + (size_t)b * nn) : nullptr;
                        double d_mn2 = do_j ? (2.0 * ds[mu_n + nu0]) : 0.0;
                        double acc_mn = 0.0;
                        for (int ic = 0; ic < nc; ++ic) {
                            int lam0 = sh_c.ao_offset + ic;
                            size_t lam_n = (size_t)lam0 * (size_t)n;
                            double d_nl = do_k_sym ? ds[nu_n + lam0] : 0.0;
                            double d_ml = do_k_sym ? ds[mu_n + lam0] : 0.0;
                            double acc_ml = 0.0, acc_nl = 0.0;
                            for (int id = 0; id < nd; ++id) {
                                double val = row[ic * nd + id];
                                if (val != 0.0) {
                                    int sig0 = sh_d.ao_offset + id;
                                    if (do_j) {
                                        acc_mn += val * ds[lam_n + sig0];
                                        atomicAdd(&oj[lam_n + sig0], val * d_mn2);
                                    }
                                    if (do_k_sym) {
                                        acc_ml += val * ds[nu_n + sig0];
                                        acc_nl += val * ds[mu_n + sig0];
                                        atomicAdd(&oks[mu_n + sig0], val * d_nl);
                                        atomicAdd(&oks[nu_n + sig0], val * d_ml);
                                    }
                                }
                            }
                            if (do_k_sym) {
                                if (acc_ml != 0.0) atomicAdd(&oks[mu_n + lam0], acc_ml);
                                if (acc_nl != 0.0) atomicAdd(&oks[nu_n + lam0], acc_nl);
                            }
                        }
                        if (do_j && acc_mn != 0.0) {
                            atomicAdd(&oj[mu_n + nu0], 2.0 * acc_mn);
                        }
                    }
                }
                if (do_k_anti) {
                    for (int b = 0; b < b_anti; ++b) {
                        const double* da = d_anti + (size_t)b * nn;
                        double* oka = out_ka + (size_t)b * nn;
                        for (int ic = 0; ic < nc; ++ic) {
                            int lam0 = sh_c.ao_offset + ic;
                            double d_nl = da[nu_n + lam0];
                            double d_ml = da[mu_n + lam0];
                            double acc_ml = 0.0, acc_nl = 0.0;
                            for (int id = 0; id < nd; ++id) {
                                double val = row[ic * nd + id];
                                if (val != 0.0) {
                                    int sig0 = sh_d.ao_offset + id;
                                    acc_ml += val * da[nu_n + sig0];
                                    acc_nl += val * da[mu_n + sig0];
                                    atomicAdd(&oka[mu_n + sig0], val * d_nl);
                                    atomicAdd(&oka[nu_n + sig0], val * d_ml);
                                }
                            }
                            if (acc_ml != 0.0) atomicAdd(&oka[mu_n + lam0], acc_ml);
                            if (acc_nl != 0.0) atomicAdd(&oka[nu_n + lam0], acc_nl);
                        }
                    }
                }
            } else {
                if (i == j && ib > ia) continue;
                for (int ic = 0; ic < nc; ++ic) {
                    int id_end = (k == l) ? (ic + 1) : nd;
                    int lam0 = sh_c.ao_offset + ic;
                    for (int id = 0; id < id_end; ++id) {
                        if (i == k && j == l && (ia < ic || (ia == ic && ib < id))) continue;
                        int sig0 = sh_d.ao_offset + id;
                        double val = row[ic * nd + id];
                        contract_quartet_element_td_canonical(
                            val, mu0, nu0, lam0, sig0, n,
                            b_sym, b_anti, do_j, do_k_sym, do_k_anti,
                            d_sym, d_anti, out_j, out_ks, out_ka
                        );
                    }
                }
            }
        }
        __syncthreads();
        if (write_cache && tid == 0) {
            cache_valid[(size_t)bra.cache_valid_offset + (size_t)item.qi] = 1;
        }
        __syncthreads();
    }
}

// ============================================================================
// M5b Analytical Gradient CUDA Kernels (Vxc Force, 3c RI-J Grad, 4c ERI Grad)
// ============================================================================

#define VXC_GRAD_MAX_ACTIVE_AO 256

extern "C" __global__ void vxc_grad_force_kernel(
    const AoShellGpu* __restrict__ shells,
    const AoPrimGpu* __restrict__ prims,
    const AoCompGpu* __restrict__ comps,
    const GridPtGpu* __restrict__ pts,
    const int* __restrict__ cand_offsets,
    const int* __restrict__ cand_shells,
    const double* __restrict__ pa_mat,
    const double* __restrict__ pb_mat,
    const double* __restrict__ wxc_in, // [n_pts * 8]: w*vra, w*uax, w*uay, w*uaz, w*vrb, w*ubx, w*uby, w*ubz
    double* __restrict__ grad_out,     // [n_atoms * 3]
    int n_pts,
    int n_ao,
    int is_rks
) {
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= n_pts) return;

    size_t wo = (size_t)g * 8;
    const double w_vra = wxc_in[wo];
    const double w_uax = wxc_in[wo + 1];
    const double w_uay = wxc_in[wo + 2];
    const double w_uaz = wxc_in[wo + 3];
    const double w_vrb = wxc_in[wo + 4];
    const double w_ubx = wxc_in[wo + 5];
    const double w_uby = wxc_in[wo + 6];
    const double w_ubz = wxc_in[wo + 7];
    if (w_vra == 0.0 && w_uax == 0.0 && w_uay == 0.0 && w_uaz == 0.0
        && w_vrb == 0.0 && w_ubx == 0.0 && w_uby == 0.0 && w_ubz == 0.0) {
        return;
    }

    const GridPtGpu pt = pts[g];
    const int c_start = cand_offsets[pt.batch_id];
    const int c_end = cand_offsets[pt.batch_id + 1];

    int act_mu[VXC_GRAD_MAX_ACTIVE_AO];
    double act_val[VXC_GRAD_MAX_ACTIVE_AO];
    double act_gx[VXC_GRAD_MAX_ACTIVE_AO];
    double act_gy[VXC_GRAD_MAX_ACTIVE_AO];
    double act_gz[VXC_GRAD_MAX_ACTIVE_AO];
    int n_act = 0;
    bool overflow = false;

    for (int ci = c_start; ci < c_end && !overflow; ++ci) {
        const AoShellGpu sh = shells[cand_shells[ci]];
        const double dx = pt.x - sh.center[0];
        const double dy = pt.y - sh.center[1];
        const double dz = pt.z - sh.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > sh.r2_cutoff) continue;

        double r0 = 0.0, r1 = 0.0, r2_rad = 0.0;
        for (int pi = 0; pi < sh.n_prims; ++pi) {
            const AoPrimGpu pr = prims[sh.prim_start + pi];
            const double e = pr.weighted_norm * exp(-pr.exponent * r2);
            r0 += e;
            r1 -= 2.0 * pr.exponent * e;
            r2_rad += 4.0 * pr.exponent * pr.exponent * e;
        }
        if (fabs(r0) < 1e-18 && fabs(r1) < 1e-18 && fabs(r2_rad) < 1e-18) continue;

        for (int ic = 0; ic < sh.n_funcs; ++ic) {
            const AoCompGpu cp = comps[sh.comp_start + ic];
            double px, dpx, d2px, py, dpy, d2py, pz, dpz, d2pz;
            pow_d0_d1_d2_dev(dx, cp.lx, px, dpx, d2px);
            pow_d0_d1_d2_dev(dy, cp.ly, py, dpy, d2py);
            pow_d0_d1_d2_dev(dz, cp.lz, pz, dpz, d2pz);
            const double m = px * py * pz;
            const double val = cp.scale * m * r0;
            const double gx = cp.scale * (dpx * py * pz * r0 + m * dx * r1);
            const double gy = cp.scale * (px * dpy * pz * r0 + m * dy * r1);
            const double gz = cp.scale * (px * py * dpz * r0 + m * dz * r1);
            if (n_act >= VXC_GRAD_MAX_ACTIVE_AO) {
                overflow = true;
                break;
            }
            act_mu[n_act] = sh.ao_offset + ic;
            act_val[n_act] = val;
            act_gx[n_act] = gx;
            act_gy[n_act] = gy;
            act_gz[n_act] = gz;
            n_act++;
        }
    }

    if (n_act == 0 && !overflow) return;

    double sum_ao_force[3] = {0.0, 0.0, 0.0};

    for (int ci = c_start; ci < c_end; ++ci) {
        const AoShellGpu sh = shells[cand_shells[ci]];
        const double dx = pt.x - sh.center[0];
        const double dy = pt.y - sh.center[1];
        const double dz = pt.z - sh.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > sh.r2_cutoff) continue;

        double r0 = 0.0, r1 = 0.0, r2_rad = 0.0;
        for (int pi = 0; pi < sh.n_prims; ++pi) {
            const AoPrimGpu pr = prims[sh.prim_start + pi];
            const double e = pr.weighted_norm * exp(-pr.exponent * r2);
            r0 += e;
            r1 -= 2.0 * pr.exponent * e;
            r2_rad += 4.0 * pr.exponent * pr.exponent * e;
        }
        if (fabs(r0) < 1e-18 && fabs(r1) < 1e-18 && fabs(r2_rad) < 1e-18) continue;

        for (int ic = 0; ic < sh.n_funcs; ++ic) {
            const AoCompGpu cp = comps[sh.comp_start + ic];
            double px, dpx, d2px, py, dpy, d2py, pz, dpz, d2pz;
            pow_d0_d1_d2_dev(dx, cp.lx, px, dpx, d2px);
            pow_d0_d1_d2_dev(dy, cp.ly, py, dpy, d2py);
            pow_d0_d1_d2_dev(dz, cp.lz, pz, dpz, d2pz);
            const double m = px * py * pz;
            const double dm0 = dpx * py * pz;
            const double dm1 = px * dpy * pz;
            const double dm2 = px * py * dpz;

            const double gx = cp.scale * (dm0 * r0 + m * dx * r1);
            const double gy = cp.scale * (dm1 * r0 + m * dy * r1);
            const double gz = cp.scale * (dm2 * r0 + m * dz * r1);

            const double h_xx = cp.scale * (d2px * py * pz * r0 + (2.0 * dm0 * dx + m) * r1 + m * dx * dx * r2_rad);
            const double h_xy = cp.scale * (dpx * dpy * pz * r0 + (dm0 * dy + dm1 * dx) * r1 + m * dx * dy * r2_rad);
            const double h_xz = cp.scale * (dpx * py * dpz * r0 + (dm0 * dz + dm2 * dx) * r1 + m * dx * dz * r2_rad);
            const double h_yy = cp.scale * (px * d2py * pz * r0 + (2.0 * dm1 * dy + m) * r1 + m * dy * dy * r2_rad);
            const double h_yz = cp.scale * (px * dpy * dpz * r0 + (dm1 * dz + dm2 * dy) * r1 + m * dy * dz * r2_rad);
            const double h_zz = cp.scale * (px * py * d2pz * r0 + (2.0 * dm2 * dz + m) * r1 + m * dz * dz * r2_rad);

            const int mu = sh.ao_offset + ic;
            const double* pa_row = pa_mat + (size_t)mu * (size_t)n_ao;
            const double* pb_row = is_rks ? nullptr : (pb_mat + (size_t)mu * (size_t)n_ao);

            double sa = 0.0, gpax = 0.0, gpay = 0.0, gpaz = 0.0;
            double sb = 0.0, gpbx = 0.0, gpby = 0.0, gpbz = 0.0;

            if (!overflow) {
                for (int j = 0; j < n_act; ++j) {
                    int mj = act_mu[j];
                    double pa_v = pa_row[mj];
                    double jv = act_val[j], jgx = act_gx[j], jgy = act_gy[j], jgz = act_gz[j];
                    sa += pa_v * jv;
                    gpax += pa_v * jgx;
                    gpay += pa_v * jgy;
                    gpaz += pa_v * jgz;
                    if (!is_rks) {
                        double pb_v = pb_row[mj];
                        sb += pb_v * jv;
                        gpbx += pb_v * jgx;
                        gpby += pb_v * jgy;
                        gpbz += pb_v * jgz;
                    }
                }
            } else {
                for (int cj = c_start; cj < c_end; ++cj) {
                    const AoShellGpu sh_j = shells[cand_shells[cj]];
                    const double jdx = pt.x - sh_j.center[0];
                    const double jdy = pt.y - sh_j.center[1];
                    const double jdz = pt.z - sh_j.center[2];
                    const double jr2 = jdx * jdx + jdy * jdy + jdz * jdz;
                    if (jr2 > sh_j.r2_cutoff) continue;
                    double jr0 = 0.0, jr1 = 0.0, jr2_rad = 0.0;
                    for (int pi = 0; pi < sh_j.n_prims; ++pi) {
                        const AoPrimGpu pr = prims[sh_j.prim_start + pi];
                        const double e = pr.weighted_norm * exp(-pr.exponent * jr2);
                        jr0 += e;
                        jr1 -= 2.0 * pr.exponent * e;
                        jr2_rad += 4.0 * pr.exponent * pr.exponent * e;
                    }
                    if (fabs(jr0) < 1e-18 && fabs(jr1) < 1e-18 && fabs(jr2_rad) < 1e-18) continue;
                    for (int jc = 0; jc < sh_j.n_funcs; ++jc) {
                        const AoCompGpu jcp = comps[sh_j.comp_start + jc];
                        double jpx, jdpx, jd2px, jpy, jdpy, jd2py, jpz, jdpz, jd2pz;
                        pow_d0_d1_d2_dev(jdx, jcp.lx, jpx, jdpx, jd2px);
                        pow_d0_d1_d2_dev(jdy, jcp.ly, jpy, jdpy, jd2py);
                        pow_d0_d1_d2_dev(jdz, jcp.lz, jpz, jdpz, jd2pz);
                        const double jm = jpx * jpy * jpz;
                        const double jv = jcp.scale * jm * jr0;
                        const double jgx = jcp.scale * (jdpx * jpy * jpz * jr0 + jm * jdx * jr1);
                        const double jgy = jcp.scale * (jpx * jdpy * jpz * jr0 + jm * jdy * jr1);
                        const double jgz = jcp.scale * (jpx * jpy * jdpz * jr0 + jm * jdz * jr1);
                        int mj = sh_j.ao_offset + jc;
                        double pa_v = pa_row[mj];
                        sa += pa_v * jv;
                        gpax += pa_v * jgx;
                        gpay += pa_v * jgy;
                        gpaz += pa_v * jgz;
                        if (!is_rks) {
                            double pb_v = pb_row[mj];
                            sb += pb_v * jv;
                            gpbx += pb_v * jgx;
                            gpby += pb_v * jgy;
                            gpbz += pb_v * jgz;
                        }
                    }
                }
            }

            if (is_rks) {
                sb = sa;
                gpbx = gpax;
                gpby = gpay;
                gpbz = gpaz;
            }

            const double ua_gpa = w_uax * gpax + w_uay * gpay + w_uaz * gpaz;
            const double ub_gpb = w_ubx * gpbx + w_uby * gpby + w_ubz * gpbz;

            const double g_arr[3] = {gx, gy, gz};
            const double h_rows[3][3] = {
                {h_xx, h_xy, h_xz},
                {h_xy, h_yy, h_yz},
                {h_xz, h_yz, h_zz}
            };
            for (int q = 0; q < 3; ++q) {
                double dphi = g_arr[q];
                double ta = w_vra * dphi * sa
                          + (w_uax * h_rows[q][0] + w_uay * h_rows[q][1] + w_uaz * h_rows[q][2]) * sa
                          + dphi * ua_gpa;
                double tb = w_vrb * dphi * sb
                          + (w_ubx * h_rows[q][0] + w_uby * h_rows[q][1] + w_ubz * h_rows[q][2]) * sb
                          + dphi * ub_gpb;
                double f_aq = -2.0 * (ta + tb);
                atomicAdd(&grad_out[(size_t)sh.atom_index * 3 + (size_t)q], f_aq);
                sum_ao_force[q] += f_aq;
            }
        }
    }

    for (int q = 0; q < 3; ++q) {
        atomicAdd(&grad_out[(size_t)pt.host_atom * 3 + (size_t)q], -sum_ao_force[q]);
    }
}

extern "C" __global__ void rij_grad_v_kernel(
    const Grad3cAuxShellGpu* __restrict__ aux_shells,
    const Grad3cAuxPrimGpu* __restrict__ aux_prims,
    const Grad3cShellPairGpu* __restrict__ shell_pairs,
    const Grad3cPrimPairGpu* __restrict__ prim_pairs,
    const double* __restrict__ p_mat,
    double* __restrict__ v_out,
    int n_aux_shells,
    int n_shell_pairs,
    int n_ao
) {
    int sp_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int p_idx = (int)blockIdx.y;
    if (sp_idx >= n_shell_pairs || p_idx >= n_aux_shells) return;

    const Grad3cAuxShellGpu aux = aux_shells[p_idx];
    const Grad3cShellPairGpu sp = shell_pairs[sp_idx];
    if (aux.bound * sp.q_ij < 1e-12) return;

    const int lp = aux.lp;
    const int li = sp.li;
    const int lj = sp.lj;
    const int np = (lp == 0) ? 1 : ((lp == 1) ? 3 : ((lp == 2) ? 6 : 10));
    const int ni = (li == 0) ? 1 : ((li == 1) ? 3 : 6);
    const int nj = (lj == 0) ? 1 : ((lj == 1) ? 3 : 6);

    double acc[360];
    const int blk_len = np * ni * nj;
    for (int k = 0; k < blk_len; ++k) acc[k] = 0.0;

    const int max_m = lp + li + lj;
    const int l_ket = li + lj;
    const int n_ket = (l_ket + 1) * (l_ket + 2) * (l_ket + 3) / 6;
    const double rp0 = aux.center[0];
    const double rp1 = aux.center[1];
    const double rp2 = aux.center[2];
    const int D_K[6] = {0, 0, 0, 1, 1, 2};
    const int D_L[6] = {0, 1, 2, 1, 2, 2};
    const int comp_off_i = (li == 0) ? 0 : ((li == 1) ? 1 : 4);

    for (int pp_i = 0; pp_i < aux.n_prims; ++pp_i) {
        const Grad3cAuxPrimGpu ap = aux_prims[aux.prim_start + pp_i];
        const double p = ap.gamma_p;
        const double inv_2p = 0.5 / p;

        for (int ij_i = 0; ij_i < sp.n_prims; ++ij_i) {
            const Grad3cPrimPairGpu pp = prim_pairs[sp.prim_start + ij_i];
            const double q = pp.q;
            const double zeta = p + q;
            const double inv_zeta = 1.0 / zeta;
            const double inv_2zeta = 0.5 * inv_zeta;
            const double rho = p * q * inv_zeta;
            const double dx = rp0 - pp.p_center[0];
            const double dy = rp1 - pp.p_center[1];
            const double dz = rp2 - pp.p_center[2];
            const double pq2 = dx * dx + dy * dy + dz * dz;

            double boys[8];
            boys_batch_dev(max_m, rho * pq2, boys);
            const double pref = pp.two_pi25_kab_w_over_q * ap.cp_lead / (p * sqrt(zeta));

            const double wx = (p * rp0 + q * pp.p_center[0]) * inv_zeta;
            const double wy = (p * rp1 + q * pp.p_center[1]) * inv_zeta;
            const double wz = (p * rp2 + q * pp.p_center[2]) * inv_zeta;
            const double wp[3] = {wx - rp0, wy - rp1, wz - rp2};
            const double wq[3] = {wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]};
            const double inv_2q = 0.5 / q;
            const double rho_over_q = rho / q;
            const double rho_over_p = rho / p;

            double V0[35][8];
            for (int m = 0; m <= max_m; ++m) V0[0][m] = pref * boys[m];
            for (int idx = 1; idx < n_ket; ++idx) {
                const Comp3C& c = COMPS_3C[idx];
                int axis = c.axis, idx1 = c.idx1, idx2 = c.idx2, n_k = c.n_k;
                for (int m = 0; m <= max_m - c.L; ++m) {
                    double val = pp.qc[axis] * V0[idx1][m] + wq[axis] * V0[idx1][m + 1];
                    if (n_k > 0) val += (double)n_k * inv_2q * (V0[idx2][m] - rho_over_q * V0[idx2][m + 1]);
                    V0[idx][m] = val;
                }
            }

            double V_P[10][35];
            if (lp == 0) {
                for (int idx = 0; idx < n_ket; ++idx) V_P[0][idx] = V0[idx][0];
            } else if (lp == 1) {
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V0[idx][1];
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V0[sub_idx][1];
                        }
                        V_P[u][idx] = val;
                    }
                }
            } else if (lp == 2) {
                double V1[3][35][2];
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        for (int m_sub = 0; m_sub < 2; ++m_sub) {
                            double val = wp[u] * V0[idx][m_sub + 1];
                            if (sub_idx >= 0) val += (double)cu * inv_2zeta * V0[sub_idx][m_sub + 1];
                            V1[u][idx][m_sub] = val;
                        }
                    }
                }
                for (int kl = 0; kl < 6; ++kl) {
                    int u = D_K[kl], v = D_L[kl];
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V1[v][idx][1];
                        if (u == v) val += inv_2p * (V0[idx][0] - rho_over_p * V0[idx][1]);
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[v][sub_idx][1];
                        }
                        V_P[kl][idx] = val;
                    }
                }
            } else {
                double V1[3][35][3];
                for (int u = 0; u < 3; ++u) {
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        for (int m_sub = 0; m_sub < 3; ++m_sub) {
                            double val = wp[u] * V0[idx][m_sub + 1];
                            if (sub_idx >= 0) val += (double)cu * inv_2zeta * V0[sub_idx][m_sub + 1];
                            V1[u][idx][m_sub] = val;
                        }
                    }
                }
                double V_D[6][35];
                for (int kl = 0; kl < 6; ++kl) {
                    int u = D_K[kl], v = D_L[kl];
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V1[v][idx][2];
                        if (u == v) val += inv_2p * (V0[idx][1] - rho_over_p * V0[idx][2]);
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[v][sub_idx][2];
                        }
                        V_D[kl][idx] = val;
                    }
                }
                for (int pf = 0; pf < 10; ++pf) {
                    const Comp3C& cp_f = COMPS_3C[10 + pf];
                    int u = cp_f.axis, d_idx = cp_f.idx1 - 4, p_idx_v1 = cp_f.idx2 - 1, n_k = cp_f.n_k;
                    for (int idx = 0; idx < n_ket; ++idx) {
                        const Comp3C& c = COMPS_3C[idx];
                        double val = wp[u] * V_D[d_idx][idx];
                        if (n_k > 0) val += (double)n_k * inv_2p * (V1[p_idx_v1][idx][0] - rho_over_p * V1[p_idx_v1][idx][1]);
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V_D[d_idx][sub_idx];
                        }
                        V_P[pf][idx] = val;
                    }
                }
            }

            for (int pf = 0; pf < np; ++pf) {
                int p_blk = pf * ni * nj;
                for (int a = 0; a < ni; ++a) {
                    int a_idx = comp_off_i + a;
                    for (int b = 0; b < nj; ++b) {
                        double val;
                        if (lj == 0) {
                            val = V_P[pf][a_idx];
                        } else if (lj == 1) {
                            int k = b;
                            int a_k = ADD_AXIS_3C[a_idx][k];
                            val = V_P[pf][a_k] + sp.ab_vec[k] * V_P[pf][a_idx];
                        } else {
                            int k = D_K[b], l = D_L[b];
                            int a_k = ADD_AXIS_3C[a_idx][k];
                            int a_kl = ADD_AXIS_3C[a_k][l];
                            int a_l = ADD_AXIS_3C[a_idx][l];
                            double t1 = V_P[pf][a_kl] + sp.ab_vec[l] * V_P[pf][a_k];
                            double t2 = V_P[pf][a_l] + sp.ab_vec[l] * V_P[pf][a_idx];
                            val = t1 + sp.ab_vec[k] * t2;
                        }
                        acc[p_blk + a * nj + b] += val;
                    }
                }
            }
        }
    }

    const double deg = sp.same_shell ? 1.0 : 2.0;
    for (int pf = 0; pf < np; ++pf) {
        double p_sum = 0.0;
        int p_blk = pf * ni * nj;
        for (int mu = 0; mu < ni; ++mu) {
            const double* p_row = p_mat + (size_t)(sp.ao_off_i + mu) * (size_t)n_ao;
            double si = sp.comp_scales_i[mu];
            for (int nu = 0; nu < nj; ++nu) {
                p_sum += acc[p_blk + mu * nj + nu] * si * sp.comp_scales_j[nu] * p_row[sp.ao_off_j + nu];
            }
        }
        double contrib = deg * aux.comp_scales[pf] * p_sum;
        if (contrib != 0.0) {
            atomicAdd(&v_out[aux.ao_offset + pf], contrib);
        }
    }
}

__device__ __forceinline__ double hrr_ket_stride_dev(
    const Comp6* __restrict__ comps7,
    const double* __restrict__ row,
    int c_idx,
    int dx, int dy, int dz,
    const double* __restrict__ cd_vec
) {
    static const double BINOM5[5][5] = {
        {1.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.0, 0.0, 0.0},
        {1.0, 3.0, 3.0, 1.0, 0.0},
        {1.0, 4.0, 6.0, 4.0, 1.0}
    };
    double px[5], py[5], pz[5];
    px[0] = 1.0; for (int k = 1; k <= dx; ++k) px[k] = px[k - 1] * cd_vec[0];
    py[0] = 1.0; for (int k = 1; k <= dy; ++k) py[k] = py[k - 1] * cd_vec[1];
    pz[0] = 1.0; for (int k = 1; k <= dz; ++k) pz[k] = pz[k - 1] * cd_vec[2];

    double sum = 0.0;
    int cx_idx = c_idx;
    for (int ux = 0; ux <= dx; ++ux) {
        double wx = BINOM5[dx][ux] * px[dx - ux];
        int cy_idx = cx_idx;
        for (int uy = 0; uy <= dy; ++uy) {
            double wxy = wx * BINOM5[dy][uy] * py[dy - uy];
            int cz_idx = cy_idx;
            for (int uz = 0; uz <= dz; ++uz) {
                sum += (wxy * BINOM5[dz][uz] * pz[dz - uz]) * row[cz_idx];
                if (uz < dz) cz_idx = comps7[cz_idx].add[2];
            }
            if (uy < dy) cy_idx = comps7[cy_idx].add[1];
        }
        if (ux < dx) cx_idx = comps7[cx_idx].add[0];
    }
    return sum;
}

__device__ __forceinline__ double hrr_bra_stride_dev(
    const Comp6* __restrict__ comps7,
    const double* __restrict__ vrr_cont,
    int row_stride,
    int a_idx,
    int bx, int by, int bz,
    int c_idx,
    int dx, int dy, int dz,
    const double* __restrict__ ab_vec,
    const double* __restrict__ cd_vec
) {
    static const double BINOM5[5][5] = {
        {1.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.0, 0.0, 0.0},
        {1.0, 3.0, 3.0, 1.0, 0.0},
        {1.0, 4.0, 6.0, 4.0, 1.0}
    };
    double px[5], py[5], pz[5];
    px[0] = 1.0; for (int k = 1; k <= bx; ++k) px[k] = px[k - 1] * ab_vec[0];
    py[0] = 1.0; for (int k = 1; k <= by; ++k) py[k] = py[k - 1] * ab_vec[1];
    pz[0] = 1.0; for (int k = 1; k <= bz; ++k) pz[k] = pz[k - 1] * ab_vec[2];

    double sum = 0.0;
    int ax_idx = a_idx;
    for (int ux = 0; ux <= bx; ++ux) {
        double wx = BINOM5[bx][ux] * px[bx - ux];
        int ay_idx = ax_idx;
        for (int uy = 0; uy <= by; ++uy) {
            double wxy = wx * BINOM5[by][uy] * py[by - uy];
            int az_idx = ay_idx;
            for (int uz = 0; uz <= bz; ++uz) {
                double w = wxy * BINOM5[bz][uz] * pz[bz - uz];
                sum += w * hrr_ket_stride_dev(
                    comps7,
                    vrr_cont + (size_t)az_idx * (size_t)row_stride,
                    c_idx, dx, dy, dz, cd_vec
                );
                if (uz < bz) az_idx = comps7[az_idx].add[2];
            }
            if (uy < by) ay_idx = comps7[ay_idx].add[1];
        }
        if (ux < bx) ax_idx = comps7[ax_idx].add[0];
    }
    return sum;
}

// 1 warp per (sp_idx, p_idx), 2 warps (64 threads) per block.
// Shared memory per warp: 1512 doubles = 12,096 bytes (24,192 bytes/block).
#define RIJ_G3C_V0_OFF 0
#define RIJ_G3C_V1_OFF (56 * 9)
#define RIJ_G3C_VD_OFF (56 * 9 + 3 * 56 * 3)
#define RIJ_G3C_U0_OFF (56 * 9 + 3 * 56 * 3 + 6 * 56)
#define RIJ_G3C_UA_OFF (RIJ_G3C_U0_OFF + 56)
#define RIJ_G3C_UB_OFF (RIJ_G3C_U0_OFF + 112)
#define RIJ_G3C_STRIDE 1512

extern "C" __global__ void rij_grad_3c_kernel(
    const Grad3cAuxShellGpu* __restrict__ aux_shells,
    const Grad3cAuxPrimGpu* __restrict__ aux_prims,
    const Grad3cShellPairGpu* __restrict__ shell_pairs,
    const Grad3cPrimPairGpu* __restrict__ prim_pairs,
    const Comp6* __restrict__ comps7,
    const double* __restrict__ x_vec,
    const double* __restrict__ p_mat,
    double* __restrict__ grad_out,
    int n_aux_shells,
    int n_shell_pairs,
    int n_ao
) {
    __shared__ double smem[2 * RIJ_G3C_STRIDE];
    const int warp_in_blk = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int sp_idx = (int)blockIdx.x * 2 + warp_in_blk;
    const int p_idx = (int)blockIdx.y;
    if (sp_idx >= n_shell_pairs || p_idx >= n_aux_shells) return;

    const Grad3cAuxShellGpu aux = aux_shells[p_idx];
    const Grad3cShellPairGpu sp = shell_pairs[sp_idx];
    if (aux.bound * sp.q_ij < 1e-12) return;
    if (aux.atom_index == sp.atom_i && sp.atom_i == sp.atom_j) return;

    const int lp = aux.lp;
    const int li = sp.li;
    const int lj = sp.lj;
    const int np = (lp == 0) ? 1 : ((lp == 1) ? 3 : ((lp == 2) ? 6 : 10));
    const int ni = (li == 0) ? 1 : ((li == 1) ? 3 : 6);
    const int nj = (lj == 0) ? 1 : ((lj == 1) ? 3 : 6);

    double w_p_eff[10];
    double max_xp = 0.0;
    for (int pf = 0; pf < np; ++pf) {
        double v = x_vec[aux.ao_offset + pf] * aux.comp_scales[pf];
        w_p_eff[pf] = v;
        double av = fabs(v);
        if (av > max_xp) max_xp = av;
    }
    if (max_xp < 1e-18) return;

    const int L_START[7] = {0, 1, 4, 10, 20, 35, 56};
    const int max_ab = li + lj + 1; // <= 5
    const int max_m = lp + max_ab;  // <= 8
    const int n_ket = L_START[max_ab + 1]; // <= 56

    double* wmem = smem + warp_in_blk * RIJ_G3C_STRIDE;
    double* V0  = wmem + RIJ_G3C_V0_OFF;
    double* V1  = wmem + RIJ_G3C_V1_OFF;
    double* V_D = wmem + RIJ_G3C_VD_OFF;
    double* U0  = wmem + RIJ_G3C_U0_OFF;
    double* Ua  = wmem + RIJ_G3C_UA_OFF;
    double* Ub  = wmem + RIJ_G3C_UB_OFF;

    for (int k = lane; k < n_ket; k += 32) {
        U0[k] = 0.0;
        Ua[k] = 0.0;
        Ub[k] = 0.0;
    }
    __syncwarp();

    const int D_K[6] = {0, 0, 0, 1, 1, 2};
    const int D_L[6] = {0, 1, 2, 1, 2, 2};
    const double rp0 = aux.center[0], rp1 = aux.center[1], rp2 = aux.center[2];

    for (int pp_i = 0; pp_i < aux.n_prims; ++pp_i) {
        const Grad3cAuxPrimGpu ap = aux_prims[aux.prim_start + pp_i];
        const double p = ap.gamma_p;
        const double inv_2p = 0.5 / p;
        const double cp_lead = ap.cp_lead;

        for (int ij_i = 0; ij_i < sp.n_prims; ++ij_i) {
            const Grad3cPrimPairGpu pp = prim_pairs[sp.prim_start + ij_i];
            const double q = pp.q;
            const double zeta = p + q;
            const double inv_zeta = 1.0 / zeta;
            const double inv_2zeta = 0.5 * inv_zeta;
            const double rho = p * q * inv_zeta;
            const double inv_2q = 0.5 / q;
            const double rho_over_q = rho / q;
            const double rho_over_p = rho / p;
            const double dx = rp0 - pp.p_center[0];
            const double dy = rp1 - pp.p_center[1];
            const double dz = rp2 - pp.p_center[2];
            const double pq2 = dx * dx + dy * dy + dz * dz;
            const double pref = pp.two_pi25_kab_w_over_q / (p * sqrt(zeta));

            if (lane == 0) {
                double boys[9];
                boys_batch_dev(max_m, rho * pq2, boys);
                for (int m = 0; m <= max_m; ++m) V0[m] = pref * boys[m];
            }
            __syncwarp();

            const double wx = (p * rp0 + q * pp.p_center[0]) * inv_zeta;
            const double wy = (p * rp1 + q * pp.p_center[1]) * inv_zeta;
            const double wz = (p * rp2 + q * pp.p_center[2]) * inv_zeta;
            const double wp[3] = {wx - rp0, wy - rp1, wz - rp2};
            const double wq[3] = {wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]};

            for (int l_cur = 1; l_cur <= max_ab; ++l_cur) {
                int k_beg = L_START[l_cur], k_end = L_START[l_cur + 1];
                for (int k = k_beg + lane; k < k_end; k += 32) {
                    const Comp6 c = comps7[k];
                    int axis = c.axis, i1 = c.idx1, i2 = c.idx2, nk = c.n_k;
                    for (int m = 0; m <= max_m - l_cur; ++m) {
                        double val = pp.qc[axis] * V0[i1 * 9 + m] + wq[axis] * V0[i1 * 9 + m + 1];
                        if (nk > 0) val += (double)nk * inv_2q * (V0[i2 * 9 + m] - rho_over_q * V0[i2 * 9 + m + 1]);
                        V0[k * 9 + m] = val;
                    }
                }
                __syncwarp();
            }

            if (lp == 0) {
                for (int k = lane; k < n_ket; k += 32) {
                    double s = w_p_eff[0] * V0[k * 9 + 0];
                    double w0 = cp_lead * s;
                    U0[k] += w0;
                    Ua[k] += (2.0 * pp.alpha) * w0;
                    Ub[k] += (2.0 * pp.beta) * w0;
                }
            } else if (lp == 1) {
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    double s = 0.0;
                    for (int u = 0; u < 3; ++u) {
                        double val = wp[u] * V0[k * 9 + 1];
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V0[sub_idx * 9 + 1];
                        }
                        s += w_p_eff[u] * val;
                    }
                    double w0 = cp_lead * s;
                    U0[k] += w0;
                    Ua[k] += (2.0 * pp.alpha) * w0;
                    Ub[k] += (2.0 * pp.beta) * w0;
                }
            } else if (lp == 2) {
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    for (int u = 0; u < 3; ++u) {
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        double val = wp[u] * V0[k * 9 + 2];
                        if (sub_idx >= 0) val += (double)cu * inv_2zeta * V0[sub_idx * 9 + 2];
                        V1[(u * 56 + k) * 3 + 1] = val;
                    }
                }
                __syncwarp();
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    double s = 0.0;
                    for (int kl = 0; kl < 6; ++kl) {
                        int u = D_K[kl], v = D_L[kl];
                        double val = wp[u] * V1[(v * 56 + k) * 3 + 1];
                        if (u == v) val += inv_2p * (V0[k * 9 + 0] - rho_over_p * V0[k * 9 + 1]);
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[(v * 56 + sub_idx) * 3 + 1];
                        }
                        s += w_p_eff[kl] * val;
                    }
                    double w0 = cp_lead * s;
                    U0[k] += w0;
                    Ua[k] += (2.0 * pp.alpha) * w0;
                    Ub[k] += (2.0 * pp.beta) * w0;
                }
            } else { // lp == 3
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    for (int u = 0; u < 3; ++u) {
                        int sub_idx = c.sub[u];
                        int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                        for (int m_sub = 0; m_sub < 3; ++m_sub) {
                            double val = wp[u] * V0[k * 9 + m_sub + 1];
                            if (sub_idx >= 0) val += (double)cu * inv_2zeta * V0[sub_idx * 9 + m_sub + 1];
                            V1[(u * 56 + k) * 3 + m_sub] = val;
                        }
                    }
                }
                __syncwarp();
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    for (int kl = 0; kl < 6; ++kl) {
                        int u = D_K[kl], v = D_L[kl];
                        double val = wp[u] * V1[(v * 56 + k) * 3 + 2];
                        if (u == v) val += inv_2p * (V0[k * 9 + 1] - rho_over_p * V0[k * 9 + 2]);
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V1[(v * 56 + sub_idx) * 3 + 2];
                        }
                        V_D[kl * 56 + k] = val;
                    }
                }
                __syncwarp();
                for (int k = lane; k < n_ket; k += 32) {
                    const Comp6 c = comps7[k];
                    double s = 0.0;
                    for (int pf = 0; pf < 10; ++pf) {
                        const Comp3C& cp_f = COMPS_3C[10 + pf];
                        int u = cp_f.axis, d_idx = cp_f.idx1 - 4, p_idx_v1 = cp_f.idx2 - 1, nk = cp_f.n_k;
                        double val = wp[u] * V_D[d_idx * 56 + k];
                        if (nk > 0) {
                            val += (double)nk * inv_2p * (V1[(p_idx_v1 * 56 + k) * 3 + 0]
                                - rho_over_p * V1[(p_idx_v1 * 56 + k) * 3 + 1]);
                        }
                        int sub_idx = c.sub[u];
                        if (sub_idx >= 0) {
                            int cu = (u == 0) ? c.x : ((u == 1) ? c.y : c.z);
                            val += (double)cu * inv_2zeta * V_D[d_idx * 56 + sub_idx];
                        }
                        s += w_p_eff[pf] * val;
                    }
                    double w0 = cp_lead * s;
                    U0[k] += w0;
                    Ua[k] += (2.0 * pp.alpha) * w0;
                    Ub[k] += (2.0 * pp.beta) * w0;
                }
            }
            __syncwarp();
        }
    }

    const int nij = ni * nj;
    const double deg = sp.same_shell ? 1.0 : 2.0;
    double loc_i[3] = {0.0, 0.0, 0.0};
    double loc_j[3] = {0.0, 0.0, 0.0};

    for (int ab = lane; ab < nij; ab += 32) {
        int mu = ab / nj;
        int nu = ab - mu * nj;
        double d_mn = p_mat[(size_t)(sp.ao_off_i + mu) * (size_t)n_ao + (size_t)(sp.ao_off_j + nu)];
        double w = deg * d_mn * sp.comp_scales_i[mu] * sp.comp_scales_j[nu];
        if (fabs(w) <= 1e-30) continue;

        int i_idx = L_START[li] + mu;
        const Comp6 ci = comps7[i_idx];
        const Comp6 cj = comps7[L_START[lj] + nu];

        for (int q = 0; q < 3; ++q) {
            int i_p = ci.add[q];
            double di = hrr_ket_stride_dev(comps7, Ua, i_p, cj.x, cj.y, cj.z, sp.ab_vec);
            int ci_q = (q == 0) ? ci.x : ((q == 1) ? ci.y : ci.z);
            if (ci_q > 0) {
                di -= (double)ci_q * hrr_ket_stride_dev(comps7, U0, ci.sub[q], cj.x, cj.y, cj.z, sp.ab_vec);
            }

            int jxp = cj.x + (q == 0 ? 1 : 0);
            int jyp = cj.y + (q == 1 ? 1 : 0);
            int jzp = cj.z + (q == 2 ? 1 : 0);
            double dj = hrr_ket_stride_dev(comps7, Ub, i_idx, jxp, jyp, jzp, sp.ab_vec);
            int cj_q = (q == 0) ? cj.x : ((q == 1) ? cj.y : cj.z);
            if (cj_q > 0) {
                int jxm = cj.x - (q == 0 ? 1 : 0);
                int jym = cj.y - (q == 1 ? 1 : 0);
                int jzm = cj.z - (q == 2 ? 1 : 0);
                dj -= (double)cj_q * hrr_ket_stride_dev(comps7, U0, i_idx, jxm, jym, jzm, sp.ab_vec);
            }

            loc_i[q] += w * di;
            loc_j[q] += w * dj;
        }
    }

    unsigned mask = 0xffffffffu;
    for (int q = 0; q < 3; ++q) {
        double vi = loc_i[q];
        double vj = loc_j[q];
        for (int off = 16; off > 0; off >>= 1) {
            vi += __shfl_down_sync(mask, vi, off);
            vj += __shfl_down_sync(mask, vj, off);
        }
        if (lane == 0) {
            double vp = -(vi + vj);
            if (vi != 0.0) atomicAdd(&grad_out[(size_t)sp.atom_i * 3 + (size_t)q], vi);
            if (vj != 0.0) atomicAdd(&grad_out[(size_t)sp.atom_j * 3 + (size_t)q], vj);
            if (vp != 0.0) atomicAdd(&grad_out[(size_t)aux.atom_index * 3 + (size_t)q], vp);
        }
    }
}

#define WL_GRAD_SSSS 0
#define WL_GRAD_SP   1
#define WL_GRAD_D    2
#define WL_GRAD_F    3
#define NUM_GRAD_WL_BUCKETS 4

extern "C" __global__ void grad4c_filter_worklist_kernel(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const int* __restrict__ ends,
    const double* __restrict__ d_tot_max,
    const double* __restrict__ d_spin_max,
    WorkItem* __restrict__ worklists,
    uint32_t wl_stride,
    int* __restrict__ wl_counts,
    int pi_start,
    int n_active,
    int n_shells,
    int do_coul,
    double exx,
    double thresh,
    int qi_start,
    int qi_end_override
) {
    int rel_pi = (int)blockIdx.x;
    if (rel_pi >= n_active) return;
    int pi = pi_start + rel_pi;
    const Grad4cShellPairGpu bra = pairs[pi];
    const int end_raw = ends[pi];
    const int end = (qi_end_override >= 0 && qi_end_override < end_raw) ? qi_end_override : end_raw;
    const int i = bra.shell_i;
    const int j = bra.shell_j;
    const int la = shells[i].l;
    const int lb = shells[j].l;
    const int aa = shells[i].atom_index;
    const int ab = shells[j].atom_index;
    const double b_ij = bra.bound;
    const double exx_abs = fabs(exx);
    const double d_ij = do_coul ? d_tot_max[i * n_shells + j] : 0.0;

    for (int qi = qi_start + (int)threadIdx.x; qi < end; qi += (int)blockDim.x) {
        const Grad4cShellPairGpu ket = pairs[qi];
        const int k = ket.shell_i;
        const int l = ket.shell_j;
        const int ac = shells[k].atom_index;
        const int ad = shells[l].atom_index;
        if (aa == ab && ab == ac && ac == ad) continue;

        const double b_ijkl = b_ij * ket.bound;
        const double d_coul = do_coul ? (d_ij * d_tot_max[k * n_shells + l]) : 0.0;
        const double d_exch = d_spin_max[i * n_shells + k] * d_spin_max[j * n_shells + l]
                            + d_spin_max[i * n_shells + l] * d_spin_max[j * n_shells + k];
        const double d_upper = 4.0 * (d_coul + exx_abs * d_exch);
        if (b_ijkl * d_upper < thresh) continue;

        const int lc = shells[k].l;
        const int ld = shells[l].l;
        const int max_l = max(max(la, lb), max(lc, ld));
        int bucket;
        if (max_l == 0) {
            bucket = WL_GRAD_SSSS;
        } else if (max_l == 1) {
            bucket = WL_GRAD_SP;
        } else if (max_l == 2) {
            bucket = WL_GRAD_D;
        } else {
            bucket = WL_GRAD_F;
        }
        int pos = atomicAdd(&wl_counts[bucket], 1);
        worklists[(size_t)bucket * (size_t)wl_stride + (size_t)pos] = {(uint32_t)pi, (uint32_t)qi};
    }
}

extern "C" __global__ void grad4c_eval_ssss_f64_kernel(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const Grad4cPrimPairGpu* __restrict__ prims,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int do_coul,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const double* __restrict__ pt,
    const double* __restrict__ pa,
    const double* __restrict__ pb,
    double* __restrict__ grad_out
) {
    int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qid >= count) return;
    const WorkItem item = wl[qid];
    const Grad4cShellPairGpu bra = pairs[item.pi];
    const Grad4cShellPairGpu ket = pairs[item.qi];
    const int i = bra.shell_i, j = bra.shell_j;
    const int k = ket.shell_i, l = ket.shell_j;
    const Grad4cShellGpu sh_a = shells[i], sh_b = shells[j];
    const Grad4cShellGpu sh_c = shells[k], sh_d = shells[l];

    const int aa = sh_a.atom_index, ab = sh_b.atom_index;
    const int ac = sh_c.atom_index, ad = sh_d.atom_index;
    if (aa == ab && ab == ac && ac == ad) return;

    const int a = sh_a.ao_offset, b = sh_b.ao_offset;
    const int c = sh_c.ao_offset, d = sh_d.ao_offset;
    const int mult = (i == j ? 1 : 2) * (k == l ? 1 : 2) * ((i == k && j == l) ? 1 : 2);
    const double weight = 0.5 * (double)mult;
    const double coul = do_coul ? (pt[a * n + b] * pt[c * n + d]) : 0.0;
    const double exch = (fabs(exx) > 1e-15)
        ? 0.5 * (pa[a * n + c] * pa[b * n + d] + pb[a * n + c] * pb[b * n + d]
               + pa[a * n + d] * pa[b * n + c] + pb[a * n + d] * pb[b * n + c])
        : 0.0;
    const double gamma_ao = weight * (coul - exx * exch)
        * sh_a.comp_scales[0] * sh_b.comp_scales[0] * sh_c.comp_scales[0] * sh_d.comp_scales[0];
    if (fabs(gamma_ao) <= 1e-30) return;

    const int need_a = (aa != ad), need_b = (ab != ad), need_c = (ac != ad);
    const int max_ab = (need_a || need_b) ? 1 : 0;
    const int max_cd = need_c ? 1 : 0;
    const int max_m = max_ab + max_cd;
    const double two_pi25 = 34.986836655249725693;

    double v2a[3] = {0.0, 0.0, 0.0};
    double v2b[4] = {0.0, 0.0, 0.0, 0.0}; // [0]=v_000, [1..3]=v_100
    double v2c[3] = {0.0, 0.0, 0.0};

    for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
        const Grad4cPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
        const double p = pab.gamma;
        const double two_a = 2.0 * pab.alpha;
        const double two_b = 2.0 * pab.beta;
        for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
            const Grad4cPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
            const double q = pcd.gamma;
            const double two_c = 2.0 * pcd.alpha;
            const double zeta = p + q;
            const double inv_zeta = 1.0 / zeta;
            const double rho = p * q * inv_zeta;
            const double w0 = (p * pab.p[0] + q * pcd.p[0]) * inv_zeta;
            const double w1 = (p * pab.p[1] + q * pcd.p[1]) * inv_zeta;
            const double w2 = (p * pab.p[2] + q * pcd.p[2]) * inv_zeta;
            const double wp[3] = {w0 - pab.p[0], w1 - pab.p[1], w2 - pab.p[2]};
            const double wq[3] = {w0 - pcd.p[0], w1 - pcd.p[1], w2 - pcd.p[2]};
            const double dx = pab.p[0] - pcd.p[0];
            const double dy = pab.p[1] - pcd.p[1];
            const double dz = pab.p[2] - pcd.p[2];
            const double pq2 = dx * dx + dy * dy + dz * dz;

            double boys[3];
            boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
            const double scale = two_pi25 / (p * q * sqrt(zeta)) * pab.kab_w * pcd.kab_w;
            const double g0 = scale * boys[0];
            const double g1 = (max_m >= 1) ? (scale * boys[1]) : 0.0;

            if (need_b) v2b[0] += two_b * g0;
            if (need_a || need_b) {
                for (int u = 0; u < 3; ++u) {
                    double v_a = pab.pa[u] * g0 + wp[u] * g1;
                    if (need_a) v2a[u] += two_a * v_a;
                    if (need_b) v2b[1 + u] += two_b * v_a;
                }
            }
            if (need_c) {
                for (int u = 0; u < 3; ++u) {
                    double v_c = pcd.pa[u] * g0 + wq[u] * g1;
                    v2c[u] += two_c * v_c;
                }
            }
        }
    }

    const double ab_vec[3] = {
        sh_a.center[0] - sh_b.center[0],
        sh_a.center[1] - sh_b.center[1],
        sh_a.center[2] - sh_b.center[2]
    };
    for (int q = 0; q < 3; ++q) {
        double da = need_a ? (gamma_ao * v2a[q]) : 0.0;
        double db = need_b ? (gamma_ao * (v2b[1 + q] + ab_vec[q] * v2b[0])) : 0.0;
        double dc = need_c ? (gamma_ao * v2c[q]) : 0.0;
        double dd = -(da + db + dc);
        if (need_a && da != 0.0) atomicAdd(&grad_out[(size_t)aa * 3 + (size_t)q], da);
        if (need_b && db != 0.0) atomicAdd(&grad_out[(size_t)ab * 3 + (size_t)q], db);
        if (need_c && dc != 0.0) atomicAdd(&grad_out[(size_t)ac * 3 + (size_t)q], dc);
        if (dd != 0.0) atomicAdd(&grad_out[(size_t)ad * 3 + (size_t)q], dd);
    }
}

template <int STRIDE, int MAX_M1>
__device__ __forceinline__ void grad4c_eval_general_dev(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const Grad4cPrimPairGpu* __restrict__ prims,
    const WorkItem* __restrict__ wl,
    int count,
    int n,
    int do_coul,
    double exx,
    double omega,
    int has_omega,
    const double* __restrict__ d_boys_f64,
    const Comp6* __restrict__ comps7,
    double* __restrict__ blk_scratch, // stride*stride*(4 + MAX_M1) doubles per block
    const double* __restrict__ pt,
    const double* __restrict__ pa,
    const double* __restrict__ pb,
    double* __restrict__ grad_out
) {
    const int tid = (int)threadIdx.x;
    const int ss = STRIDE * STRIDE;
    double* vrr_0  = blk_scratch;
    double* vrr_2a = vrr_0 + ss;
    double* vrr_2b = vrr_2a + ss;
    double* vrr_2c = vrr_2b + ss;
    double* V      = vrr_2c + ss; // [STRIDE * STRIDE * MAX_M1]

    const int L_START[9] = {0, 1, 4, 10, 20, 35, 56, 84, 120};
    const int has_exch = (fabs(exx) > 1e-15) ? 1 : 0;

    for (int qid = (int)blockIdx.x; qid < count; qid += (int)gridDim.x) {
        const WorkItem item = wl[qid];
        const Grad4cShellPairGpu bra = pairs[item.pi];
        const Grad4cShellPairGpu ket = pairs[item.qi];
        const int i = bra.shell_i, j = bra.shell_j;
        const int k = ket.shell_i, l = ket.shell_j;
        const Grad4cShellGpu sh_a = shells[i], sh_b = shells[j];
        const Grad4cShellGpu sh_c = shells[k], sh_d = shells[l];
        const int aa = sh_a.atom_index, ab = sh_b.atom_index;
        const int ac = sh_c.atom_index, ad = sh_d.atom_index;
        if (aa == ab && ab == ac && ac == ad) continue;

        const int need_a = (aa != ad) ? 1 : 0;
        const int need_b = (ab != ad) ? 1 : 0;
        const int need_c = (ac != ad) ? 1 : 0;

        const int la = sh_a.l, lb = sh_b.l, lc = sh_c.l, ld = sh_d.l;
        const int na = sh_a.n_funcs, nb = sh_b.n_funcs, nc = sh_c.n_funcs, nd = sh_d.n_funcs;
        const int max_ab = la + lb + ((need_a || need_b) ? 1 : 0);
        const int max_cd = lc + ld + (need_c ? 1 : 0);
        const int max_m = max_ab + max_cd;
        const int n_a = L_START[max_ab + 1];
        const int n_c = L_START[max_cd + 1];

        for (int idx = tid; idx < n_a * STRIDE; idx += (int)blockDim.x) {
            vrr_0[idx] = 0.0;
            if (need_a) vrr_2a[idx] = 0.0;
            if (need_b) vrr_2b[idx] = 0.0;
            if (need_c) vrr_2c[idx] = 0.0;
        }
        __syncthreads();

        const double two_pi25 = 34.986836655249725693;
        for (int pab_i = 0; pab_i < bra.n_prim_pairs; ++pab_i) {
            const Grad4cPrimPairGpu pab = prims[bra.prim_pair_start + pab_i];
            const double p = pab.gamma;
            const double inv_2p = 0.5 / p;
            const double two_a = 2.0 * pab.alpha;
            const double two_b = 2.0 * pab.beta;

            for (int pcd_i = 0; pcd_i < ket.n_prim_pairs; ++pcd_i) {
                const Grad4cPrimPairGpu pcd = prims[ket.prim_pair_start + pcd_i];
                const double q = pcd.gamma;
                const double inv_2q = 0.5 / q;
                const double two_c = 2.0 * pcd.alpha;
                const double zeta = p + q;
                const double inv_zeta = 1.0 / zeta;
                const double inv_2zeta = 0.5 * inv_zeta;
                const double rho = p * q * inv_zeta;
                const double w0 = (p * pab.p[0] + q * pcd.p[0]) * inv_zeta;
                const double w1 = (p * pab.p[1] + q * pcd.p[1]) * inv_zeta;
                const double w2 = (p * pab.p[2] + q * pcd.p[2]) * inv_zeta;
                const double wp[3] = {w0 - pab.p[0], w1 - pab.p[1], w2 - pab.p[2]};
                const double wq[3] = {w0 - pcd.p[0], w1 - pcd.p[1], w2 - pcd.p[2]};
                const double dx = pab.p[0] - pcd.p[0];
                const double dy = pab.p[1] - pcd.p[1];
                const double dz = pab.p[2] - pcd.p[2];
                const double pq2 = dx * dx + dy * dy + dz * dz;
                const double rho_over_p = rho / p;
                const double rho_over_q = rho / q;
                const double scale = two_pi25 / (p * q * sqrt(zeta)) * pab.kab_w * pcd.kab_w;

                if (tid == 0) {
                    double boys[16];
                    boys_batch_table_f64(max_m, rho, pq2, omega, has_omega, d_boys_f64, boys);
                    for (int m = 0; m <= max_m; ++m) V[m] = scale * boys[m];
                }
                __syncthreads();

                for (int la_cur = 1; la_cur <= max_ab; ++la_cur) {
                    int a_beg = L_START[la_cur], a_end = L_START[la_cur + 1];
                    for (int a = a_beg + tid; a < a_end; a += (int)blockDim.x) {
                        const Comp6 ca = comps7[a];
                        int u = ca.axis, i1 = ca.idx1, i2 = ca.idx2, nk = ca.n_k;
                        for (int m = 0; m <= max_m - la_cur; ++m) {
                            double val = pab.pa[u] * V[(size_t)i1 * STRIDE * MAX_M1 + m]
                                       + wp[u]     * V[(size_t)i1 * STRIDE * MAX_M1 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2p * (V[(size_t)i2 * STRIDE * MAX_M1 + m]
                                    - rho_over_p * V[(size_t)i2 * STRIDE * MAX_M1 + m + 1]);
                            }
                            V[(size_t)a * STRIDE * MAX_M1 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                for (int lc_cur = 1; lc_cur <= max_cd; ++lc_cur) {
                    int c_beg = L_START[lc_cur], c_end = L_START[lc_cur + 1];
                    int n_c_lvl = c_end - c_beg;
                    int total_ac = n_a * n_c_lvl;
                    for (int ac_i = tid; ac_i < total_ac; ac_i += (int)blockDim.x) {
                        int a = ac_i / n_c_lvl;
                        int c = c_beg + (ac_i - a * n_c_lvl);
                        const Comp6 ca = comps7[a];
                        const Comp6 cc = comps7[c];
                        int w = cc.axis, c1 = cc.idx1, c2 = cc.idx2, nk = cc.n_k;
                        int sub_a = ca.sub[w];
                        int aw = (w == 0) ? ca.x : ((w == 1) ? ca.y : ca.z);
                        int m_top = max_m - ca.L - lc_cur;
                        for (int m = 0; m <= m_top; ++m) {
                            double val = pcd.pa[w] * V[((size_t)a * STRIDE + c1) * MAX_M1 + m]
                                       + wq[w]     * V[((size_t)a * STRIDE + c1) * MAX_M1 + m + 1];
                            if (nk > 0) {
                                val += (double)nk * inv_2q * (V[((size_t)a * STRIDE + c2) * MAX_M1 + m]
                                    - rho_over_q * V[((size_t)a * STRIDE + c2) * MAX_M1 + m + 1]);
                            }
                            if (sub_a >= 0) {
                                val += (double)aw * inv_2zeta * V[((size_t)sub_a * STRIDE + c1) * MAX_M1 + m + 1];
                            }
                            V[((size_t)a * STRIDE + c) * MAX_M1 + m] = val;
                        }
                    }
                    __syncthreads();
                }

                for (int idx = tid; idx < n_a * n_c; idx += (int)blockDim.x) {
                    int a = idx / n_c;
                    int c = idx - a * n_c;
                    double val = V[((size_t)a * STRIDE + c) * MAX_M1 + 0];
                    int dst = a * STRIDE + c;
                    vrr_0[dst] += val;
                    if (need_a) vrr_2a[dst] += two_a * val;
                    if (need_b) vrr_2b[dst] += two_b * val;
                    if (need_c) vrr_2c[dst] += two_c * val;
                }
                __syncthreads();
            }
        }

        const double ab_vec[3] = {
            sh_a.center[0] - sh_b.center[0],
            sh_a.center[1] - sh_b.center[1],
            sh_a.center[2] - sh_b.center[2]
        };
        const double cd_vec[3] = {
            sh_c.center[0] - sh_d.center[0],
            sh_c.center[1] - sh_d.center[1],
            sh_c.center[2] - sh_d.center[2]
        };
        const int mult = (i == j ? 1 : 2) * (k == l ? 1 : 2) * ((i == k && j == l) ? 1 : 2);
        const double weight = 0.5 * (double)mult;
        const int n_cd = nc * nd;
        const int n_bcd = nb * n_cd;
        const int total_abcd = na * n_bcd;

        double loc_a[3] = {0.0, 0.0, 0.0};
        double loc_b[3] = {0.0, 0.0, 0.0};
        double loc_c[3] = {0.0, 0.0, 0.0};

        for (int abcd = tid; abcd < total_abcd; abcd += (int)blockDim.x) {
            int ia = abcd / n_bcd;
            int rem = abcd - ia * n_bcd;
            int ib = rem / n_cd;
            rem -= ib * n_cd;
            int ic = rem / nd;
            int id = rem - ic * nd;

            int a = sh_a.ao_offset + ia;
            int b = sh_b.ao_offset + ib;
            int c = sh_c.ao_offset + ic;
            int d = sh_d.ao_offset + id;

            double coul = do_coul ? (pt[a * n + b] * pt[c * n + d]) : 0.0;
            double exch = has_exch
                ? 0.5 * (pa[a * n + c] * pa[b * n + d] + pb[a * n + c] * pb[b * n + d]
                       + pa[a * n + d] * pa[b * n + c] + pb[a * n + d] * pb[b * n + c])
                : 0.0;
            double gamma_ao = weight * (coul - exx * exch)
                * sh_a.comp_scales[ia] * sh_b.comp_scales[ib] * sh_c.comp_scales[ic] * sh_d.comp_scales[id];
            if (fabs(gamma_ao) <= 1e-30) continue;

            int ca_idx = L_START[la] + ia;
            int cb_idx = L_START[lb] + ib;
            int cc_idx = L_START[lc] + ic;
            int cd_idx = L_START[ld] + id;
            const Comp6 ca = comps7[ca_idx];
            const Comp6 cb = comps7[cb_idx];
            const Comp6 cc = comps7[cc_idx];
            const Comp6 cd = comps7[cd_idx];

            if (need_a) {
                for (int q = 0; q < 3; ++q) {
                    int ca_p = ca.add[q];
                    double dv = hrr_bra_stride_dev(comps7, vrr_2a, STRIDE, ca_p, cb.x, cb.y, cb.z, cc_idx, cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    int ca_q = (q == 0) ? ca.x : ((q == 1) ? ca.y : ca.z);
                    if (ca_q > 0) {
                        dv -= (double)ca_q * hrr_bra_stride_dev(comps7, vrr_0, STRIDE, ca.sub[q], cb.x, cb.y, cb.z, cc_idx, cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    }
                    loc_a[q] += gamma_ao * dv;
                }
            }
            if (need_b) {
                for (int q = 0; q < 3; ++q) {
                    int bxp = cb.x + (q == 0 ? 1 : 0);
                    int byp = cb.y + (q == 1 ? 1 : 0);
                    int bzp = cb.z + (q == 2 ? 1 : 0);
                    double dv = hrr_bra_stride_dev(comps7, vrr_2b, STRIDE, ca_idx, bxp, byp, bzp, cc_idx, cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    int cb_q = (q == 0) ? cb.x : ((q == 1) ? cb.y : cb.z);
                    if (cb_q > 0) {
                        int bxm = cb.x - (q == 0 ? 1 : 0);
                        int bym = cb.y - (q == 1 ? 1 : 0);
                        int bzm = cb.z - (q == 2 ? 1 : 0);
                        dv -= (double)cb_q * hrr_bra_stride_dev(comps7, vrr_0, STRIDE, ca_idx, bxm, bym, bzm, cc_idx, cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    }
                    loc_b[q] += gamma_ao * dv;
                }
            }
            if (need_c) {
                for (int q = 0; q < 3; ++q) {
                    int cc_p = cc.add[q];
                    double dv = hrr_bra_stride_dev(comps7, vrr_2c, STRIDE, ca_idx, cb.x, cb.y, cb.z, cc_p, cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    int cc_q = (q == 0) ? cc.x : ((q == 1) ? cc.y : cc.z);
                    if (cc_q > 0) {
                        dv -= (double)cc_q * hrr_bra_stride_dev(comps7, vrr_0, STRIDE, ca_idx, cb.x, cb.y, cb.z, cc.sub[q], cd.x, cd.y, cd.z, ab_vec, cd_vec);
                    }
                    loc_c[q] += gamma_ao * dv;
                }
            }
        }

        // Warp reduction -> 1 atomicAdd per warp per center
        const int lane = tid & 31;
        unsigned mask = 0xffffffffu;
        for (int q = 0; q < 3; ++q) {
            double va = loc_a[q], vb = loc_b[q], vc = loc_c[q];
            for (int off = 16; off > 0; off >>= 1) {
                va += __shfl_down_sync(mask, va, off);
                vb += __shfl_down_sync(mask, vb, off);
                vc += __shfl_down_sync(mask, vc, off);
            }
            if (lane == 0) {
                double vd = -(va + vb + vc);
                if (need_a && va != 0.0) atomicAdd(&grad_out[(size_t)aa * 3 + (size_t)q], va);
                if (need_b && vb != 0.0) atomicAdd(&grad_out[(size_t)ab * 3 + (size_t)q], vb);
                if (need_c && vc != 0.0) atomicAdd(&grad_out[(size_t)ac * 3 + (size_t)q], vc);
                if (vd != 0.0) atomicAdd(&grad_out[(size_t)ad * 3 + (size_t)q], vd);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void grad4c_eval_sp_f64_kernel(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const Grad4cPrimPairGpu* __restrict__ prims,
    const WorkItem* __restrict__ wl,
    int count, int n, int do_coul, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const Comp6* __restrict__ comps7,
    const double* __restrict__ pt, const double* __restrict__ pa, const double* __restrict__ pb,
    double* __restrict__ grad_out
) {
    // 20 * 20 * (4 + 7) = 4400 doubles = 35,200 bytes dynamic shared memory
    extern __shared__ double sp_smem[];
    grad4c_eval_general_dev<20, 7>(
        shells, pairs, prims, wl, count, n, do_coul, exx, omega, has_omega,
        d_boys_f64, comps7, sp_smem, pt, pa, pb, grad_out
    );
}

extern "C" __global__ void grad4c_eval_d_f64_kernel(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const Grad4cPrimPairGpu* __restrict__ prims,
    const WorkItem* __restrict__ wl,
    int count, int n, int do_coul, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const Comp6* __restrict__ comps7,
    double* __restrict__ d_grad_scratch, // gridDim.x * 56 * 56 * 15
    const double* __restrict__ pt, const double* __restrict__ pa, const double* __restrict__ pb,
    double* __restrict__ grad_out
) {
    double* blk_scratch = d_grad_scratch + (size_t)blockIdx.x * (size_t)(56 * 56 * 15);
    grad4c_eval_general_dev<56, 11>(
        shells, pairs, prims, wl, count, n, do_coul, exx, omega, has_omega,
        d_boys_f64, comps7, blk_scratch, pt, pa, pb, grad_out
    );
}

extern "C" __global__ void grad4c_eval_f_f64_kernel(
    const Grad4cShellGpu* __restrict__ shells,
    const Grad4cShellPairGpu* __restrict__ pairs,
    const Grad4cPrimPairGpu* __restrict__ prims,
    const WorkItem* __restrict__ wl,
    int count, int n, int do_coul, double exx, double omega, int has_omega,
    const double* __restrict__ d_boys_f64, const Comp6* __restrict__ comps7,
    double* __restrict__ f_grad_scratch, // gridDim.x * 120 * 120 * 19
    const double* __restrict__ pt, const double* __restrict__ pa, const double* __restrict__ pb,
    double* __restrict__ grad_out
) {
    double* blk_scratch = f_grad_scratch + (size_t)blockIdx.x * (size_t)(120 * 120 * 19);
    grad4c_eval_general_dev<120, 15>(
        shells, pairs, prims, wl, count, n, do_coul, exx, omega, has_omega,
        d_boys_f64, comps7, blk_scratch, pt, pa, pb, grad_out
    );
}

#ifdef HAVE_CALCHEM_CUBIN_H
#include "calchem_cubin.h"
#endif

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

static std::mutex rij_mutex;

static double host_kummer_scaled(int n, double t) {
    double term = 1.0 / (2.0 * (double)n + 1.0);
    double sum = term;
    double corr = 0.0;
    for (int k = 1; k < 10000; ++k) {
        term *= t / ((double)n + (double)k + 0.5);
        double y = term - corr;
        double next = sum + y;
        corr = (next - sum) - y;
        sum = next;
        if (term <= sum * 5e-17) break;
    }
    return sum;
}

static void host_boys_row(double t, double* row26) {
    const int max_n = 25;
    if (t < 1e-14) {
        for (int n = 0; n <= max_n; ++n) {
            double nf = (double)n;
            row26[n] = 1.0 / (2.0 * nf + 1.0) - t / (2.0 * nf + 3.0) + 0.5 * t * t / (2.0 * nf + 5.0);
        }
        return;
    }
    double et = exp(-t);
    if (t > (double)max_n + 40.0) {
        row26[0] = 0.5 * sqrt(3.14159265358979323846 / t) * erf(sqrt(t));
        int last_up = (max_n < (int)t) ? max_n : (int)t;
        for (int n = 1; n <= last_up; ++n) {
            row26[n] = ((double)(2 * n - 1) * row26[n - 1] - et) * (0.5 / t);
        }
        for (int n = last_up + 1; n <= max_n; ++n) {
            row26[n] = et * host_kummer_scaled(n, t);
        }
        return;
    }
    row26[max_n] = et * host_kummer_scaled(max_n, t);
    for (int n = max_n; n >= 1; --n) {
        row26[n - 1] = (2.0 * t * row26[n] + et) / (double)(2 * n - 1);
    }
}

static void build_host_comps6(Comp6* out120) {
    int idx = 0;
    for (int L = 0; L <= 7; ++L) {
        for (int x = L; x >= 0; --x) {
            for (int y = L - x; y >= 0; --y) {
                int z = L - x - y;
                Comp6 c{};
                c.x = (signed char)x;
                c.y = (signed char)y;
                c.z = (signed char)z;
                c.L = (signed char)L;
                c.axis = (x > 0) ? 0 : ((y > 0) ? 1 : ((z > 0) ? 2 : 0));
                c.n_k = 0;
                c.idx1 = -1;
                c.idx2 = -1;
                c.sub[0] = c.sub[1] = c.sub[2] = -1;
                c.add[0] = c.add[1] = c.add[2] = -1;
                out120[idx++] = c;
            }
        }
    }
    auto find_comp = [&](int x, int y, int z) -> signed char {
        if (x < 0 || y < 0 || z < 0 || x + y + z > 7) return -1;
        for (int k = 0; k < 120; ++k) {
            if (out120[k].x == x && out120[k].y == y && out120[k].z == z) return (signed char)k;
        }
        return -1;
    };
    for (int k = 0; k < 120; ++k) {
        int x = out120[k].x, y = out120[k].y, z = out120[k].z;
        out120[k].sub[0] = find_comp(x - 1, y, z);
        out120[k].sub[1] = find_comp(x, y - 1, z);
        out120[k].sub[2] = find_comp(x, y, z - 1);
        out120[k].add[0] = find_comp(x + 1, y, z);
        out120[k].add[1] = find_comp(x, y + 1, z);
        out120[k].add[2] = find_comp(x, y, z + 1);
        if (out120[k].L > 0) {
            int ax = out120[k].axis;
            int dx = (ax == 0) ? 1 : 0;
            int dy = (ax == 1) ? 1 : 0;
            int dz = (ax == 2) ? 1 : 0;
            out120[k].idx1 = find_comp(x - dx, y - dy, z - dz);
            int c1 = (ax == 0) ? (x - 1) : ((ax == 1) ? (y - 1) : (z - 1));
            out120[k].n_k = (signed char)(c1 > 0 ? c1 : 0);
            if (c1 > 0) {
                out120[k].idx2 = find_comp(x - 2 * dx, y - 2 * dy, z - 2 * dz);
            }
        }
    }
}

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
typedef int (__stdcall *PFN_cuMemsetD8_v2)(uint64_t, unsigned char, size_t);
typedef int (__stdcall *PFN_cuModuleLoadData)(void**, const void*);
typedef int (__stdcall *PFN_cuModuleUnload)(void*);
typedef int (__stdcall *PFN_cuModuleGetFunction)(void**, void*, const char*);
typedef int (__stdcall *PFN_cuFuncSetAttribute)(void*, int, int);
typedef int (__stdcall *PFN_cuLaunchKernel)(
    void*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, void*, void**, void**
);
typedef int (__stdcall *PFN_cuCtxSynchronize)(void);

struct VxcGridEntry {
    int n_pts = 0;
    int n_ao = 0;
    size_t bytes = 0;
    uint64_t d_shells = 0;
    uint64_t d_prims = 0;
    uint64_t d_comps = 0;
    uint64_t d_pts = 0;
    uint64_t d_cand_off = 0;
    uint64_t d_cand_sh = 0;
    uint64_t d_p = 0;
    uint64_t d_v = 0;
    uint64_t d_rg = 0;
    uint64_t d_wv = 0;
    uint64_t d_ovf = 0;

    void free_all(PFN_cuMemFree_v2 cuMemFree) {
        if (cuMemFree) {
            if (d_shells) cuMemFree(d_shells);
            if (d_prims) cuMemFree(d_prims);
            if (d_comps) cuMemFree(d_comps);
            if (d_pts) cuMemFree(d_pts);
            if (d_cand_off) cuMemFree(d_cand_off);
            if (d_cand_sh) cuMemFree(d_cand_sh);
            if (d_p) cuMemFree(d_p);
            if (d_v) cuMemFree(d_v);
            if (d_rg) cuMemFree(d_rg);
            if (d_wv) cuMemFree(d_wv);
            if (d_ovf) cuMemFree(d_ovf);
        }
        d_shells = d_prims = d_comps = d_pts = 0;
        d_cand_off = d_cand_sh = 0;
        d_p = d_v = d_rg = d_wv = d_ovf = 0;
        n_pts = 0;
        n_ao = 0;
        bytes = 0;
    }
};

struct RijTensorEntry {
    uint64_t b_ptr = 0;
    size_t b_bytes = 0;
    const double* h_b = nullptr;
    int n_p = 0;
    int n_ao_sq = 0;
    bool device_only = false;

    void free_all(PFN_cuMemFree_v2 cuMemFree) {
        if (b_ptr && cuMemFree) cuMemFree(b_ptr);
        b_ptr = 0;
        b_bytes = 0;
        h_b = nullptr;
        n_p = 0;
        n_ao_sq = 0;
        device_only = false;
    }
};

struct RijStreamEntry {
    size_t bytes = 0;
    uint64_t d_aux_sh = 0;
    uint64_t d_aux_pr = 0;
    uint64_t d_sp = 0;
    uint64_t d_pp = 0;
    uint64_t d_b_chunk = 0;
    uint64_t d_p = 0;
    uint64_t d_v = 0;
    uint64_t d_x = 0;
    uint64_t d_j = 0;
    int max_chunk_n_aux = 0;
    int width = 0;
    int n_shell_pairs = 0;
    int n_aux_shells = 0;
    bool p_uploaded = false;
    std::vector<AuxShellGpu> aux_shells;

    void free_all(PFN_cuMemFree_v2 cuMemFree) {
        if (cuMemFree) {
            if (d_aux_sh) cuMemFree(d_aux_sh);
            if (d_aux_pr) cuMemFree(d_aux_pr);
            if (d_sp) cuMemFree(d_sp);
            if (d_pp) cuMemFree(d_pp);
            if (d_b_chunk) cuMemFree(d_b_chunk);
            if (d_p) cuMemFree(d_p);
            if (d_v) cuMemFree(d_v);
            if (d_x) cuMemFree(d_x);
            if (d_j) cuMemFree(d_j);
        }
        d_aux_sh = d_aux_pr = d_sp = d_pp = 0;
        d_b_chunk = d_p = d_v = d_x = d_j = 0;
        bytes = 0;
        max_chunk_n_aux = 0;
        width = 0;
        n_shell_pairs = 0;
        n_aux_shells = 0;
        p_uploaded = false;
        aux_shells.clear();
    }
};

struct DirectContext {
    int n_ao = 0;
    int n_shells = 0;
    int n_pairs = 0;
    size_t bytes = 0;
    size_t memory_limit_bytes = 0;
    uint64_t d_shells = 0;
    uint64_t d_pairs = 0;
    uint64_t d_prims = 0;
    uint64_t d_prefix = 0;
    uint64_t d_cache = 0;
    uint64_t d_valid = 0;
    uint64_t d_pa = 0;
    uint64_t d_pb = 0;
    uint64_t d_dt = 0;
    uint64_t d_da = 0;
    uint64_t d_db = 0;
    uint64_t d_ends = 0;
    uint64_t d_out_a = 0;
    uint64_t d_out_b = 0;
    uint64_t d_wl = 0;
    uint32_t wl_stride = 0;
    uint64_t d_wl_counts = 0;
    uint64_t d_stats = 0;

    // Lazy batched TD-DFT scratch buffers
    uint64_t d_td_sym = 0;
    size_t td_sym_cap_bytes = 0;
    uint64_t d_td_anti = 0;
    size_t td_anti_cap_bytes = 0;
    uint64_t d_td_out_j = 0;
    size_t td_out_j_cap_bytes = 0;
    uint64_t d_td_out_ks = 0;
    size_t td_out_ks_cap_bytes = 0;
    uint64_t d_td_out_ka = 0;
    size_t td_out_ka_cap_bytes = 0;

    size_t td_scratch_bytes() const {
        return td_sym_cap_bytes + td_anti_cap_bytes
             + td_out_j_cap_bytes + td_out_ks_cap_bytes + td_out_ka_cap_bytes;
    }

    size_t total_entry_bytes() const {
        return bytes + td_scratch_bytes();
    }

    void free_td_scratch(PFN_cuMemFree_v2 cuMemFree) {
        if (cuMemFree) {
            if (d_td_sym) cuMemFree(d_td_sym);
            if (d_td_anti) cuMemFree(d_td_anti);
            if (d_td_out_j) cuMemFree(d_td_out_j);
            if (d_td_out_ks) cuMemFree(d_td_out_ks);
            if (d_td_out_ka) cuMemFree(d_td_out_ka);
        }
        d_td_sym = d_td_anti = d_td_out_j = d_td_out_ks = d_td_out_ka = 0;
        td_sym_cap_bytes = td_anti_cap_bytes = 0;
        td_out_j_cap_bytes = td_out_ks_cap_bytes = td_out_ka_cap_bytes = 0;
    }

    void free_all(PFN_cuMemFree_v2 cuMemFree) {
        free_td_scratch(cuMemFree);
        if (cuMemFree) {
            if (d_shells) cuMemFree(d_shells);
            if (d_pairs) cuMemFree(d_pairs);
            if (d_prims) cuMemFree(d_prims);
            if (d_prefix) cuMemFree(d_prefix);
            if (d_cache) cuMemFree(d_cache);
            if (d_valid) cuMemFree(d_valid);
            if (d_pa) cuMemFree(d_pa);
            if (d_pb) cuMemFree(d_pb);
            if (d_dt) cuMemFree(d_dt);
            if (d_da) cuMemFree(d_da);
            if (d_db) cuMemFree(d_db);
            if (d_ends) cuMemFree(d_ends);
            if (d_out_a) cuMemFree(d_out_a);
            if (d_out_b) cuMemFree(d_out_b);
            if (d_wl) cuMemFree(d_wl);
            if (d_wl_counts) cuMemFree(d_wl_counts);
            if (d_stats) cuMemFree(d_stats);
        }
        d_shells = d_pairs = d_prims = d_prefix = 0;
        d_cache = d_valid = 0;
        d_pa = d_pb = d_dt = d_da = d_db = d_ends = 0;
        d_out_a = d_out_b = d_wl = d_wl_counts = d_stats = 0;
        n_ao = n_shells = n_pairs = 0;
        bytes = 0;
        memory_limit_bytes = 0;
        wl_stride = 0;
    }
};

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
    PFN_cuMemsetD8_v2 cuMemsetD8 = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuFuncSetAttribute cuFuncSetAttribute = nullptr;
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
    void* fn_direct_filter = nullptr;
    void* fn_direct_cached = nullptr;
    void* fn_direct_ssss_64 = nullptr;
    void* fn_direct_ssss_32 = nullptr;
    void* fn_direct_sp_64 = nullptr;
    void* fn_direct_sp_32 = nullptr;
    void* fn_direct_d_64 = nullptr;
    void* fn_direct_d_32 = nullptr;
    void* fn_direct_f_64 = nullptr;
    void* fn_direct_cached_td = nullptr;
    void* fn_direct_ssss_td_64 = nullptr;
    void* fn_direct_ssss_td_32 = nullptr;
    void* fn_direct_sp_td_64 = nullptr;
    void* fn_direct_sp_td_32 = nullptr;
    void* fn_direct_d_td_64 = nullptr;
    void* fn_direct_d_td_32 = nullptr;
    void* fn_direct_f_td_64 = nullptr;
    void* fn_vxc_grad_force = nullptr;
    void* fn_rij_grad_v = nullptr;
    void* fn_rij_grad_3c = nullptr;
    void* fn_grad4c_filter = nullptr;
    void* fn_grad4c_ssss = nullptr;
    void* fn_grad4c_sp = nullptr;
    void* fn_grad4c_d = nullptr;
    void* fn_grad4c_f = nullptr;

    // L2-cached static tables
    uint64_t d_boys_f64 = 0;
    uint64_t d_boys_f32 = 0;
    uint64_t d_comps6 = 0;
    size_t static_table_bytes = 0;

    // Multi-context maps keyed by handle ID
    std::unordered_map<uint64_t, VxcGridEntry> vxc_grids;
    std::unordered_map<uint64_t, RijTensorEntry> rij_tensors;
    std::unordered_map<uint64_t, RijStreamEntry> rij_streams;
    std::unordered_map<uint64_t, DirectContext> direct_contexts;

    // Shared transient RI-J contraction scratch (serialized by rij_mutex)
    uint64_t scratch_p_ptr = 0;
    size_t scratch_p_cap = 0;
    uint64_t scratch_ao_ptr = 0;
    size_t scratch_ao_cap = 0;

    // Shared F_BLOCK scratch across DirectContexts (serialized by rij_mutex, freed when direct_contexts.empty())
    uint64_t d_dir_f_scratch = 0;
    size_t dir_f_scratch_bytes = 0;
    size_t transient_grad_bytes = 0;

    size_t scratch_bytes() const {
        return (scratch_p_cap + scratch_ao_cap) * sizeof(double);
    }

    size_t total_allocated_bytes() const {
        size_t total = static_table_bytes + scratch_bytes() + dir_f_scratch_bytes + transient_grad_bytes;
        for (const auto& kv : vxc_grids) total += kv.second.bytes;
        for (const auto& kv : rij_tensors) total += kv.second.b_bytes;
        for (const auto& kv : rij_streams) total += kv.second.bytes;
        for (const auto& kv : direct_contexts) total += kv.second.total_entry_bytes();
        return total;
    }

    void release_vxc_resources() {
        for (auto& kv : vxc_grids) kv.second.free_all(cuMemFree);
        vxc_grids.clear();
    }

    void release_rij_scratch_if_idle() {
        if (rij_tensors.empty()) {
            if (scratch_p_ptr && cuMemFree) cuMemFree(scratch_p_ptr);
            scratch_p_ptr = 0;
            scratch_p_cap = 0;
            if (scratch_ao_ptr && cuMemFree) cuMemFree(scratch_ao_ptr);
            scratch_ao_ptr = 0;
            scratch_ao_cap = 0;
        }
    }

    void release_rij_cache() {
        for (auto& kv : rij_tensors) kv.second.free_all(cuMemFree);
        rij_tensors.clear();
        release_rij_scratch_if_idle();
    }

    void release_stream_resources() {
        for (auto& kv : rij_streams) kv.second.free_all(cuMemFree);
        rij_streams.clear();
    }

    void release_direct_resources() {
        for (auto& kv : direct_contexts) kv.second.free_all(cuMemFree);
        direct_contexts.clear();
        if (d_dir_f_scratch && cuMemFree) cuMemFree(d_dir_f_scratch);
        d_dir_f_scratch = 0;
        dir_f_scratch_bytes = 0;
    }

    void release_cached_resources() {
        release_rij_cache();
        if (scratch_p_ptr && cuMemFree) cuMemFree(scratch_p_ptr);
        scratch_p_ptr = 0;
        scratch_p_cap = 0;
        if (scratch_ao_ptr && cuMemFree) cuMemFree(scratch_ao_ptr);
        scratch_ao_ptr = 0;
        scratch_ao_cap = 0;
        release_stream_resources();
        release_vxc_resources();
        release_direct_resources();
        if (cuMemFree) {
            if (d_boys_f64) cuMemFree(d_boys_f64);
            if (d_boys_f32) cuMemFree(d_boys_f32);
            if (d_comps6) cuMemFree(d_comps6);
        }
        d_boys_f64 = d_boys_f32 = d_comps6 = 0;
        static_table_bytes = 0;
        if (rij_mod && cuModuleUnload) {
            cuModuleUnload(rij_mod);
        }
        rij_mod = nullptr;
        fn_rij_accum = fn_rij_v = fn_eri3c = fn_zero_doubles = nullptr;
        fn_vxc_rho = fn_vxc_accum = nullptr;
        fn_direct_filter = fn_direct_cached = nullptr;
        fn_direct_ssss_64 = fn_direct_ssss_32 = nullptr;
        fn_direct_sp_64 = fn_direct_sp_32 = nullptr;
        fn_direct_d_64 = fn_direct_d_32 = fn_direct_f_64 = nullptr;
        fn_direct_cached_td = nullptr;
        fn_direct_ssss_td_64 = fn_direct_ssss_td_32 = nullptr;
        fn_direct_sp_td_64 = fn_direct_sp_td_32 = nullptr;
        fn_direct_d_td_64 = fn_direct_d_td_32 = fn_direct_f_td_64 = nullptr;
        fn_vxc_grad_force = fn_rij_grad_v = fn_rij_grad_3c = nullptr;
        fn_grad4c_filter = fn_grad4c_ssss = fn_grad4c_sp = fn_grad4c_d = fn_grad4c_f = nullptr;
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
        cuMemsetD8 = (PFN_cuMemsetD8_v2)GetProcAddress(hMod, "cuMemsetD8_v2");
        if (!cuMemsetD8) cuMemsetD8 = (PFN_cuMemsetD8_v2)GetProcAddress(hMod, "cuMemsetD8");
        cuModuleLoadData = (PFN_cuModuleLoadData)GetProcAddress(hMod, "cuModuleLoadData");
        cuModuleUnload = (PFN_cuModuleUnload)GetProcAddress(hMod, "cuModuleUnload");
        cuModuleGetFunction = (PFN_cuModuleGetFunction)GetProcAddress(hMod, "cuModuleGetFunction");
        cuFuncSetAttribute = (PFN_cuFuncSetAttribute)GetProcAddress(hMod, "cuFuncSetAttribute");
        cuLaunchKernel = (PFN_cuLaunchKernel)GetProcAddress(hMod, "cuLaunchKernel");
        cuCtxSynchronize = (PFN_cuCtxSynchronize)GetProcAddress(hMod, "cuCtxSynchronize");

        if (!cuInit || !cuDriverGetVersion || !cuDeviceGetCount || !cuDeviceGet ||
            !cuDevicePrimaryCtxRetain || !cuDevicePrimaryCtxRelease || !cuCtxSetCurrent || !cuMemGetInfo ||
            !cuMemAlloc || !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH || !cuMemsetD8 ||
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

    bool ensure_static_tables() {
        if (d_boys_f64 && d_boys_f32 && d_comps6) return true;
        const size_t n_entries = 441 * 26;
        double* h_f64 = (double*)malloc(n_entries * sizeof(double));
        float* h_f32 = (float*)malloc(n_entries * sizeof(float));
        Comp6 h_comps[120];
        if (!h_f64 || !h_f32) {
            if (h_f64) free(h_f64);
            if (h_f32) free(h_f32);
            return false;
        }
        for (int i = 0; i <= 440; ++i) {
            double* row = h_f64 + (size_t)i * 26;
            host_boys_row((double)i * 0.125, row);
            for (int m = 0; m < 26; ++m) {
                h_f32[(size_t)i * 26 + m] = (float)row[m];
            }
        }
        build_host_comps6(h_comps);

        size_t b64 = n_entries * sizeof(double);
        size_t b32 = n_entries * sizeof(float);
        size_t bc6 = sizeof(h_comps);
        if (cuMemAlloc(&d_boys_f64, b64) != 0
            || cuMemAlloc(&d_boys_f32, b32) != 0
            || cuMemAlloc(&d_comps6, bc6) != 0
            || cuMemcpyHtoD(d_boys_f64, h_f64, b64) != 0
            || cuMemcpyHtoD(d_boys_f32, h_f32, b32) != 0
            || cuMemcpyHtoD(d_comps6, h_comps, bc6) != 0) {
            if (d_boys_f64) cuMemFree(d_boys_f64);
            if (d_boys_f32) cuMemFree(d_boys_f32);
            if (d_comps6) cuMemFree(d_comps6);
            d_boys_f64 = d_boys_f32 = d_comps6 = 0;
            free(h_f64);
            free(h_f32);
            return false;
        }
        static_table_bytes = b64 + b32 + bc6;
        free(h_f64);
        free(h_f32);
        return true;
    }

    bool ensure_kernels() {
        if (rij_mod && fn_rij_accum && fn_rij_v) return ensure_static_tables();
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
            fn_rij_accum = fn_rij_v = fn_eri3c = nullptr;
            return false;
        }
        cuModuleGetFunction(&fn_eri3c, rij_mod, "eri3c_build_packed_kernel");
        cuModuleGetFunction(&fn_zero_doubles, rij_mod, "zero_doubles_kernel");
        cuModuleGetFunction(&fn_vxc_rho, rij_mod, "vxc_rho_grad_kernel");
        cuModuleGetFunction(&fn_vxc_accum, rij_mod, "vxc_accum_kernel");
        cuModuleGetFunction(&fn_direct_filter, rij_mod, "direct_filter_worklist_kernel");
        cuModuleGetFunction(&fn_direct_cached, rij_mod, "direct_contract_cached_kernel");
        cuModuleGetFunction(&fn_direct_ssss_64, rij_mod, "direct_eval_ssss_f64_kernel");
        cuModuleGetFunction(&fn_direct_ssss_32, rij_mod, "direct_eval_ssss_f32_kernel");
        cuModuleGetFunction(&fn_direct_sp_64, rij_mod, "direct_eval_sp_f64_kernel");
        cuModuleGetFunction(&fn_direct_sp_32, rij_mod, "direct_eval_sp_f32_kernel");
        cuModuleGetFunction(&fn_direct_d_64, rij_mod, "direct_eval_d_f64_kernel");
        cuModuleGetFunction(&fn_direct_d_32, rij_mod, "direct_eval_d_f32_kernel");
        cuModuleGetFunction(&fn_direct_f_64, rij_mod, "direct_eval_f_f64_kernel");
        cuModuleGetFunction(&fn_direct_cached_td, rij_mod, "direct_contract_cached_td_kernel");
        cuModuleGetFunction(&fn_direct_ssss_td_64, rij_mod, "direct_eval_ssss_td_f64_kernel");
        cuModuleGetFunction(&fn_direct_ssss_td_32, rij_mod, "direct_eval_ssss_td_f32_kernel");
        cuModuleGetFunction(&fn_direct_sp_td_64, rij_mod, "direct_eval_sp_td_f64_kernel");
        cuModuleGetFunction(&fn_direct_sp_td_32, rij_mod, "direct_eval_sp_td_f32_kernel");
        cuModuleGetFunction(&fn_direct_d_td_64, rij_mod, "direct_eval_d_td_f64_kernel");
        cuModuleGetFunction(&fn_direct_d_td_32, rij_mod, "direct_eval_d_td_f32_kernel");
        cuModuleGetFunction(&fn_direct_f_td_64, rij_mod, "direct_eval_f_td_f64_kernel");
        cuModuleGetFunction(&fn_vxc_grad_force, rij_mod, "vxc_grad_force_kernel");
        cuModuleGetFunction(&fn_rij_grad_v, rij_mod, "rij_grad_v_kernel");
        cuModuleGetFunction(&fn_rij_grad_3c, rij_mod, "rij_grad_3c_kernel");
        cuModuleGetFunction(&fn_grad4c_filter, rij_mod, "grad4c_filter_worklist_kernel");
        cuModuleGetFunction(&fn_grad4c_ssss, rij_mod, "grad4c_eval_ssss_f64_kernel");
        cuModuleGetFunction(&fn_grad4c_sp, rij_mod, "grad4c_eval_sp_f64_kernel");
        cuModuleGetFunction(&fn_grad4c_d, rij_mod, "grad4c_eval_d_f64_kernel");
        cuModuleGetFunction(&fn_grad4c_f, rij_mod, "grad4c_eval_f_f64_kernel");

        if (cuFuncSetAttribute) {
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES == 8
            if (fn_direct_d_64) cuFuncSetAttribute(fn_direct_d_64, 8, 65536);
            if (fn_direct_d_32) cuFuncSetAttribute(fn_direct_d_32, 8, 65536);
            if (fn_direct_f_64) cuFuncSetAttribute(fn_direct_f_64, 8, 65536);
            if (fn_direct_d_td_64) cuFuncSetAttribute(fn_direct_d_td_64, 8, 65536);
            if (fn_direct_d_td_32) cuFuncSetAttribute(fn_direct_d_td_32, 8, 65536);
            if (fn_direct_f_td_64) cuFuncSetAttribute(fn_direct_f_td_64, 8, 65536);
        }
        return ensure_static_tables();
    }

    bool check_budget_for_replace(size_t alloc_needed, size_t will_free_bytes, size_t memory_limit_bytes, int* err_out) {
        size_t cur = total_allocated_bytes();
        size_t base = (cur >= will_free_bytes) ? (cur - will_free_bytes) : 0;
        if (alloc_needed > memory_limit_bytes || base > memory_limit_bytes - alloc_needed) {
            *err_out = -2;
            return false;
        }
        if (alloc_needed > 0) {
            size_t free_mem = 0, total_mem = 0;
            if (cuMemGetInfo(&free_mem, &total_mem) != 0) {
                *err_out = -3;
                return false;
            }
            size_t effective_free = free_mem + will_free_bytes;
            if (effective_free < alloc_needed || effective_free - alloc_needed < 64ULL * 1024ULL * 1024ULL) {
                *err_out = -3;
                return false;
            }
        }
        return true;
    }

    bool check_budget_and_free(size_t alloc_needed, size_t memory_limit_bytes, int* err_out) {
        return check_budget_for_replace(alloc_needed, 0, memory_limit_bytes, err_out);
    }

    int ensure_b_and_scratch(const double* h_b_block, int n_p, int n_ao_sq, size_t memory_limit_bytes, uint64_t tensor_id) {
        size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
        auto it = rij_tensors.find(tensor_id);
        bool has_entry = (it != rij_tensors.end());
        bool same_shape = has_entry && it->second.b_ptr && it->second.n_p == n_p && it->second.n_ao_sq == n_ao_sq;
        bool b_hit = same_shape && tensor_id != 0
            && (h_b_block == nullptr ? it->second.device_only : (it->second.h_b == h_b_block));
        if (h_b_block == nullptr && !b_hit) {
            return -2;
        }

        size_t alloc_needed = 0;
        size_t will_free = 0;
        if (!same_shape) {
            if (has_entry) will_free += it->second.b_bytes;
            alloc_needed += b_count * sizeof(double);
        }
        bool grow_p = (size_t)n_p > scratch_p_cap;
        bool grow_ao = (size_t)n_ao_sq > scratch_ao_cap;
        if (grow_p) {
            will_free += scratch_p_cap * sizeof(double);
            alloc_needed += (size_t)n_p * sizeof(double);
        }
        if (grow_ao) {
            will_free += scratch_ao_cap * sizeof(double);
            alloc_needed += (size_t)n_ao_sq * sizeof(double);
        }

        int budget_err = -3;
        if (!check_budget_for_replace(alloc_needed, will_free, memory_limit_bytes, &budget_err)) {
            return budget_err;
        }

        if (!same_shape && has_entry) {
            it->second.free_all(cuMemFree);
            rij_tensors.erase(it);
            has_entry = false;
        }
        if (grow_p) {
            if (scratch_p_ptr) {
                cuMemFree(scratch_p_ptr);
                scratch_p_ptr = 0;
                scratch_p_cap = 0;
            }
            if (cuMemAlloc(&scratch_p_ptr, (size_t)n_p * sizeof(double)) != 0) {
                scratch_p_ptr = 0;
                release_rij_scratch_if_idle();
                return -4;
            }
            scratch_p_cap = (size_t)n_p;
        }
        if (grow_ao) {
            if (scratch_ao_ptr) {
                cuMemFree(scratch_ao_ptr);
                scratch_ao_ptr = 0;
                scratch_ao_cap = 0;
            }
            if (cuMemAlloc(&scratch_ao_ptr, (size_t)n_ao_sq * sizeof(double)) != 0) {
                scratch_ao_ptr = 0;
                release_rij_scratch_if_idle();
                return -4;
            }
            scratch_ao_cap = (size_t)n_ao_sq;
        }

        if (!b_hit) {
            size_t b_bytes = b_count * sizeof(double);
            RijTensorEntry& entry = rij_tensors[tensor_id];
            if (!same_shape) {
                if (cuMemAlloc(&entry.b_ptr, b_bytes) != 0) {
                    entry.b_ptr = 0;
                    entry.b_bytes = 0;
                    rij_tensors.erase(tensor_id);
                    release_rij_scratch_if_idle();
                    return -4;
                }
                entry.b_bytes = b_bytes;
            }
            if (cuMemcpyHtoD(entry.b_ptr, h_b_block, b_bytes) != 0) {
                entry.free_all(cuMemFree);
                rij_tensors.erase(tensor_id);
                release_rij_scratch_if_idle();
                return -5;
            }
            entry.h_b = h_b_block;
            entry.n_p = n_p;
            entry.n_ao_sq = n_ao_sq;
            entry.device_only = false;
        }
        return 0;
    }
};

static CudaDriverApi g_drv;
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

size_t calchem_cuda_total_allocated_bytes() {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    return g_drv.total_allocated_bytes();
#else
    return 0;
#endif
}

static bool safe_mul_sz(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool safe_add_sz(size_t a, size_t b, size_t* out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
}

static int check_rij_budget(int n_p, int n_ao_sq, size_t memory_limit_bytes) {
    size_t b_count = (size_t)n_p * (size_t)n_ao_sq;
    if (n_ao_sq > 0 && b_count / (size_t)n_ao_sq != (size_t)n_p) return -2;
    if (b_count > SIZE_MAX / sizeof(double) - (size_t)n_p - (size_t)n_ao_sq) return -2;
    size_t req_bytes = (b_count + (size_t)n_p + (size_t)n_ao_sq) * sizeof(double);
    if (req_bytes > memory_limit_bytes) {
        return -2;
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
    if ((!h_b_block && tensor_id == 0) || !h_x_block || !h_j || n_p <= 0 || n_ao_sq <= 0) {
        return -1;
    }
    int budget_rc = check_rij_budget(n_p, n_ao_sq, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels()) return -5;
    struct TransientRijGuard {
        uint64_t id;
        ~TransientRijGuard() {
            if (id == 0) {
                auto it = g_drv.rij_tensors.find(0);
                if (it != g_drv.rij_tensors.end()) {
                    it->second.free_all(g_drv.cuMemFree);
                    g_drv.rij_tensors.erase(it);
                }
                g_drv.release_rij_scratch_if_idle();
            }
        }
    } transient_guard{tensor_id};
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, memory_limit_bytes, tensor_id);
    if (prep != 0) return prep;

    auto it = g_drv.rij_tensors.find(tensor_id);
    if (it == g_drv.rij_tensors.end() || !it->second.b_ptr) return -2;
    uint64_t b_ptr = it->second.b_ptr;

    size_t j_bytes = (size_t)n_ao_sq * sizeof(double);
    if (g_drv.cuMemcpyHtoD(g_drv.scratch_p_ptr, h_x_block, (size_t)n_p * sizeof(double)) != 0
        || g_drv.cuMemcpyHtoD(g_drv.scratch_ao_ptr, h_j, j_bytes) != 0) {
        return -5;
    }

    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)(((size_t)n_ao_sq + block_size - 1) / block_size);
    void* params[5] = {
        &b_ptr,
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
    return -5;
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
    if ((!h_b_block && tensor_id == 0) || !h_p_block || !h_v || n_p <= 0 || n_ao_sq <= 0) {
        return -1;
    }
    int budget_rc = check_rij_budget(n_p, n_ao_sq, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels()) return -5;
    struct TransientRijGuard {
        uint64_t id;
        ~TransientRijGuard() {
            if (id == 0) {
                auto it = g_drv.rij_tensors.find(0);
                if (it != g_drv.rij_tensors.end()) {
                    it->second.free_all(g_drv.cuMemFree);
                    g_drv.rij_tensors.erase(it);
                }
                g_drv.release_rij_scratch_if_idle();
            }
        }
    } transient_guard{tensor_id};
    int prep = g_drv.ensure_b_and_scratch(h_b_block, n_p, n_ao_sq, memory_limit_bytes, tensor_id);
    if (prep != 0) return prep;

    auto it = g_drv.rij_tensors.find(tensor_id);
    if (it == g_drv.rij_tensors.end() || !it->second.b_ptr) return -2;
    uint64_t b_ptr = it->second.b_ptr;

    size_t p_bytes = (size_t)n_ao_sq * sizeof(double);
    size_t v_bytes = (size_t)n_p * sizeof(double);
    if (g_drv.cuMemcpyHtoD(g_drv.scratch_ao_ptr, h_p_block, p_bytes) != 0) {
        return -5;
    }

    const unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)n_p;
    void* params[5] = {
        &b_ptr,
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
    return -5;
#endif
}

int calchem_cuda_rij_contract_coulomb(const double* b, const double* x, double* j,
                                      int n, int width, size_t limit) {
    return calchem_cuda_rij_contract_coulomb_tagged(b, x, j, n, width, limit, 0);
}

int calchem_cuda_rij_contract_v(const double* b, const double* p, double* v,
                                int n, int width, size_t limit) {
    return calchem_cuda_rij_contract_v_tagged(b, p, v, n, width, limit, 0);
}

static int aux_n_funcs_host(int lp) {
    switch (lp) {
        case 0: return 1;
        case 1: return 3;
        case 2: return 6;
        case 3: return 10;
        default: return 0;
    }
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
    if (!h_aux_shells || !h_aux_prims || !h_shell_pairs || !h_prim_pairs
        || n_aux_shells <= 0 || n_aux_prims <= 0 || n_shell_pairs <= 0 || n_prim_pairs <= 0
        || n_aux <= 0 || width <= 0) {
        return -1;
    }
    for (int i = 0; i < n_aux_shells; ++i) {
        int np = aux_n_funcs_host(h_aux_shells[i].lp);
        if (np <= 0 || h_aux_shells[i].ao_offset < 0
            || h_aux_shells[i].prim_start < 0 || h_aux_shells[i].n_prims <= 0
            || (size_t)h_aux_shells[i].ao_offset + (size_t)np > (size_t)n_aux
            || (size_t)h_aux_shells[i].prim_start + (size_t)h_aux_shells[i].n_prims > (size_t)n_aux_prims) {
            return -1;
        }
    }
    int budget_rc = check_rij_budget(n_aux, width, memory_limit_bytes);
    if (budget_rc != 0) return budget_rc;

    size_t b_cnt = 0, b_bytes = 0;
    size_t aux_sh_bytes = 0, aux_pr_bytes = 0, sp_bytes = 0, pp_bytes = 0;
    size_t meta_bytes = 0;
    if (!safe_mul_sz((size_t)n_aux, (size_t)width, &b_cnt)
        || !safe_mul_sz(b_cnt, sizeof(double), &b_bytes)
        || !safe_mul_sz((size_t)n_aux_shells, sizeof(AuxShellGpu), &aux_sh_bytes)
        || !safe_mul_sz((size_t)n_aux_prims, sizeof(AuxPrimGpu), &aux_pr_bytes)
        || !safe_mul_sz((size_t)n_shell_pairs, sizeof(ShellPair3CGpu), &sp_bytes)
        || !safe_mul_sz((size_t)n_prim_pairs, sizeof(PrimPair3CGpu), &pp_bytes)
        || !safe_add_sz(aux_sh_bytes, aux_pr_bytes, &meta_bytes)
        || !safe_add_sz(meta_bytes, sp_bytes, &meta_bytes)
        || !safe_add_sz(meta_bytes, pp_bytes, &meta_bytes)) {
        return -2;
    }

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_eri3c) return -5;

    auto it = g_drv.rij_tensors.find(tensor_id);
    bool has_entry = (it != g_drv.rij_tensors.end());
    bool same_shape = has_entry && it->second.b_ptr && it->second.n_p == n_aux && it->second.n_ao_sq == width;
    size_t will_free = (!same_shape && has_entry) ? it->second.b_bytes : 0;
    size_t alloc_needed = 0;
    if (!safe_add_sz(meta_bytes, same_shape ? 0 : b_bytes, &alloc_needed)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_for_replace(alloc_needed, will_free, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }
    if (!same_shape && has_entry) {
        it->second.free_all(g_drv.cuMemFree);
        g_drv.rij_tensors.erase(it);
        has_entry = false;
    }

    RijTensorEntry& entry = g_drv.rij_tensors[tensor_id];
    if (!same_shape) {
        if (g_drv.cuMemAlloc(&entry.b_ptr, b_bytes) != 0) {
            entry.b_ptr = 0;
            entry.b_bytes = 0;
            g_drv.rij_tensors.erase(tensor_id);
            return -4;
        }
        entry.b_bytes = b_bytes;
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
        entry.free_all(g_drv.cuMemFree);
        g_drv.rij_tensors.erase(tensor_id);
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(d_aux_sh, h_aux_shells, aux_sh_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_aux_pr, h_aux_prims, aux_pr_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_sp, h_shell_pairs, sp_bytes) != 0
        || g_drv.cuMemcpyHtoD(d_pp, h_prim_pairs, pp_bytes) != 0) {
        cleanup_meta();
        entry.free_all(g_drv.cuMemFree);
        g_drv.rij_tensors.erase(tensor_id);
        return -5;
    }

    const unsigned int block_x = 128;
    unsigned int grid_x = ((unsigned int)n_shell_pairs + block_x - 1) / block_x;
    unsigned int grid_y = (unsigned int)n_aux_shells;
    int aux_ao_base = 0;
    void* params[10] = {
        &d_aux_sh, &d_aux_pr, &d_sp, &d_pp, &entry.b_ptr,
        &n_aux_shells, &n_shell_pairs, &width, &aux_ao_base, &n_aux
    };
    int rc = g_drv.cuLaunchKernel(
        g_drv.fn_eri3c, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, params, nullptr);
    if (rc == 0) rc = g_drv.cuCtxSynchronize();
    cleanup_meta();
    if (rc != 0) {
        entry.free_all(g_drv.cuMemFree);
        g_drv.rij_tensors.erase(tensor_id);
        return -5;
    }

    if (h_b_out != nullptr) {
        if (g_drv.cuMemcpyDtoH(h_b_out, entry.b_ptr, b_bytes) != 0) {
            entry.free_all(g_drv.cuMemFree);
            g_drv.rij_tensors.erase(tensor_id);
            return -5;
        }
        entry.h_b = h_b_out;
        entry.device_only = false;
    } else {
        entry.h_b = nullptr;
        entry.device_only = true;
    }
    entry.n_p = n_aux;
    entry.n_ao_sq = width;
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_rij_release(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    auto it = g_drv.rij_tensors.find(tensor_id);
    if (it != g_drv.rij_tensors.end()) {
        it->second.free_all(g_drv.cuMemFree);
        g_drv.rij_tensors.erase(it);
    }
    g_drv.release_rij_scratch_if_idle();
    return 0;
#else
    (void)tensor_id;
    return 0;
#endif
}

#ifdef _WIN32
static bool validate_rij_stream_chunk_host(
    const RijStreamEntry& entry,
    int shell_start,
    int chunk_n_shells,
    int ao_start,
    int chunk_n_aux
) {
    if (shell_start < 0 || chunk_n_shells <= 0 || ao_start < 0 || chunk_n_aux <= 0) return false;
    if (chunk_n_aux > entry.max_chunk_n_aux) return false;
    if ((size_t)shell_start + (size_t)chunk_n_shells > entry.aux_shells.size()) return false;
    if (entry.aux_shells[(size_t)shell_start].ao_offset != ao_start) return false;
    size_t ao_end = (size_t)ao_start + (size_t)chunk_n_aux;
    size_t max_end = (size_t)ao_start;
    size_t sum_np = 0;
    for (int i = 0; i < chunk_n_shells; ++i) {
        const AuxShellGpu& sh = entry.aux_shells[(size_t)shell_start + (size_t)i];
        int np = aux_n_funcs_host(sh.lp);
        if (np <= 0 || sh.ao_offset < ao_start) return false;
        size_t end = (size_t)sh.ao_offset + (size_t)np;
        if (end > ao_end) return false;
        if (end > max_end) max_end = end;
        sum_np += (size_t)np;
    }
    return (max_end - (size_t)ao_start == (size_t)chunk_n_aux) && (sum_np == (size_t)chunk_n_aux);
}
#endif

int calchem_cuda_rij_stream_upload(
    const AuxShellGpu* h_aux_shells,
    int n_aux_shells,
    const AuxPrimGpu* h_aux_prims,
    int n_aux_prims,
    const ShellPair3CGpu* h_shell_pairs,
    int n_shell_pairs,
    const PrimPair3CGpu* h_prim_pairs,
    int n_prim_pairs,
    int max_chunk_n_aux,
    int width,
    size_t memory_limit_bytes,
    uint64_t stream_id
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_aux_shells || !h_aux_prims || !h_shell_pairs || !h_prim_pairs
        || n_aux_shells <= 0 || n_aux_prims <= 0 || n_shell_pairs <= 0 || n_prim_pairs <= 0
        || max_chunk_n_aux <= 0 || width <= 0 || stream_id == 0) {
        return -1;
    }
    for (int i = 0; i < n_aux_shells; ++i) {
        int np = aux_n_funcs_host(h_aux_shells[i].lp);
        if (np <= 0 || h_aux_shells[i].ao_offset < 0
            || h_aux_shells[i].prim_start < 0 || h_aux_shells[i].n_prims <= 0
            || (size_t)h_aux_shells[i].prim_start + (size_t)h_aux_shells[i].n_prims > (size_t)n_aux_prims) {
            return -1;
        }
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_eri3c || !g_drv.fn_zero_doubles) return -5;
    if (g_drv.rij_streams.find(stream_id) != g_drv.rij_streams.end()) {
        return -2;
    }

    size_t ash_b = 0, apr_b = 0, sp_b = 0, pp_b = 0, chk_cnt = 0, chk_b = 0, w_b = 0, a_b = 0;
    size_t w2_b = 0, a2_b = 0, total_req = 0;
    if (!safe_mul_sz((size_t)n_aux_shells, sizeof(AuxShellGpu), &ash_b)
        || !safe_mul_sz((size_t)n_aux_prims, sizeof(AuxPrimGpu), &apr_b)
        || !safe_mul_sz((size_t)n_shell_pairs, sizeof(ShellPair3CGpu), &sp_b)
        || !safe_mul_sz((size_t)n_prim_pairs, sizeof(PrimPair3CGpu), &pp_b)
        || !safe_mul_sz((size_t)max_chunk_n_aux, (size_t)width, &chk_cnt)
        || !safe_mul_sz(chk_cnt, sizeof(double), &chk_b)
        || !safe_mul_sz((size_t)width, sizeof(double), &w_b)
        || !safe_mul_sz((size_t)max_chunk_n_aux, sizeof(double), &a_b)
        || !safe_mul_sz(2, w_b, &w2_b)
        || !safe_mul_sz(2, a_b, &a2_b)
        || !safe_add_sz(ash_b, apr_b, &total_req)
        || !safe_add_sz(total_req, sp_b, &total_req)
        || !safe_add_sz(total_req, pp_b, &total_req)
        || !safe_add_sz(total_req, chk_b, &total_req)
        || !safe_add_sz(total_req, w2_b, &total_req)
        || !safe_add_sz(total_req, a2_b, &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }

    RijStreamEntry entry;
    if (g_drv.cuMemAlloc(&entry.d_aux_sh, ash_b) != 0
        || g_drv.cuMemAlloc(&entry.d_aux_pr, apr_b) != 0
        || g_drv.cuMemAlloc(&entry.d_sp, sp_b) != 0
        || g_drv.cuMemAlloc(&entry.d_pp, pp_b) != 0
        || g_drv.cuMemAlloc(&entry.d_b_chunk, chk_b) != 0
        || g_drv.cuMemAlloc(&entry.d_p, w_b) != 0
        || g_drv.cuMemAlloc(&entry.d_j, w_b) != 0
        || g_drv.cuMemAlloc(&entry.d_v, a_b) != 0
        || g_drv.cuMemAlloc(&entry.d_x, a_b) != 0) {
        entry.free_all(g_drv.cuMemFree);
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(entry.d_aux_sh, h_aux_shells, ash_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_aux_pr, h_aux_prims, apr_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_sp, h_shell_pairs, sp_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_pp, h_prim_pairs, pp_b) != 0) {
        entry.free_all(g_drv.cuMemFree);
        return -5;
    }
    entry.bytes = total_req;
    entry.max_chunk_n_aux = max_chunk_n_aux;
    entry.width = width;
    entry.n_shell_pairs = n_shell_pairs;
    entry.n_aux_shells = n_aux_shells;
    entry.p_uploaded = false;
    entry.aux_shells.assign(h_aux_shells, h_aux_shells + n_aux_shells);
    g_drv.rij_streams.emplace(stream_id, std::move(entry));
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_pass1_begin(uint64_t stream_id, const double* h_p_packed, int width) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_p_packed || stream_id == 0 || width <= 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.rij_streams.find(stream_id);
    if (it == g_drv.rij_streams.end() || it->second.width != width) return -2;
    RijStreamEntry& entry = it->second;
    if (g_drv.cuMemcpyHtoD(entry.d_p, h_p_packed, (size_t)width * sizeof(double)) != 0) {
        entry.p_uploaded = false;
        return -5;
    }
    entry.p_uploaded = true;
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_pass1_v(
    uint64_t stream_id,
    const double* h_p_packed,
    int width,
    int shell_start,
    int chunk_n_shells,
    int ao_start,
    int chunk_n_aux,
    double* h_v_out
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_v_out || stream_id == 0 || width <= 0 || shell_start < 0 || chunk_n_shells <= 0
        || ao_start < 0 || chunk_n_aux <= 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.rij_streams.find(stream_id);
    if (it == g_drv.rij_streams.end() || it->second.width != width
        || !validate_rij_stream_chunk_host(it->second, shell_start, chunk_n_shells, ao_start, chunk_n_aux)) {
        return -2;
    }
    RijStreamEntry& entry = it->second;
    if (h_p_packed != nullptr) {
        if (g_drv.cuMemcpyHtoD(entry.d_p, h_p_packed, (size_t)width * sizeof(double)) != 0) {
            entry.p_uploaded = false;
            return -5;
        }
        entry.p_uploaded = true;
    } else if (!entry.p_uploaded) {
        return -2;
    }
    uint64_t d_aux_sh_slice = entry.d_aux_sh + (size_t)shell_start * sizeof(AuxShellGpu);
    const unsigned int block_x = 128;
    unsigned int grid_x = ((unsigned int)entry.n_shell_pairs + block_x - 1) / block_x;
    unsigned int grid_y = (unsigned int)chunk_n_shells;
    void* p_eri[10] = {
        &d_aux_sh_slice, &entry.d_aux_pr, &entry.d_sp, &entry.d_pp,
        &entry.d_b_chunk, &chunk_n_shells, &entry.n_shell_pairs, &width, &ao_start, &chunk_n_aux
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_eri3c, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, p_eri, nullptr) != 0) {
        return -5;
    }
    const unsigned int v_block = 256;
    unsigned int v_grid = (unsigned int)chunk_n_aux;
    void* p_v[5] = {
        &entry.d_b_chunk, &entry.d_p, &entry.d_v, &chunk_n_aux, &width
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_rij_v, v_grid, 1, 1, v_block, 1, 1, 0, nullptr, p_v, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_v_out, entry.d_v, (size_t)chunk_n_aux * sizeof(double)) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_pass2_begin(uint64_t stream_id, int width) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (stream_id == 0 || width <= 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.rij_streams.find(stream_id);
    if (it == g_drv.rij_streams.end() || it->second.width != width) return -2;
    RijStreamEntry& entry = it->second;
    entry.p_uploaded = false;
    size_t len = (size_t)width;
    const unsigned int block = 256;
    unsigned int grid = (unsigned int)((len + block - 1) / block);
    void* params[2] = { &entry.d_j, &len };
    return (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_pass2_accum(
    uint64_t stream_id,
    const double* h_x_chunk,
    int shell_start,
    int chunk_n_shells,
    int ao_start,
    int chunk_n_aux,
    int width
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_x_chunk || stream_id == 0 || width <= 0 || shell_start < 0 || chunk_n_shells <= 0
        || ao_start < 0 || chunk_n_aux <= 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.rij_streams.find(stream_id);
    if (it == g_drv.rij_streams.end() || it->second.width != width
        || !validate_rij_stream_chunk_host(it->second, shell_start, chunk_n_shells, ao_start, chunk_n_aux)) {
        return -2;
    }
    RijStreamEntry& entry = it->second;
    if (g_drv.cuMemcpyHtoD(entry.d_x, h_x_chunk, (size_t)chunk_n_aux * sizeof(double)) != 0) {
        return -5;
    }
    uint64_t d_aux_sh_slice = entry.d_aux_sh + (size_t)shell_start * sizeof(AuxShellGpu);
    const unsigned int block_x = 128;
    unsigned int grid_x = ((unsigned int)entry.n_shell_pairs + block_x - 1) / block_x;
    unsigned int grid_y = (unsigned int)chunk_n_shells;
    void* p_eri[10] = {
        &d_aux_sh_slice, &entry.d_aux_pr, &entry.d_sp, &entry.d_pp,
        &entry.d_b_chunk, &chunk_n_shells, &entry.n_shell_pairs, &width, &ao_start, &chunk_n_aux
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_eri3c, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, p_eri, nullptr) != 0) {
        return -5;
    }
    const unsigned int a_block = 256;
    unsigned int a_grid = (unsigned int)(((size_t)width + a_block - 1) / a_block);
    void* p_acc[5] = {
        &entry.d_b_chunk, &entry.d_x, &entry.d_j, &chunk_n_aux, &width
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_rij_accum, a_grid, 1, 1, a_block, 1, 1, 0, nullptr, p_acc, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_pass2_finish(uint64_t stream_id, double* h_j_packed, int width) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_j_packed || stream_id == 0 || width <= 0) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.rij_streams.find(stream_id);
    if (it == g_drv.rij_streams.end() || it->second.width != width) return -2;
    if (g_drv.cuCtxSynchronize() != 0) return -5;
    return (g_drv.cuMemcpyDtoH(h_j_packed, it->second.d_j, (size_t)width * sizeof(double)) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_rij_stream_release(uint64_t stream_id) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    if (stream_id != 0) {
        auto it = g_drv.rij_streams.find(stream_id);
        if (it != g_drv.rij_streams.end()) {
            it->second.free_all(g_drv.cuMemFree);
            g_drv.rij_streams.erase(it);
        }
    }
    return 0;
#else
    (void)stream_id;
    return 0;
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
    uint64_t grid_id,
    size_t memory_limit_bytes
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
    auto it = g_drv.vxc_grids.find(grid_id);
    if (it != g_drv.vxc_grids.end()) {
        if (it->second.n_pts == n_pts && it->second.n_ao == n_ao) return 0;
        return -2;
    }

    size_t sh_b = 0, pr_b = 0, cp_b = 0, pt_b = 0, co_b = 0, cs_b = 0;
    size_t nn = 0, mat_b = 0, mat2_b = 0, pts4 = 0, rg_b = 0, rg2_b = 0, total_req = 0;
    if (!safe_mul_sz((size_t)n_shells, sizeof(AoShellGpu), &sh_b)
        || !safe_mul_sz((size_t)n_prims, sizeof(AoPrimGpu), &pr_b)
        || !safe_mul_sz((size_t)n_comps, sizeof(AoCompGpu), &cp_b)
        || !safe_mul_sz((size_t)n_pts, sizeof(GridPtGpu), &pt_b)
        || !safe_mul_sz((size_t)n_cand_offsets, sizeof(int), &co_b)
        || !safe_mul_sz((size_t)n_cand_shells, sizeof(int), &cs_b)
        || !safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &mat_b)
        || !safe_mul_sz(2, mat_b, &mat2_b)
        || !safe_mul_sz((size_t)n_pts, 4, &pts4)
        || !safe_mul_sz(pts4, sizeof(double), &rg_b)
        || !safe_mul_sz(2, rg_b, &rg2_b)
        || !safe_add_sz(sh_b, pr_b, &total_req)
        || !safe_add_sz(total_req, cp_b, &total_req)
        || !safe_add_sz(total_req, pt_b, &total_req)
        || !safe_add_sz(total_req, co_b, &total_req)
        || !safe_add_sz(total_req, cs_b, &total_req)
        || !safe_add_sz(total_req, mat2_b, &total_req)
        || !safe_add_sz(total_req, rg2_b, &total_req)
        || !safe_add_sz(total_req, sizeof(int), &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }

    VxcGridEntry entry;
    if (g_drv.cuMemAlloc(&entry.d_shells, sh_b) != 0
        || g_drv.cuMemAlloc(&entry.d_prims, pr_b) != 0
        || g_drv.cuMemAlloc(&entry.d_comps, cp_b) != 0
        || g_drv.cuMemAlloc(&entry.d_pts, pt_b) != 0
        || g_drv.cuMemAlloc(&entry.d_cand_off, co_b) != 0
        || g_drv.cuMemAlloc(&entry.d_cand_sh, cs_b) != 0
        || g_drv.cuMemAlloc(&entry.d_p, mat_b) != 0
        || g_drv.cuMemAlloc(&entry.d_v, mat_b) != 0
        || g_drv.cuMemAlloc(&entry.d_rg, rg_b) != 0
        || g_drv.cuMemAlloc(&entry.d_wv, rg_b) != 0
        || g_drv.cuMemAlloc(&entry.d_ovf, sizeof(int)) != 0) {
        entry.free_all(g_drv.cuMemFree);
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(entry.d_shells, h_shells, sh_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_prims, h_prims, pr_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_comps, h_comps, cp_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_pts, h_pts, pt_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_cand_off, h_cand_offsets, co_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_cand_sh, h_cand_shells, cs_b) != 0) {
        entry.free_all(g_drv.cuMemFree);
        return -5;
    }
    entry.n_pts = n_pts;
    entry.n_ao = n_ao;
    entry.bytes = total_req;
    g_drv.vxc_grids.emplace(grid_id, entry);
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
    auto it = g_drv.vxc_grids.find(grid_id);
    if (it == g_drv.vxc_grids.end() || it->second.n_pts != n_pts || it->second.n_ao != n_ao) return -2;
    VxcGridEntry& entry = it->second;

    size_t mat_b = (size_t)n_ao * (size_t)n_ao * sizeof(double);
    size_t rg_b = (size_t)n_pts * 4 * sizeof(double);
    int zero_ovf = 0;
    if (g_drv.cuMemcpyHtoD(entry.d_p, h_p_mat, mat_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_ovf, &zero_ovf, sizeof(int)) != 0) {
        return -5;
    }
    const unsigned int block = 128;
    unsigned int grid = ((unsigned int)n_pts + block - 1) / block;
    void* params[11] = {
        &entry.d_shells, &entry.d_prims, &entry.d_comps,
        &entry.d_pts, &entry.d_cand_off, &entry.d_cand_sh,
        &entry.d_p, &entry.d_rg, &entry.d_ovf,
        &n_pts, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_vxc_rho, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    int ovf = 0;
    if (g_drv.cuMemcpyDtoH(&ovf, entry.d_ovf, sizeof(int)) != 0 || ovf != 0) {
        return -6;
    }
    return (g_drv.cuMemcpyDtoH(h_rho_grad_out, entry.d_rg, rg_b) == 0) ? 0 : -5;
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
    auto it = g_drv.vxc_grids.find(grid_id);
    if (it == g_drv.vxc_grids.end() || it->second.n_pts != n_pts || it->second.n_ao != n_ao) return -2;
    VxcGridEntry& entry = it->second;

    size_t mat_len = (size_t)n_ao * (size_t)n_ao;
    size_t mat_b = mat_len * sizeof(double);
    size_t wv_b = (size_t)n_pts * 4 * sizeof(double);
    int zero_ovf = 0;
    if (g_drv.cuMemcpyHtoD(entry.d_wv, h_wv_in, wv_b) != 0
        || g_drv.cuMemcpyHtoD(entry.d_ovf, &zero_ovf, sizeof(int)) != 0) {
        return -5;
    }
    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((mat_len + z_block - 1) / z_block);
    void* z_params[2] = { &entry.d_v, &mat_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_params, nullptr) != 0) {
        return -5;
    }
    const unsigned int block = 128;
    unsigned int grid = ((unsigned int)n_pts + block - 1) / block;
    void* params[11] = {
        &entry.d_shells, &entry.d_prims, &entry.d_comps,
        &entry.d_pts, &entry.d_cand_off, &entry.d_cand_sh,
        &entry.d_wv, &entry.d_v, &entry.d_ovf,
        &n_pts, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_vxc_accum, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    int ovf = 0;
    if (g_drv.cuMemcpyDtoH(&ovf, entry.d_ovf, sizeof(int)) != 0 || ovf != 0) {
        return -6;
    }
    return (g_drv.cuMemcpyDtoH(h_vxc_out, entry.d_v, mat_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_vxc_release(uint64_t grid_id) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    if (grid_id != 0) {
        auto it = g_drv.vxc_grids.find(grid_id);
        if (it != g_drv.vxc_grids.end()) {
            it->second.free_all(g_drv.cuMemFree);
            g_drv.vxc_grids.erase(it);
        }
    }
    return 0;
#else
    (void)grid_id;
    return 0;
#endif
}

size_t calchem_cuda_direct_recommended_cache_bytes(
    size_t memory_limit_bytes,
    size_t max_cache_cap
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    if (!g_drv.bind_context()) return 0;
    size_t free_mem = 0, total_mem = 0;
    if (g_drv.cuMemGetInfo(&free_mem, &total_mem) != 0) return 0;
    size_t alloc_cur = g_drv.total_allocated_bytes();
    size_t reserve = 512ULL * 1024ULL * 1024ULL;
    size_t lim_avail = (memory_limit_bytes > alloc_cur + reserve)
        ? (memory_limit_bytes - alloc_cur - reserve) : 0;
    size_t free_avail = (free_mem > reserve) ? (free_mem - reserve) : 0;
    size_t best = max_cache_cap;
    if (lim_avail < best) best = lim_avail;
    if (free_avail < best) best = free_avail;
    return best;
#else
    (void)memory_limit_bytes;
    (void)max_cache_cap;
    return 0;
#endif
}

size_t calchem_cuda_direct_td_scratch_bytes(uint64_t direct_id) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    auto it = g_drv.direct_contexts.find(direct_id);
    if (it == g_drv.direct_contexts.end()) return 0;
    return it->second.td_scratch_bytes();
#else
    (void)direct_id;
    return 0;
#endif
}

int calchem_cuda_direct_upload(
    const DirectShellGpu* h_shells,
    int n_shells,
    const DirectShellPairGpu* h_pairs,
    int n_pairs,
    const DirectPrimPairGpu* h_prims,
    int n_prims,
    const uint64_t* h_pair_prefix,
    int n_prefix,
    int n_ao,
    size_t cache_elements,
    size_t valid_slots,
    size_t memory_limit_bytes,
    uint32_t wl_stride_override,
    uint64_t direct_id
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_shells || !h_pairs || !h_prims || !h_pair_prefix
        || n_shells <= 0 || n_pairs <= 0 || n_prims <= 0 || n_prefix != n_pairs + 1
        || n_ao <= 0 || direct_id == 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_direct_filter || !g_drv.fn_direct_cached
        || !g_drv.fn_direct_ssss_64 || !g_drv.fn_direct_ssss_32
        || !g_drv.fn_direct_sp_64 || !g_drv.fn_direct_sp_32
        || !g_drv.fn_direct_d_64 || !g_drv.fn_direct_d_32
        || !g_drv.fn_direct_f_64 || !g_drv.fn_zero_doubles) {
        return -5;
    }
    if (g_drv.direct_contexts.find(direct_id) != g_drv.direct_contexts.end()) {
        return 0;
    }

    size_t sh_b = 0, pr_b = 0, pp_b = 0, px_b = 0;
    size_t nn = 0, mat_b = 0, mat4_b = 0, ss = 0, d_b = 0, d3_b = 0;
    size_t end_b = 0, c_b = 0, v_b = 0, cnt_b = 0, st_b = 0;
    if (!safe_mul_sz((size_t)n_shells, sizeof(DirectShellGpu), &sh_b)
        || !safe_mul_sz((size_t)n_pairs, sizeof(DirectShellPairGpu), &pr_b)
        || !safe_mul_sz((size_t)n_prims, sizeof(DirectPrimPairGpu), &pp_b)
        || !safe_mul_sz((size_t)n_prefix, sizeof(uint64_t), &px_b)
        || !safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &mat_b)
        || !safe_mul_sz(4, mat_b, &mat4_b)
        || !safe_mul_sz((size_t)n_shells, (size_t)n_shells, &ss)
        || !safe_mul_sz(ss, sizeof(double), &d_b)
        || !safe_mul_sz(3, d_b, &d3_b)
        || !safe_mul_sz((size_t)n_pairs, sizeof(int), &end_b)
        || !safe_mul_sz(cache_elements, sizeof(double), &c_b)
        || !safe_mul_sz(valid_slots, sizeof(uint8_t), &v_b)
        || !safe_mul_sz((size_t)NUM_WL_BUCKETS, sizeof(int), &cnt_b)
        || !safe_mul_sz(6, sizeof(unsigned long long), &st_b)) {
        return -2;
    }

    // Choose worklist stride up to 2,000,000 items (max 128 MiB across 8 buckets)
    uint32_t wl_stride = 2000000u;
    if ((size_t)n_pairs <= 65536u) {
        size_t max_pairs_cand = (size_t)n_pairs * (size_t)(n_pairs + 1) / 2;
        if (max_pairs_cand < (size_t)wl_stride) {
            wl_stride = (uint32_t)(max_pairs_cand > 1024 ? max_pairs_cand : 1024);
        }
    }
    if (wl_stride_override > 0 && wl_stride_override < wl_stride) {
        wl_stride = wl_stride_override;
    }
    size_t base_req = 0;
    if (!safe_add_sz(sh_b, pr_b, &base_req)
        || !safe_add_sz(base_req, pp_b, &base_req)
        || !safe_add_sz(base_req, px_b, &base_req)
        || !safe_add_sz(base_req, mat4_b, &base_req)
        || !safe_add_sz(base_req, d3_b, &base_req)
        || !safe_add_sz(base_req, end_b, &base_req)
        || !safe_add_sz(base_req, c_b, &base_req)
        || !safe_add_sz(base_req, v_b, &base_req)
        || !safe_add_sz(base_req, cnt_b, &base_req)
        || !safe_add_sz(base_req, st_b, &base_req)) {
        return -2;
    }
    size_t wl_cnt = 0, wl_b = 0, total_req = 0;
    if (!safe_mul_sz((size_t)NUM_WL_BUCKETS, (size_t)wl_stride, &wl_cnt)
        || !safe_mul_sz(wl_cnt, sizeof(WorkItem), &wl_b)
        || !safe_add_sz(base_req, wl_b, &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        if (wl_stride > 250000u) {
            wl_stride = 250000u;
            if (!safe_mul_sz((size_t)NUM_WL_BUCKETS, (size_t)wl_stride, &wl_cnt)
                || !safe_mul_sz(wl_cnt, sizeof(WorkItem), &wl_b)
                || !safe_add_sz(base_req, wl_b, &total_req)) {
                return -2;
            }
            if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
                return budget_err;
            }
        } else {
            return budget_err;
        }
    }

    DirectContext ctx;
    if (g_drv.cuMemAlloc(&ctx.d_shells, sh_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_pairs, pr_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_prims, pp_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_prefix, px_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_pa, mat_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_pb, mat_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_out_a, mat_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_out_b, mat_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_dt, d_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_da, d_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_db, d_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_ends, end_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_wl_counts, cnt_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_stats, st_b) != 0
        || g_drv.cuMemAlloc(&ctx.d_wl, wl_b) != 0
        || (c_b > 0 && g_drv.cuMemAlloc(&ctx.d_cache, c_b) != 0)
        || (v_b > 0 && g_drv.cuMemAlloc(&ctx.d_valid, v_b) != 0)) {
        ctx.free_all(g_drv.cuMemFree);
        return -4;
    }
    if (g_drv.cuMemcpyHtoD(ctx.d_shells, h_shells, sh_b) != 0
        || g_drv.cuMemcpyHtoD(ctx.d_pairs, h_pairs, pr_b) != 0
        || g_drv.cuMemcpyHtoD(ctx.d_prims, h_prims, pp_b) != 0
        || g_drv.cuMemcpyHtoD(ctx.d_prefix, h_pair_prefix, px_b) != 0
        || (v_b > 0 && g_drv.cuMemsetD8(ctx.d_valid, 0, v_b) != 0)) {
        ctx.free_all(g_drv.cuMemFree);
        return -5;
    }

    ctx.n_ao = n_ao;
    ctx.n_shells = n_shells;
    ctx.n_pairs = n_pairs;
    ctx.bytes = total_req;
    ctx.memory_limit_bytes = memory_limit_bytes;
    ctx.wl_stride = wl_stride;
    g_drv.direct_contexts.emplace(direct_id, ctx);
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_direct_contract(
    uint64_t direct_id,
    int mode,
    const double* h_p_a,
    const double* h_p_b,
    const double* h_d_t,
    const double* h_d_a,
    const double* h_d_b,
    int active_pairs,
    const int* h_ends,
    double exx,
    double omega,
    int has_omega,
    double thresh,
    double fp32_thresh,
    double* h_out_a,
    double* h_out_b,
    DirectStatsGpu* out_stats
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (direct_id == 0 || !h_p_a || !h_ends || !h_out_a || active_pairs <= 0) return -1;
    if ((mode == 2 || mode == 3) && (!h_p_b || !h_out_b)) return -1;
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.direct_contexts.find(direct_id);
    if (it == g_drv.direct_contexts.end() || active_pairs > it->second.n_pairs) return -2;
    DirectContext& ctx = it->second;

    for (int pi = 0; pi < active_pairs; ++pi) {
        if (h_ends[pi] < 0 || h_ends[pi] > ctx.n_pairs) return -1;
    }

    int n = ctx.n_ao;
    int n_sh = ctx.n_shells;
    size_t nn = (size_t)n * (size_t)n;
    size_t mat_b = nn * sizeof(double);
    size_t d_b = (size_t)n_sh * (size_t)n_sh * sizeof(double);

    if (g_drv.cuMemcpyHtoD(ctx.d_pa, h_p_a, mat_b) != 0) return -5;
    if ((mode == 2 || mode == 3) && g_drv.cuMemcpyHtoD(ctx.d_pb, h_p_b, mat_b) != 0) return -5;
    if (h_d_t && g_drv.cuMemcpyHtoD(ctx.d_dt, h_d_t, d_b) != 0) return -5;
    if (h_d_a && g_drv.cuMemcpyHtoD(ctx.d_da, h_d_a, d_b) != 0) return -5;
    if (h_d_b && g_drv.cuMemcpyHtoD(ctx.d_db, h_d_b, d_b) != 0) return -5;
    if (g_drv.cuMemcpyHtoD(ctx.d_ends, h_ends, (size_t)active_pairs * sizeof(int)) != 0) return -5;

    // Zero output buffers and stats
    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((nn + z_block - 1) / z_block);
    void* z_pa[2] = { &ctx.d_out_a, &nn };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_pa, nullptr) != 0) return -5;
    if (mode == 2 || mode == 3) {
        void* z_pb[2] = { &ctx.d_out_b, &nn };
        if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_pb, nullptr) != 0) return -5;
    }
    if (g_drv.cuMemsetD8(ctx.d_stats, 0, 6 * sizeof(unsigned long long)) != 0) return -5;

    double k_weight_dummy = 0.0;
    int pi_start = 0;
    while (pi_start < active_pairs) {
        uint64_t chunk_sum = 0;
        int pi_end = pi_start;
        while (pi_end < active_pairs) {
            uint64_t e = (uint64_t)(h_ends[pi_end] > 0 ? h_ends[pi_end] : 0);
            if (pi_end > pi_start && chunk_sum + e > (uint64_t)ctx.wl_stride) break;
            chunk_sum += e;
            pi_end++;
            if (chunk_sum > (uint64_t)ctx.wl_stride) break;
        }
        int n_act = pi_end - pi_start;
        if (chunk_sum == 0) {
            pi_start = pi_end;
            continue;
        }

        int total_ket_end = (n_act == 1 && chunk_sum > (uint64_t)ctx.wl_stride) ? (int)chunk_sum : -1;
        int qi_s = 0;
        do {
            int qi_e = -1;
            if (total_ket_end > 0) {
                int64_t next_e = (int64_t)qi_s + (int64_t)ctx.wl_stride;
                qi_e = (next_e < (int64_t)total_ket_end) ? (int)next_e : total_ket_end;
            }

            if (g_drv.cuMemsetD8(ctx.d_wl_counts, 0, NUM_WL_BUCKETS * sizeof(int)) != 0) return -5;
            void* f_params[22] = {
                &ctx.d_shells, &ctx.d_pairs, &ctx.d_ends,
                &ctx.d_dt, &ctx.d_da, &ctx.d_db,
                &ctx.d_valid, &ctx.d_wl, &ctx.wl_stride,
                &ctx.d_wl_counts, &ctx.d_stats,
                &pi_start, &n_act, &n_sh, &mode, &exx, &has_omega, &thresh, &fp32_thresh,
                &qi_s, &qi_e, &k_weight_dummy
            };
            if (g_drv.cuLaunchKernel(g_drv.fn_direct_filter, (unsigned int)n_act, 1, 1, 128, 1, 1, 0, nullptr, f_params, nullptr) != 0
                || g_drv.cuCtxSynchronize() != 0) {
                return -5;
            }
            int counts[NUM_WL_BUCKETS] = {};
            if (g_drv.cuMemcpyDtoH(counts, ctx.d_wl_counts, sizeof(counts)) != 0) return -5;

            if (counts[WL_HIT] > 0) {
                int nh = counts[WL_HIT];
                uint64_t wl_ptr = ctx.d_wl + (size_t)WL_HIT * (size_t)ctx.wl_stride * sizeof(WorkItem);
                const unsigned int blk = 128; // 4 warps
                unsigned int grd = ((unsigned int)nh + 3) / 4;
                void* p_hit[13] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prefix,
                    &ctx.d_cache, &wl_ptr, &nh, &n, &mode, &exx,
                    &ctx.d_pa, &ctx.d_pb, &ctx.d_out_a, &ctx.d_out_b
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_direct_cached, grd, 1, 1, blk, 1, 1, 0, nullptr, p_hit, nullptr) != 0) return -5;
            }

            auto launch_bucket = [&](int b_idx, void* fn, int items_per_blk, int blk_threads, size_t dyn_smem) -> int {
                int cnt = counts[b_idx];
                if (cnt <= 0) return 0;
                if (!fn) return -5;
                uint64_t wl_ptr = ctx.d_wl + (size_t)b_idx * (size_t)ctx.wl_stride * sizeof(WorkItem);
                unsigned int grd = (unsigned int)((cnt + items_per_blk - 1) / items_per_blk);
                void* params[19] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prims, &ctx.d_prefix,
                    &ctx.d_cache, &ctx.d_valid, &wl_ptr, &cnt, &n, &mode,
                    &exx, &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_boys_f32,
                    &ctx.d_pa, &ctx.d_pb, &ctx.d_out_a, &ctx.d_out_b
                };
                return (g_drv.cuLaunchKernel(fn, grd, 1, 1, (unsigned int)blk_threads, 1, 1, (unsigned int)dyn_smem, nullptr, params, nullptr) == 0) ? 0 : -5;
            };

            if (launch_bucket(WL_SSSS_64, g_drv.fn_direct_ssss_64, 128, 128, 0) != 0) return -5;
            if (launch_bucket(WL_SSSS_32, g_drv.fn_direct_ssss_32, 128, 128, 0) != 0) return -5;
            if (launch_bucket(WL_SP_64, g_drv.fn_direct_sp_64, 128, 128, 0) != 0) return -5;
            if (launch_bucket(WL_SP_32, g_drv.fn_direct_sp_32, 128, 128, 0) != 0) return -5;
            if (launch_bucket(WL_D_64, g_drv.fn_direct_d_64, 2, 64, (size_t)(2 * DWARP_STRIDE_T) * sizeof(double)) != 0) return -5;
            if (launch_bucket(WL_D_32, g_drv.fn_direct_d_32, 2, 64, (size_t)(2 * DWARP_STRIDE_T) * sizeof(float)) != 0) return -5;

            if (counts[WL_F_64] > 0) {
                if (!g_drv.fn_direct_f_64) return -5;
                int nf = counts[WL_F_64];
                unsigned int f_grid = (unsigned int)(nf < 64 ? nf : 64);
                const size_t block_f_bytes = (size_t)(84 * 84 * 13) * sizeof(double);
                size_t need_f = (size_t)f_grid * block_f_bytes;
                if (need_f > g_drv.dir_f_scratch_bytes) {
                    bool fit = false;
                    while (f_grid >= 1) {
                        need_f = (size_t)f_grid * block_f_bytes;
                        if (need_f <= g_drv.dir_f_scratch_bytes) {
                            fit = true;
                            break;
                        }
                        int budget_err = -3;
                        if (g_drv.check_budget_for_replace(need_f, g_drv.dir_f_scratch_bytes, ctx.memory_limit_bytes, &budget_err)) {
                            fit = true;
                            break;
                        }
                        if (f_grid == 1) break;
                        f_grid /= 2;
                    }
                    if (!fit) {
                        return -4;
                    }
                    if (need_f > g_drv.dir_f_scratch_bytes) {
                        if (g_drv.d_dir_f_scratch) {
                            g_drv.cuMemFree(g_drv.d_dir_f_scratch);
                            g_drv.d_dir_f_scratch = 0;
                            g_drv.dir_f_scratch_bytes = 0;
                        }
                        if (g_drv.cuMemAlloc(&g_drv.d_dir_f_scratch, need_f) != 0) {
                            g_drv.d_dir_f_scratch = 0;
                            return -4;
                        }
                        g_drv.dir_f_scratch_bytes = need_f;
                    }
                }
                uint64_t wl_ptr = ctx.d_wl + (size_t)WL_F_64 * (size_t)ctx.wl_stride * sizeof(WorkItem);
                void* f_arg[20] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prims, &ctx.d_prefix,
                    &ctx.d_cache, &ctx.d_valid, &wl_ptr, &nf, &n, &mode,
                    &exx, &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_comps6, &g_drv.d_dir_f_scratch,
                    &ctx.d_pa, &ctx.d_pb, &ctx.d_out_a, &ctx.d_out_b
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_direct_f_64, f_grid, 1, 1, 64, 1, 1, (unsigned int)(84 * 84 * sizeof(double)), nullptr, f_arg, nullptr) != 0) return -5;
            }

            if (g_drv.cuCtxSynchronize() != 0) return -5;
            if (total_ket_end > 0) {
                qi_s = qi_e;
            }
        } while (total_ket_end > 0 && qi_s < total_ket_end);

        pi_start = pi_end;
    }

    if (g_drv.cuMemcpyDtoH(h_out_a, ctx.d_out_a, mat_b) != 0) return -5;
    if ((mode == 2 || mode == 3) && g_drv.cuMemcpyDtoH(h_out_b, ctx.d_out_b, mat_b) != 0) return -5;
    if (out_stats) {
        unsigned long long raw_stats[6] = {};
        if (g_drv.cuMemcpyDtoH(raw_stats, ctx.d_stats, sizeof(raw_stats)) != 0) return -5;
        out_stats->computed_quartets = (uint64_t)raw_stats[0];
        out_stats->skipped_quartets = (uint64_t)raw_stats[1];
        out_stats->cache_hits = (uint64_t)raw_stats[2];
        out_stats->cache_misses = (uint64_t)raw_stats[3];
        out_stats->fp64_quartets = (uint64_t)raw_stats[4];
        out_stats->fp32_quartets = (uint64_t)raw_stats[5];
    }
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_direct_contract_tddft_batch(
    uint64_t direct_id,
    int b_sym,
    int b_anti,
    int do_j,
    int do_k_sym,
    int do_k_anti,
    const double* h_d_sym_packed,
    const double* h_d_anti_packed,
    const double* h_dj_env,
    const double* h_dk_env,
    int active_pairs,
    const int* h_ends,
    double j_weight,
    double k_weight,
    double omega,
    int has_omega,
    double thresh,
    double fp32_thresh,
    double* h_out_j_packed,
    double* h_out_ks_packed,
    double* h_out_ka_packed,
    DirectStatsGpu* out_stats
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (direct_id == 0 || !h_ends || active_pairs <= 0 || b_sym < 0 || b_anti < 0) return -1;
    int eff_do_j = (do_j && b_sym > 0) ? 1 : 0;
    int eff_do_ks = (do_k_sym && b_sym > 0) ? 1 : 0;
    int eff_do_ka = (do_k_anti && b_anti > 0) ? 1 : 0;
    if (!eff_do_j && !eff_do_ks && !eff_do_ka) {
        if (out_stats) memset(out_stats, 0, sizeof(DirectStatsGpu));
        return 0;
    }
    if ((eff_do_j || eff_do_ks) && !h_d_sym_packed) return -1;
    if (eff_do_ka && !h_d_anti_packed) return -1;
    if (eff_do_j && (!h_dj_env || !h_out_j_packed)) return -1;
    if (eff_do_ks && (!h_dk_env || !h_out_ks_packed)) return -1;
    if (eff_do_ka && (!h_dk_env || !h_out_ka_packed)) return -1;

#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    auto it = g_drv.direct_contexts.find(direct_id);
    if (it == g_drv.direct_contexts.end() || active_pairs > it->second.n_pairs) return -2;
    if (!g_drv.fn_direct_filter || !g_drv.fn_zero_doubles
        || !g_drv.fn_direct_cached_td
        || !g_drv.fn_direct_ssss_td_64 || !g_drv.fn_direct_ssss_td_32
        || !g_drv.fn_direct_sp_td_64 || !g_drv.fn_direct_sp_td_32
        || !g_drv.fn_direct_d_td_64 || !g_drv.fn_direct_d_td_32
        || !g_drv.fn_direct_f_td_64) {
        return -5;
    }
    DirectContext& ctx = it->second;

    for (int pi = 0; pi < active_pairs; ++pi) {
        if (h_ends[pi] < 0 || h_ends[pi] > ctx.n_pairs) return -1;
    }

    int n = ctx.n_ao;
    int n_sh = ctx.n_shells;
    size_t nn = (size_t)n * (size_t)n;
    size_t mat_b = nn * sizeof(double);
    size_t d_b = (size_t)n_sh * (size_t)n_sh * sizeof(double);

    size_t sym_need_b    = (eff_do_j || eff_do_ks) ? ((size_t)b_sym * mat_b) : 0;
    size_t anti_need_b   = eff_do_ka ? ((size_t)b_anti * mat_b) : 0;
    size_t out_j_need_b  = eff_do_j  ? ((size_t)b_sym * mat_b) : 0;
    size_t out_ks_need_b = eff_do_ks ? ((size_t)b_sym * mat_b) : 0;
    size_t out_ka_need_b = eff_do_ka ? ((size_t)b_anti * mat_b) : 0;

    size_t will_free = 0;
    size_t alloc_needed = 0;
    if (sym_need_b > ctx.td_sym_cap_bytes) { will_free += ctx.td_sym_cap_bytes; alloc_needed += sym_need_b; }
    if (anti_need_b > ctx.td_anti_cap_bytes) { will_free += ctx.td_anti_cap_bytes; alloc_needed += anti_need_b; }
    if (out_j_need_b > ctx.td_out_j_cap_bytes) { will_free += ctx.td_out_j_cap_bytes; alloc_needed += out_j_need_b; }
    if (out_ks_need_b > ctx.td_out_ks_cap_bytes) { will_free += ctx.td_out_ks_cap_bytes; alloc_needed += out_ks_need_b; }
    if (out_ka_need_b > ctx.td_out_ka_cap_bytes) { will_free += ctx.td_out_ka_cap_bytes; alloc_needed += out_ka_need_b; }

    if (alloc_needed > 0) {
        int budget_err = -3;
        if (!g_drv.check_budget_for_replace(alloc_needed, will_free, ctx.memory_limit_bytes, &budget_err)) {
            return budget_err;
        }
        auto ensure_buf = [&](uint64_t& ptr, size_t& cap, size_t need) -> bool {
            if (need <= cap) return true;
            if (ptr) {
                g_drv.cuMemFree(ptr);
                ptr = 0;
                cap = 0;
            }
            if (g_drv.cuMemAlloc(&ptr, need) != 0) {
                ptr = 0;
                cap = 0;
                return false;
            }
            cap = need;
            return true;
        };
        if (!ensure_buf(ctx.d_td_sym, ctx.td_sym_cap_bytes, sym_need_b)
            || !ensure_buf(ctx.d_td_anti, ctx.td_anti_cap_bytes, anti_need_b)
            || !ensure_buf(ctx.d_td_out_j, ctx.td_out_j_cap_bytes, out_j_need_b)
            || !ensure_buf(ctx.d_td_out_ks, ctx.td_out_ks_cap_bytes, out_ks_need_b)
            || !ensure_buf(ctx.d_td_out_ka, ctx.td_out_ka_cap_bytes, out_ka_need_b)) {
            ctx.free_td_scratch(g_drv.cuMemFree);
            return -4;
        }
    }

    if (sym_need_b > 0 && g_drv.cuMemcpyHtoD(ctx.d_td_sym, h_d_sym_packed, sym_need_b) != 0) return -5;
    if (anti_need_b > 0 && g_drv.cuMemcpyHtoD(ctx.d_td_anti, h_d_anti_packed, anti_need_b) != 0) return -5;
    if (eff_do_j && g_drv.cuMemcpyHtoD(ctx.d_dt, h_dj_env, d_b) != 0) return -5;
    if ((eff_do_ks || eff_do_ka) && g_drv.cuMemcpyHtoD(ctx.d_da, h_dk_env, d_b) != 0) return -5;
    if (g_drv.cuMemcpyHtoD(ctx.d_ends, h_ends, (size_t)active_pairs * sizeof(int)) != 0) return -5;

    const unsigned int z_block = 256;
    if (eff_do_j) {
        size_t len = (size_t)b_sym * nn;
        unsigned int z_grid = (unsigned int)((len + z_block - 1) / z_block);
        void* z_p[2] = { &ctx.d_td_out_j, &len };
        if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) return -5;
    }
    if (eff_do_ks) {
        size_t len = (size_t)b_sym * nn;
        unsigned int z_grid = (unsigned int)((len + z_block - 1) / z_block);
        void* z_p[2] = { &ctx.d_td_out_ks, &len };
        if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) return -5;
    }
    if (eff_do_ka) {
        size_t len = (size_t)b_anti * nn;
        unsigned int z_grid = (unsigned int)((len + z_block - 1) / z_block);
        void* z_p[2] = { &ctx.d_td_out_ka, &len };
        if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) return -5;
    }
    if (g_drv.cuMemsetD8(ctx.d_stats, 0, 6 * sizeof(unsigned long long)) != 0) return -5;

    int mode_td = 4;
    double filt_jw = eff_do_j ? j_weight : 0.0;
    double filt_kw = (eff_do_ks || eff_do_ka) ? k_weight : 0.0;
    int eff_b_sym = (eff_do_j || eff_do_ks) ? b_sym : 0;
    int eff_b_anti = eff_do_ka ? b_anti : 0;

    int pi_start = 0;
    while (pi_start < active_pairs) {
        uint64_t chunk_sum = 0;
        int pi_end = pi_start;
        while (pi_end < active_pairs) {
            uint64_t e = (uint64_t)(h_ends[pi_end] > 0 ? h_ends[pi_end] : 0);
            if (pi_end > pi_start && chunk_sum + e > (uint64_t)ctx.wl_stride) break;
            chunk_sum += e;
            pi_end++;
            if (chunk_sum > (uint64_t)ctx.wl_stride) break;
        }
        int n_act = pi_end - pi_start;
        if (chunk_sum == 0) {
            pi_start = pi_end;
            continue;
        }

        int total_ket_end = (n_act == 1 && chunk_sum > (uint64_t)ctx.wl_stride) ? (int)chunk_sum : -1;
        int qi_s = 0;
        do {
            int qi_e = -1;
            if (total_ket_end > 0) {
                int64_t next_e = (int64_t)qi_s + (int64_t)ctx.wl_stride;
                qi_e = (next_e < (int64_t)total_ket_end) ? (int)next_e : total_ket_end;
            }

            if (g_drv.cuMemsetD8(ctx.d_wl_counts, 0, NUM_WL_BUCKETS * sizeof(int)) != 0) return -5;
            void* f_params[22] = {
                &ctx.d_shells, &ctx.d_pairs, &ctx.d_ends,
                &ctx.d_dt, &ctx.d_da, &ctx.d_db,
                &ctx.d_valid, &ctx.d_wl, &ctx.wl_stride,
                &ctx.d_wl_counts, &ctx.d_stats,
                &pi_start, &n_act, &n_sh, &mode_td, &filt_jw, &has_omega, &thresh, &fp32_thresh,
                &qi_s, &qi_e, &filt_kw
            };
            if (g_drv.cuLaunchKernel(g_drv.fn_direct_filter, (unsigned int)n_act, 1, 1, 128, 1, 1, 0, nullptr, f_params, nullptr) != 0
                || g_drv.cuCtxSynchronize() != 0) {
                return -5;
            }
            int counts[NUM_WL_BUCKETS] = {};
            if (g_drv.cuMemcpyDtoH(counts, ctx.d_wl_counts, sizeof(counts)) != 0) return -5;

            if (counts[WL_HIT] > 0) {
                int nh = counts[WL_HIT];
                uint64_t wl_ptr = ctx.d_wl + (size_t)WL_HIT * (size_t)ctx.wl_stride * sizeof(WorkItem);
                const unsigned int blk = 128; // 4 warps
                unsigned int grd = ((unsigned int)nh + 3) / 4;
                void* p_hit[17] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prefix,
                    &ctx.d_cache, &wl_ptr, &nh, &n,
                    &eff_b_sym, &eff_b_anti, &eff_do_j, &eff_do_ks, &eff_do_ka,
                    &ctx.d_td_sym, &ctx.d_td_anti,
                    &ctx.d_td_out_j, &ctx.d_td_out_ks, &ctx.d_td_out_ka
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_direct_cached_td, grd, 1, 1, blk, 1, 1, 0, nullptr, p_hit, nullptr) != 0) return -5;
            }

            auto launch_td_bucket = [&](int b_idx, void* fn, int items_per_blk, int blk_threads, size_t dyn_smem) -> int {
                int cnt = counts[b_idx];
                if (cnt <= 0) return 0;
                if (!fn) return -5;
                uint64_t wl_ptr = ctx.d_wl + (size_t)b_idx * (size_t)ctx.wl_stride * sizeof(WorkItem);
                unsigned int grd = (unsigned int)((cnt + items_per_blk - 1) / items_per_blk);
                void* params[23] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prims, &ctx.d_prefix,
                    &ctx.d_cache, &ctx.d_valid, &wl_ptr, &cnt, &n,
                    &eff_b_sym, &eff_b_anti, &eff_do_j, &eff_do_ks, &eff_do_ka,
                    &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_boys_f32,
                    &ctx.d_td_sym, &ctx.d_td_anti,
                    &ctx.d_td_out_j, &ctx.d_td_out_ks, &ctx.d_td_out_ka
                };
                return (g_drv.cuLaunchKernel(fn, grd, 1, 1, (unsigned int)blk_threads, 1, 1, (unsigned int)dyn_smem, nullptr, params, nullptr) == 0) ? 0 : -5;
            };

            if (launch_td_bucket(WL_SSSS_64, g_drv.fn_direct_ssss_td_64, 128, 128, 0) != 0) return -5;
            if (launch_td_bucket(WL_SSSS_32, g_drv.fn_direct_ssss_td_32, 128, 128, 0) != 0) return -5;
            if (launch_td_bucket(WL_SP_64, g_drv.fn_direct_sp_td_64, 128, 128, 0) != 0) return -5;
            if (launch_td_bucket(WL_SP_32, g_drv.fn_direct_sp_td_32, 128, 128, 0) != 0) return -5;
            if (launch_td_bucket(WL_D_64, g_drv.fn_direct_d_td_64, 2, 64, (size_t)(2 * DWARP_STRIDE_T) * sizeof(double)) != 0) return -5;
            if (launch_td_bucket(WL_D_32, g_drv.fn_direct_d_td_32, 2, 64, (size_t)(2 * DWARP_STRIDE_T) * sizeof(float)) != 0) return -5;

            if (counts[WL_F_64] > 0) {
                int nf = counts[WL_F_64];
                unsigned int f_grid = (unsigned int)(nf < 64 ? nf : 64);
                const size_t block_f_bytes = (size_t)(84 * 84 * 13) * sizeof(double);
                size_t need_f = (size_t)f_grid * block_f_bytes;
                if (need_f > g_drv.dir_f_scratch_bytes) {
                    bool fit = false;
                    while (f_grid >= 1) {
                        need_f = (size_t)f_grid * block_f_bytes;
                        if (need_f <= g_drv.dir_f_scratch_bytes) {
                            fit = true;
                            break;
                        }
                        int budget_err = -3;
                        if (g_drv.check_budget_for_replace(need_f, g_drv.dir_f_scratch_bytes, ctx.memory_limit_bytes, &budget_err)) {
                            fit = true;
                            break;
                        }
                        if (f_grid == 1) break;
                        f_grid /= 2;
                    }
                    if (!fit) {
                        return -4;
                    }
                    if (need_f > g_drv.dir_f_scratch_bytes) {
                        if (g_drv.d_dir_f_scratch) {
                            g_drv.cuMemFree(g_drv.d_dir_f_scratch);
                            g_drv.d_dir_f_scratch = 0;
                            g_drv.dir_f_scratch_bytes = 0;
                        }
                        if (g_drv.cuMemAlloc(&g_drv.d_dir_f_scratch, need_f) != 0) {
                            g_drv.d_dir_f_scratch = 0;
                            return -4;
                        }
                        g_drv.dir_f_scratch_bytes = need_f;
                    }
                }
                uint64_t wl_ptr = ctx.d_wl + (size_t)WL_F_64 * (size_t)ctx.wl_stride * sizeof(WorkItem);
                void* f_arg[24] = {
                    &ctx.d_shells, &ctx.d_pairs, &ctx.d_prims, &ctx.d_prefix,
                    &ctx.d_cache, &ctx.d_valid, &wl_ptr, &nf, &n,
                    &eff_b_sym, &eff_b_anti, &eff_do_j, &eff_do_ks, &eff_do_ka,
                    &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_comps6, &g_drv.d_dir_f_scratch,
                    &ctx.d_td_sym, &ctx.d_td_anti,
                    &ctx.d_td_out_j, &ctx.d_td_out_ks, &ctx.d_td_out_ka
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_direct_f_td_64, f_grid, 1, 1, 64, 1, 1, (unsigned int)(84 * 84 * sizeof(double)), nullptr, f_arg, nullptr) != 0) return -5;
            }

            if (g_drv.cuCtxSynchronize() != 0) return -5;
            if (total_ket_end > 0) {
                qi_s = qi_e;
            }
        } while (total_ket_end > 0 && qi_s < total_ket_end);

        pi_start = pi_end;
    }

    if (eff_do_j && g_drv.cuMemcpyDtoH(h_out_j_packed, ctx.d_td_out_j, out_j_need_b) != 0) return -5;
    if (eff_do_ks && g_drv.cuMemcpyDtoH(h_out_ks_packed, ctx.d_td_out_ks, out_ks_need_b) != 0) return -5;
    if (eff_do_ka && g_drv.cuMemcpyDtoH(h_out_ka_packed, ctx.d_td_out_ka, out_ka_need_b) != 0) return -5;
    if (out_stats) {
        unsigned long long raw_stats[6] = {};
        if (g_drv.cuMemcpyDtoH(raw_stats, ctx.d_stats, sizeof(raw_stats)) != 0) return -5;
        out_stats->computed_quartets = (uint64_t)raw_stats[0];
        out_stats->skipped_quartets = (uint64_t)raw_stats[1];
        out_stats->cache_hits = (uint64_t)raw_stats[2];
        out_stats->cache_misses = (uint64_t)raw_stats[3];
        out_stats->fp64_quartets = (uint64_t)raw_stats[4];
        out_stats->fp32_quartets = (uint64_t)raw_stats[5];
    }
    return 0;
#else
    return -5;
#endif
}

int calchem_cuda_direct_release(uint64_t direct_id) {
    std::lock_guard<std::mutex> lock(rij_mutex);
#ifdef _WIN32
    if (direct_id != 0) {
        auto it = g_drv.direct_contexts.find(direct_id);
        if (it != g_drv.direct_contexts.end()) {
            it->second.free_all(g_drv.cuMemFree);
            g_drv.direct_contexts.erase(it);
        }
        if (g_drv.direct_contexts.empty() && g_drv.d_dir_f_scratch) {
            g_drv.cuMemFree(g_drv.d_dir_f_scratch);
            g_drv.d_dir_f_scratch = 0;
            g_drv.dir_f_scratch_bytes = 0;
        }
    }
    return 0;
#else
    (void)direct_id;
    return 0;
#endif
}

int calchem_cuda_vxc_grad_force(
    uint64_t grid_id,
    const double* h_pa,
    const double* h_pb,
    int is_rks,
    const double* h_wxc_in,
    double* h_grad_out,
    int n_pts,
    int n_ao,
    int n_atoms,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (grid_id == 0 || !h_pa || (!is_rks && !h_pb) || !h_wxc_in || !h_grad_out
        || n_pts <= 0 || n_ao <= 0 || n_atoms <= 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_vxc_grad_force || !g_drv.fn_zero_doubles) return -5;
    auto it = g_drv.vxc_grids.find(grid_id);
    if (it == g_drv.vxc_grids.end() || it->second.n_pts != n_pts || it->second.n_ao != n_ao) return -2;
    VxcGridEntry& entry = it->second;

    size_t nn = 0, mat_b = 0, pts8 = 0, wxc_b = 0, grad_len = 0, grad_b = 0, tmp_req = 0;
    if (!safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &mat_b)
        || !safe_mul_sz((size_t)n_pts, 8, &pts8)
        || !safe_mul_sz(pts8, sizeof(double), &wxc_b)
        || !safe_mul_sz((size_t)n_atoms, 3, &grad_len)
        || !safe_mul_sz(grad_len, sizeof(double), &grad_b)
        || !safe_add_sz(is_rks ? 0 : mat_b, wxc_b, &tmp_req)
        || !safe_add_sz(tmp_req, grad_b, &tmp_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(tmp_req, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }

    struct Guard {
        uint64_t d_pb = 0, d_wxc = 0, d_grad = 0;
        ~Guard() {
            if (g_drv.cuMemFree) {
                if (d_pb) g_drv.cuMemFree(d_pb);
                if (d_wxc) g_drv.cuMemFree(d_wxc);
                if (d_grad) g_drv.cuMemFree(d_grad);
            }
            g_drv.transient_grad_bytes = 0;
        }
    } g;

    if ((!is_rks && g_drv.cuMemAlloc(&g.d_pb, mat_b) != 0)
        || g_drv.cuMemAlloc(&g.d_wxc, wxc_b) != 0
        || g_drv.cuMemAlloc(&g.d_grad, grad_b) != 0) {
        return -4;
    }
    g_drv.transient_grad_bytes = tmp_req;

    if (g_drv.cuMemcpyHtoD(entry.d_p, h_pa, mat_b) != 0
        || (!is_rks && g_drv.cuMemcpyHtoD(g.d_pb, h_pb, mat_b) != 0)
        || g_drv.cuMemcpyHtoD(g.d_wxc, h_wxc_in, wxc_b) != 0) {
        return -5;
    }

    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((grad_len + z_block - 1) / z_block);
    void* z_p[2] = { &g.d_grad, &grad_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) {
        return -5;
    }

    uint64_t pb_ptr = is_rks ? entry.d_p : g.d_pb;
    const unsigned int block = 128;
    unsigned int grid = ((unsigned int)n_pts + block - 1) / block;
    void* params[13] = {
        &entry.d_shells, &entry.d_prims, &entry.d_comps,
        &entry.d_pts, &entry.d_cand_off, &entry.d_cand_sh,
        &entry.d_p, &pb_ptr, &g.d_wxc, &g.d_grad,
        &n_pts, &n_ao, &is_rks
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_vxc_grad_force, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_grad_out, g.d_grad, grad_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_rij_grad_v(
    const Grad3cAuxShellGpu* h_aux_shells,
    int n_aux_shells,
    const Grad3cAuxPrimGpu* h_aux_prims,
    int n_aux_prims,
    const Grad3cShellPairGpu* h_shell_pairs,
    int n_shell_pairs,
    const Grad3cPrimPairGpu* h_prim_pairs,
    int n_prim_pairs,
    const double* h_p_mat,
    double* h_v_out,
    int n_aux,
    int n_ao,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_aux_shells || !h_aux_prims || !h_shell_pairs || !h_prim_pairs || !h_p_mat || !h_v_out
        || n_aux_shells <= 0 || n_aux_prims <= 0 || n_shell_pairs <= 0 || n_prim_pairs <= 0
        || n_aux <= 0 || n_ao <= 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_rij_grad_v || !g_drv.fn_zero_doubles) return -5;

    size_t ash_b = 0, apr_b = 0, sp_b = 0, pp_b = 0, nn = 0, p_b = 0, v_b = 0, total_req = 0;
    size_t v_len = (size_t)n_aux;
    if (!safe_mul_sz((size_t)n_aux_shells, sizeof(Grad3cAuxShellGpu), &ash_b)
        || !safe_mul_sz((size_t)n_aux_prims, sizeof(Grad3cAuxPrimGpu), &apr_b)
        || !safe_mul_sz((size_t)n_shell_pairs, sizeof(Grad3cShellPairGpu), &sp_b)
        || !safe_mul_sz((size_t)n_prim_pairs, sizeof(Grad3cPrimPairGpu), &pp_b)
        || !safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &p_b)
        || !safe_mul_sz(v_len, sizeof(double), &v_b)
        || !safe_add_sz(ash_b, apr_b, &total_req)
        || !safe_add_sz(total_req, sp_b, &total_req)
        || !safe_add_sz(total_req, pp_b, &total_req)
        || !safe_add_sz(total_req, p_b, &total_req)
        || !safe_add_sz(total_req, v_b, &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }

    struct Guard {
        uint64_t d_ash = 0, d_apr = 0, d_sp = 0, d_pp = 0, d_p = 0, d_v = 0;
        ~Guard() {
            if (g_drv.cuMemFree) {
                if (d_ash) g_drv.cuMemFree(d_ash);
                if (d_apr) g_drv.cuMemFree(d_apr);
                if (d_sp)  g_drv.cuMemFree(d_sp);
                if (d_pp)  g_drv.cuMemFree(d_pp);
                if (d_p)   g_drv.cuMemFree(d_p);
                if (d_v)   g_drv.cuMemFree(d_v);
            }
            g_drv.transient_grad_bytes = 0;
        }
    } g;

    if (g_drv.cuMemAlloc(&g.d_ash, ash_b) != 0
        || g_drv.cuMemAlloc(&g.d_apr, apr_b) != 0
        || g_drv.cuMemAlloc(&g.d_sp, sp_b) != 0
        || g_drv.cuMemAlloc(&g.d_pp, pp_b) != 0
        || g_drv.cuMemAlloc(&g.d_p, p_b) != 0
        || g_drv.cuMemAlloc(&g.d_v, v_b) != 0) {
        return -4;
    }
    g_drv.transient_grad_bytes = total_req;

    if (g_drv.cuMemcpyHtoD(g.d_ash, h_aux_shells, ash_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_apr, h_aux_prims, apr_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_sp, h_shell_pairs, sp_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pp, h_prim_pairs, pp_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_p, h_p_mat, p_b) != 0) {
        return -5;
    }

    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((v_len + z_block - 1) / z_block);
    void* z_p[2] = { &g.d_v, &v_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) {
        return -5;
    }

    const unsigned int block = 64;
    unsigned int grid_x = ((unsigned int)n_shell_pairs + block - 1) / block;
    unsigned int grid_y = (unsigned int)n_aux_shells;
    void* params[9] = {
        &g.d_ash, &g.d_apr, &g.d_sp, &g.d_pp, &g.d_p, &g.d_v,
        &n_aux_shells, &n_shell_pairs, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_rij_grad_v, grid_x, grid_y, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_v_out, g.d_v, v_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_rij_grad_3c(
    const Grad3cAuxShellGpu* h_aux_shells,
    int n_aux_shells,
    const Grad3cAuxPrimGpu* h_aux_prims,
    int n_aux_prims,
    const Grad3cShellPairGpu* h_shell_pairs,
    int n_shell_pairs,
    const Grad3cPrimPairGpu* h_prim_pairs,
    int n_prim_pairs,
    const double* h_x_vec,
    const double* h_p_mat,
    double* h_grad_out,
    int n_aux,
    int n_ao,
    int n_atoms,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_aux_shells || !h_aux_prims || !h_shell_pairs || !h_prim_pairs || !h_x_vec || !h_p_mat || !h_grad_out
        || n_aux_shells <= 0 || n_aux_prims <= 0 || n_shell_pairs <= 0 || n_prim_pairs <= 0
        || n_aux <= 0 || n_ao <= 0 || n_atoms <= 0) {
        return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_rij_grad_3c || !g_drv.fn_zero_doubles) return -5;

    size_t ash_b = 0, apr_b = 0, sp_b = 0, pp_b = 0, x_b = 0, nn = 0, p_b = 0, grad_len = 0, grad_b = 0, total_req = 0;
    if (!safe_mul_sz((size_t)n_aux_shells, sizeof(Grad3cAuxShellGpu), &ash_b)
        || !safe_mul_sz((size_t)n_aux_prims, sizeof(Grad3cAuxPrimGpu), &apr_b)
        || !safe_mul_sz((size_t)n_shell_pairs, sizeof(Grad3cShellPairGpu), &sp_b)
        || !safe_mul_sz((size_t)n_prim_pairs, sizeof(Grad3cPrimPairGpu), &pp_b)
        || !safe_mul_sz((size_t)n_aux, sizeof(double), &x_b)
        || !safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &p_b)
        || !safe_mul_sz((size_t)n_atoms, 3, &grad_len)
        || !safe_mul_sz(grad_len, sizeof(double), &grad_b)
        || !safe_add_sz(ash_b, apr_b, &total_req)
        || !safe_add_sz(total_req, sp_b, &total_req)
        || !safe_add_sz(total_req, pp_b, &total_req)
        || !safe_add_sz(total_req, x_b, &total_req)
        || !safe_add_sz(total_req, p_b, &total_req)
        || !safe_add_sz(total_req, grad_b, &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        return budget_err;
    }

    struct Guard {
        uint64_t d_ash = 0, d_apr = 0, d_sp = 0, d_pp = 0, d_x = 0, d_p = 0, d_grad = 0;
        ~Guard() {
            if (g_drv.cuMemFree) {
                if (d_ash)  g_drv.cuMemFree(d_ash);
                if (d_apr)  g_drv.cuMemFree(d_apr);
                if (d_sp)   g_drv.cuMemFree(d_sp);
                if (d_pp)   g_drv.cuMemFree(d_pp);
                if (d_x)    g_drv.cuMemFree(d_x);
                if (d_p)    g_drv.cuMemFree(d_p);
                if (d_grad) g_drv.cuMemFree(d_grad);
            }
            g_drv.transient_grad_bytes = 0;
        }
    } g;

    if (g_drv.cuMemAlloc(&g.d_ash, ash_b) != 0
        || g_drv.cuMemAlloc(&g.d_apr, apr_b) != 0
        || g_drv.cuMemAlloc(&g.d_sp, sp_b) != 0
        || g_drv.cuMemAlloc(&g.d_pp, pp_b) != 0
        || g_drv.cuMemAlloc(&g.d_x, x_b) != 0
        || g_drv.cuMemAlloc(&g.d_p, p_b) != 0
        || g_drv.cuMemAlloc(&g.d_grad, grad_b) != 0) {
        return -4;
    }
    g_drv.transient_grad_bytes = total_req;

    if (g_drv.cuMemcpyHtoD(g.d_ash, h_aux_shells, ash_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_apr, h_aux_prims, apr_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_sp, h_shell_pairs, sp_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pp, h_prim_pairs, pp_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_x, h_x_vec, x_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_p, h_p_mat, p_b) != 0) {
        return -5;
    }

    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((grad_len + z_block - 1) / z_block);
    void* z_p[2] = { &g.d_grad, &grad_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) {
        return -5;
    }

    const unsigned int block = 64; // 2 warps per block
    unsigned int grid_x = ((unsigned int)n_shell_pairs + 1) / 2;
    unsigned int grid_y = (unsigned int)n_aux_shells;
    void* params[11] = {
        &g.d_ash, &g.d_apr, &g.d_sp, &g.d_pp, &g_drv.d_comps6,
        &g.d_x, &g.d_p, &g.d_grad,
        &n_aux_shells, &n_shell_pairs, &n_ao
    };
    if (g_drv.cuLaunchKernel(g_drv.fn_rij_grad_3c, grid_x, grid_y, 1, block, 1, 1, 0, nullptr, params, nullptr) != 0
        || g_drv.cuCtxSynchronize() != 0) {
        return -5;
    }
    return (g_drv.cuMemcpyDtoH(h_grad_out, g.d_grad, grad_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

int calchem_cuda_grad4c_contract(
    const Grad4cShellGpu* h_shells,
    int n_shells,
    const Grad4cShellPairGpu* h_pairs,
    int n_pairs,
    const Grad4cPrimPairGpu* h_prims,
    int n_prims,
    const double* h_pt,
    const double* h_pa,
    const double* h_pb,
    const double* h_d_tot_max,
    const double* h_d_spin_max,
    const int* h_ends,
    int n_ao,
    int n_atoms,
    int do_coul,
    double exx,
    double omega,
    int has_omega,
    double thresh,
    double* h_grad_out,
    size_t memory_limit_bytes
) {
    std::lock_guard<std::mutex> lock(rij_mutex);
    if (!h_shells || !h_pairs || !h_prims || !h_pt || !h_pa || !h_pb
        || !h_d_tot_max || !h_d_spin_max || !h_ends || !h_grad_out
        || n_shells <= 0 || n_pairs <= 0 || n_prims <= 0
        || n_ao <= 0 || n_atoms <= 0) {
        return -1;
    }
    for (int pi = 0; pi < n_pairs; ++pi) {
        if (h_ends[pi] < 0 || h_ends[pi] > n_pairs) return -1;
    }
#ifdef _WIN32
    if (!g_drv.bind_context()) return -3;
    if (!g_drv.ensure_kernels() || !g_drv.fn_grad4c_filter || !g_drv.fn_grad4c_ssss
        || !g_drv.fn_grad4c_sp || !g_drv.fn_grad4c_d || !g_drv.fn_grad4c_f
        || !g_drv.fn_zero_doubles) {
        return -5;
    }

    size_t sh_b = 0, pr_b = 0, pp_b = 0, nn = 0, mat_b = 0, mat3_b = 0;
    size_t ss = 0, d_b = 0, d2_b = 0, end_b = 0, grad_len = 0, grad_b = 0, cnt_b = 0;
    if (!safe_mul_sz((size_t)n_shells, sizeof(Grad4cShellGpu), &sh_b)
        || !safe_mul_sz((size_t)n_pairs, sizeof(Grad4cShellPairGpu), &pr_b)
        || !safe_mul_sz((size_t)n_prims, sizeof(Grad4cPrimPairGpu), &pp_b)
        || !safe_mul_sz((size_t)n_ao, (size_t)n_ao, &nn)
        || !safe_mul_sz(nn, sizeof(double), &mat_b)
        || !safe_mul_sz(3, mat_b, &mat3_b)
        || !safe_mul_sz((size_t)n_shells, (size_t)n_shells, &ss)
        || !safe_mul_sz(ss, sizeof(double), &d_b)
        || !safe_mul_sz(2, d_b, &d2_b)
        || !safe_mul_sz((size_t)n_pairs, sizeof(int), &end_b)
        || !safe_mul_sz((size_t)n_atoms, 3, &grad_len)
        || !safe_mul_sz(grad_len, sizeof(double), &grad_b)
        || !safe_mul_sz((size_t)NUM_GRAD_WL_BUCKETS, sizeof(int), &cnt_b)) {
        return -2;
    }

    uint32_t wl_stride = 2000000u;
    if ((size_t)n_pairs <= 65536u) {
        size_t max_pairs_cand = (size_t)n_pairs * (size_t)(n_pairs + 1) / 2;
        if (max_pairs_cand < (size_t)wl_stride) {
            wl_stride = (uint32_t)(max_pairs_cand > 1024 ? max_pairs_cand : 1024);
        }
    }
    size_t base_req = 0;
    if (!safe_add_sz(sh_b, pr_b, &base_req)
        || !safe_add_sz(base_req, pp_b, &base_req)
        || !safe_add_sz(base_req, mat3_b, &base_req)
        || !safe_add_sz(base_req, d2_b, &base_req)
        || !safe_add_sz(base_req, end_b, &base_req)
        || !safe_add_sz(base_req, grad_b, &base_req)
        || !safe_add_sz(base_req, cnt_b, &base_req)) {
        return -2;
    }
    size_t wl_cnt = 0, wl_b = 0, total_req = 0;
    if (!safe_mul_sz((size_t)NUM_GRAD_WL_BUCKETS, (size_t)wl_stride, &wl_cnt)
        || !safe_mul_sz(wl_cnt, sizeof(WorkItem), &wl_b)
        || !safe_add_sz(base_req, wl_b, &total_req)) {
        return -2;
    }

    int budget_err = -3;
    if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
        if (wl_stride > 65536u) {
            wl_stride = 65536u;
            if (!safe_mul_sz((size_t)NUM_GRAD_WL_BUCKETS, (size_t)wl_stride, &wl_cnt)
                || !safe_mul_sz(wl_cnt, sizeof(WorkItem), &wl_b)
                || !safe_add_sz(base_req, wl_b, &total_req)) {
                return -2;
            }
            if (!g_drv.check_budget_and_free(total_req, memory_limit_bytes, &budget_err)) {
                return budget_err;
            }
        } else {
            return budget_err;
        }
    }

    struct Guard {
        uint64_t d_sh = 0, d_pr = 0, d_pp = 0;
        uint64_t d_pt = 0, d_pa = 0, d_pb = 0;
        uint64_t d_dt = 0, d_ds = 0, d_ends = 0;
        uint64_t d_grad = 0, d_wl = 0, d_wl_cnt = 0;
        uint64_t d_scr = 0;
        size_t scr_b = 0;
        size_t base_b = 0;
        ~Guard() {
            if (g_drv.cuMemFree) {
                if (d_sh)     g_drv.cuMemFree(d_sh);
                if (d_pr)     g_drv.cuMemFree(d_pr);
                if (d_pp)     g_drv.cuMemFree(d_pp);
                if (d_pt)     g_drv.cuMemFree(d_pt);
                if (d_pa)     g_drv.cuMemFree(d_pa);
                if (d_pb)     g_drv.cuMemFree(d_pb);
                if (d_dt)     g_drv.cuMemFree(d_dt);
                if (d_ds)     g_drv.cuMemFree(d_ds);
                if (d_ends)   g_drv.cuMemFree(d_ends);
                if (d_grad)   g_drv.cuMemFree(d_grad);
                if (d_wl)     g_drv.cuMemFree(d_wl);
                if (d_wl_cnt) g_drv.cuMemFree(d_wl_cnt);
                if (d_scr)    g_drv.cuMemFree(d_scr);
            }
            g_drv.transient_grad_bytes = 0;
        }
    } g;

    if (g_drv.cuMemAlloc(&g.d_sh, sh_b) != 0
        || g_drv.cuMemAlloc(&g.d_pr, pr_b) != 0
        || g_drv.cuMemAlloc(&g.d_pp, pp_b) != 0
        || g_drv.cuMemAlloc(&g.d_pt, mat_b) != 0
        || g_drv.cuMemAlloc(&g.d_pa, mat_b) != 0
        || g_drv.cuMemAlloc(&g.d_pb, mat_b) != 0
        || g_drv.cuMemAlloc(&g.d_dt, d_b) != 0
        || g_drv.cuMemAlloc(&g.d_ds, d_b) != 0
        || g_drv.cuMemAlloc(&g.d_ends, end_b) != 0
        || g_drv.cuMemAlloc(&g.d_grad, grad_b) != 0
        || g_drv.cuMemAlloc(&g.d_wl, wl_b) != 0
        || g_drv.cuMemAlloc(&g.d_wl_cnt, cnt_b) != 0) {
        return -4;
    }
    g.base_b = total_req;
    g_drv.transient_grad_bytes = total_req;

    if (g_drv.cuMemcpyHtoD(g.d_sh, h_shells, sh_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pr, h_pairs, pr_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pp, h_prims, pp_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pt, h_pt, mat_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pa, h_pa, mat_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_pb, h_pb, mat_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_dt, h_d_tot_max, d_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_ds, h_d_spin_max, d_b) != 0
        || g_drv.cuMemcpyHtoD(g.d_ends, h_ends, end_b) != 0) {
        return -5;
    }

    const unsigned int z_block = 256;
    unsigned int z_grid = (unsigned int)((grad_len + z_block - 1) / z_block);
    void* z_p[2] = { &g.d_grad, &grad_len };
    if (g_drv.cuLaunchKernel(g_drv.fn_zero_doubles, z_grid, 1, 1, z_block, 1, 1, 0, nullptr, z_p, nullptr) != 0) {
        return -5;
    }

    auto ensure_grad_scratch = [&](size_t blk_bytes, int cnt, unsigned int max_grid, unsigned int& out_grid) -> int {
        unsigned int grd = (unsigned int)(cnt < (int)max_grid ? cnt : (int)max_grid);
        size_t need = (size_t)grd * blk_bytes;
        if (need <= g.scr_b) {
            out_grid = grd;
            return 0;
        }
        bool fit = false;
        while (grd >= 1) {
            need = (size_t)grd * blk_bytes;
            if (need <= g.scr_b) {
                fit = true;
                break;
            }
            int b_err = -3;
            if (g_drv.check_budget_for_replace(need, g.scr_b, memory_limit_bytes, &b_err)) {
                fit = true;
                break;
            }
            if (grd == 1) break;
            grd /= 2;
        }
        if (!fit) return -4;
        if (need > g.scr_b) {
            if (g.d_scr) {
                g_drv.cuMemFree(g.d_scr);
                g.d_scr = 0;
                g.scr_b = 0;
                g_drv.transient_grad_bytes = g.base_b;
            }
            if (g_drv.cuMemAlloc(&g.d_scr, need) != 0) {
                g.d_scr = 0;
                return -4;
            }
            g.scr_b = need;
            g_drv.transient_grad_bytes = g.base_b + need;
        }
        out_grid = grd;
        return 0;
    };

    int pi_start = 0;
    while (pi_start < n_pairs) {
        uint64_t chunk_sum = 0;
        int pi_end = pi_start;
        while (pi_end < n_pairs) {
            uint64_t e = (uint64_t)(h_ends[pi_end] > 0 ? h_ends[pi_end] : 0);
            if (pi_end > pi_start && chunk_sum + e > (uint64_t)wl_stride) break;
            chunk_sum += e;
            pi_end++;
            if (chunk_sum > (uint64_t)wl_stride) break;
        }
        int n_act = pi_end - pi_start;
        if (chunk_sum == 0) {
            pi_start = pi_end;
            continue;
        }

        int total_ket_end = (n_act == 1 && chunk_sum > (uint64_t)wl_stride) ? (int)chunk_sum : -1;
        int qi_s = 0;
        do {
            int qi_e = -1;
            if (total_ket_end > 0) {
                int64_t next_e = (int64_t)qi_s + (int64_t)wl_stride;
                qi_e = (next_e < (int64_t)total_ket_end) ? (int)next_e : total_ket_end;
            }

            if (g_drv.cuMemsetD8(g.d_wl_cnt, 0, cnt_b) != 0) return -5;
            void* f_params[16] = {
                &g.d_sh, &g.d_pr, &g.d_ends,
                &g.d_dt, &g.d_ds,
                &g.d_wl, &wl_stride, &g.d_wl_cnt,
                &pi_start, &n_act, &n_shells, &do_coul, &exx, &thresh,
                &qi_s, &qi_e
            };
            if (g_drv.cuLaunchKernel(g_drv.fn_grad4c_filter, (unsigned int)n_act, 1, 1, 128, 1, 1, 0, nullptr, f_params, nullptr) != 0
                || g_drv.cuCtxSynchronize() != 0) {
                return -5;
            }
            int counts[NUM_GRAD_WL_BUCKETS] = {};
            if (g_drv.cuMemcpyDtoH(counts, g.d_wl_cnt, sizeof(counts)) != 0) return -5;

            if (counts[WL_GRAD_SSSS] > 0) {
                int cnt = counts[WL_GRAD_SSSS];
                uint64_t wl_ptr = g.d_wl + (size_t)WL_GRAD_SSSS * (size_t)wl_stride * sizeof(WorkItem);
                const unsigned int blk = 128;
                unsigned int grd = ((unsigned int)cnt + blk - 1) / blk;
                void* p_s[15] = {
                    &g.d_sh, &g.d_pr, &g.d_pp, &wl_ptr, &cnt, &n_ao,
                    &do_coul, &exx, &omega, &has_omega, &g_drv.d_boys_f64,
                    &g.d_pt, &g.d_pa, &g.d_pb, &g.d_grad
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_grad4c_ssss, grd, 1, 1, blk, 1, 1, 0, nullptr, p_s, nullptr) != 0) return -5;
            }

            if (counts[WL_GRAD_SP] > 0) {
                int cnt = counts[WL_GRAD_SP];
                uint64_t wl_ptr = g.d_wl + (size_t)WL_GRAD_SP * (size_t)wl_stride * sizeof(WorkItem);
                unsigned int smem_b = 20 * 20 * (4 + 7) * (unsigned int)sizeof(double);
                void* p_sp[16] = {
                    &g.d_sh, &g.d_pr, &g.d_pp, &wl_ptr, &cnt, &n_ao,
                    &do_coul, &exx, &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_comps6,
                    &g.d_pt, &g.d_pa, &g.d_pb, &g.d_grad
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_grad4c_sp, (unsigned int)cnt, 1, 1, 32, 1, 1, smem_b, nullptr, p_sp, nullptr) != 0) return -5;
            }

            if (counts[WL_GRAD_D] > 0) {
                int cnt = counts[WL_GRAD_D];
                unsigned int d_grid = 0;
                int scr_rc = ensure_grad_scratch((size_t)(56 * 56 * 15) * sizeof(double), cnt, 64, d_grid);
                if (scr_rc != 0) return scr_rc;
                uint64_t wl_ptr = g.d_wl + (size_t)WL_GRAD_D * (size_t)wl_stride * sizeof(WorkItem);
                void* p_d[17] = {
                    &g.d_sh, &g.d_pr, &g.d_pp, &wl_ptr, &cnt, &n_ao,
                    &do_coul, &exx, &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_comps6,
                    &g.d_scr, &g.d_pt, &g.d_pa, &g.d_pb, &g.d_grad
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_grad4c_d, d_grid, 1, 1, 64, 1, 1, 0, nullptr, p_d, nullptr) != 0) return -5;
            }

            if (counts[WL_GRAD_F] > 0) {
                int cnt = counts[WL_GRAD_F];
                unsigned int f_grid = 0;
                int scr_rc = ensure_grad_scratch((size_t)(120 * 120 * 19) * sizeof(double), cnt, 32, f_grid);
                if (scr_rc != 0) return scr_rc;
                uint64_t wl_ptr = g.d_wl + (size_t)WL_GRAD_F * (size_t)wl_stride * sizeof(WorkItem);
                void* p_f[17] = {
                    &g.d_sh, &g.d_pr, &g.d_pp, &wl_ptr, &cnt, &n_ao,
                    &do_coul, &exx, &omega, &has_omega, &g_drv.d_boys_f64, &g_drv.d_comps6,
                    &g.d_scr, &g.d_pt, &g.d_pa, &g.d_pb, &g.d_grad
                };
                if (g_drv.cuLaunchKernel(g_drv.fn_grad4c_f, f_grid, 1, 1, 64, 1, 1, 0, nullptr, p_f, nullptr) != 0) return -5;
            }

            if (g_drv.cuCtxSynchronize() != 0) return -5;
            if (total_ket_end > 0) {
                qi_s = qi_e;
            }
        } while (total_ket_end > 0 && qi_s < total_ket_end);

        pi_start = pi_end;
    }

    return (g_drv.cuMemcpyDtoH(h_grad_out, g.d_grad, grad_b) == 0) ? 0 : -5;
#else
    return -5;
#endif
}

}
