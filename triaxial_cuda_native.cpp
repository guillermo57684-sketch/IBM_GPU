/*
 * triaxial_cuda_native.cpp
 *
 * Simulador Triaxial DEM 100% C++ / CUDA Nativo (libnvrtc + libcuda Driver API)
 * Replicación de YADE (test-4-triaxial-opt.py + InterlockingBondModel.cpp).
 *
 * Versión validada (16-10-2026, ver validacion_ibm/PLAN_VALIDACION_IBM.md):
 *   FIX-1 factor anti-ratcheting alpha=(r1+r2)/d en el corte (como ScGeom de YADE)
 *   FIX-2 amortiguamiento de Cundall de 2.º orden (NewtonIntegrator::cundallDamp2nd)
 *   FIX-3 recontacto con penetración = 0
 *   FIX-4 estado incremental de corte/momento acumulado en FP64
 *   [CONTACTOS] detección dinámica de contactos esfera-esfera y esfera-faceta (lista de Verlet)
 *   [DET] suma de fuerzas determinista en FP64 (DEM_ATOMICS=1 para la suma con atomicAdd)
 *   [VTK] DEM_VTK_EVERY=N exporta esferas, interacciones y facetas como exp_vtk() de YADE (DEM_VTK_ASCII, DEM_VTK_INTRS)
 *   [PAR-1..5] paridad con YADE: ley esfera-esfera en el orden de Law2_ScGeom_IBPhys_IBLaw con rotación exacta
 *              del corte (normal previa), rotación exacta en facetas, servo cada 50 iteraciones con rigidez inicial
 *              analítica, calendario alineado con O.iter, enlaces y facetas calculados como YADE; ley en FP64 por defecto (DEM_FP32=1)
 *   [CASO] (17-10-2026) parámetros del material, del ensayo y del contorno en un archivo de caso en tiempo de ejecución
 *          (--caso archivo.cfg y/o clave=valor; ver --ayuda); tamaños del contorno derivados de la geometría;
 *          NVRTC y arquitectura de la GPU detectados en tiempo de ejecución; resumen.txt y caso_usado.cfg en la salida.
 *   [FIX-5/6] (19-10-2026) contacto inicial esfera-plato con penetración de 1 ulp (antes se descartaba) e initD de los enlaces
 *          calculado en la GPU (en el host, el redondeo con FMA dejaba D = ±1e-24 y activaba Ks_eff en el primer paso).
 *          Versión anterior en validacion_ibm/cuda_corregido/pre_fix_2026-10-19/.
 *   [PARADA] parada automática de checkProgress() de YADE (parada_yade=1 por defecto; pasos = tope).
 *          Con los valores por defecto el PTX y los resultados son idénticos bit a bit a la versión anterior
 *          (validacion_ibm/cuda_corregido/pre_config/).
 * La versión anterior se conserva en triaxial_cuda_native_orig.cpp.
 *
 * Uso: ./triaxial_cuda_native [pasos] [salida] [static] [--caso archivo.cfg] [clave=valor ...]
 *      ./triaxial_cuda_native --ayuda
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <set>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <random>
#include <sys/stat.h>
#include <glob.h>

// Tipos básicos de CUDA Driver API y NVRTC cargados dinámicamente
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;
typedef int nvrtcResult;
typedef void* nvrtcProgram;

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != 0) { \
        std::cerr << "CUDA Driver Error " << err << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_NVRTC(call) do { \
    nvrtcResult err = (call); \
    if (err != 0) { \
        std::cerr << "NVRTC Error " << err << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Código fuente CUDA C++ de los kernels (se compila con NVRTC en tiempo de ejecución para la GPU detectada)
const char* CUDA_KERNEL_SOURCE = R"CUDA(

// Tamaños de la malla de contorno. El host los fija con -D según la geometría del caso;
// los valores por defecto son los de la probeta de referencia (24x19x2 facetas por plato, 24x20 anillos de membrana).
#ifndef N_PLATE
#define N_PLATE 456          // facetas por plato: [0, N_PLATE) superior, [N_PLATE, 2·N_PLATE) inferior
#endif
#ifndef N_SEG
#define N_SEG 24             // segmentos angulares de la membrana
#endif
#ifndef N_RINGS
#define N_RINGS 20           // anillos verticales de la membrana (2·N_SEG facetas por anillo)
#endif
#define N_FACETS (2 * N_PLATE + 2 * N_SEG * N_RINGS)

extern "C" {

struct double3_t {
    double x, y, z;
};
struct float3_t {
    float x, y, z;
};

__device__ __forceinline__ double3_t make_d3(double x, double y, double z) {
    double3_t r; r.x = x; r.y = y; r.z = z; return r;
}
__device__ __forceinline__ double3_t add_d3(double3_t a, double3_t b) {
    return make_d3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ double3_t sub_d3(double3_t a, double3_t b) {
    return make_d3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ double3_t scale_d3(double3_t a, double s) {
    return make_d3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ double dot_d3(double3_t a, double3_t b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ double3_t cross_d3(double3_t a, double3_t b) {
    return make_d3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__device__ __forceinline__ double norm_d3(double3_t a) {
    return sqrt(dot_d3(a, a));
}

__device__ double3_t closest_pt_tri_3d(double3_t p, double3_t a, double3_t b, double3_t c) {
    double3_t ab = sub_d3(b, a);
    double3_t ac = sub_d3(c, a);
    double3_t ap = sub_d3(p, a);
    double d1 = dot_d3(ab, ap);
    double d2 = dot_d3(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;

    double3_t bp = sub_d3(p, b);
    double d3 = dot_d3(ab, bp);
    double d4 = dot_d3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double v = d1 / (d1 - d3);
        return add_d3(a, scale_d3(ab, v));
    }

    double3_t cp = sub_d3(p, c);
    double d5 = dot_d3(ab, cp);
    double d6 = dot_d3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / (d2 - d6);
        return add_d3(a, scale_d3(ac, w));
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add_d3(b, scale_d3(sub_d3(c, b), w));
    }

    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom;
    double w = vc * denom;
    return add_d3(a, add_d3(scale_d3(ab, v), scale_d3(ac, w)));
}

__global__ void kernel_reset_forces(
    int n_particles,
    int n_facets,
    float3_t* __restrict__ force,
    float3_t* __restrict__ torque,
    double3_t* __restrict__ f_force
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_facets) {
        f_force[idx] = make_d3(0.0, 0.0, 0.0);
    }
}

#ifdef DEM_FP64
typedef double real;
#else
typedef float real;
#endif
struct real3 { real x, y, z; };
__device__ __forceinline__ real3 mk3(real x, real y, real z) { real3 r; r.x = x; r.y = y; r.z = z; return r; }
__device__ __forceinline__ real3 sub3(real3 a, real3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ real3 add3(real3 a, real3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ real3 scl3(real3 a, real s) { return mk3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ real dot3(real3 a, real3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ real3 crs3(real3 a, real3 b) { return mk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// [PAR-1] Réplica de Law2_ScGeom_IBPhys_IBLaw::go + ScGeom::precompute/rotate (YADE), mismo orden de operaciones.
__global__ void kernel_sphere_sphere(
    int n_bonds,
    double dt_val,
    double tgPhi,
    double mu,
    double rollFric,
    const double3_t* __restrict__ pos,
    const double3_t* __restrict__ vel,
    const double3_t* __restrict__ angVel,
    const double* __restrict__ radius,
    const int* __restrict__ b_p1,
    const int* __restrict__ b_p2,
    const double* __restrict__ b_kn,
    const double* __restrict__ b_ks,
    const double* __restrict__ b_kb,
    const double* __restrict__ b_cn,
    const double* __restrict__ b_FnMax,
    const double* __restrict__ b_FsMax,
    const double* __restrict__ b_kseff,      // ks + 12·kb/beamLength² (acople bimodular)
    double* __restrict__ b_initD,
    unsigned char* __restrict__ b_state,
    double* __restrict__ b_sfx,
    double* __restrict__ b_sfy,
    double* __restrict__ b_sfz,
    double* __restrict__ b_mbx,
    double* __restrict__ b_mby,
    double* __restrict__ b_mbz,
    double* __restrict__ b_pnx,              // normal del paso anterior (ScGeom::normal antes de actualizar)
    double* __restrict__ b_pny,
    double* __restrict__ b_pnz,
    float3_t* __restrict__ force,
    float3_t* __restrict__ torque,
    int* __restrict__ nb_tens_cracks,
    int* __restrict__ nb_shear_cracks,
    double* __restrict__ pf_out,   // [DET] 9 valores por par: F (sobre j), T_i, T_j
    int det,
    int init_pass                  // [FIX-6] 1 = solo fija initD y la normal inicial de los enlaces (isFresh de la iteración 0)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_bonds) return;
    if (det) { double* o = pf_out + 9 * (size_t)k; for (int c = 0; c < 9; ++c) o[c] = 0.0; }

    const int i = b_p1[k];
    const int j = b_p2[k];
    const double3_t r12 = sub_d3(pos[j], pos[i]);
    const double dist = sqrt(dot_d3(r12, r12));
    const double r1 = radius[i], r2 = radius[j];
    const double pen = r1 + r2 - dist;                      // geom->penetrationDepth

    unsigned char st = b_state[k];
    if (init_pass) {   // [FIX-6] initD con la misma aritmética del kernel: D = 0 exacto mientras no hay movimiento
        if (st & 1) { b_initD[k] = pen; const double id = 1.0 / dist; b_pnx[k] = r12.x * id; b_pny[k] = r12.y * id; b_pnz[k] = r12.z * id; }
        return;
    }
    int act = st & 1;
    int coh = (st >> 1) & 1;
    bool isNew = false;
    if (act == 0) {
        if (pen < 0.0) return;                              // Ig2: nuevo contacto solo con penetración >= 0
        act = 1; st |= 1; b_state[k] = st; isNew = true;
        b_initD[k] = pen;                                   // isFresh: initD = penetrationDepth
        b_sfx[k] = 0.0; b_sfy[k] = 0.0; b_sfz[k] = 0.0;
        b_mbx[k] = 0.0; b_mby[k] = 0.0; b_mbz[k] = 0.0;
    }

    const real D = (real)(pen - b_initD[k]);
    real Fn = (real)b_kn[k] * D;

    // STEP B: contacto no cohesivo en tracción -> se borra la interacción
    if (coh == 0 && D < (real)0) {
        b_state[k] = st & ~1; b_initD[k] = 0.0;
        b_sfx[k] = 0.0; b_sfy[k] = 0.0; b_sfz[k] = 0.0; b_mbx[k] = 0.0; b_mby[k] = 0.0; b_mbz[k] = 0.0;
        return;
    }
    // STEP C: rotura por tracción pura -> isCohesive=false y, como D<0, la interacción se borra (STEP D)
    if (coh == 1 && -Fn > (real)b_FnMax[k]) {
        b_state[k] = 4; b_initD[k] = 0.0;
        b_sfx[k] = 0.0; b_sfy[k] = 0.0; b_sfz[k] = 0.0; b_mbx[k] = 0.0; b_mby[k] = 0.0; b_mbz[k] = 0.0;
        atomicAdd(nb_tens_cracks, 1);
        return;
    }

    // ScGeom::precompute
    const double invd = 1.0 / dist;
    const real3 n = mk3((real)(r12.x * invd), (real)(r12.y * invd), (real)(r12.z * invd));
    const real3 w1 = mk3((real)angVel[i].x, (real)angVel[i].y, (real)angVel[i].z);
    const real3 w2 = mk3((real)angVel[j].x, (real)angVel[j].y, (real)angVel[j].z);
    const real dt = (real)dt_val;
    real3 ortho = mk3(0, 0, 0), twist = mk3(0, 0, 0);
    if (!isNew) {
        const real3 np = mk3((real)b_pnx[k], (real)b_pny[k], (real)b_pnz[k]);
        ortho = crs3(np, n);                                            // normal.cross(currentNormal)
        const real angle = dt * (real)0.5 * dot3(np, add3(w1, w2));     // con la normal previa
        twist = scl3(np, angle);
    }
    b_pnx[k] = (double)n.x; b_pny[k] = (double)n.y; b_pnz[k] = (double)n.z;

    const real3 dv = mk3((real)(vel[j].x - vel[i].x), (real)(vel[j].y - vel[i].y), (real)(vel[j].z - vel[i].z));
    // getIncidentVel(avoidGranularRatcheting=true): alpha·(v2−v1) + w2×(−r2·n) − w1×(r1·n)
    const real alpha = (real)((r1 + r2) / (r1 + r2 - pen));
    real3 relV = sub3(sub3(scl3(dv, alpha), crs3(w2, scl3(n, (real)r2))), crs3(w1, scl3(n, (real)r1)));
    relV = sub3(relV, scl3(n, dot3(n, relV)));
    const real3 shearInc = scl3(relV, dt);

    // STEP A: viscoso normal (getIncidentVel sin anti-ratcheting: la parte normal es n·(v2−v1))
    const real vn = dot3(n, dv);
    const real3 Fvisc = scl3(n, -(real)b_cn[k] * vn);

    // STEP E: corte. El estado acumulado (fuerza de corte y momento) se integra siempre en FP64 (FIX-4).
    const double3_t orthoD = make_d3(ortho.x, ortho.y, ortho.z), twistD = make_d3(twist.x, twist.y, twist.z);
    double3_t sfd = make_d3(b_sfx[k], b_sfy[k], b_sfz[k]);
    sfd = sub_d3(sfd, cross_d3(sfd, orthoD));                          // ScGeom::rotate
    sfd = sub_d3(sfd, cross_d3(sfd, twistD));
    double Ks_eff = b_ks[k];
    if (coh == 1 && Fn > (real)0 && b_kb[k] > 0.0) Ks_eff = b_kseff[k];
    sfd = sub_d3(sfd, scale_d3(make_d3(shearInc.x, shearInc.y, shearInc.z), Ks_eff));
    double maxFs;
    if (coh == 1) { maxFs = b_FsMax[k] + (double)Fn * tgPhi; if (maxFs < 0.0) maxFs = 0.0; }
    else { maxFs = ((Fn > (real)0) ? (double)Fn : 0.0) * mu; }
    const double curFs = sqrt(dot_d3(sfd, sfd));
    if (curFs > maxFs) {
        if (maxFs > 0.0) sfd = scale_d3(sfd, maxFs / curFs); else sfd = make_d3(0.0, 0.0, 0.0);
        if (coh == 1) { coh = 0; b_state[k] = 5; atomicAdd(nb_shear_cracks, 1); }
    }
    b_sfx[k] = sfd.x; b_sfy[k] = sfd.y; b_sfz[k] = sfd.z;
    const real3 sf = mk3((real)sfd.x, (real)sfd.y, (real)sfd.z);

    // STEP F: flexión / rodadura
    double3_t mbd = make_d3(b_mbx[k], b_mby[k], b_mbz[k]);
    mbd = sub_d3(mbd, cross_d3(mbd, orthoD));
    mbd = sub_d3(mbd, cross_d3(mbd, twistD));
    real3 relW = sub3(w2, w1);
    relW = sub3(relW, scl3(n, dot3(n, relW)));
    mbd = sub_d3(mbd, scale_d3(make_d3(relW.x, relW.y, relW.z), b_kb[k] * dt_val));
    if (coh == 0) {
        const double Mlim = ((Fn > (real)0) ? (double)Fn : 0.0) * fmin(r1, r2) * rollFric;
        const double Mm = sqrt(dot_d3(mbd, mbd));
        if (Mm > Mlim) { if (Mlim > 0.0) mbd = scale_d3(mbd, Mlim / Mm); else mbd = make_d3(0.0, 0.0, 0.0); }
    }
    b_mbx[k] = mbd.x; b_mby[k] = mbd.y; b_mbz[k] = mbd.z;
    const real3 mb = mk3((real)mbd.x, (real)mbd.y, (real)mbd.z);

    // STEP G: fuerzas y momentos
    const real3 fT = add3(add3(scl3(n, Fn), Fvisc), sf);
    const real3 ncs = crs3(n, sf);
    const real arm1 = -(real)(r1 - 0.5 * pen), arm2 = -(real)(r2 - 0.5 * pen);
    const real3 t1 = sub3(scl3(ncs, arm1), mb);
    const real3 t2 = add3(scl3(ncs, arm2), mb);
    if (det) {
        double* o = pf_out + 9 * (size_t)k;
        o[0] = fT.x; o[1] = fT.y; o[2] = fT.z; o[3] = t1.x; o[4] = t1.y; o[5] = t1.z; o[6] = t2.x; o[7] = t2.y; o[8] = t2.z;
        return;
    }
    atomicAdd(&force[i].x, (float)-fT.x); atomicAdd(&force[i].y, (float)-fT.y); atomicAdd(&force[i].z, (float)-fT.z);
    atomicAdd(&force[j].x, (float)fT.x); atomicAdd(&force[j].y, (float)fT.y); atomicAdd(&force[j].z, (float)fT.z);
    atomicAdd(&torque[i].x, (float)t1.x); atomicAdd(&torque[i].y, (float)t1.y); atomicAdd(&torque[i].z, (float)t1.z);
    atomicAdd(&torque[j].x, (float)t2.x); atomicAdd(&torque[j].y, (float)t2.y); atomicAdd(&torque[j].z, (float)t2.z);
}

__global__ void kernel_facet_sphere(
    int n_cand_fs,
    double dt_val,
    double mu,
    const double3_t* __restrict__ pos,
    const double3_t* __restrict__ vel,
    const double3_t* __restrict__ angVel,
    const double* __restrict__ radius,
    const double3_t* __restrict__ f_v0,
    const double3_t* __restrict__ f_v1,
    const double3_t* __restrict__ f_v2,
    const double3_t* __restrict__ f_vel,
    const int* __restrict__ fs_f,
    const int* __restrict__ fs_s,
    const double* __restrict__ fs_kn,
    const double* __restrict__ fs_ks,
    const double* __restrict__ fs_cn,
    int* __restrict__ fs_active,
    double* __restrict__ fs_initD,
    double3_t* __restrict__ fs_shearForce,
    double3_t* __restrict__ fs_prevNormal,
    double3_t* __restrict__ f_force,
    float3_t* __restrict__ force,
    float3_t* __restrict__ torque,
    double* __restrict__ fs_out,   // [DET] 6 valores por par: F sobre la esfera, T sobre la esfera
    int det
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_cand_fs) return;
    if (det) { double* o = fs_out + 6 * (size_t)k; for (int c = 0; c < 6; ++c) o[c] = 0.0; }

    int fi = fs_f[k];
    int si = fs_s[k];
    double r2 = radius[si];
    double3_t p = pos[si];

    // Filtro rápido por coordenada Z
    if (fi < N_PLATE) {
        if (f_v0[0].z - p.z >= r2) {   // [FIX-5] sin contacto si r − L <= 0 (como Ig2_Facet_Sphere_ScGeom)
            if (fs_active[k] != 0) {
                fs_active[k] = 0;
                fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            }
            return;
        }
    } else if (fi < 2 * N_PLATE) {
        if (p.z - f_v0[N_PLATE].z >= r2) {
            if (fs_active[k] != 0) {
                fs_active[k] = 0;
                fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            }
            return;
        }
    } else {
        if (f_v0[fi].z - p.z >= r2 || p.z - f_v2[fi].z >= r2) {
            if (fs_active[k] != 0) {
                fs_active[k] = 0;
                fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            }
            return;
        }
    }

    double3_t v0 = f_v0[fi], v1 = f_v1[fi], v2 = f_v2[fi];
    double minx = fmin(fmin(v0.x, v1.x), v2.x);
    double maxx = fmax(fmax(v0.x, v1.x), v2.x);
    if (minx - p.x >= r2 || p.x - maxx >= r2) {
        if (fs_active[k] != 0) { fs_active[k] = 0; fs_shearForce[k] = make_d3(0.0, 0.0, 0.0); }
        return;
    }
    double miny = fmin(fmin(v0.y, v1.y), v2.y);
    double maxy = fmax(fmax(v0.y, v1.y), v2.y);
    if (miny - p.y >= r2 || p.y - maxy >= r2) {
        if (fs_active[k] != 0) { fs_active[k] = 0; fs_shearForce[k] = make_d3(0.0, 0.0, 0.0); }
        return;
    }
    double3_t cp = closest_pt_tri_3d(p, v0, v1, v2);
    double3_t vec = sub_d3(p, cp);
    double dist_sq = dot_d3(vec, vec);

    // [FIX-5] contacto nuevo si penetrationDepth = r − distancia > 0, con sqrt correctamente redondeada (= |L| exacta en el
    // interior de un plato plano, como YADE); con dist² < r² se perdía la esfera que toca el plato en t=0 con penetración de 1 ulp.
    const double dist_c = (dist_sq > 1e-36) ? sqrt(dist_sq) : 0.0;
    if (dist_sq > 1e-36 && r2 - dist_c > 0.0) {
        double dist = dist_c;
        double3_t normal = scale_d3(vec, 1.0 / dist);
        double overlap = r2 - dist;
        bool isNewF = false;
        if (fs_active[k] == 0) {
            fs_active[k] = 1;
            fs_initD[k] = overlap;
            fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            isNewF = true;
        }

        double D = overlap - fs_initD[k];
        if (D < 0.0) {
            fs_active[k] = 0;
            fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            return;
        }

        double Fn = fs_kn[k] * D;
        double3_t c2x = scale_d3(normal, -(r2 - 0.5 * overlap));
        double3_t v1 = f_vel[fi];
        double3_t v2 = add_d3(vel[si], cross_d3(angVel[si], c2x));
        double3_t incidentV = sub_d3(v2, v1);
        double vn_s = dot_d3(normal, incidentV);
        double3_t incidentVn = scale_d3(normal, vn_s);
        double3_t F_visc_vec = scale_d3(incidentVn, -fs_cn[k]);

        double3_t vTang = sub_d3(incidentV, incidentVn);
        double3_t shearInc = scale_d3(vTang, dt_val);

        // [PAR-2] ScGeom::precompute/rotate: orthonormal_axis = n_prev × n, twist con n_prev (w_faceta = 0)
        double3_t ortho_f = make_d3(0.0, 0.0, 0.0), twist_axis = make_d3(0.0, 0.0, 0.0);
        if (!isNewF) {
            double3_t np = fs_prevNormal[k];
            ortho_f = cross_d3(np, normal);
            twist_axis = scale_d3(np, dt_val * 0.5 * dot_d3(np, angVel[si]));
        }
        fs_prevNormal[k] = normal;

        double3_t sf = fs_shearForce[k];
        sf = sub_d3(sf, cross_d3(sf, ortho_f));
        sf = sub_d3(sf, cross_d3(sf, twist_axis));
        sf = sub_d3(sf, scale_d3(shearInc, fs_ks[k]));

        double maxFs = ((Fn > 0.0) ? Fn : 0.0) * mu;
        double sf_sq = dot_d3(sf, sf);
        if (sf_sq > maxFs * maxFs) {
            if (maxFs > 0.0) {
                sf = scale_d3(sf, maxFs / sqrt(sf_sq));
            } else {
                sf = make_d3(0.0, 0.0, 0.0);
            }
        }
        fs_shearForce[k] = sf;

        double3_t fTotal = add_d3(add_d3(scale_d3(normal, Fn), F_visc_vec), sf);
        if (det) {
            double3_t dt2 = scale_d3(cross_d3(normal, sf), -(r2 - 0.5 * overlap));
            double* o = fs_out + 6 * (size_t)k;
            o[0] = fTotal.x; o[1] = fTotal.y; o[2] = fTotal.z; o[3] = dt2.x; o[4] = dt2.y; o[5] = dt2.z;
            return;
        }
        atomicAdd(&f_force[fi].x, -fTotal.x);
        atomicAdd(&f_force[fi].y, -fTotal.y);
        atomicAdd(&f_force[fi].z, -fTotal.z);

        atomicAdd(&force[si].x, (float)fTotal.x);
        atomicAdd(&force[si].y, (float)fTotal.y);
        atomicAdd(&force[si].z, (float)fTotal.z);

        double3_t t2 = scale_d3(cross_d3(normal, sf), -(r2 - 0.5 * overlap));
        atomicAdd(&torque[si].x, (float)t2.x);
        atomicAdd(&torque[si].y, (float)t2.y);
        atomicAdd(&torque[si].z, (float)t2.z);
    } else {
        if (fs_active[k] != 0) {
            fs_active[k] = 0;
            fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
        }
    }
}

__global__ void kernel_integrate_newton(
    int n_particles,
    int n_facets,
    double dt_val,
    const double* __restrict__ mass,
    const double* __restrict__ inertia,
    float3_t* __restrict__ force,
    float3_t* __restrict__ torque,
    double3_t* __restrict__ pos,
    double3_t* __restrict__ vel,
    double3_t* __restrict__ angVel,
    const double3_t* __restrict__ f_vel,
    double3_t* __restrict__ f_pos,
    double3_t* __restrict__ f_v0,
    double3_t* __restrict__ f_v1,
    double3_t* __restrict__ f_v2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
#ifdef DEM_DAMPING
        double cundall = DEM_DAMPING;
#else
        double cundall = 0.01;
#endif
        double3_t f = make_d3((double)force[i].x, (double)force[i].y, (double)force[i].z);
        double3_t v = vel[i];
        // [FIX-2] YADE NewtonIntegrator::cundallDamp2nd: a_i *= 1 - d*sign(a_i*(v_i + 0.5*dt*a_i))
        double inv_m = 1.0 / mass[i];
        double3_t acc = scale_d3(f, inv_m);
        acc.x *= 1.0 - cundall * ((acc.x * (v.x + 0.5 * dt_val * acc.x) > 0.0) ? 1.0 : ((acc.x * (v.x + 0.5 * dt_val * acc.x) < 0.0) ? -1.0 : 0.0));
        acc.y *= 1.0 - cundall * ((acc.y * (v.y + 0.5 * dt_val * acc.y) > 0.0) ? 1.0 : ((acc.y * (v.y + 0.5 * dt_val * acc.y) < 0.0) ? -1.0 : 0.0));
        acc.z *= 1.0 - cundall * ((acc.z * (v.z + 0.5 * dt_val * acc.z) > 0.0) ? 1.0 : ((acc.z * (v.z + 0.5 * dt_val * acc.z) < 0.0) ? -1.0 : 0.0));
        v = add_d3(v, scale_d3(acc, dt_val));
        vel[i] = v;
        pos[i] = add_d3(pos[i], scale_d3(v, dt_val));

        double3_t t = make_d3((double)torque[i].x, (double)torque[i].y, (double)torque[i].z);
        double3_t w = angVel[i];
        double inv_I = 1.0 / inertia[i];
        double3_t angAcc = scale_d3(t, inv_I);
        angAcc.x *= 1.0 - cundall * ((angAcc.x * (w.x + 0.5 * dt_val * angAcc.x) > 0.0) ? 1.0 : ((angAcc.x * (w.x + 0.5 * dt_val * angAcc.x) < 0.0) ? -1.0 : 0.0));
        angAcc.y *= 1.0 - cundall * ((angAcc.y * (w.y + 0.5 * dt_val * angAcc.y) > 0.0) ? 1.0 : ((angAcc.y * (w.y + 0.5 * dt_val * angAcc.y) < 0.0) ? -1.0 : 0.0));
        angAcc.z *= 1.0 - cundall * ((angAcc.z * (w.z + 0.5 * dt_val * angAcc.z) > 0.0) ? 1.0 : ((angAcc.z * (w.z + 0.5 * dt_val * angAcc.z) < 0.0) ? -1.0 : 0.0));
        angVel[i] = add_d3(w, scale_d3(angAcc, dt_val));
        force[i].x = 0.0f; force[i].y = 0.0f; force[i].z = 0.0f;
        torque[i].x = 0.0f; torque[i].y = 0.0f; torque[i].z = 0.0f;
    }

    if (i < n_facets) {
        double3_t dp = scale_d3(f_vel[i], dt_val);
        f_pos[i] = add_d3(f_pos[i], dp);
        f_v0[i] = add_d3(f_v0[i], dp);
        f_v1[i] = add_d3(f_v1[i], dp);
        f_v2[i] = add_d3(f_v2[i], dp);
    }
}

__global__ void kernel_measure_stress(
    const double3_t* __restrict__ f_force,
    const double3_t* __restrict__ f_pos,
    const int* __restrict__ f_ring,
    const double3_t* __restrict__ f_nrad,
    double h_p_0,
    double r_avg_0,
    double* __restrict__ f_rad_ring,
    double* __restrict__ r_cum_ring,
    double* __restrict__ out_metrics // [0]=s_ax, [1]=s_rad, [2]=eps_ax, [3]=eps_rad, [4]=fz, [5]=h_p, [6]=r_avg, [7]=A_rad
) {
    // Ejecutado en 1 solo bloque de 256 hilos
    int tid = threadIdx.x;
    if (tid < N_RINGS) {
        f_rad_ring[tid] = 0.0;
        r_cum_ring[tid] = 0.0;
    }
    __shared__ double sh_f1z[256];
    __shared__ double sh_f2z[256];
    sh_f1z[tid] = 0.0;
    sh_f2z[tid] = 0.0;
    __syncthreads();

    for (int fi = tid; fi < N_PLATE; fi += 256) {
        sh_f1z[tid] += f_force[fi].z;
    }
    for (int fi = N_PLATE + tid; fi < 2 * N_PLATE; fi += 256) {
        sh_f2z[tid] += f_force[fi].z;
    }
    __syncthreads();
    // [DET] suma por anillo en orden fijo (antes: atomicAdd en orden no determinista)
    if (tid < N_RINGS) {
        double fr = 0.0, rc = 0.0;
        for (int fi = 2 * N_PLATE; fi < N_FACETS; ++fi) {
            if (f_ring[fi] != tid) continue;
            fr += f_force[fi].x * f_nrad[fi].x + f_force[fi].y * f_nrad[fi].y;
            rc += sqrt(f_pos[fi].x * f_pos[fi].x + f_pos[fi].y * f_pos[fi].y);
        }
        f_rad_ring[tid] = fr;
        r_cum_ring[tid] = rc;
    }
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sh_f1z[tid] += sh_f1z[tid + s];
            sh_f2z[tid] += sh_f2z[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double f1 = sh_f1z[0];
        double f2 = sh_f2z[0];
        double h_p = f_pos[0].z - f_pos[N_PLATE].z;
        double r_cum_total = 0.0;
        double f_rad_total = 0.0;
        for (int h = 0; h < N_RINGS; ++h) {
            r_cum_total += r_cum_ring[h];
            f_rad_total += f_rad_ring[h];
        }
        double r_avg = r_cum_total / (double)(2 * N_SEG * N_RINGS);
        double pi = 3.14159265358979323846;
        double A_plate = pi * r_avg * r_avg;
        double A_rad = 2.0 * pi * r_avg * h_p;

        out_metrics[0] = -0.5 * (f2 - f1) / A_plate;
        out_metrics[1] = f_rad_total / A_rad;
        out_metrics[2] = -(h_p - h_p_0) / h_p_0;
        out_metrics[3] = -(r_avg - r_avg_0) / r_avg_0;
        out_metrics[4] = 0.5 * (f1 - f2);
        out_metrics[5] = h_p;
        out_metrics[6] = r_avg;
        out_metrics[7] = A_rad;
    }
}

__global__ void kernel_update_ring_stiff(
    int n_cand_fs,
    const int* __restrict__ fs_active,
    const int* __restrict__ fs_f,
    const double* __restrict__ fs_kn,
    const int* __restrict__ f_ring,
    double* __restrict__ cached_ring_stiff
) {
    // [DET] un hilo por anillo, suma en orden fijo
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < N_RINGS) {
        double s = 0.0;
        for (int k = 0; k < n_cand_fs; ++k) {
            if (fs_active[k] != 1) continue;
            int fi = fs_f[k];
            if (fi >= 2 * N_PLATE && f_ring[fi] == h) s += fs_kn[k];
        }
        cached_ring_stiff[h] = s;
    }
}

__global__ void kernel_apply_servo(
    double confining,
    double dt_pwave,
    double r_vel_max,
    const double* __restrict__ f_rad_ring,
    const double* __restrict__ r_cum_ring,
    const double* __restrict__ cached_ring_stiff,
    const double* __restrict__ out_metrics,
    const int* __restrict__ f_ring,
    const double3_t* __restrict__ f_nrad,
    double3_t* __restrict__ f_vel
) {
    int fi = 2 * N_PLATE + blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= N_FACETS) return;

    int h = f_ring[fi];
    double h_p = out_metrics[5];
    double A_rad = out_metrics[7];
    double r_avg_ring = r_cum_ring[h] / (double)(2 * N_SEG);
    double pi = 3.14159265358979323846;
    double area_ring = 2.0 * pi * r_avg_ring * h_p / (double)N_RINGS;
    double stress_in_ring = f_rad_ring[h] / area_ring;
    double gr1 = (cached_ring_stiff[h] > 1.0) ? cached_ring_stiff[h] : 1.0;
    double v_factor = 0.5 * A_rad / (gr1 * dt_pwave);
    double dv = stress_in_ring - confining;
    double rv = v_factor * dv;
    if (rv > r_vel_max) rv = r_vel_max;
    if (rv < -r_vel_max) rv = -r_vel_max;
    f_vel[fi] = scale_d3(f_nrad[fi], rv);
}

// [CONTACTOS] desplazamiento máximo de esferas (out[0]) y facetas (out[1]) desde la última reconstrucción
__global__ void kernel_max_disp(
    int n_particles,
    int n_facets,
    const double3_t* __restrict__ pos,
    const double3_t* __restrict__ pos_ref,
    const double3_t* __restrict__ f_v0,
    const double3_t* __restrict__ f_v0_ref,
    unsigned long long* __restrict__ out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        double d = norm_d3(sub_d3(pos[i], pos_ref[i]));
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(d));
    }
    if (i < n_facets) {
        double d = norm_d3(sub_d3(f_v0[i], f_v0_ref[i]));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(d));
    }
}

// [DET] suma determinista por partícula (CSR: entradas 2k+rol para pares esfera-esfera, k para esfera-faceta)
__global__ void kernel_gather_particles(
    int n_particles,
    const int* __restrict__ ss_start, const int* __restrict__ ss_idx, const double* __restrict__ pf_out,
    const int* __restrict__ fsp_start, const int* __restrict__ fsp_idx, const double* __restrict__ fs_out,
    float3_t* __restrict__ force, float3_t* __restrict__ torque
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    double fx = 0, fy = 0, fz = 0, tx = 0, ty = 0, tz = 0;
    for (int q = ss_start[i]; q < ss_start[i + 1]; ++q) {
        int e = ss_idx[q];
        const double* o = pf_out + 9 * (size_t)(e >> 1);
        if ((e & 1) == 0) { fx -= o[0]; fy -= o[1]; fz -= o[2]; tx += o[3]; ty += o[4]; tz += o[5]; }
        else              { fx += o[0]; fy += o[1]; fz += o[2]; tx += o[6]; ty += o[7]; tz += o[8]; }
    }
    for (int q = fsp_start[i]; q < fsp_start[i + 1]; ++q) {
        const double* o = fs_out + 6 * (size_t)fsp_idx[q];
        fx += o[0]; fy += o[1]; fz += o[2]; tx += o[3]; ty += o[4]; tz += o[5];
    }
    force[i].x = (float)fx; force[i].y = (float)fy; force[i].z = (float)fz;
    torque[i].x = (float)tx; torque[i].y = (float)ty; torque[i].z = (float)tz;
}

__global__ void kernel_gather_facets(
    int n_facets, const int* __restrict__ ff_start, const int* __restrict__ ff_idx,
    const double* __restrict__ fs_out, double3_t* __restrict__ f_force
) {
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= n_facets) return;
    double fx = 0, fy = 0, fz = 0;
    for (int q = ff_start[fi]; q < ff_start[fi + 1]; ++q) {
        const double* o = fs_out + 6 * (size_t)ff_idx[q];
        fx -= o[0]; fy -= o[1]; fz -= o[2];
    }
    f_force[fi] = make_d3(fx, fy, fz);
}

// [VTK] NewtonIntegrator::leapfrogSphericalRotate: ori = q(|w|·dt, w/|w|) · ori, normalizado (solo para exportar rotAngle)
__global__ void kernel_rotate_ori(int n, double dt, const double3_t* __restrict__ angVel, double* __restrict__ ori) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double3_t w = angVel[i];
    double a2 = dot_d3(w, w);
    double* q = ori + 4 * (size_t)i;   // (w, x, y, z)
    if (a2 != 0.0) {
        double a = sqrt(a2), h = 0.5 * a * dt, s = sin(h) / a;
        double rw = cos(h), rx = w.x * s, ry = w.y * s, rz = w.z * s;
        double nw = rw * q[0] - rx * q[1] - ry * q[2] - rz * q[3];
        double nx = rw * q[1] + rx * q[0] + ry * q[3] - rz * q[2];
        double ny = rw * q[2] - rx * q[3] + ry * q[0] + rz * q[1];
        double nz = rw * q[3] + rx * q[2] - ry * q[1] + rz * q[0];
        q[0] = nw; q[1] = nx; q[2] = ny; q[3] = nz;
    }
    double nn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0] /= nn; q[1] /= nn; q[2] /= nn; q[3] /= nn;
}

__global__ void kernel_set_plate_vel(
    double vz_top,
    double vz_bot,
    double3_t* __restrict__ f_vel
) {
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi < N_PLATE) {
        f_vel[fi] = make_d3(0.0, 0.0, vz_top);
    } else if (fi < 2 * N_PLATE) {
        f_vel[fi] = make_d3(0.0, 0.0, vz_bot);
    }
}

} // extern "C"
)CUDA";

struct Vec3 {
    double x, y, z;
};

// =====================================================================================
// [CASO] Parámetros del caso en tiempo de ejecución.
//   Orden de prioridad (de menor a mayor): valores por defecto (= caso de referencia) <
//   variables de entorno antiguas (DEM_GEO, TRIAX_ZVEL, ...) < archivo de caso (--caso archivo.cfg) <
//   argumentos clave=valor en la línea de comandos.
//   Formato del archivo de caso: una línea "clave = valor" por parámetro; '#' inicia un comentario.
// =====================================================================================
struct CfgKey { const char* key; const char* def; const char* env; const char* help; };
static const CfgKey CFG_KEYS[] = {
    // ejecución
    {"pasos",            "200000",               nullptr,               "pasos dinámicos a simular"},
    {"salida",           "resultados/cuda_gpu",  nullptr,               "directorio de salida"},
    {"sobrescribir",     "0",                    nullptr,               "1 = permite escribir sobre una salida existente (facet_force_exact.txt)"},
    {"contactos",        "dinamicos",            nullptr,               "dinamicos (lista de Verlet) | estaticos (lista fija de t=0)"},
    {"gpu",              "0",                    "DEM_GPU",             "índice de la GPU"},
    {"arch",             "auto",                 "DEM_ARCH",            "arquitectura NVRTC (auto = la de la GPU, p. ej. 89 para compute_89)"},
    {"fp32",             "0",                    "DEM_FP32",            "1 = ley de contacto en FP32 (por defecto FP64)"},
    {"atomics",          "0",                    "DEM_ATOMICS",         "1 = suma de fuerzas con atomicAdd (no reproducible)"},
    {"parar_eps1",       "0",                    nullptr,               "detener al superar esta deformación axial (fracción, 0 = no)"},
    {"parar_caida_q",    "0",                    nullptr,               "detener cuando q cae a (1 - valor)·q_pico tras el pico (0 = no)"},
    {"parada_yade",      "1",                    nullptr,               "1 = parada automática de checkProgress() de YADE (fractura: fz < fraccion·fz_max; sigue hasta t = (1+extra)·t_fractura)"},
    {"parada_fraccion",  "0.7",                  nullptr,               "parada_yade: fractura cuando fz/fz_max < fraccion"},
    {"parada_extra",     "0.05",                 nullptr,               "parada_yade: tras la fractura se sigue hasta O.time >= (1+extra)·t_fractura"},
    {"parada_fz_min",    "2000",                 nullptr,               "parada_yade: solo si fz_max > este valor [N]"},
    // entrada
    {"geo",              "inputs/probeta_1.geo", "DEM_GEO",             "probeta (.geo de gengeo: x y z r id tipo tras BeginParticles)"},
    {"enlaces",          "",                     "DEM_BONDS",           "archivo de enlaces (vacío = calcularlos como YADE con intRadius)"},
    {"intRadius",        "1.3",                  "DEM_INTRADIUS",       "factor de detección de enlaces: d <= intRadius·(ri+rj)"},
    {"recentrar",        "auto",                 nullptr,               "auto | 1 | 0: trasladar la probeta al eje z si su centro está desplazado"},
    {"perturb_seed",     "0",                    "DEM_PERTURB_SEED",    "semilla de perturbación de posiciones (0 = sin perturbar)"},
    {"perturb_amp",      "1e-9",                 "DEM_PERTURB_AMP",     "amplitud de la perturbación (fracción del radio)"},
    // material IBM (mismos nombres que test-4-triaxial-opt.py)
    {"young",            "6.671e10",             nullptr,               "módulo de Young de las partículas [Pa]"},
    {"kn_ks_ratio",      "25.0",                 nullptr,               "kn/ks partícula-partícula"},
    {"tensile",          "9.62e6",               nullptr,               "resistencia a tracción del enlace [Pa]"},
    {"cohesion",         "6.704e7",              nullptr,               "cohesión del enlace [Pa]"},
    {"tgPhi",            "0.466",                nullptr,               "tan(phi) del criterio de Mohr-Coulomb del enlace"},
    {"mu",               "0.466",                nullptr,               "fricción de contacto (partícula-partícula y partícula-pared)"},
    {"density",          "1.00e8",               nullptr,               "densidad [kg/m3] (escalada)"},
    {"rollFric",         "0.100",                nullptr,               "coeficiente de rodadura"},
    {"bendingIn",        "1.000",                nullptr,               "factor de rigidez a flexión"},
    {"viscAlpha",        "0.100",                nullptr,               "amortiguamiento viscoso normal (fracción del crítico)"},
    {"damping",          "0.01",                 nullptr,               "amortiguamiento de Cundall (NewtonIntegrator)"},
    {"young_wall",       "1.00e10",              nullptr,               "módulo de Young de las facetas [Pa]"},
    {"kn_ks_wall_ratio", "2.5",                  nullptr,               "kn/ks de las facetas"},
    {"dt_factor",        "0.5",                  nullptr,               "dt = dt_factor · r_min·sqrt(density/young)"},
    // ensayo
    {"confining",        "4.00e7",               nullptr,               "presión de confinamiento [Pa]"},
    {"z_vel",            "1.0e-5",               "TRIAX_ZVEL",          "velocidad de cada plato en la carga desviadora [m/s] (precarga: 2·z_vel)"},
    {"r_vel_max_factor", "3.0",                  nullptr,               "velocidad radial máxima del servo = factor·z_vel"},
    {"servo_preload",    "50",                   "TRIAX_SERVO_PRELOAD", "periodo del servo en la precarga (10 = corrida de referencia resultados/yade_cpu)"},
    {"servo_period",     "50",                   nullptr,               "periodo del servo en la carga desviadora"},
    {"preload_hold",     "50",                   nullptr,               "comprobaciones (cada 100 iter) que se mantiene el confinamiento antes de cargar"},
    // contorno
    {"plate_n_radial",   "10",                   nullptr,               "divisiones radiales de cada plato"},
    {"plate_n_angular",  "24",                   nullptr,               "divisiones angulares de cada plato"},
    {"plate_R_factor",   "1.05",                 nullptr,               "radio del plato = factor·R_probeta"},
    {"n_s",              "24",                   nullptr,               "segmentos angulares de la membrana"},
    {"n_h",              "auto",                 nullptr,               "anillos de la membrana (auto = fórmula de YADE: max(20, min(40, ceil(L/(10·r_max)))))"},
    // contactos y salida
    {"skin_factor",      "0.3",                  "CONTACT_SKIN_FACTOR", "skin de Verlet = factor·r_min"},
    {"contact_verify",   "0",                    "CONTACT_VERIFY",      "cada N pasos cuenta contactos perdidos (0 = no)"},
    {"vtk_every",        "0",                    "DEM_VTK_EVERY",       "exportar VTK cada N iteraciones (0 = no)"},
    {"vtk_ascii",        "0",                    "DEM_VTK_ASCII",       "1 = VTK en ASCII"},
    {"vtk_intrs",        "1",                    "DEM_VTK_INTRS",       "0 = no exportar interacciones"},
};

struct Config {
    std::map<std::string, std::string> v;
    std::map<std::string, std::string> src;   // procedencia de cada valor (para caso_usado.cfg)
    Config() { for (auto& k : CFG_KEYS) { v[k.key] = k.def; src[k.key] = "defecto"; } }
    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? "" : s.substr(a, b - a + 1);
    }
    bool set(const std::string& k, const std::string& val, const std::string& from) {
        if (!v.count(k)) { std::cerr << "Parámetro desconocido '" << k << "' (" << from << "). Use --ayuda para ver la lista.\n"; return false; }
        v[k] = val; src[k] = from; return true;
    }
    bool set_kv(const std::string& kv, const std::string& from) {
        size_t e = kv.find('=');
        if (e == std::string::npos) { std::cerr << "Se esperaba clave=valor: '" << kv << "' (" << from << ")\n"; return false; }
        return set(trim(kv.substr(0, e)), trim(kv.substr(e + 1)), from);
    }
    bool load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) { std::cerr << "No se pudo abrir el archivo de caso " << path << "\n"; return false; }
        std::string line; int n = 0;
        while (std::getline(f, line)) {
            ++n;
            size_t h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
            line = trim(line); if (line.empty()) continue;
            if (!set_kv(line, path + ":" + std::to_string(n))) return false;
        }
        return true;
    }
    void load_env() { for (auto& k : CFG_KEYS) if (k.env && getenv(k.env)) set(k.key, getenv(k.env), std::string("env ") + k.env); }
    const std::string& s(const char* k) const { return v.at(k); }
    double d(const char* k) const {
        char* e = nullptr; double x = strtod(v.at(k).c_str(), &e);
        if (v.at(k).empty() || *e) { std::cerr << "Valor no numérico para " << k << ": '" << v.at(k) << "'\n"; exit(1); }
        return x;
    }
    int i(const char* k) const { return (int)std::lround(d(k)); }
    void write(const std::string& path) const {
        std::ofstream f(path);
        f << "# Parámetros efectivos de la corrida (reutilizable con --caso)\n";
        for (auto& k : CFG_KEYS) f << std::left << std::setw(18) << k.key << " = " << v.at(k.key) << "    # " << src.at(k.key) << "\n";
    }
    static void help() {
        std::cout << "Uso: ./triaxial_cuda_native [pasos] [salida] [static] [--caso archivo.cfg]... [clave=valor]...\n\nParámetros (por defecto = caso de referencia):\n";
        for (auto& k : CFG_KEYS) std::cout << "  " << std::left << std::setw(18) << k.key << " = " << std::setw(22) << (*k.def ? k.def : "\"\"")
                                           << (k.env ? std::string(" [env ") + k.env + "]" : std::string()) << "  " << k.help << "\n";
    }
};

// [PORTABLE] Carga de NVRTC sin rutas fijas: DEM_NVRTC (archivo o directorio) > CUDA_HOME/CUDA_PATH > búsqueda estándar del
// enlazador > /usr/local/cuda* > paquetes pip nvidia-cuda-nvrtc (VIRTUAL_ENV, CONDA_PREFIX, ~/.local) > ruta hallada por compile.sh.
#ifndef NVRTC_HINT
#define NVRTC_HINT ""
#endif
static std::vector<std::string> glob_paths(const std::string& pat) {
    std::vector<std::string> out; glob_t g;
    if (glob(pat.c_str(), 0, nullptr, &g) == 0) for (size_t i = 0; i < g.gl_pathc; ++i) out.push_back(g.gl_pathv[i]);
    globfree(&g);
    std::sort(out.rbegin(), out.rend());   // versiones más nuevas primero
    return out;
}
static void* try_nvrtc_dir(const std::string& dir, std::string& used) {
    for (auto& f : glob_paths(dir + "/libnvrtc.so*")) {
        if (f.find(".alt.") != std::string::npos) continue;
        std::string ver = f.substr(f.find("libnvrtc.so") + 11);   // ".13", ".12", "" ...
        for (auto& b : glob_paths(dir + "/libnvrtc-builtins.so" + ver + "*")) if (b.find(".alt.") == std::string::npos) { dlopen(b.c_str(), RTLD_LAZY | RTLD_GLOBAL); break; }
        if (void* h = dlopen(f.c_str(), RTLD_LAZY)) { used = f; return h; }
    }
    return nullptr;
}
static void* load_nvrtc(std::string& used) {
    std::vector<std::string> dirs;
    if (const char* e = getenv("DEM_NVRTC")) {
        struct stat st;
        if (stat(e, &st) == 0 && S_ISREG(st.st_mode)) {
            std::string f(e), d = f.substr(0, f.find_last_of('/'));
            for (auto& b : glob_paths(d + "/libnvrtc-builtins.so*")) if (b.find(".alt.") == std::string::npos) { dlopen(b.c_str(), RTLD_LAZY | RTLD_GLOBAL); break; }
            if (void* h = dlopen(e, RTLD_LAZY)) { used = e; return h; }
        }
        dirs.push_back(e);
    }
    for (const char* v : {"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"}) if (getenv(v)) { dirs.push_back(std::string(getenv(v)) + "/lib64"); dirs.push_back(std::string(getenv(v)) + "/lib"); }
    for (auto& d : dirs) if (void* h = try_nvrtc_dir(d, used)) return h;
    for (const char* so : {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"})
        if (void* h = dlopen(so, RTLD_LAZY)) { used = so; return h; }
    dirs.clear();
    dirs.push_back("/usr/local/cuda/lib64");
    for (auto& d : glob_paths("/usr/local/cuda-*/lib64")) dirs.push_back(d);
    std::vector<std::string> py_roots;
    for (const char* v : {"VIRTUAL_ENV", "CONDA_PREFIX"}) if (getenv(v)) py_roots.push_back(getenv(v));
    if (getenv("HOME")) py_roots.push_back(std::string(getenv("HOME")) + "/.local");
    py_roots.push_back("/usr/local"); py_roots.push_back("/usr");
    for (auto& r : py_roots) for (const char* sub : {"nvidia/cu13/lib", "nvidia/cu12/lib", "nvidia/cuda_nvrtc/lib"})
        for (auto& d : glob_paths(r + "/lib/python3*/*-packages/" + sub)) dirs.push_back(d);
    if (*NVRTC_HINT) dirs.push_back(NVRTC_HINT);
    for (auto& d : dirs) if (void* h = try_nvrtc_dir(d, used)) return h;
    return nullptr;
}
static void* load_libcuda() {
    for (const char* so : {"libcuda.so.1", "libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"})
        if (void* h = dlopen(so, RTLD_LAZY)) return h;
    return nullptr;
}

int main(int argc, char** argv) {
    // Argumentos: [pasos] [salida] [static] (compatibilidad), --caso archivo.cfg (repetible), clave=valor, --ayuda
    Config cfg;
    cfg.load_env();
    {
        std::vector<std::string> pos_args;
        for (int a = 1; a < argc; ++a) {
            std::string s = argv[a];
            if (s == "--ayuda" || s == "-h" || s == "--help") { Config::help(); return 0; }
            if (s == "--caso" || s == "-c") { if (a + 1 >= argc || !cfg.load_file(argv[++a])) return 1; continue; }
            if (s.find('=') != std::string::npos) { if (!cfg.set_kv(s, "línea de comandos")) return 1; continue; }
            pos_args.push_back(s);
        }
        // los posicionales se aplican después del archivo de caso (tienen prioridad, como clave=valor)
        if (pos_args.size() > 0) cfg.set("pasos", pos_args[0], "línea de comandos");
        if (pos_args.size() > 1) cfg.set("salida", pos_args[1], "línea de comandos");
        if (pos_args.size() > 2) { if (pos_args[2] != "static") { std::cerr << "Argumento no reconocido: " << pos_args[2] << "\n"; return 1; }
                                   cfg.set("contactos", "estaticos", "línea de comandos"); }
        if (pos_args.size() > 3) { std::cerr << "Demasiados argumentos posicionales\n"; return 1; }
    }
    int max_steps = cfg.i("pasos");
    std::string out_dir = cfg.s("salida");
    if (cfg.s("contactos") != "dinamicos" && cfg.s("contactos") != "estaticos") { std::cerr << "contactos debe ser dinamicos o estaticos\n"; return 1; }
    bool dynamic_contacts = cfg.s("contactos") == "dinamicos";
    if (std::ifstream(out_dir + "/facet_force_exact.txt").good() && cfg.i("sobrescribir") != 1) {
        std::cerr << "La salida " << out_dir << " ya contiene resultados (facet_force_exact.txt). Use otra carpeta o sobrescribir=1.\n";
        return 1;
    }

    // 1. Cargar NVRTC y el driver CUDA (sin rutas fijas, ver load_nvrtc)
    std::string nvrtc_used;
    void* h_nvrtc = load_nvrtc(nvrtc_used);
    if (!h_nvrtc) { std::cerr << "No se encontró libnvrtc. Instale el CUDA Toolkit o 'pip install nvidia-cuda-nvrtc' e indique la ruta con DEM_NVRTC=/ruta/a/lib (o CUDA_HOME).\n"; return 1; }
    void* h_cuda = load_libcuda();
    if (!h_cuda) { std::cerr << "No se pudo cargar libcuda.so.1 (driver NVIDIA): " << dlerror() << std::endl; return 1; }
    std::cout << "[CUDA] NVRTC: " << nvrtc_used << "\n";

    auto nvrtcCreateProgram = (nvrtcResult(*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*))dlsym(h_nvrtc, "nvrtcCreateProgram");
    auto nvrtcCompileProgram = (nvrtcResult(*)(nvrtcProgram, int, const char* const*))dlsym(h_nvrtc, "nvrtcCompileProgram");
    auto nvrtcGetPTXSize = (nvrtcResult(*)(nvrtcProgram, size_t*))dlsym(h_nvrtc, "nvrtcGetPTXSize");
    auto nvrtcGetPTX = (nvrtcResult(*)(nvrtcProgram, char*))dlsym(h_nvrtc, "nvrtcGetPTX");
    auto nvrtcGetProgramLogSize = (nvrtcResult(*)(nvrtcProgram, size_t*))dlsym(h_nvrtc, "nvrtcGetProgramLogSize");
    auto nvrtcGetProgramLog = (nvrtcResult(*)(nvrtcProgram, char*))dlsym(h_nvrtc, "nvrtcGetProgramLog");

    auto cuInit = (CUresult(*)(unsigned int))dlsym(h_cuda, "cuInit");
    auto cuDeviceGet = (CUresult(*)(CUdevice*, int))dlsym(h_cuda, "cuDeviceGet");
    auto cuCtxCreate = (CUresult(*)(CUcontext*, unsigned int, CUdevice))dlsym(h_cuda, "cuCtxCreate_v2");
    auto cuModuleLoadData = (CUresult(*)(CUmodule*, const void*))dlsym(h_cuda, "cuModuleLoadData");
    auto cuModuleGetFunction = (CUresult(*)(CUfunction*, CUmodule, const char*))dlsym(h_cuda, "cuModuleGetFunction");
    auto cuMemAlloc = (CUresult(*)(CUdeviceptr*, size_t))dlsym(h_cuda, "cuMemAlloc_v2");
    auto cuMemcpyHtoD = (CUresult(*)(CUdeviceptr, const void*, size_t))dlsym(h_cuda, "cuMemcpyHtoD_v2");
    auto cuMemcpyDtoH = (CUresult(*)(void*, CUdeviceptr, size_t))dlsym(h_cuda, "cuMemcpyDtoH_v2");
    auto cuMemsetD8 = (CUresult(*)(CUdeviceptr, unsigned char, size_t))dlsym(h_cuda, "cuMemsetD8_v2");
    auto cuLaunchKernel = (CUresult(*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**))dlsym(h_cuda, "cuLaunchKernel");
    auto cuCtxSynchronize = (CUresult(*)())dlsym(h_cuda, "cuCtxSynchronize");
    auto cuMemcpyDtoD = (CUresult(*)(CUdeviceptr, CUdeviceptr, size_t))dlsym(h_cuda, "cuMemcpyDtoD_v2");
    auto cuMemFree = (CUresult(*)(CUdeviceptr))dlsym(h_cuda, "cuMemFree_v2");

    auto cuDeviceGetAttribute = (CUresult(*)(int*, int, CUdevice))dlsym(h_cuda, "cuDeviceGetAttribute");
    auto cuDeviceGetName = (CUresult(*)(char*, int, CUdevice))dlsym(h_cuda, "cuDeviceGetName");
    auto nvrtcGetSupportedArchs = (nvrtcResult(*)(int*))dlsym(h_nvrtc, "nvrtcGetSupportedArchs");
    auto nvrtcGetNumSupportedArchs = (nvrtcResult(*)(int*))dlsym(h_nvrtc, "nvrtcGetNumSupportedArchs");

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, cfg.i("gpu")));
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));
    // [PORTABLE] arquitectura de compilación = la de la GPU (compute_XY), limitada a la máxima que soporte esta NVRTC
    int arch = 0;
    {
        char name[256] = {0}; cuDeviceGetName(name, 255, device);
        int cc_major = 0, cc_minor = 0;
        CHECK_CU(cuDeviceGetAttribute(&cc_major, 75, device));   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        CHECK_CU(cuDeviceGetAttribute(&cc_minor, 76, device));   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        arch = cfg.s("arch") == "auto" ? 10 * cc_major + cc_minor : cfg.i("arch");
        if (cfg.s("arch") == "auto" && nvrtcGetNumSupportedArchs && nvrtcGetSupportedArchs) {
            int na = 0; nvrtcGetNumSupportedArchs(&na);
            std::vector<int> archs(na); if (na > 0) nvrtcGetSupportedArchs(archs.data());
            int best = 0; for (int a : archs) if (a <= arch) best = std::max(best, a);
            if (best > 0 && best != arch) std::cout << "[CUDA] NVRTC no soporta compute_" << arch << "; se usa compute_" << best << " (PTX compilado por el driver)\n";
            if (best > 0) arch = best;
        }
        std::cout << "[CUDA] GPU " << cfg.i("gpu") << ": " << name << " (sm_" << cc_major << cc_minor << "), compilación para compute_" << arch << "\n";
    }

    // 2. Parámetros físicos (archivo de caso; por defecto, los del caso de referencia)
    const double PI = 3.14159265358979323846;
    const double young = cfg.d("young");
    const double kn_ks_ratio = cfg.d("kn_ks_ratio");
    const double tensile = cfg.d("tensile");
    const double cohesion = cfg.d("cohesion");
    const double tgPhi = cfg.d("tgPhi");
    const double mu = cfg.d("mu");
    const double density = cfg.d("density");
    const double rollFric = cfg.d("rollFric");
    const double bendingIn = cfg.d("bendingIn");
    const double viscAlpha = cfg.d("viscAlpha");
    const double confining = cfg.d("confining");
    const double young_wall = cfg.d("young_wall");
    const double kn_ks_wall_ratio = cfg.d("kn_ks_wall_ratio");

    // 3. Cargar esferas desde probeta_1.geo
    std::string geo_path = cfg.s("geo");
    if (!std::ifstream(geo_path).good() && cfg.src.at("geo") == "defecto") geo_path = "probeta_1.geo";
    if (!std::ifstream(geo_path).good()) { std::cerr << "No se encontró la probeta " << geo_path << "\n"; return 1; }
    std::cout << "[Entrada] probeta: " << geo_path << "\n";
    std::ifstream geo(geo_path);
    std::string line;
    bool in_p = false;
    std::vector<Vec3> coords0;
    std::vector<double> radii;
    while (std::getline(geo, line)) {
        if (line.find("BeginParticles") != std::string::npos) { in_p = true; continue; }
        if (in_p) {
            std::istringstream ss(line);
            double x, y, z, r; int id, t;
            if (ss >> x >> y >> z >> r >> id >> t) {
                coords0.push_back({x, y, z});
                radii.push_back(r);
            }
        }
    }
    int n_particles = radii.size();
    if (n_particles == 0) { std::cerr << "La probeta " << geo_path << " no tiene partículas legibles (se esperan líneas 'x y z r id tipo' tras BeginParticles, con punto decimal)\n"; return 1; }
    // DEM_PERTURB_SEED=s (>0): desplaza cada centro inicial un valor uniforme en [-a, a]·r, a = DEM_PERTURB_AMP (1e-9 por defecto).
    // Los enlaces se instalan después, en la posición perturbada, así que no se introducen tensiones iniciales.
    if (cfg.i("perturb_seed") > 0) {
        std::mt19937_64 rng((uint64_t)cfg.i("perturb_seed"));
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        double amp = cfg.d("perturb_amp");
        for (int i = 0; i < n_particles; ++i) { coords0[i].x += U(rng) * amp * radii[i]; coords0[i].y += U(rng) * amp * radii[i]; coords0[i].z += U(rng) * amp * radii[i]; }
        std::cout << "[Perturbación] semilla=" << cfg.i("perturb_seed") << " amplitud=" << amp << "·r\n";
    }
    double xMin = 1e9, xMax = -1e9, yMin = 1e9, yMax = -1e9, zMin = 1e9, zMax = -1e9, min_r = 1e9;
    for (int i = 0; i < n_particles; ++i) {
        xMin = std::min(xMin, coords0[i].x - radii[i]);
        xMax = std::max(xMax, coords0[i].x + radii[i]);
        yMin = std::min(yMin, coords0[i].y - radii[i]);
        yMax = std::max(yMax, coords0[i].y + radii[i]);
        zMin = std::min(zMin, coords0[i].z - radii[i]);
        zMax = std::max(zMax, coords0[i].z + radii[i]);
        min_r = std::min(min_r, radii[i]);
    }
    {   // [CASO] el contorno (platos y membrana) se construye alrededor del eje z, como en test-4-triaxial-opt.py.
        // recentrar=auto traslada la probeta en x-y solo si su centro se aparta más de un 1 % del radio (la referencia no se toca).
        double cx = 0.5 * (xMin + xMax), cy = 0.5 * (yMin + yMax), R0 = 0.25 * (xMax - xMin + yMax - yMin);
        const std::string rc = cfg.s("recentrar");
        bool off = std::sqrt(cx * cx + cy * cy) > 0.01 * R0;
        if ((rc == "auto" && off) || rc == "1") {
            for (auto& c : coords0) { c.x -= cx; c.y -= cy; }
            xMin -= cx; xMax -= cx; yMin -= cy; yMax -= cy;
            std::cout << "[Entrada] probeta trasladada al eje z: (" << -cx << ", " << -cy << ") m\n";
        } else if (off) std::cerr << "AVISO: el centro de la probeta está en (" << cx << ", " << cy << ") m, fuera del eje z del contorno (recentrar=0)\n";
        double ex = xMax - xMin, ey = yMax - yMin;
        if (std::fabs(ex - ey) > 0.05 * std::max(ex, ey)) std::cerr << "AVISO: la sección de la probeta no es circular (" << ex << " x " << ey << " m); la membrana es cilíndrica\n";
    }
    double spec_D = 0.5 * (xMax - xMin + yMax - yMin);
    double spec_L = zMax - zMin;
    double spec_R = 0.5 * spec_D;
    double dt_pwave = min_r * std::sqrt(density / young);
    double dt = cfg.d("dt_factor") * dt_pwave;

    std::vector<double> mass(n_particles), inertia(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        mass[i] = (4.0 / 3.0) * PI * std::pow(radii[i], 3) * density;
        inertia[i] = 0.4 * mass[i] * radii[i] * radii[i];
    }

    // 4. Cargar enlaces de bonds_JCFP.txt
    std::vector<std::pair<int,int>> raw_b_pre;
    // [PAR-5] Por defecto los enlaces se calculan como YADE en la iteración 0: Ig2_Sphere_Sphere_ScGeom con
    // interactionDetectionFactor = intRadius (1,3) acepta el par si d <= 1,3·(ri+rj). DEM_BONDS=<archivo> fuerza a leerlos.
    const double intRadius = cfg.d("intRadius");
    std::string bonds_path = cfg.s("enlaces");
    if (!bonds_path.empty() && !std::ifstream(bonds_path).good()) { std::cerr << "No se encontró el archivo de enlaces " << bonds_path << "\n"; return 1; }
    std::ifstream bfile(bonds_path);
    if (bonds_path.empty()) {
        // malla uniforme para encontrar los pares
        double rmx = 0; for (double r : radii) rmx = std::max(rmx, r);
        double h = 2.0 * intRadius * rmx;
        double lo[3] = {1e30, 1e30, 1e30};
        for (auto& p : coords0) { lo[0] = std::min(lo[0], p.x); lo[1] = std::min(lo[1], p.y); lo[2] = std::min(lo[2], p.z); }
        std::map<std::tuple<int,int,int>, std::vector<int>> cells;
        auto ci = [&](const Vec3& p) { return std::make_tuple((int)std::floor((p.x - lo[0]) / h), (int)std::floor((p.y - lo[1]) / h), (int)std::floor((p.z - lo[2]) / h)); };
        for (int i = 0; i < n_particles; ++i) cells[ci(coords0[i])].push_back(i);
        std::vector<std::pair<int,int>> prs;
        for (int i = 0; i < n_particles; ++i) {
            auto c = ci(coords0[i]);
            for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                auto it_c = cells.find(std::make_tuple(std::get<0>(c) + dx, std::get<1>(c) + dy, std::get<2>(c) + dz));
                if (it_c == cells.end()) continue;
                for (int j : it_c->second) {
                    if (j <= i) continue;
                    double ddx = coords0[j].x - coords0[i].x, ddy = coords0[j].y - coords0[i].y, ddz = coords0[j].z - coords0[i].z;
                    double lim = intRadius * (radii[i] + radii[j]);
                    if (lim * lim - (ddx * ddx + ddy * ddy + ddz * ddz) >= 0.0) prs.push_back({i, j});
                }
            }
        }
        std::sort(prs.begin(), prs.end());
        std::cout << "[Entrada] enlaces calculados (d <= " << intRadius << "·(ri+rj)): " << prs.size() << "\n";
        for (auto& pr : prs) { raw_b_pre.push_back(pr); }
    } else std::cout << "[Entrada] enlaces leídos de " << bonds_path << "\n";
    std::vector<int> b_p1, b_p2;
    int id1, id2;
    struct BondInit { int i, j; unsigned char st; };
    std::vector<BondInit> raw_b;
    std::set<std::pair<int,int>> bonded_set;
    for (auto& pr : raw_b_pre) {
        raw_b.push_back({pr.first, pr.second, 3});
        bonded_set.insert({std::min(pr.first, pr.second), std::max(pr.first, pr.second)});
    }
    while (!bonds_path.empty() && bfile >> id1 >> id2) {
        raw_b.push_back({id1, id2, 3}); // act=1, coh=1
        bonded_set.insert({std::min(id1, id2), std::max(id1, id2)});
    }
    // Agregar pares vecinos cercanos no enlazados (para capturar nuevos contactos friccionales post-fractura)
    for (int i = 0; i < n_particles && !dynamic_contacts; ++i) {
        for (int j = i + 1; j < n_particles; ++j) {
            double dz = std::abs(coords0[j].z - coords0[i].z);
            double r_sum = radii[i] + radii[j];
            if (dz >= 1.32 * r_sum) continue;
            double dx = coords0[j].x - coords0[i].x;
            if (std::abs(dx) >= 1.32 * r_sum) continue;
            double dy = coords0[j].y - coords0[i].y;
            if (std::abs(dy) >= 1.32 * r_sum) continue;
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < 1.32 * r_sum && bonded_set.find({i, j}) == bonded_set.end()) {
                raw_b.push_back({i, j, 0}); // act=0, coh=0
            }
        }
    }
    std::sort(raw_b.begin(), raw_b.end(), [&](const BondInit& a, const BondInit& b) {
        double za = coords0[a.i].z + coords0[a.j].z;
        double zb = coords0[b.i].z + coords0[b.j].z;
        if (std::abs(za - zb) > 2e-3) return za < zb;
        double ya = coords0[a.i].y + coords0[a.j].y;
        double yb = coords0[b.i].y + coords0[b.j].y;
        return ya < yb;
    });
    int nb_raw = raw_b.size();
    std::vector<unsigned char> b_state_init;
    for (int base = 0; base < nb_raw; base += 256) {
        int chunk = std::min(256, nb_raw - base);
        if (chunk == 256) {
            for (int t = 0; t < 256; ++t) {
                int idx_p = base + (t % 16) * 16 + (t / 16);
                b_p1.push_back(raw_b[idx_p].i);
                b_p2.push_back(raw_b[idx_p].j);
                b_state_init.push_back(raw_b[idx_p].st);
            }
        } else {
            for (int t = 0; t < chunk; ++t) {
                b_p1.push_back(raw_b[base + t].i);
                b_p2.push_back(raw_b[base + t].j);
                b_state_init.push_back(raw_b[base + t].st);
            }
        }
    }
    int n_bonds = b_p1.size();

    std::vector<double> b_kn(n_bonds), b_ks(n_bonds), b_kb(n_bonds), b_cn(n_bonds);
    std::vector<double> b_FnMax(n_bonds), b_FsMax(n_bonds), b_beamLenSq(n_bonds);
    std::vector<double> b_initD(n_bonds);
    std::vector<unsigned char> b_state = b_state_init;
    std::vector<Vec3> b_prevNormal(n_bonds);

    for (int k = 0; k < n_bonds; ++k) {
        int i = b_p1[k], j = b_p2[k];
        double r1 = radii[i], r2 = radii[j];
        double minR = std::min(r1, r2);
        double area = PI * minR * minR;
        double kn1 = 2.0 * young * r1, kn2 = 2.0 * young * r2;
        double kn_val = (kn1 * kn2) / (kn1 + kn2);
        b_kn[k] = kn_val;
        double ks1 = kn1 / kn_ks_ratio, ks2 = kn2 / kn_ks_ratio;
        b_ks[k] = (ks1 * ks2) / (ks1 + ks2);
        double kr1 = 0.5 * young * r1 * r1 * r1, kr2 = 0.5 * young * r2 * r2 * r2;
        b_kb[k] = ((kr1 * kr2) / (kr1 + kr2)) * bendingIn;
        double meff = (mass[i] * mass[j]) / (mass[i] + mass[j]);
        b_cn[k] = viscAlpha * 2.0 * std::sqrt(kn_val * meff);
        b_FnMax[k] = tensile * area;
        b_FsMax[k] = cohesion * area;

        double dx = coords0[j].x - coords0[i].x;
        double dy = coords0[j].y - coords0[i].y;
        double dz = coords0[j].z - coords0[i].z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        double ov = (r1 + r2) - dist;
        b_initD[k] = ov;
        double bl = r1 + r2 - ov;
        b_beamLenSq[k] = b_ks[k] + (12.0 * b_kb[k]) / (bl * bl);
        b_prevNormal[k] = {dx / dist, dy / dist, dz / dist};
    }

    // 5. Contorno: dos platos y la membrana radial, generados como en test-4-triaxial-opt.py.
    // [CASO] Los tamaños se derivan de la geometría: n_plate = plate_n_angular·(2·plate_n_radial − 1) facetas por plato,
    // membrana de n_s segmentos × n_h anillos (2 facetas por celda). Orden: plato superior, plato inferior, membrana.
    // [PAR-5] f_pos es el centro del círculo inscrito de cada faceta (yade.utils.facet)
    const int plate_n_radial = cfg.i("plate_n_radial"), plate_n_angular = cfg.i("plate_n_angular");
    const int n_s = cfg.i("n_s");
    int n_h;
    {   // misma fórmula que el script YADE (ring_height_target = 10·r_max), salvo que el caso fije n_h
        double rmx = 0; for (double r : radii) rmx = std::max(rmx, r);
        n_h = cfg.s("n_h") == "auto" ? std::max(20, std::min(40, (int)std::ceil(spec_L / (10.0 * rmx)))) : cfg.i("n_h");
    }
    if (plate_n_radial < 1 || plate_n_angular < 3 || n_s < 3 || n_h < 1 || n_h > 256) { std::cerr << "Contorno no válido (plate_n_radial>=1, plate_n_angular>=3, n_s>=3, 1<=n_h<=256)\n"; return 1; }
    const int n_plate = plate_n_angular * (2 * plate_n_radial - 1);
    const int n_mem = 2 * n_s * n_h;
    int n_facets = 2 * n_plate + n_mem;
    std::cout << "[Contorno] platos " << plate_n_radial << "x" << plate_n_angular << " (" << n_plate << " facetas c/u), membrana "
              << n_s << "x" << n_h << " (" << n_mem << " facetas, anillo de " << spec_L / n_h * 1e3 << " mm), total " << n_facets << "\n";
    std::vector<Vec3> f_pos(n_facets);

    double plate_R = spec_R * cfg.d("plate_R_factor");
    std::vector<Vec3> f_v0, f_v1, f_v2;

    auto add_plate = [&](double z_p) {
        for (int ia = 0; ia < plate_n_angular; ++ia) {
            double a1 = 2.0 * PI * ia / plate_n_angular;
            double a2 = 2.0 * PI * (ia + 1) / plate_n_angular;
            double r_out = plate_R / plate_n_radial;
            f_v0.push_back({0.0, 0.0, z_p});
            f_v1.push_back({r_out * std::cos(a1), r_out * std::sin(a1), z_p});
            f_v2.push_back({r_out * std::cos(a2), r_out * std::sin(a2), z_p});
        }
        for (int ir = 1; ir < plate_n_radial; ++ir) {
            double r_in = plate_R * ir / plate_n_radial;
            double r_out = plate_R * (ir + 1) / plate_n_radial;
            for (int ia = 0; ia < plate_n_angular; ++ia) {
                double a1 = 2.0 * PI * ia / plate_n_angular;
                double a2 = 2.0 * PI * (ia + 1) / plate_n_angular;
                Vec3 v1 = {r_in * std::cos(a1), r_in * std::sin(a1), z_p};
                Vec3 v2 = {r_out * std::cos(a1), r_out * std::sin(a1), z_p};
                Vec3 v3 = {r_out * std::cos(a2), r_out * std::sin(a2), z_p};
                Vec3 v4 = {r_in * std::cos(a2), r_in * std::sin(a2), z_p};
                f_v0.push_back(v1); f_v1.push_back(v2); f_v2.push_back(v3);
                f_v0.push_back(v1); f_v1.push_back(v3); f_v2.push_back(v4);
            }
        }
    };
    add_plate(zMax);
    add_plate(zMin);
    // yade.utils.facet: centro = inscribedCircleCenter(v0, v1, v2)
    auto incenter = [](Vec3 a, Vec3 b, Vec3 c) -> Vec3 {
        auto nrm = [](Vec3 u, Vec3 v) { return std::sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y) + (u.z - v.z) * (u.z - v.z)); };
        double l01 = nrm(b, a), l02 = nrm(c, a), l12 = nrm(c, b), s = l01 + l12 + l02;
        return {a.x + ((c.x - a.x) * l01 + (b.x - a.x) * l02) / s, a.y + ((c.y - a.y) * l01 + (b.y - a.y) * l02) / s, a.z + ((c.z - a.z) * l01 + (b.z - a.z) * l02) / s};
    };
    for (int k = 0; k < 2 * n_plate; ++k) f_pos[k] = incenter(f_v0[k], f_v1[k], f_v2[k]);

    // la membrana se genera primero en f_v0..f_v2 y luego se calculan todos los centros
    std::vector<Vec3> mem_v0, mem_v1, mem_v2;
    double spec_R_cyl = spec_R / std::cos(PI / n_s);
    std::vector<int> f_ring(n_facets, -1);
    std::vector<Vec3> f_nrad(n_facets, {0.0, 0.0, 0.0});
    int idx_f = 2 * n_plate;
    double r_avg_0 = 0.0;
    for (int r = 0; r < n_s; ++r) {
        for (int h = 0; h < n_h; ++h) {
            double a_lo = 2.0 * PI * (r - 0.5) / n_s;
            double a_hi = 2.0 * PI * (r + 0.5) / n_s;
            double z_lo = zMin + spec_L * h / n_h;
            double z_hi = zMin + spec_L * (h + 1) / n_h;
            Vec3 v1 = {spec_R_cyl * std::cos(a_lo), spec_R_cyl * std::sin(a_lo), z_lo};
            Vec3 v2 = {spec_R_cyl * std::cos(a_hi), spec_R_cyl * std::sin(a_hi), z_lo};
            Vec3 v3 = {spec_R_cyl * std::cos(a_hi), spec_R_cyl * std::sin(a_hi), z_hi};
            Vec3 v4 = {spec_R_cyl * std::cos(a_lo), spec_R_cyl * std::sin(a_lo), z_hi};
            f_v0.push_back(v1); f_v1.push_back(v2); f_v2.push_back(v3);
            f_v0.push_back(v1); f_v1.push_back(v3); f_v2.push_back(v4);
            for (int t = 0; t < 2; ++t) {
                f_pos[idx_f] = incenter(f_v0[idx_f], f_v1[idx_f], f_v2[idx_f]);
                f_ring[idx_f] = h;
                double px = f_pos[idx_f].x, py = f_pos[idx_f].y;
                double rr = std::sqrt(px * px + py * py);
                f_nrad[idx_f] = {px / rr, py / rr, 0.0};
                r_avg_0 += rr;
                idx_f++;
            }
        }
    }
    r_avg_0 /= (double)n_mem;
    double h_p_0 = zMax - zMin;

    // 6. Pares candidatos Facet-Sphere usando distancia 3D exacta al triángulo en t=0
    auto host_closest_pt_tri = [](Vec3 p, Vec3 a, Vec3 b, Vec3 c) -> Vec3 {
        auto sub = [](Vec3 u, Vec3 v) -> Vec3 { return {u.x-v.x, u.y-v.y, u.z-v.z}; };
        auto add = [](Vec3 u, Vec3 v) -> Vec3 { return {u.x+v.x, u.y+v.y, u.z+v.z}; };
        auto sc = [](Vec3 u, double k) -> Vec3 { return {u.x*k, u.y*k, u.z*k}; };
        auto dot = [](Vec3 u, Vec3 v) -> double { return u.x*v.x + u.y*v.y + u.z*v.z; };
        Vec3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
        double d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0.0 && d2 <= 0.0) return a;
        Vec3 bp = sub(p, b);
        double d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) return b;
        double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return add(a, sc(ab, d1 / (d1 - d3)));
        Vec3 cp = sub(p, c);
        double d5 = dot(ab, cp), d6 = dot(ac, cp);
        if (d6 >= 0.0 && d5 <= d6) return c;
        double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return add(a, sc(ac, d2 / (d2 - d6)));
        double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) return add(b, sc(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
        double denom = 1.0 / (va + vb + vc);
        return add(a, add(sc(ab, vb * denom), sc(ac, vc * denom)));
    };
    std::vector<int> fs_f, fs_s;
    for (int fi = 0; fi < n_facets; ++fi) {
        double margin = (fi < 2 * n_plate) ? 1.5e-3 : 0.8e-3;   // solo modo estático (la lista dinámica usa el skin)
        double minx = std::min({f_v0[fi].x, f_v1[fi].x, f_v2[fi].x}) - margin;
        double maxx = std::max({f_v0[fi].x, f_v1[fi].x, f_v2[fi].x}) + margin;
        double miny = std::min({f_v0[fi].y, f_v1[fi].y, f_v2[fi].y}) - margin;
        double maxy = std::max({f_v0[fi].y, f_v1[fi].y, f_v2[fi].y}) + margin;
        double minz = std::min({f_v0[fi].z, f_v1[fi].z, f_v2[fi].z}) - margin;
        double maxz = std::max({f_v0[fi].z, f_v1[fi].z, f_v2[fi].z}) + margin;
        for (int si = 0; si < n_particles; ++si) {
            Vec3 p = coords0[si];
            double r = radii[si];
            if (p.x + r < minx || p.x - r > maxx ||
                p.y + r < miny || p.y - r > maxy ||
                p.z + r < minz || p.z - r > maxz) continue;
            Vec3 cp = host_closest_pt_tri(p, f_v0[fi], f_v1[fi], f_v2[fi]);
            double dx = p.x - cp.x, dy = p.y - cp.y, dz = p.z - cp.z;
            if (std::sqrt(dx*dx + dy*dy + dz*dz) <= r + margin) {
                fs_f.push_back(fi);
                fs_s.push_back(si);
            }
        }
    }
    std::cout << "n_cand_fs tightened to: " << fs_f.size() << "\n";
    int n_cand_fs = fs_f.size();
    std::vector<double> fs_kn(n_cand_fs), fs_ks(n_cand_fs), fs_cn(n_cand_fs), fs_initD(n_cand_fs, 0.0);
    std::vector<int> fs_active(n_cand_fs, 0);
    std::vector<Vec3> fs_zero(n_cand_fs, {0.0, 0.0, 0.0}), fs_prevNormal(n_cand_fs, {0.0, 0.0, 1.0});

    for (int k = 0; k < n_cand_fs; ++k) {
        int si = fs_s[k];
        double r2 = radii[si];
        double r1 = 2.0 * r2;
        double kn1 = 2.0 * young_wall * r1;
        double kn2 = 2.0 * young * r2;
        double kn_val = (kn1 * kn2) / (kn1 + kn2);
        fs_kn[k] = kn_val;
        double ks1 = kn1 / kn_ks_wall_ratio;
        double ks2 = kn2 / kn_ks_ratio;
        fs_ks[k] = (ks1 * ks2) / (ks1 + ks2);
        fs_cn[k] = viscAlpha * 2.0 * std::sqrt(kn_val * mass[si]);
    }

    // Compilación de los kernels con NVRTC para la GPU detectada
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, CUDA_KERNEL_SOURCE, "triaxial_kernels.cu", 0, nullptr, nullptr));
    // FP64 por defecto (reproduce la trayectoria de YADE hasta que el caos la separa); fp32=1 para la ley en FP32
    bool fp64_law = cfg.i("fp32") != 1;
    // [CASO] tamaños del contorno y amortiguamiento de Cundall como constantes de compilación del kernel
    std::vector<std::string> opt_s = {"--gpu-architecture=compute_" + std::to_string(arch)};
    if (fp64_law) opt_s.push_back("-DDEM_FP64");
    opt_s.push_back("-DN_PLATE=" + std::to_string(n_plate));
    opt_s.push_back("-DN_SEG=" + std::to_string(n_s));
    opt_s.push_back("-DN_RINGS=" + std::to_string(n_h));
    { char b[64]; snprintf(b, sizeof(b), "-DDEM_DAMPING=%.17g", cfg.d("damping")); opt_s.push_back(b); }
    std::vector<const char*> opts; for (auto& o : opt_s) opts.push_back(o.c_str());
    nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    std::cout << (fp64_law ? "[Precisión] ley de contacto en FP64\n" : "[Precisión] ley de contacto en FP32 (fp32=1)\n");
    if (compileRes != 0) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        std::cerr << "NVRTC Compile Log:\n" << log << std::endl;
        return 1;
    }
    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
    if (getenv("DEM_DUMP_PTX")) { std::ofstream(getenv("DEM_DUMP_PTX")) << ptx.data(); }

    CUmodule module;
    CHECK_CU(cuModuleLoadData(&module, ptx.data()));
    CUfunction fn_reset, fn_ss, fn_fs, fn_newton, fn_measure, fn_stiff, fn_servo, fn_plate_vel, fn_maxdisp, fn_gather_p, fn_gather_f, fn_rotori;
    CHECK_CU(cuModuleGetFunction(&fn_reset, module, "kernel_reset_forces"));
    CHECK_CU(cuModuleGetFunction(&fn_ss, module, "kernel_sphere_sphere"));
    CHECK_CU(cuModuleGetFunction(&fn_fs, module, "kernel_facet_sphere"));
    CHECK_CU(cuModuleGetFunction(&fn_newton, module, "kernel_integrate_newton"));
    CHECK_CU(cuModuleGetFunction(&fn_measure, module, "kernel_measure_stress"));
    CHECK_CU(cuModuleGetFunction(&fn_stiff, module, "kernel_update_ring_stiff"));
    CHECK_CU(cuModuleGetFunction(&fn_servo, module, "kernel_apply_servo"));
    CHECK_CU(cuModuleGetFunction(&fn_plate_vel, module, "kernel_set_plate_vel"));
    CHECK_CU(cuModuleGetFunction(&fn_maxdisp, module, "kernel_max_disp"));
    CHECK_CU(cuModuleGetFunction(&fn_gather_p, module, "kernel_gather_particles"));
    CHECK_CU(cuModuleGetFunction(&fn_gather_f, module, "kernel_gather_facets"));
    CHECK_CU(cuModuleGetFunction(&fn_rotori, module, "kernel_rotate_ori"));

    // 7. Reservar y copiar memoria en GPU
    auto alloc_and_copy = [&](const void* src, size_t bytes) -> CUdeviceptr {
        CUdeviceptr d;
        CHECK_CU(cuMemAlloc(&d, bytes));
        if (src) CHECK_CU(cuMemcpyHtoD(d, src, bytes));
        else CHECK_CU(cuMemsetD8(d, 0, bytes));
        return d;
    };

    // [CONTACTOS] reserva con capacidad: la parte [0, n_bonds) son los enlaces (estáticos);
    // la parte [n_bonds, n_ss) son contactos no enlazados que la lista de Verlet reconstruye.
    size_t cap_ss = dynamic_contacts ? (size_t)n_bonds + std::max<size_t>(200000, 2 * (size_t)n_bonds) : (size_t)n_bonds;
    size_t cap_fs = dynamic_contacts ? std::max<size_t>(50000, 4 * (size_t)n_cand_fs) : (size_t)n_cand_fs;
    auto alloc_cap = [&](const void* src, size_t used_bytes, size_t cap_bytes) -> CUdeviceptr {
        CUdeviceptr d;
        CHECK_CU(cuMemAlloc(&d, cap_bytes));
        CHECK_CU(cuMemsetD8(d, 0, cap_bytes));
        if (src && used_bytes) CHECK_CU(cuMemcpyHtoD(d, src, used_bytes));
        return d;
    };
    CUdeviceptr d_pos = alloc_and_copy(coords0.data(), n_particles * sizeof(Vec3));
    CUdeviceptr d_vel = alloc_and_copy(nullptr, n_particles * sizeof(Vec3));
    CUdeviceptr d_angVel = alloc_and_copy(nullptr, n_particles * sizeof(Vec3));
    CUdeviceptr d_force = alloc_and_copy(nullptr, n_particles * sizeof(Vec3));
    CUdeviceptr d_torque = alloc_and_copy(nullptr, n_particles * sizeof(Vec3));
    CUdeviceptr d_radius = alloc_and_copy(radii.data(), n_particles * sizeof(double));
    CUdeviceptr d_mass = alloc_and_copy(mass.data(), n_particles * sizeof(double));
    CUdeviceptr d_inertia = alloc_and_copy(inertia.data(), n_particles * sizeof(double));

    CUdeviceptr d_bp1 = alloc_cap(b_p1.data(), n_bonds * sizeof(int), cap_ss * sizeof(int));
    CUdeviceptr d_bp2 = alloc_cap(b_p2.data(), n_bonds * sizeof(int), cap_ss * sizeof(int));
    CUdeviceptr d_bkn = alloc_cap(b_kn.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bks = alloc_cap(b_ks.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bkb = alloc_cap(b_kb.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bcn = alloc_cap(b_cn.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bFnMax = alloc_cap(b_FnMax.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bFsMax = alloc_cap(b_FsMax.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bbeamLenSq = alloc_cap(b_beamLenSq.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_binitD = alloc_cap(b_initD.data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bstate = alloc_cap(b_state.data(), n_bonds * sizeof(unsigned char), cap_ss * sizeof(unsigned char));
    CUdeviceptr d_bsfx = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bsfy = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bsfz = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bmbx = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bmby = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bmbz = alloc_cap(nullptr, n_bonds * sizeof(double), cap_ss * sizeof(double));
    std::vector<double> pn0[3];
    for (int c = 0; c < 3; ++c) pn0[c].resize(n_bonds);
    for (int k = 0; k < n_bonds; ++k) { pn0[0][k] = b_prevNormal[k].x; pn0[1][k] = b_prevNormal[k].y; pn0[2][k] = b_prevNormal[k].z; }
    CUdeviceptr d_bpnx = alloc_cap(pn0[0].data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bpny = alloc_cap(pn0[1].data(), n_bonds * sizeof(double), cap_ss * sizeof(double));
    CUdeviceptr d_bpnz = alloc_cap(pn0[2].data(), n_bonds * sizeof(double), cap_ss * sizeof(double));

    CUdeviceptr d_fv0 = alloc_and_copy(f_v0.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fv1 = alloc_and_copy(f_v1.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fv2 = alloc_and_copy(f_v2.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fpos = alloc_and_copy(f_pos.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fvel = alloc_and_copy(nullptr, n_facets * sizeof(Vec3));
    CUdeviceptr d_fforce = alloc_and_copy(nullptr, n_facets * sizeof(Vec3));
    CUdeviceptr d_fring = alloc_and_copy(f_ring.data(), n_facets * sizeof(int));
    CUdeviceptr d_fnrad = alloc_and_copy(f_nrad.data(), n_facets * sizeof(Vec3));

    CUdeviceptr d_fsf = alloc_cap(fs_f.data(), n_cand_fs * sizeof(int), cap_fs * sizeof(int));
    CUdeviceptr d_fss = alloc_cap(fs_s.data(), n_cand_fs * sizeof(int), cap_fs * sizeof(int));
    CUdeviceptr d_fskn = alloc_cap(fs_kn.data(), n_cand_fs * sizeof(double), cap_fs * sizeof(double));
    CUdeviceptr d_fsks = alloc_cap(fs_ks.data(), n_cand_fs * sizeof(double), cap_fs * sizeof(double));
    CUdeviceptr d_fscn = alloc_cap(fs_cn.data(), n_cand_fs * sizeof(double), cap_fs * sizeof(double));
    CUdeviceptr d_fsactive = alloc_cap(fs_active.data(), n_cand_fs * sizeof(int), cap_fs * sizeof(int));
    CUdeviceptr d_fsinitD = alloc_cap(fs_initD.data(), n_cand_fs * sizeof(double), cap_fs * sizeof(double));
    CUdeviceptr d_fsshear = alloc_cap(fs_zero.data(), n_cand_fs * sizeof(Vec3), cap_fs * sizeof(Vec3));
    CUdeviceptr d_fsprevN = alloc_cap(fs_prevNormal.data(), n_cand_fs * sizeof(Vec3), cap_fs * sizeof(Vec3));

    CUdeviceptr d_nb_tens = alloc_and_copy(nullptr, sizeof(int));
    CUdeviceptr d_nb_shear = alloc_and_copy(nullptr, sizeof(int));
    CUdeviceptr d_frad_ring = alloc_and_copy(nullptr, n_h * sizeof(double));
    CUdeviceptr d_rcum_ring = alloc_and_copy(nullptr, n_h * sizeof(double));
    // [PAR-3] bondCreation(): est_kn_ring = max(1, 2·young·spec_R·(spec_L/n_h)/r_max)
    double r_max_sph = 0; for (double r : radii) r_max_sph = std::max(r_max_sph, r);
    std::vector<double> init_stiff(n_h, std::max(1.0, 2.0 * young * spec_R * (spec_L / n_h) / r_max_sph));
    CUdeviceptr d_ring_stiff = alloc_and_copy(init_stiff.data(), n_h * sizeof(double));
    CUdeviceptr d_metrics = alloc_and_copy(nullptr, 8 * sizeof(double));

    // =====================================================================================
    // [CONTACTOS] Detección dinámica de contactos (lista de Verlet reconstruida en el host)
    //   * Esfera-esfera: pares no enlazados con d < ri + rj + skin (los enlaces viven aparte).
    //   * Esfera-faceta: pares con distancia centro-triángulo < r + skin.
    //   * Se reconstruye cuando el desplazamiento acumulado puede haber consumido el skin:
    //       2·max|Δx_esfera| > skin  o  max|Δx_esfera| + max|Δx_faceta| > skin  (con margen).
    //   * La historia (initD, estado, corte, momento) se transfiere por clave de par.
    //   * Verificación: en cada reconstrucción se cuentan los contactos con solapamiento que la
    //     lista anterior NO contenía (contactos perdidos). Debe ser 0.
    // =====================================================================================
    int n_ss = n_bonds;           // número total de pares que procesa kernel_sphere_sphere
    // [DET] suma de fuerzas determinista (por defecto). DEM_ATOMICS=1 -> suma con atomicAdd (no reproducible)
    int det = cfg.i("atomics") == 1 ? 0 : 1;
    CUdeviceptr d_pfout = 0, d_fsout = 0, d_ss_start = 0, d_ss_idx = 0, d_fsp_start = 0, d_fsp_idx = 0, d_ff_start = 0, d_ff_idx = 0;
    size_t pf_cap = 0, fsout_cap = 0, ssidx_cap = 0, fspidx_cap = 0, ffidx_cap = 0;
    const int nA = n_bonds;       // enlaces (estáticos)
    double r_max = 0.0;
    for (double r : radii) r_max = std::max(r_max, r);
    // skin de Verlet = 0.3·r_min (CONTACT_SKIN_FACTOR permite cambiar el 0.3, p. ej. para pruebas de estrés)
    const double skin = cfg.d("skin_factor") * min_r;
    std::vector<int> h_bp1 = b_p1, h_bp2 = b_p2;
    std::vector<int> h_fsf = fs_f, h_fss = fs_s;
    auto pkey = [](int a, int b) -> uint64_t { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    std::unordered_set<uint64_t> bonded_keys;
    for (auto& pr : bonded_set) bonded_keys.insert(pkey(pr.first, pr.second));
    CUdeviceptr d_pos_ref = alloc_and_copy(nullptr, n_particles * sizeof(Vec3));
    CUdeviceptr d_fv0_ref = alloc_and_copy(nullptr, n_facets * sizeof(Vec3));
    CUdeviceptr d_maxdisp = alloc_and_copy(nullptr, 2 * sizeof(unsigned long long));
    // CONTACT_VERIFY=N: cada N pasos se cuentan, sin modificar nada, los contactos con solapamiento ausentes de la lista actual
    int verify_every = cfg.i("contact_verify");
    long long n_verifications = 0;
    long long n_rebuilds = 0, missed_ss = 0, missed_fs = 0, new_ss_contacts = 0;
    std::unordered_set<uint64_t> ever_ss;   // pares no enlazados que alguna vez estuvieron en la lista

    auto grow = [&](std::vector<std::pair<CUdeviceptr*, size_t>> arrs, size_t keep, size_t newcap) {
        for (auto& a : arrs) {
            CUdeviceptr nd;
            CHECK_CU(cuMemAlloc(&nd, newcap * a.second));
            CHECK_CU(cuMemsetD8(nd, 0, newcap * a.second));
            if (keep) CHECK_CU(cuMemcpyDtoD(nd, *a.first, keep * a.second));
            CHECK_CU(cuMemFree(*a.first));
            *a.first = nd;
        }
    };

    auto build_csr = [&]() {
        if (!det) return;
        auto realloc_if = [&](CUdeviceptr& p, size_t& cap, size_t need) {
            if (need <= cap && p) return;
            if (p) CHECK_CU(cuMemFree(p));
            cap = std::max<size_t>(need, 1);
            CHECK_CU(cuMemAlloc(&p, cap));
        };
        realloc_if(d_pfout, pf_cap, cap_ss * 9 * sizeof(double));
        realloc_if(d_fsout, fsout_cap, cap_fs * 6 * sizeof(double));
        // esfera-esfera
        std::vector<int> st(n_particles + 1, 0), idx(2 * (size_t)n_ss);
        for (int k = 0; k < n_ss; ++k) { st[h_bp1[k] + 1]++; st[h_bp2[k] + 1]++; }
        for (int i = 0; i < n_particles; ++i) st[i + 1] += st[i];
        { std::vector<int> f(st.begin(), st.end() - 1);
          for (int k = 0; k < n_ss; ++k) { idx[f[h_bp1[k]]++] = 2 * k; idx[f[h_bp2[k]]++] = 2 * k + 1; } }
        // esfera-faceta (por partícula y por faceta)
        std::vector<int> pst(n_particles + 1, 0), pidx(n_cand_fs), fst(n_facets + 1, 0), fidx(n_cand_fs);
        for (int k = 0; k < n_cand_fs; ++k) { pst[h_fss[k] + 1]++; fst[h_fsf[k] + 1]++; }
        for (int i = 0; i < n_particles; ++i) pst[i + 1] += pst[i];
        for (int f = 0; f < n_facets; ++f) fst[f + 1] += fst[f];
        { std::vector<int> a(pst.begin(), pst.end() - 1), b(fst.begin(), fst.end() - 1);
          for (int k = 0; k < n_cand_fs; ++k) { pidx[a[h_fss[k]]++] = k; fidx[b[h_fsf[k]]++] = k; } }
        if (!d_ss_start) { CHECK_CU(cuMemAlloc(&d_ss_start, (n_particles + 1) * 4)); CHECK_CU(cuMemAlloc(&d_fsp_start, (n_particles + 1) * 4));
                           CHECK_CU(cuMemAlloc(&d_ff_start, (n_facets + 1) * 4)); }
        realloc_if(d_ss_idx, ssidx_cap, std::max<size_t>(idx.size(), 1) * 4);
        realloc_if(d_fsp_idx, fspidx_cap, std::max<size_t>(pidx.size(), 1) * 4);
        realloc_if(d_ff_idx, ffidx_cap, std::max<size_t>(fidx.size(), 1) * 4);
        CHECK_CU(cuMemcpyHtoD(d_ss_start, st.data(), st.size() * 4));
        if (!idx.empty()) CHECK_CU(cuMemcpyHtoD(d_ss_idx, idx.data(), idx.size() * 4));
        CHECK_CU(cuMemcpyHtoD(d_fsp_start, pst.data(), pst.size() * 4));
        if (!pidx.empty()) CHECK_CU(cuMemcpyHtoD(d_fsp_idx, pidx.data(), pidx.size() * 4));
        CHECK_CU(cuMemcpyHtoD(d_ff_start, fst.data(), fst.size() * 4));
        if (!fidx.empty()) CHECK_CU(cuMemcpyHtoD(d_ff_idx, fidx.data(), fidx.size() * 4));
    };
    auto rebuild_contacts = [&](bool verify, bool apply) {
        CHECK_CU(cuCtxSynchronize());
        std::vector<Vec3> P(n_particles), V0(n_facets), V1(n_facets), V2(n_facets);
        CHECK_CU(cuMemcpyDtoH(P.data(), d_pos, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V0.data(), d_fv0, n_facets * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V1.data(), d_fv1, n_facets * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V2.data(), d_fv2, n_facets * sizeof(Vec3)));

        // ---- malla uniforme ----
        double h = 2.0 * r_max + skin;
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (auto& p : P) {
            lo[0] = std::min(lo[0], p.x); lo[1] = std::min(lo[1], p.y); lo[2] = std::min(lo[2], p.z);
            hi[0] = std::max(hi[0], p.x); hi[1] = std::max(hi[1], p.y); hi[2] = std::max(hi[2], p.z);
        }
        int nc[3];
        for (int a = 0; a < 3; ++a) nc[a] = std::max(1, (int)std::floor((hi[a] - lo[a]) / h) + 1);
        auto cidx = [&](double x, int a) { int c = (int)std::floor((x - lo[a]) / h); return std::min(std::max(c, 0), nc[a] - 1); };
        size_t ncell = (size_t)nc[0] * nc[1] * nc[2];
        std::vector<int> cstart(ncell + 1, 0), cell_of(n_particles), sorted(n_particles);
        for (int i = 0; i < n_particles; ++i) {
            cell_of[i] = (cidx(P[i].z, 2) * nc[1] + cidx(P[i].y, 1)) * nc[0] + cidx(P[i].x, 0);
            cstart[cell_of[i] + 1]++;
        }
        for (size_t c = 0; c < ncell; ++c) cstart[c + 1] += cstart[c];
        { std::vector<int> fill(cstart.begin(), cstart.end() - 1); for (int i = 0; i < n_particles; ++i) sorted[fill[cell_of[i]]++] = i; }

        // ---- esfera-esfera ----
        int nB_old = n_ss - nA;
        std::vector<double> o_initD(nB_old), o_sf[3], o_mb[3], o_pn[3];
        CUdeviceptr* pnp[3] = {&d_bpnx, &d_bpny, &d_bpnz};
        std::vector<unsigned char> o_st(nB_old);
        CUdeviceptr* sfp[3] = {&d_bsfx, &d_bsfy, &d_bsfz};
        CUdeviceptr* mbp[3] = {&d_bmbx, &d_bmby, &d_bmbz};
        if (nB_old > 0) {
            CHECK_CU(cuMemcpyDtoH(o_initD.data(), d_binitD + nA * sizeof(double), nB_old * sizeof(double)));
            CHECK_CU(cuMemcpyDtoH(o_st.data(), d_bstate + nA, nB_old));
            for (int c = 0; c < 3; ++c) {
                o_sf[c].resize(nB_old); o_mb[c].resize(nB_old);
                CHECK_CU(cuMemcpyDtoH(o_sf[c].data(), *sfp[c] + nA * sizeof(double), nB_old * sizeof(double)));
                CHECK_CU(cuMemcpyDtoH(o_mb[c].data(), *mbp[c] + nA * sizeof(double), nB_old * sizeof(double)));
                o_pn[c].resize(nB_old);
                CHECK_CU(cuMemcpyDtoH(o_pn[c].data(), *pnp[c] + nA * sizeof(double), nB_old * sizeof(double)));
            }
        }
        std::unordered_map<uint64_t, int> old_ss;
        old_ss.reserve(nB_old * 2 + 1);
        for (int b = 0; b < nB_old; ++b) old_ss[pkey(h_bp1[nA + b], h_bp2[nA + b])] = b;

        std::vector<int> np1, np2;
        for (int i = 0; i < n_particles; ++i) {
            int cx = cidx(P[i].x, 0), cy = cidx(P[i].y, 1), cz = cidx(P[i].z, 2);
            for (int zz = std::max(cz - 1, 0); zz <= std::min(cz + 1, nc[2] - 1); ++zz)
            for (int yy = std::max(cy - 1, 0); yy <= std::min(cy + 1, nc[1] - 1); ++yy)
            for (int xx = std::max(cx - 1, 0); xx <= std::min(cx + 1, nc[0] - 1); ++xx) {
                int c = (zz * nc[1] + yy) * nc[0] + xx;
                for (int q = cstart[c]; q < cstart[c + 1]; ++q) {
                    int j = sorted[q];
                    if (j <= i) continue;
                    double dx = P[j].x - P[i].x, dy = P[j].y - P[i].y, dz = P[j].z - P[i].z;
                    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    double rs = radii[i] + radii[j];
                    if (d >= rs + skin) continue;
                    uint64_t k = pkey(i, j);
                    if (bonded_keys.count(k)) continue;
                    if (verify && d < rs && !old_ss.count(k)) missed_ss++;
                    if (apply && ever_ss.insert(k).second && n_rebuilds > 0) new_ss_contacts++;
                    np1.push_back(i); np2.push_back(j);
                }
            }
        }
        int nB = np1.size();
        if (apply) {
        if ((size_t)(nA + nB) > cap_ss) {
            size_t nc_ = (size_t)(nA + nB) * 3 / 2;
            grow({{&d_bp1, 4}, {&d_bp2, 4}, {&d_bkn, 8}, {&d_bks, 8}, {&d_bkb, 8}, {&d_bcn, 8}, {&d_bFnMax, 8}, {&d_bFsMax, 8},
                  {&d_bpnx, 8}, {&d_bpny, 8}, {&d_bpnz, 8}, {&d_bbeamLenSq, 8}, {&d_binitD, 8}, {&d_bstate, 1}, {&d_bsfx, 8}, {&d_bsfy, 8}, {&d_bsfz, 8},
                  {&d_bmbx, 8}, {&d_bmby, 8}, {&d_bmbz, 8}}, nA, nc_);
            cap_ss = nc_;
        }
        std::vector<double> nkn(nB), nks(nB), nkb(nB), ncn(nB), nFn(nB), nFs(nB), nbl(nB), npn[3];
        for (int c = 0; c < 3; ++c) npn[c].assign(nB, 0.0);
        std::vector<double> ninitD(nB, 0.0), nsf[3], nmb[3];
        std::vector<unsigned char> nst(nB, 0);
        for (int c = 0; c < 3; ++c) { nsf[c].assign(nB, 0.0); nmb[c].assign(nB, 0.0); }
        for (int b = 0; b < nB; ++b) {
            int i = np1[b], j = np2[b];
            double r1 = radii[i], r2 = radii[j], minR = std::min(r1, r2), area = PI * minR * minR;
            double kn1 = 2.0 * young * r1, kn2 = 2.0 * young * r2, kn_val = kn1 * kn2 / (kn1 + kn2);
            double ks1 = kn1 / kn_ks_ratio, ks2 = kn2 / kn_ks_ratio, ks_val = ks1 * ks2 / (ks1 + ks2);
            double kr1 = 0.5 * young * r1 * r1 * r1, kr2 = 0.5 * young * r2 * r2 * r2, kb_val = kr1 * kr2 / (kr1 + kr2) * bendingIn;
            double meff = mass[i] * mass[j] / (mass[i] + mass[j]);
            nkn[b] = kn_val; nks[b] = ks_val; nkb[b] = kb_val; ncn[b] = viscAlpha * 2.0 * std::sqrt(kn_val * meff);
            nFn[b] = tensile * area; nFs[b] = cohesion * area;
            nbl[b] = ks_val + 12.0 * kb_val / ((r1 + r2) * (r1 + r2));   // solo se usa con enlace cohesivo (nunca aquí)
            auto it_o = old_ss.find(pkey(i, j));
            if (it_o != old_ss.end()) {
                int o = it_o->second;
                ninitD[b] = o_initD[o]; nst[b] = o_st[o];
                for (int c = 0; c < 3; ++c) { nsf[c][b] = o_sf[c][o]; nmb[c][b] = o_mb[c][o]; npn[c][b] = o_pn[c][o]; }
            }
        }
        if (nB > 0) {
            CHECK_CU(cuMemcpyHtoD(d_bp1 + nA * 4, np1.data(), nB * 4));
            CHECK_CU(cuMemcpyHtoD(d_bp2 + nA * 4, np2.data(), nB * 4));
            CHECK_CU(cuMemcpyHtoD(d_bkn + nA * 8, nkn.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bks + nA * 8, nks.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bkb + nA * 8, nkb.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bcn + nA * 8, ncn.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bFnMax + nA * 8, nFn.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bFsMax + nA * 8, nFs.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bbeamLenSq + nA * 8, nbl.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_binitD + nA * 8, ninitD.data(), nB * 8));
            CHECK_CU(cuMemcpyHtoD(d_bstate + nA, nst.data(), nB));
            for (int c = 0; c < 3; ++c) {
                CHECK_CU(cuMemcpyHtoD(*sfp[c] + nA * 8, nsf[c].data(), nB * 8));
                CHECK_CU(cuMemcpyHtoD(*mbp[c] + nA * 8, nmb[c].data(), nB * 8));
                CHECK_CU(cuMemcpyHtoD(*pnp[c] + nA * 8, npn[c].data(), nB * 8));
            }
        }
        n_ss = nA + nB;
        h_bp1.resize(nA); h_bp2.resize(nA);
        h_bp1.insert(h_bp1.end(), np1.begin(), np1.end());
        h_bp2.insert(h_bp2.end(), np2.begin(), np2.end());
        }

        // ---- esfera-faceta ----
        int nF_old = n_cand_fs;
        std::vector<int> o_act(nF_old);
        std::vector<double> o_fid(nF_old);
        std::vector<Vec3> o_fsh(nF_old), o_fpn(nF_old);
        if (nF_old > 0) {
            CHECK_CU(cuMemcpyDtoH(o_act.data(), d_fsactive, nF_old * sizeof(int)));
            CHECK_CU(cuMemcpyDtoH(o_fid.data(), d_fsinitD, nF_old * sizeof(double)));
            CHECK_CU(cuMemcpyDtoH(o_fsh.data(), d_fsshear, nF_old * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(o_fpn.data(), d_fsprevN, nF_old * sizeof(Vec3)));
        }
        std::unordered_map<uint64_t, int> old_fs;
        old_fs.reserve(nF_old * 2 + 1);
        for (int k = 0; k < nF_old; ++k) old_fs[pkey(h_fsf[k], h_fss[k])] = k;
        std::vector<int> nff, nfs;
        for (int fi = 0; fi < n_facets; ++fi) {
            double ext = r_max + skin;
            double mn[3] = {std::min({V0[fi].x, V1[fi].x, V2[fi].x}) - ext, std::min({V0[fi].y, V1[fi].y, V2[fi].y}) - ext, std::min({V0[fi].z, V1[fi].z, V2[fi].z}) - ext};
            double mx[3] = {std::max({V0[fi].x, V1[fi].x, V2[fi].x}) + ext, std::max({V0[fi].y, V1[fi].y, V2[fi].y}) + ext, std::max({V0[fi].z, V1[fi].z, V2[fi].z}) + ext};
            bool outside = false;
            for (int a = 0; a < 3; ++a) if (mx[a] < lo[a] - h || mn[a] > hi[a] + h) outside = true;
            if (outside) continue;
            for (int zz = cidx(mn[2], 2); zz <= cidx(mx[2], 2); ++zz)
            for (int yy = cidx(mn[1], 1); yy <= cidx(mx[1], 1); ++yy)
            for (int xx = cidx(mn[0], 0); xx <= cidx(mx[0], 0); ++xx) {
                int c = (zz * nc[1] + yy) * nc[0] + xx;
                for (int q = cstart[c]; q < cstart[c + 1]; ++q) {
                    int si = sorted[q];
                    Vec3 cp = host_closest_pt_tri(P[si], V0[fi], V1[fi], V2[fi]);
                    double dx = P[si].x - cp.x, dy = P[si].y - cp.y, dz = P[si].z - cp.z;
                    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d >= radii[si] + skin) continue;
                    if (verify && d < radii[si] && !old_fs.count(pkey(fi, si))) missed_fs++;
                    nff.push_back(fi); nfs.push_back(si);
                }
            }
        }
        int nF = nff.size();
        if (!apply) return;
        if ((size_t)nF > cap_fs) {
            size_t nc_ = (size_t)nF * 3 / 2;
            grow({{&d_fsf, 4}, {&d_fss, 4}, {&d_fskn, 8}, {&d_fsks, 8}, {&d_fscn, 8}, {&d_fsactive, 4}, {&d_fsinitD, 8},
                  {&d_fsshear, sizeof(Vec3)}, {&d_fsprevN, sizeof(Vec3)}}, 0, nc_);
            cap_fs = nc_;
        }
        std::vector<double> fkn(nF), fks(nF), fcn(nF), fid(nF, 0.0);
        std::vector<int> fact(nF, 0);
        std::vector<Vec3> fsh(nF, {0.0, 0.0, 0.0}), fpn(nF, {0.0, 0.0, 1.0});
        for (int k = 0; k < nF; ++k) {
            int si = nfs[k];
            double r2 = radii[si], r1 = 2.0 * r2;
            double kn1 = 2.0 * young_wall * r1, kn2 = 2.0 * young * r2, kn_val = kn1 * kn2 / (kn1 + kn2);
            double ks1 = kn1 / kn_ks_wall_ratio, ks2 = kn2 / kn_ks_ratio;
            fkn[k] = kn_val; fks[k] = ks1 * ks2 / (ks1 + ks2); fcn[k] = viscAlpha * 2.0 * std::sqrt(kn_val * mass[si]);
            auto it_o = old_fs.find(pkey(nff[k], si));
            if (it_o != old_fs.end()) { int o = it_o->second; fact[k] = o_act[o]; fid[k] = o_fid[o]; fsh[k] = o_fsh[o]; fpn[k] = o_fpn[o]; }
        }
        if (nF > 0) {
            CHECK_CU(cuMemcpyHtoD(d_fsf, nff.data(), nF * 4));
            CHECK_CU(cuMemcpyHtoD(d_fss, nfs.data(), nF * 4));
            CHECK_CU(cuMemcpyHtoD(d_fskn, fkn.data(), nF * 8));
            CHECK_CU(cuMemcpyHtoD(d_fsks, fks.data(), nF * 8));
            CHECK_CU(cuMemcpyHtoD(d_fscn, fcn.data(), nF * 8));
            CHECK_CU(cuMemcpyHtoD(d_fsactive, fact.data(), nF * 4));
            CHECK_CU(cuMemcpyHtoD(d_fsinitD, fid.data(), nF * 8));
            CHECK_CU(cuMemcpyHtoD(d_fsshear, fsh.data(), nF * sizeof(Vec3)));
            CHECK_CU(cuMemcpyHtoD(d_fsprevN, fpn.data(), nF * sizeof(Vec3)));
        }
        n_cand_fs = nF;
        h_fsf = nff; h_fss = nfs;

        // referencias para el criterio de reconstrucción
        CHECK_CU(cuMemcpyDtoD(d_pos_ref, d_pos, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoD(d_fv0_ref, d_fv0, n_facets * sizeof(Vec3)));
        n_rebuilds++;
        build_csr();
    };
    double last_disp = 0.0;
    auto check_rebuild = [&](int it) {
        CHECK_CU(cuMemsetD8(d_maxdisp, 0, 2 * sizeof(unsigned long long)));
        void* a[] = {&n_particles, &n_facets, &d_pos, &d_pos_ref, &d_fv0, &d_fv0_ref, &d_maxdisp};
        CHECK_CU(cuLaunchKernel(fn_maxdisp, (std::max(n_particles, n_facets) + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        double md[2];
        CHECK_CU(cuMemcpyDtoH(md, d_maxdisp, sizeof(md)));
        double need = std::max(2.0 * md[0], md[0] + md[1]);
        double growth = std::max(0.0, need - last_disp);   // crecimiento en el último intervalo de comprobación
        last_disp = need;
        if (need + 2.0 * growth > skin) { rebuild_contacts(true, true); last_disp = 0.0; }
    };
    if (dynamic_contacts) {
        rebuild_contacts(false, true);
        std::cout << "[Contactos] detección dinámica: skin=" << skin * 1e3 << " mm, pares esfera-esfera no enlazados="
                  << (n_ss - nA) << ", pares esfera-faceta=" << n_cand_fs << "\n";
    } else {
        std::cout << "[Contactos] lista estática de t=0 (modo static)\n";
        build_csr();
    }
    std::cout << (det ? "[Fuerzas] suma determinista (orden fijo, FP64)\n" : "[Fuerzas] suma con atomicAdd (DEM_ATOMICS=1, no reproducible)\n");

    // Paso 0 (O.step() con dt=0 para registrar los 28 contactos iniciales Facet-Sphere con su initD)
    double dt_zero = 0.0;
    void* args_fs0[] = {&n_cand_fs, &dt_zero, (void*)&mu, &d_pos, &d_vel, &d_angVel, &d_radius,
                        &d_fv0, &d_fv1, &d_fv2, &d_fvel, &d_fsf, &d_fss, &d_fskn, &d_fsks, &d_fscn,
                        &d_fsactive, &d_fsinitD, &d_fsshear, &d_fsprevN, &d_fforce, &d_force, &d_torque, &d_fsout, &det};
    CHECK_CU(cuLaunchKernel(fn_fs, (n_cand_fs + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_fs0, nullptr));
    CHECK_CU(cuMemsetD8(d_force, 0, n_particles * 12));
    CHECK_CU(cuMemsetD8(d_torque, 0, n_particles * 12));
    CHECK_CU(cuCtxSynchronize());

    // Bucle principal de simulación en GPU
    double z_vel = cfg.d("z_vel");
    double r_vel_max = cfg.d("r_vel_max_factor") * z_vel;
    bool reached_threshold = false;
    int hold_counter = 0;
    bool preload_dead = false;
    bool load_dead = true;
    // [PAR-3] test-4-triaxial-opt.py: PyRunner(iterPeriod=50, 'servo_radial()') desde el inicio.
    // La corrida de referencia resultados/yade_cpu (ago-2026) se hizo con una versión anterior del script con el servo
    // cada 10 iteraciones durante la precarga (load() lo pasa a 50): TRIAX_SERVO_PRELOAD=10 la reproduce.
    int servo_period = cfg.i("servo_preload");
    const int servo_period_load = cfg.i("servo_period");
    const int preload_hold = cfg.i("preload_hold");

    auto launch_plate_vel = [&](double vzt, double vzb) {
        void* args[] = {&vzt, &vzb, &d_fvel};
        CHECK_CU(cuLaunchKernel(fn_plate_vel, (2 * n_plate + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    };

    auto launch_measure = [&]() {
        void* args[] = {&d_fforce, &d_fpos, &d_fring, &d_fnrad, &h_p_0, &r_avg_0, &d_frad_ring, &d_rcum_ring, &d_metrics};
        CHECK_CU(cuLaunchKernel(fn_measure, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    };

    int init_pass = 1;
    void* args_reset[] = {&n_particles, &n_facets, &d_force, &d_torque, &d_fforce};
    void* args_ss[] = {&n_ss, &dt, (void*)&tgPhi, (void*)&mu, (void*)&rollFric,
                       &d_pos, &d_vel, &d_angVel, &d_radius, &d_bp1, &d_bp2, &d_bkn, &d_bks, &d_bkb, &d_bcn,
                       &d_bFnMax, &d_bFsMax, &d_bbeamLenSq, &d_binitD, &d_bstate,
                       &d_bsfx, &d_bsfy, &d_bsfz, &d_bmbx, &d_bmby, &d_bmbz, &d_bpnx, &d_bpny, &d_bpnz, &d_force, &d_torque, &d_nb_tens, &d_nb_shear, &d_pfout, &det, &init_pass};
    // [FIX-6] iteración 0 (isFresh): initD y normal inicial de los enlaces calculados en la GPU, no en el host
    CHECK_CU(cuLaunchKernel(fn_ss, (n_ss + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_ss, nullptr));
    CHECK_CU(cuCtxSynchronize());
    init_pass = 0;
    void* args_fs[] = {&n_cand_fs, &dt, (void*)&mu, &d_pos, &d_vel, &d_angVel, &d_radius,
                       &d_fv0, &d_fv1, &d_fv2, &d_fvel, &d_fsf, &d_fss, &d_fskn, &d_fsks, &d_fscn,
                       &d_fsactive, &d_fsinitD, &d_fsshear, &d_fsprevN, &d_fforce, &d_force, &d_torque, &d_fsout, &det};
    void* args_newton[] = {&n_particles, &n_facets, &dt, &d_mass, &d_inertia, &d_force, &d_torque,
                           &d_pos, &d_vel, &d_angVel, &d_fvel, &d_fpos, &d_fv0, &d_fv1, &d_fv2};
    void* args_stiff[] = {&n_cand_fs, &d_fsactive, &d_fsf, &d_fskn, &d_fring, &d_ring_stiff};
    void* args_servo[] = {(void*)&confining, &dt_pwave, &r_vel_max, &d_frad_ring, &d_rcum_ring,
                          &d_ring_stiff, &d_metrics, &d_fring, &d_fnrad, &d_fvel};

    { std::string acc; std::istringstream ps(out_dir); std::string part;
      while (std::getline(ps, part, '/')) { acc += part; if (!acc.empty()) mkdir(acc.c_str(), 0777); acc += "/"; } }
    cfg.write(out_dir + "/caso_usado.cfg");
    std::ofstream f_facet(out_dir + "/facet_force_exact.txt");
    f_facet << "# time iter wall_dspl stress_axial stress_radial q p eps1_facet eps3_facet evol_geom evol_calc eps1_particles eps3_particles nu_particles fz un_force n_tens n_shear n_bonds_active n_bonds_broken\n";
    std::ofstream f_cracks(out_dir + "/cracks_summary_exact.txt");
    f_cracks << "# time iter nb_tens nb_shear\n";


    // =====================================================================================
    // [VTK] Exportación equivalente a exp_vtk() del script YADE (VTKExporter)
    // =====================================================================================
    const int vtk_every = cfg.i("vtk_every");
    const bool vtk_ascii = cfg.i("vtk_ascii") == 1;
    const bool vtk_intrs = cfg.i("vtk_intrs") != 0;
    int vtk_index = 0;
    CUdeviceptr d_ori = 0;
    if (vtk_every > 0) {
        std::vector<double> q0(4 * (size_t)n_particles, 0.0);
        for (int i = 0; i < n_particles; ++i) q0[4 * i] = 1.0;
        d_ori = alloc_and_copy(q0.data(), q0.size() * sizeof(double));
    }
    double vtk_seconds = 0.0;
    std::vector<Vec3> f_v0_init = f_v0, f_v1_init = f_v1, f_v2_init = f_v2;
    auto export_vtk = [&]() {
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CU(cuCtxSynchronize());
        std::string dir = out_dir + "/vtk";
        mkdir(dir.c_str(), 0777);
        char num[16]; snprintf(num, sizeof(num), "%08d", vtk_index++);
        // ---- escritura (ASCII o binario big-endian float32) ----
        struct W {
            FILE* f; bool ascii;
            void hdr(const char* t) { fprintf(f, "# vtk DataFile Version 3.0.\n%s\n%s\nDATASET POLYDATA\n", t, ascii ? "ASCII" : "BINARY"); }
            void flt(double v) {
                if (ascii) { fprintf(f, "%.10g\n", v); return; }
                float x = (float)v; uint32_t u; memcpy(&u, &x, 4); u = __builtin_bswap32(u); fwrite(&u, 4, 1, f);
            }
            void vec(double a, double b, double c) {
                if (ascii) { fprintf(f, "%.10g %.10g %.10g\n", a, b, c); return; }
                flt(a); flt(b); flt(c);
            }
            void i32(int v) { if (ascii) { fprintf(f, "%d ", v); return; } uint32_t u = __builtin_bswap32((uint32_t)v); fwrite(&u, 4, 1, f); }
            void nl() { if (ascii) fprintf(f, "\n"); }
            void end() { if (!ascii) fprintf(f, "\n"); }
            const char* tp() { return ascii ? "double" : "float"; }
        };
        std::vector<Vec3> P(n_particles), V(n_particles), Wv(n_particles);
        CHECK_CU(cuMemcpyDtoH(P.data(), d_pos, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V.data(), d_vel, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(Wv.data(), d_angVel, n_particles * sizeof(Vec3)));
        {   // ---- esferas ----
            W w{fopen((dir + "/data-spheres-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
            w.hdr("comment");
            fprintf(w.f, "POINTS %d %s\n", n_particles, w.tp());
            for (auto& p : P) w.vec(p.x, p.y, p.z);
            w.end();
            fprintf(w.f, "POINT_DATA %d\nSCALARS radius %s 1\nLOOKUP_TABLE default\n", n_particles, w.tp());
            for (double r : radii) w.flt(r);
            w.end();
            fprintf(w.f, "VECTORS pos %s\n", w.tp()); for (auto& p : P) w.vec(p.x, p.y, p.z); w.end();
            fprintf(w.f, "VECTORS vel %s\n", w.tp()); for (auto& v : V) w.vec(v.x, v.y, v.z); w.end();
            fprintf(w.f, "VECTORS disp %s\n", w.tp());
            for (int i = 0; i < n_particles; ++i) w.vec(P[i].x - coords0[i].x, P[i].y - coords0[i].y, P[i].z - coords0[i].z);
            w.end();
            fprintf(w.f, "VECTORS color %s\n", w.tp()); for (int i = 0; i < n_particles; ++i) w.vec(1, 0, 0); w.end();
            fprintf(w.f, "VECTORS angVel %s\n", w.tp()); for (auto& v : Wv) w.vec(v.x, v.y, v.z); w.end();
            fprintf(w.f, "SCALARS mask %s 1\nLOOKUP_TABLE default\n", w.tp()); for (int i = 0; i < n_particles; ++i) w.flt(1); w.end();
            std::vector<double> q(4 * (size_t)n_particles);
            CHECK_CU(cuMemcpyDtoH(q.data(), d_ori, q.size() * sizeof(double)));
            fprintf(w.f, "SCALARS rotAngle %s 1\nLOOKUP_TABLE default\n", w.tp());   // Eigen AngleAxis(ori).angle()
            for (int i = 0; i < n_particles; ++i) w.flt(2.0 * std::atan2(std::sqrt(q[4*i+1]*q[4*i+1] + q[4*i+2]*q[4*i+2] + q[4*i+3]*q[4*i+3]), std::fabs(q[4*i])));
            w.end();
            fclose(w.f);
        }
        std::vector<Vec3> FV0(n_facets), FV1(n_facets), FV2(n_facets), FF(n_facets);
        CHECK_CU(cuMemcpyDtoH(FV0.data(), d_fv0, n_facets * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(FV1.data(), d_fv1, n_facets * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(FV2.data(), d_fv2, n_facets * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(FF.data(), d_fforce, n_facets * sizeof(Vec3)));
        {   // ---- facetas ----
            W w{fopen((dir + "/data-facets-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
            w.hdr("comment");
            fprintf(w.f, "POINTS %d %s\n", 3 * n_facets, w.tp());
            for (int k = 0; k < n_facets; ++k) { w.vec(FV0[k].x, FV0[k].y, FV0[k].z); w.vec(FV1[k].x, FV1[k].y, FV1[k].z); w.vec(FV2[k].x, FV2[k].y, FV2[k].z); }
            w.end();
            fprintf(w.f, "POLYGONS %d %d\n", n_facets, 4 * n_facets);
            for (int k = 0; k < n_facets; ++k) { w.i32(3); w.i32(3 * k); w.i32(3 * k + 1); w.i32(3 * k + 2); w.nl(); }
            w.end();
            fprintf(w.f, "CELL_DATA %d\nVECTORS pos %s\n", n_facets, w.tp());
            std::vector<Vec3> fp(n_facets);
            for (int k = 0; k < n_facets; ++k) {  // el centro se desplaza igual que los vértices
                fp[k] = {f_pos[k].x + FV0[k].x - f_v0_init[k].x, f_pos[k].y + FV0[k].y - f_v0_init[k].y, f_pos[k].z + FV0[k].z - f_v0_init[k].z};
                w.vec(fp[k].x, fp[k].y, fp[k].z);
            }
            w.end();
            fprintf(w.f, "VECTORS displ %s\n", w.tp());
            for (int k = 0; k < n_facets; ++k) w.vec(FV0[k].x - f_v0_init[k].x, FV0[k].y - f_v0_init[k].y, FV0[k].z - f_v0_init[k].z);
            w.end();
            fprintf(w.f, "VECTORS forceVec %s\n", w.tp()); for (auto& v : FF) w.vec(v.x, v.y, v.z); w.end();
            fclose(w.f);
        }
        if (vtk_intrs) {   // ---- interacciones (esfera-esfera activas) ----
            std::vector<unsigned char> st(n_ss);
            std::vector<double> kn(n_ss), cn(n_ss), initD(n_ss), sx(n_ss), sy(n_ss), sz(n_ss), mx(n_ss), my(n_ss), mz(n_ss);
            CHECK_CU(cuMemcpyDtoH(st.data(), d_bstate, n_ss));
            CHECK_CU(cuMemcpyDtoH(kn.data(), d_bkn, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(cn.data(), d_bcn, n_ss * 8));
            CHECK_CU(cuMemcpyDtoH(initD.data(), d_binitD, n_ss * 8));
            CHECK_CU(cuMemcpyDtoH(sx.data(), d_bsfx, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(sy.data(), d_bsfy, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(sz.data(), d_bsfz, n_ss * 8));
            CHECK_CU(cuMemcpyDtoH(mx.data(), d_bmbx, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(my.data(), d_bmby, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(mz.data(), d_bmbz, n_ss * 8));
            std::vector<int> ids;
            for (int k = 0; k < n_ss; ++k) if (st[k] & 1) ids.push_back(k);
            int nssi = ids.size();
            std::vector<int> fact(n_cand_fs); std::vector<double> fkn(n_cand_fs), fcn(n_cand_fs), fid(n_cand_fs); std::vector<Vec3> fsh(n_cand_fs), FVel(n_facets);
            if (n_cand_fs > 0) {
                CHECK_CU(cuMemcpyDtoH(fact.data(), d_fsactive, n_cand_fs * 4)); CHECK_CU(cuMemcpyDtoH(fkn.data(), d_fskn, n_cand_fs * 8));
                CHECK_CU(cuMemcpyDtoH(fcn.data(), d_fscn, n_cand_fs * 8)); CHECK_CU(cuMemcpyDtoH(fid.data(), d_fsinitD, n_cand_fs * 8));
                CHECK_CU(cuMemcpyDtoH(fsh.data(), d_fsshear, n_cand_fs * sizeof(Vec3)));
            }
            CHECK_CU(cuMemcpyDtoH(FVel.data(), d_fvel, n_facets * sizeof(Vec3)));
            std::vector<int> fids;
            for (int k = 0; k < n_cand_fs; ++k) if (fact[k]) fids.push_back(k);
            int n = nssi + (int)fids.size();
            auto fcen = [&](int fi) -> Vec3 { return {f_pos[fi].x + FV0[fi].x - f_v0_init[fi].x, f_pos[fi].y + FV0[fi].y - f_v0_init[fi].y, f_pos[fi].z + FV0[fi].z - f_v0_init[fi].z}; };
            W w{fopen((dir + "/data-intrs-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
            w.hdr("comment");
            fprintf(w.f, "POINTS %d %s\n", 2 * n, w.tp());
            for (int k : ids) { const Vec3& a = P[h_bp1[k]]; const Vec3& b = P[h_bp2[k]]; w.vec(a.x, a.y, a.z); w.vec(b.x, b.y, b.z); }
            for (int k : fids) { Vec3 a = fcen(h_fsf[k]); const Vec3& b = P[h_fss[k]]; w.vec(a.x, a.y, a.z); w.vec(b.x, b.y, b.z); }
            w.end();
            fprintf(w.f, "LINES %d %d\n", n, 3 * n);
            for (int q = 0; q < n; ++q) { w.i32(2); w.i32(2 * q); w.i32(2 * q + 1); w.nl(); }
            w.end();
            std::vector<double> Fn(n), Ft(n), Mm(n), Fvm(n), ov(n), coh(n), brk(n); std::vector<Vec3> Fv(n);
            for (int q = nssi; q < n; ++q) {   // esfera-faceta: misma geometría que kernel_facet_sphere
                int k = fids[q - nssi], fi = h_fsf[k], si = h_fss[k];
                Vec3 cp = host_closest_pt_tri(P[si], FV0[fi], FV1[fi], FV2[fi]);
                double dx = P[si].x - cp.x, dy = P[si].y - cp.y, dz = P[si].z - cp.z, d = std::sqrt(dx * dx + dy * dy + dz * dz);
                double nx = dx / d, ny = dy / d, nz = dz / d, pen = radii[si] - d;
                double vn = nx * (V[si].x - FVel[fi].x) + ny * (V[si].y - FVel[fi].y) + nz * (V[si].z - FVel[fi].z);
                Fn[q] = -fkn[k] * (pen - fid[k]);
                Ft[q] = std::sqrt(fsh[k].x * fsh[k].x + fsh[k].y * fsh[k].y + fsh[k].z * fsh[k].z); Mm[q] = 0.0;
                Fv[q] = {-fcn[k] * vn * nx, -fcn[k] * vn * ny, -fcn[k] * vn * nz};
                Fvm[q] = std::fabs(fcn[k] * vn); ov[q] = pen; coh[q] = 0; brk[q] = 0;
            }
            for (int q = 0; q < nssi; ++q) {
                int k = ids[q], i = h_bp1[k], j = h_bp2[k];
                double dx = P[j].x - P[i].x, dy = P[j].y - P[i].y, dz = P[j].z - P[i].z, d = std::sqrt(dx * dx + dy * dy + dz * dz);
                double nx = dx / d, ny = dy / d, nz = dz / d, pen = radii[i] + radii[j] - d;
                double vn = nx * (V[j].x - V[i].x) + ny * (V[j].y - V[i].y) + nz * (V[j].z - V[i].z);
                Fn[q] = -kn[k] * (pen - initD[k]);                          // IBPhys::sigmaN (positivo = tracción)
                Ft[q] = std::sqrt(sx[k] * sx[k] + sy[k] * sy[k] + sz[k] * sz[k]);
                Mm[q] = std::sqrt(mx[k] * mx[k] + my[k] * my[k] + mz[k] * mz[k]);
                Fv[q] = {-cn[k] * vn * nx, -cn[k] * vn * ny, -cn[k] * vn * nz};
                Fvm[q] = std::fabs(cn[k] * vn); ov[q] = pen; coh[q] = (st[k] >> 1) & 1; brk[q] = (st[k] >> 2) & 1;
            }
            auto sc = [&](const char* nm, const std::vector<double>& v) { fprintf(w.f, "SCALARS %s %s 1\nLOOKUP_TABLE default\n", nm, w.tp()); for (double x : v) w.flt(x); w.end(); };
            fprintf(w.f, "CELL_DATA %d\n", n);
            sc("Fn", Fn); sc("Ft", Ft); sc("momentMag", Mm);
            fprintf(w.f, "VECTORS Fvisc %s\n", w.tp()); for (auto& v : Fv) w.vec(v.x, v.y, v.z); w.end();
            sc("FviscMag", Fvm); sc("Fadh", std::vector<double>(n, 0.0)); sc("overlap", ov); sc("cohesive", coh); sc("broken", brk);
            sc("isAdhesive", std::vector<double>(n, 0.0));
            fclose(w.f);
        }
        vtk_seconds += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    };

    if (vtk_every > 0) { export_vtk(); std::cout << "[VTK] exportación cada " << vtk_every << " iteraciones en " << out_dir << "/vtk ("
                                                  << (vtk_ascii ? "ASCII" : "binario float32") << (vtk_intrs ? "" : ", sin interacciones") << ")\n"; }
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "Ejecutando " << max_steps << " pasos en GPU CUDA nativo (RTX 4060)...\n";

    // fila de facet_force_exact.txt / cracks_summary_exact.txt (extract_Data() de YADE)
    auto write_row = [&](int it, double sim_t, const double* m, int nt, int ns) {
        double wall_dspl = h_p_0 - m[5];   // YADE: -2·displ_z(placa superior)
        double q_val = m[0] - m[1];
        double p_val = (m[0] + 2.0 * m[1]) / 3.0;
        double v0_cyl = 3.14159265358979323846 * r_avg_0 * r_avg_0 * h_p_0;
        double v_cyl = 3.14159265358979323846 * m[6] * m[6] * m[5];
        double evol_geom = -(v_cyl - v0_cyl) / v0_cyl;
        double evol_calc = m[2] + 2.0 * m[3];
        int n_brok = nt + ns;
        int n_act_b = nA - n_brok;
        f_facet << std::scientific << std::setprecision(10)
                << sim_t << " " << it << " " << wall_dspl << " "
                << m[0] << " " << m[1] << " " << q_val << " " << p_val << " "
                << m[2] << " " << m[3] << " " << evol_geom << " " << evol_calc << " "
                << 0.0 << " " << 0.0 << " " << 0.0 << " "
                << m[4] << " " << 0.0 << " "
                << nt << " " << ns << " " << n_act_b << " " << n_brok << "\n";
        f_cracks << std::scientific << std::setprecision(6)
                 << sim_t << " " << it << " " << nt << " " << ns << "\n";
    };
    // [CASO] criterios de parada opcionales (se evalúan en las salidas cada 1000 iteraciones; no alteran la física)
    const double stop_eps1 = cfg.d("parar_eps1"), stop_qdrop = cfg.d("parar_caida_q");
    double q_peak = -1e300, eps1_peak = 0.0, sax_peak = 0.0, srad_peak = 0.0; int it_peak = 0;
    double last_m[8] = {0}; int last_it = 0, last_nt = 0, last_ns = 0, it_conf = -1, it_dev = -1;
    std::string stop_reason = "pasos";
    // [PARADA] réplica de checkProgress() de test-4-triaxial-opt.py (PyRunner cada 100 iteraciones tras alcanzar el confinamiento):
    // fz = 0,5·(f1z − f2z) en los platos; fz_max se actualiza aquí y en extract_Data() (cada 1000 iteraciones, desde el inicio).
    // Fractura: fz/fz_max < 0,7 con fz_max > 2000 N -> t_fractura = O.time; parada cuando O.time >= 1,05·t_fractura.
    const bool stop_yade = cfg.i("parada_yade") == 1;
    const double stop_frac = cfg.d("parada_fraccion"), stop_extra = cfg.d("parada_extra"), stop_fzmin = cfg.d("parada_fz_min");
    double fz_max = 1.0, t_fracture = -1.0, fz_max_fracture = 0.0; int it_fracture = -1;
    // [PAR-4] YADE ejecuta dos pasos con dt=0 (iter 0 y 1); el paso dinámico 'step' es O.iter = step + 1.
    // Todas las condiciones periódicas usan 'it' = O.iter; la salida 'iter' y 'time' coinciden con facet_force.txt de YADE.
    for (int step = 1; step <= max_steps; ++step) {
        const int it = step + 1;
        bool need_meas = (!preload_dead && (it % 100 == 0)) || (stop_yade && reached_threshold && it % 100 == 0) || (it % servo_period == 0) || (it % 1000 == 0) || (step == max_steps) || (vtk_every > 0 && it % vtk_every == 0);
        if (need_meas) {
            CHECK_CU(cuLaunchKernel(fn_reset, (n_facets + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_reset, nullptr));
        }
        if (dynamic_contacts && it % 10 == 0) check_rebuild(it);
        if (verify_every > 0 && it % verify_every == 0) { rebuild_contacts(true, false); n_verifications++; }
        CHECK_CU(cuLaunchKernel(fn_ss, (n_ss + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_ss, nullptr));
        if (n_cand_fs > 0) CHECK_CU(cuLaunchKernel(fn_fs, (n_cand_fs + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_fs, nullptr));
        if (det) {
            void* ag[] = {&n_particles, &d_ss_start, &d_ss_idx, &d_pfout, &d_fsp_start, &d_fsp_idx, &d_fsout, &d_force, &d_torque};
            CHECK_CU(cuLaunchKernel(fn_gather_p, (n_particles + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, ag, nullptr));
            if (need_meas) {
                void* af[] = {&n_facets, &d_ff_start, &d_ff_idx, &d_fsout, &d_fforce};
                CHECK_CU(cuLaunchKernel(fn_gather_f, (n_facets + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, af, nullptr));
            }
        }
        // [PAR-6] el mismo kernel integra esferas (i < n_particles) y facetas (i < n_facets): la malla debe cubrir ambos
        CHECK_CU(cuLaunchKernel(fn_newton, (std::max(n_particles, n_facets) + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_newton, nullptr));
        if (vtk_every > 0) { void* ar[] = {&n_particles, &dt, &d_angVel, &d_ori}; CHECK_CU(cuLaunchKernel(fn_rotori, (n_particles + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, ar, nullptr)); }

        if (!preload_dead && (it % 100 == 0)) {
            launch_measure();
            double m[8];
            CHECK_CU(cuMemcpyDtoH(m, d_metrics, 8 * sizeof(double)));
            if (!reached_threshold) {
                if (m[0] >= confining && m[1] >= confining) {
                    reached_threshold = true;
                    hold_counter = 0;
                    launch_plate_vel(0.0, 0.0);
                    it_conf = it;
                    std::cout << "  [Confinement reached at iter " << it << ": s_ax=" << m[0]*1e-6 << " MPa, s_rad=" << m[1]*1e-6 << " MPa]\n";
                } else {
                    launch_plate_vel(-2.0 * z_vel, 2.0 * z_vel);
                }
            } else {
                hold_counter++;
                launch_plate_vel(0.0, 0.0);
                if (hold_counter >= preload_hold) {
                    preload_dead = true;
                    load_dead = false;
                }
            }
        }

        if (!load_dead && (it % 2 == 0)) {
            launch_plate_vel(-z_vel, z_vel);
            load_dead = true;
            servo_period = servo_period_load;
            it_dev = it;
            std::cout << "  [Deviatoric loading started at iter " << it << "]\n";
        }

        if (it % servo_period == 0) {
            launch_measure();
            if (it % 1000 == 0) {
                CHECK_CU(cuMemsetD8(d_ring_stiff, 0, n_h * sizeof(double)));
                CHECK_CU(cuLaunchKernel(fn_stiff, (n_cand_fs + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_stiff, nullptr));
            }
            CHECK_CU(cuLaunchKernel(fn_servo, (n_mem + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_servo, nullptr));
        }

        if (vtk_every > 0 && it % vtk_every == 0) export_vtk();   // f_force válido: need_meas incluye este paso
        if (it % 1000 == 0 || step == max_steps) {
            launch_measure();
            double m[8];
            int nt = 0, ns = 0;
            CHECK_CU(cuMemcpyDtoH(m, d_metrics, 8 * sizeof(double)));
            CHECK_CU(cuMemcpyDtoH(&nt, d_nb_tens, sizeof(int)));
            CHECK_CU(cuMemcpyDtoH(&ns, d_nb_shear, sizeof(int)));
            double sim_t = (it - 1) * dt;   // O.time en la iteración it
            if (it % 1000 == 0) {
                write_row(it, sim_t, m, nt, ns);
                const double q_val = m[0] - m[1];
                if (m[4] > fz_max) fz_max = m[4];   // extract_Data()
                if (it_dev > 0 && q_val > q_peak) { q_peak = q_val; eps1_peak = m[2]; sax_peak = m[0]; srad_peak = m[1]; it_peak = it; }
                if (it % 10000 == 0) {
                    f_facet.flush();
                    f_cracks.flush();
                }
            }
            if (it % 50000 == 0 || it == 1000 || it == 5000 || it == 100000 || step == max_steps) {
                double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
                std::cout << "Iter " << std::setw(7) << it
                          << " | s_ax=" << std::fixed << std::setprecision(4) << m[0]*1e-6 << " MPa"
                          << " | s_rad=" << m[1]*1e-6 << " MPa"
                          << " | q=" << (m[0]-m[1])*1e-6 << " MPa"
                          << " | eps1=" << std::setprecision(5) << m[2]*100.0 << "%"
                          << " | tens=" << nt << " shear=" << ns
                          << " | " << std::setprecision(1) << (step / el) << " it/s";
                if (dynamic_contacts) std::cout << " | rebuilds=" << n_rebuilds << " nuevos_ss=" << new_ss_contacts
                                                << " perdidos ss/fs=" << missed_ss << "/" << missed_fs << " (verif=" << n_verifications << ")";
                std::cout << std::endl;
            }
            for (int c = 0; c < 8; ++c) last_m[c] = m[c];
            last_it = it; last_nt = nt; last_ns = ns;
            if (it % 1000 == 0 && it_dev > 0) {
                const double q_now = m[0] - m[1];
                if (stop_eps1 > 0.0 && m[2] >= stop_eps1) stop_reason = "parar_eps1";
                else if (stop_qdrop > 0.0 && it_peak > 0 && q_now < (1.0 - stop_qdrop) * q_peak) stop_reason = "parar_caida_q";
                if (stop_reason != "pasos") {
                    std::cout << "  [Parada por " << stop_reason << " en iter " << it << ": q=" << q_now * 1e-6 << " MPa, eps1=" << m[2] * 100.0 << "%]\n";
                    max_steps = step;
                    break;
                }
            }
        }

        if (stop_yade && reached_threshold && it % 100 == 0) {   // [PARADA] checkProgress()
            launch_measure();
            double m[8];
            CHECK_CU(cuMemcpyDtoH(m, d_metrics, 8 * sizeof(double)));
            const double fz = m[4], sim_t = (it - 1) * dt;
            if (fz > fz_max) fz_max = fz;
            if (it_fracture > 0) {
                if (sim_t >= t_fracture * (1.0 + stop_extra)) {
                    int nt = 0, ns = 0;
                    CHECK_CU(cuMemcpyDtoH(&nt, d_nb_tens, sizeof(int)));
                    CHECK_CU(cuMemcpyDtoH(&ns, d_nb_shear, sizeof(int)));
                    write_row(it, sim_t, m, nt, ns);   // extract_Data() final, como YADE (puede repetir la fila de esta iteración)
                    if (vtk_every > 0) export_vtk();
                    for (int c = 0; c < 8; ++c) last_m[c] = m[c];
                    last_it = it; last_nt = nt; last_ns = ns;
                    stop_reason = "fractura_yade";
                    std::cout << "  [Parada automática (YADE): t=" << sim_t << " s >= " << (1.0 + stop_extra) << "·t_fractura, iter " << it
                              << ", q=" << (m[0] - m[1]) * 1e-6 << " MPa, eps1=" << m[2] * 100.0 << "%]\n";
                    max_steps = step;
                    break;
                }
            } else if (fz / fz_max < stop_frac && fz_max > stop_fzmin) {
                t_fracture = sim_t; it_fracture = it; fz_max_fracture = fz_max;
                if (vtk_every > 0) export_vtk();
                std::cout << "  [Fractura detectada en iter " << it << " (t=" << sim_t << " s, fz/fz_max=" << fz / fz_max
                          << "): se continúa hasta t=" << t_fracture * (1.0 + stop_extra) << " s]\n";
            }
        }
    }

    CHECK_CU(cuCtxSynchronize());
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    if (vtk_every > 0) std::cout << "[VTK] " << vtk_index << " instantáneas, " << vtk_seconds << " s de escritura\n";
    std::cout << "\nCompletado en " << elapsed << " s (" << (max_steps / elapsed) << " it/s)!\n";
    f_facet.flush(); f_cracks.flush();
    {   // [CASO] resumen de la corrida (clave = valor), pensado para barridos de parámetros
        std::ofstream r(out_dir + "/resumen.txt");
        r << std::setprecision(10);
        r << "estado = completado\nparada = " << stop_reason << "\npasos = " << max_steps << "\niter_final = " << last_it
          << "\nsegundos = " << elapsed << "\nit_por_s = " << max_steps / elapsed
          << "\niter_confinamiento = " << it_conf << "\niter_inicio_carga = " << it_dev
          << "\nq_pico = " << (it_peak > 0 ? q_peak : 0.0) << "\neps1_pico = " << eps1_peak << "\ns_ax_pico = " << sax_peak << "\ns_rad_pico = " << srad_peak
          << "\niter_pico = " << it_peak << "\ns_ax_final = " << last_m[0] << "\ns_rad_final = " << last_m[1] << "\nq_final = " << last_m[0] - last_m[1]
          << "\neps1_final = " << last_m[2] << "\neps3_final = " << last_m[3] << "\nfisuras_traccion = " << last_nt << "\nfisuras_corte = " << last_ns
          << "\niter_fractura = " << it_fracture << "\ntiempo_fractura = " << t_fracture << "\nfz_max = " << (it_fracture > 0 ? fz_max_fracture : fz_max)
          << "\nenlaces = " << nA << "\nparticulas = " << n_particles << "\nfacetas = " << n_facets << "\n";
    }
    if (getenv("DEM_DUMP_STATE")) {   // regresión: estado final binario (pos, vel, angVel de esferas y v0 de facetas)
        std::vector<Vec3> P(n_particles), V(n_particles), W(n_particles), F(n_facets);
        CHECK_CU(cuMemcpyDtoH(P.data(), d_pos, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V.data(), d_vel, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(W.data(), d_angVel, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(F.data(), d_fv0, n_facets * sizeof(Vec3)));
        FILE* fd = fopen(getenv("DEM_DUMP_STATE"), "wb");
        fwrite(P.data(), sizeof(Vec3), n_particles, fd); fwrite(V.data(), sizeof(Vec3), n_particles, fd);
        fwrite(W.data(), sizeof(Vec3), n_particles, fd); fwrite(F.data(), sizeof(Vec3), n_facets, fd); fclose(fd);
    }
    return 0;
}
