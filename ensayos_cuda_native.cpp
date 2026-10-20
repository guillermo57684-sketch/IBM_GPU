/*
 * ensayos_cuda_native.cpp
 *
 * Ensayos BTS (brasileño), UCS (compresión simple) y UTS (tracción directa) del modelo IBM en CUDA nativo
 * (libnvrtc + libcuda Driver API). Réplica de test-1-BTS.py, test-2-UCS.py y test-3-UTS.py (YADE) con la misma
 * física que triaxial_cuda_native.cpp (ley esfera-esfera y esfera-faceta de paridad, FP64, suma determinista,
 * contactos dinámicos con lista de Verlet). Añade respecto al triaxial:
 *   - energías por interacción como Law2_ScGeom_IBPhys_IBLaw (elásticas N/S/B, viscosa, plásticas, fractura,
 *     energía cinética al fallar) y O.energy['nonviscDamp'] / ['kinetic'] de NewtonIntegrator;
 *   - unbalancedForce() de YADE (Shop::unbalancedForce: media |F| de los cuerpos dinámicos / media |normalForce+shearForce|
 *     de las interacciones reales; la ley IBM no escribe normalForce, así que el denominador es la media de |shearForce|);
 *   - grados de libertad bloqueados (blockedDOFs='z' de las partículas de agarre del UTS);
 *   - archivo de fisuras como recordCracks (iter time x y z failureMode energy area nx ny nz);
 *   - acople corte-flexión opcional (coupledShearBending, argumento 16) y sección de Weibull (argumentos 13 y 14).
 *
 * Uso: ./ensayos_cuda_native <bts|ucs|uts> young kn_ks_ratio tensile cohesion tgPhi density identifier intRadius
 *                             input_file bendingIn rollFric viscAlpha weibShape weibScale mu coupledShearBending [clave=valor ...]
 *      (los 16 argumentos de los scripts YADE, en el mismo orden). ./ensayos_cuda_native --ayuda
 * La salida va a output_tribo_<identifier>/ (como YADE) salvo salida=<carpeta>.
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

#include <charconv>
#include <array>

// Código fuente CUDA C++ de los kernels (se compila con NVRTC en tiempo de ejecución para la GPU detectada)
const char* CUDA_KERNEL_SOURCE = R"CUDA(

// Facetas por plato: [0, N_PLATE) superior, [N_PLATE, 2·N_PLATE) inferior (0 en el UTS, que no tiene facetas)
#ifndef N_PLATE
#define N_PLATE 0
#endif

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

// [FISURAS] registro como recordCracks de Law2_ScGeom_IBPhys_IBLaw: iter, par, modo, energía, punto de contacto, normal, tiempo
__device__ void record_crack(double* crk, int* crk_n, int cap, int iter, double time_now, int k, int mode, double en,
                             double cx, double cy, double cz, double nx, double ny, double nz) {
    int q = atomicAdd(crk_n, 1);
    if (q >= cap) return;
    double* c = crk + 11 * (size_t)q;
    c[0] = (double)iter; c[1] = (double)k; c[2] = (double)mode; c[3] = en;
    c[4] = cx; c[5] = cy; c[6] = cz; c[7] = nx; c[8] = ny; c[9] = nz; c[10] = time_now;
}

// [PAR-1] Réplica de Law2_ScGeom_IBPhys_IBLaw::go + ScGeom::precompute/rotate (YADE), mismo orden de operaciones.
// [ENERGIA] b_en: 9 valores por par = elasticEnergy_Normal, _Shear, _Bending, dissipatedEnergy_Visc, _Plastic,
//           _PlasticFr, _PlasticRoll, _Fracture, kineticEnergyAtFailure (se ponen a 0 cuando YADE borraría la interacción).
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
    const double* __restrict__ mass,
    const double* __restrict__ inertia,
    const int* __restrict__ b_p1,
    const int* __restrict__ b_p2,
    const double* __restrict__ b_kn,
    const double* __restrict__ b_ks,
    const double* __restrict__ b_kb,
    const double* __restrict__ b_cn,
    const double* __restrict__ b_FnMax,
    const double* __restrict__ b_FsMax,
    const double* __restrict__ b_kseff,      // ks + 12·kb/beamLength² (acople bimodular; = ks si coupledShearBending = 0)
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
    int* __restrict__ nb_tens_cracks,
    int* __restrict__ nb_shear_cracks,
    double* __restrict__ pf_out,   // [DET] 9 valores por par: F (sobre j), T_i, T_j
    double* __restrict__ b_en,
    double* __restrict__ crk,
    int* __restrict__ crk_n,
    int crk_cap,
    int iter_now,
    double time_now,
    int init_pass               // 1 = solo fija initD y la normal inicial de los enlaces (isFresh de la iteración 0)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_bonds) return;
    double* o = pf_out + 9 * (size_t)k; for (int c = 0; c < 9; ++c) o[c] = 0.0;
    double* e = b_en + 9 * (size_t)k;

    const int i = b_p1[k];
    const int j = b_p2[k];
    const double3_t r12 = sub_d3(pos[j], pos[i]);
    const double dist = sqrt(dot_d3(r12, r12));
    const double r1 = radius[i], r2 = radius[j];
    const double pen = r1 + r2 - dist;                      // geom->penetrationDepth

    unsigned char st = b_state[k];
    if (init_pass) {   // [PAR-7] initD = penetrationDepth con la misma aritmética del kernel: D = 0 exacto mientras no haya movimiento
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
        for (int c = 0; c < 9; ++c) e[c] = 0.0;             // IBPhys nuevo
    }

    const real D = (real)(pen - b_initD[k]);
    real Fn = (real)b_kn[k] * D;

    // STEP B: contacto no cohesivo en tracción -> se borra la interacción
    if (coh == 0 && D < (real)0) {
        b_state[k] = st & ~1; b_initD[k] = 0.0;
        b_sfx[k] = 0.0; b_sfy[k] = 0.0; b_sfz[k] = 0.0; b_mbx[k] = 0.0; b_mby[k] = 0.0; b_mbz[k] = 0.0;
        for (int c = 0; c < 9; ++c) e[c] = 0.0;
        return;
    }

    // ScGeom::precompute (normal actual)
    const double invd = 1.0 / dist;
    const real3 n = mk3((real)(r12.x * invd), (real)(r12.y * invd), (real)(r12.z * invd));

    // STEP C (energía elástica con el corte y el momento del paso anterior, antes de rotarlos)
    const double sfx0 = b_sfx[k], sfy0 = b_sfy[k], sfz0 = b_sfz[k];
    const double mbx0 = b_mbx[k], mby0 = b_mby[k], mbz0 = b_mbz[k];
    const double eN = 0.5 * ((double)Fn * (double)Fn) / b_kn[k];
    const double eS = 0.5 * (sfx0 * sfx0 + sfy0 * sfy0 + sfz0 * sfz0) / b_ks[k];
    const double eB = (b_kb[k] > 0.0) ? 0.5 * (mbx0 * mbx0 + mby0 * mby0 + mbz0 * mbz0) / b_kb[k] : 0.0;
    // geom->contactPoint = pos1 + (radius1 - 0.5·penetrationDepth)·normal
    const double cpx = pos[i].x + (r1 - 0.5 * pen) * (double)n.x;
    const double cpy = pos[i].y + (r1 - 0.5 * pen) * (double)n.y;
    const double cpz = pos[i].z + (r1 - 0.5 * pen) * (double)n.z;

    // STEP C: rotura por tracción pura -> isCohesive=false y, como D<0, la interacción se borra (STEP D)
    if (coh == 1 && -Fn > (real)b_FnMax[k]) {
        record_crack(crk, crk_n, crk_cap, iter_now, time_now, k, 1, eN + eS + eB, cpx, cpy, cpz, n.x, n.y, n.z);
        b_state[k] = 4; b_initD[k] = 0.0;
        b_sfx[k] = 0.0; b_sfy[k] = 0.0; b_sfz[k] = 0.0; b_mbx[k] = 0.0; b_mby[k] = 0.0; b_mbz[k] = 0.0;
        for (int c = 0; c < 9; ++c) e[c] = 0.0;
        atomicAdd(nb_tens_cracks, 1);
        return;
    }
    e[0] = eN; e[1] = eS; e[2] = eB;

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
    e[3] -= (double)dot3(Fvisc, scl3(n, vn)) * dt_val;                 // dissipatedEnergy_Visc -= F_visc·incidentVn·dt

    // STEP E: corte. El estado acumulado (fuerza de corte y momento) se integra siempre en FP64 (FIX-4).
    const double3_t orthoD = make_d3(ortho.x, ortho.y, ortho.z), twistD = make_d3(twist.x, twist.y, twist.z);
    double3_t sfd = make_d3(sfx0, sfy0, sfz0);
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
        const double3_t trial = sfd;
        if (maxFs > 0.0) sfd = scale_d3(sfd, maxFs / curFs); else sfd = make_d3(0.0, 0.0, 0.0);
        const double dissip = (1.0 / Ks_eff) * dot_d3(sub_d3(trial, sfd), sfd);
        if (dissip > 0.0) { e[4] += dissip; e[5] += dissip; }
        if (coh == 1) {
            coh = 0; b_state[k] = 5; atomicAdd(nb_shear_cracks, 1);
            e[7] = eN + eS + eB;
            const double3_t v1 = vel[i], v2 = vel[j], a1 = angVel[i], a2 = angVel[j];
            const double ke1 = 0.5 * (mass[i] * dot_d3(v1, v1) + (inertia[i] * (a1.x * a1.x) + inertia[i] * (a1.y * a1.y) + inertia[i] * (a1.z * a1.z)));
            const double ke2 = 0.5 * (mass[j] * dot_d3(v2, v2) + (inertia[j] * (a2.x * a2.x) + inertia[j] * (a2.y * a2.y) + inertia[j] * (a2.z * a2.z)));
            e[8] = ke1 + ke2;
            record_crack(crk, crk_n, crk_cap, iter_now, time_now, k, 2, e[7], cpx, cpy, cpz, n.x, n.y, n.z);
        }
    }
    b_sfx[k] = sfd.x; b_sfy[k] = sfd.y; b_sfz[k] = sfd.z;
    const real3 sf = mk3((real)sfd.x, (real)sfd.y, (real)sfd.z);

    // STEP F: flexión / rodadura
    double3_t mbd = make_d3(mbx0, mby0, mbz0);
    mbd = sub_d3(mbd, cross_d3(mbd, orthoD));
    mbd = sub_d3(mbd, cross_d3(mbd, twistD));
    real3 relW = sub3(w2, w1);
    relW = sub3(relW, scl3(n, dot3(n, relW)));
    mbd = sub_d3(mbd, scale_d3(make_d3(relW.x, relW.y, relW.z), b_kb[k] * dt_val));
    if (coh == 0) {
        const double Mlim = ((Fn > (real)0) ? (double)Fn : 0.0) * fmin(r1, r2) * rollFric;
        const double Mm = sqrt(dot_d3(mbd, mbd));
        if (Mm > Mlim) {
            const double3_t trial = mbd;
            if (Mlim > 0.0) mbd = scale_d3(mbd, Mlim / Mm); else mbd = make_d3(0.0, 0.0, 0.0);
            if (b_kb[k] > 0.0) {
                const double dissip = (1.0 / b_kb[k]) * dot_d3(sub_d3(trial, mbd), mbd);
                if (dissip > 0.0) { e[4] += dissip; e[6] += dissip; }
            }
        }
    }
    b_mbx[k] = mbd.x; b_mby[k] = mbd.y; b_mbz[k] = mbd.z;
    const real3 mb = mk3((real)mbd.x, (real)mbd.y, (real)mbd.z);

    // STEP G: fuerzas y momentos
    const real3 fT = add3(add3(scl3(n, Fn), Fvisc), sf);
    const real3 ncs = crs3(n, sf);
    const real arm1 = -(real)(r1 - 0.5 * pen), arm2 = -(real)(r2 - 0.5 * pen);
    const real3 t1 = sub3(scl3(ncs, arm1), mb);
    const real3 t2 = add3(scl3(ncs, arm2), mb);
    o[0] = fT.x; o[1] = fT.y; o[2] = fT.z; o[3] = t1.x; o[4] = t1.y; o[5] = t1.z; o[6] = t2.x; o[7] = t2.y; o[8] = t2.z;
}

// [ENERGIA] fs_en: 4 valores por par = elasticEnergy_Normal, _Shear, dissipatedEnergy_Visc, _Plastic (= _PlasticFr)
#define FS_OFF() do { if (fs_active[k] != 0) { fs_active[k] = 0; fs_shearForce[k] = make_d3(0.0, 0.0, 0.0); \
                       fs_en[4 * (size_t)k] = 0.0; fs_en[4 * (size_t)k + 1] = 0.0; fs_en[4 * (size_t)k + 2] = 0.0; fs_en[4 * (size_t)k + 3] = 0.0; } } while (0)

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
    double* __restrict__ fs_out,   // [DET] 6 valores por par: F sobre la esfera, T sobre la esfera
    double* __restrict__ fs_en
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_cand_fs) return;
    { double* o = fs_out + 6 * (size_t)k; for (int c = 0; c < 6; ++c) o[c] = 0.0; }

    int fi = fs_f[k];
    int si = fs_s[k];
    double r2 = radius[si];
    double3_t p = pos[si];

    // Filtro rápido por coordenada Z (platos horizontales)
    if (fi < N_PLATE) {
        if (f_v0[0].z - p.z >= r2) { FS_OFF(); return; }   // L >= r (como Ig2: sin contacto si r − L <= 0)
    } else {
        if (p.z - f_v0[N_PLATE].z >= r2) { FS_OFF(); return; }
    }

    double3_t v0 = f_v0[fi], v1 = f_v1[fi], v2 = f_v2[fi];
    double minx = fmin(fmin(v0.x, v1.x), v2.x);
    double maxx = fmax(fmax(v0.x, v1.x), v2.x);
    if (minx - p.x >= r2 || p.x - maxx >= r2) { FS_OFF(); return; }
    double miny = fmin(fmin(v0.y, v1.y), v2.y);
    double maxy = fmax(fmax(v0.y, v1.y), v2.y);
    if (miny - p.y >= r2 || p.y - maxy >= r2) { FS_OFF(); return; }
    double3_t cp = closest_pt_tri_3d(p, v0, v1, v2);
    double3_t vec = sub_d3(p, cp);
    double dist_sq = dot_d3(vec, vec);

    // Ig2_Facet_Sphere_ScGeom: contacto nuevo si penetrationDepth = r − distancia > 0. La distancia se toma como sqrt
    // correctamente redondeada (= |L| exacta en el interior de un plato plano, como YADE), no como dist² < r²: con la esfera
    // que toca el plato en t=0 (penetración de 1 ulp) dist² < r² falla y el contacto inicial se perdía (desfase de un paso).
    const double dist_c = (dist_sq > 1e-36) ? sqrt(dist_sq) : 0.0;
    if (dist_sq > 1e-36 && r2 - dist_c > 0.0) {
        double dist = dist_c;
        double3_t normal = scale_d3(vec, 1.0 / dist);
        double overlap = r2 - dist;
        double* e = fs_en + 4 * (size_t)k;
        bool isNewF = false;
        if (fs_active[k] == 0) {
            fs_active[k] = 1;
            fs_initD[k] = overlap;
            fs_shearForce[k] = make_d3(0.0, 0.0, 0.0);
            e[0] = 0.0; e[1] = 0.0; e[2] = 0.0; e[3] = 0.0;
            isNewF = true;
        }

        double D = overlap - fs_initD[k];
        if (D < 0.0) { FS_OFF(); return; }

        double Fn = fs_kn[k] * D;
        {   // STEP C (no cohesivo): energías elásticas con el corte del paso anterior
            double3_t s0 = fs_shearForce[k];
            e[0] = 0.5 * (Fn * Fn) / fs_kn[k];
            e[1] = 0.5 * dot_d3(s0, s0) / fs_ks[k];
        }
        double3_t c2x = scale_d3(normal, -(r2 - 0.5 * overlap));
        double3_t v1 = f_vel[fi];
        double3_t v2 = add_d3(vel[si], cross_d3(angVel[si], c2x));
        double3_t incidentV = sub_d3(v2, v1);
        double vn_s = dot_d3(normal, incidentV);
        double3_t incidentVn = scale_d3(normal, vn_s);
        double3_t F_visc_vec = scale_d3(incidentVn, -fs_cn[k]);
        e[2] -= dot_d3(F_visc_vec, incidentVn) * dt_val;

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
            const double3_t trial = sf;
            if (maxFs > 0.0) {
                sf = scale_d3(sf, maxFs / sqrt(sf_sq));
            } else {
                sf = make_d3(0.0, 0.0, 0.0);
            }
            const double dissip = (1.0 / fs_ks[k]) * dot_d3(sub_d3(trial, sf), sf);
            if (dissip > 0.0) e[3] += dissip;
        }
        fs_shearForce[k] = sf;

        double3_t fTotal = add_d3(add_d3(scale_d3(normal, Fn), F_visc_vec), sf);
        double3_t dt2 = scale_d3(cross_d3(normal, sf), -(r2 - 0.5 * overlap));
        double* o = fs_out + 6 * (size_t)k;
        o[0] = fTotal.x; o[1] = fTotal.y; o[2] = fTotal.z; o[3] = dt2.x; o[4] = dt2.y; o[5] = dt2.z;
    } else {
        FS_OFF();
    }
}

// NewtonIntegrator (cundallDamp2nd) + [ENERGIA] nonviscDamp (updateEnergy: |v|·|F|·damping·dt con la velocidad previa)
// + [DOF] blockedDOFs: bit 0/1/2 = x/y/z, bit 3/4/5 = rotaciones; la aceleración bloqueada es 0 (velocidad impuesta)
__global__ void kernel_integrate_newton(
    int n_particles,
    int n_facets,
    double dt_val,
    const double* __restrict__ mass,
    const double* __restrict__ inertia,
    double3_t* __restrict__ force,
    double3_t* __restrict__ torque,
    double3_t* __restrict__ pos,
    double3_t* __restrict__ vel,
    double3_t* __restrict__ angVel,
    const double3_t* __restrict__ f_vel,
    double3_t* __restrict__ f_pos,
    double3_t* __restrict__ f_v0,
    double3_t* __restrict__ f_v1,
    double3_t* __restrict__ f_v2,
    const unsigned char* __restrict__ blocked,
    const double3_t* __restrict__ f_force,
    double* __restrict__ e_damp,
    double* __restrict__ f_edamp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef DEM_DAMPING
    const double cundall = DEM_DAMPING;
#else
    const double cundall = 0.01;
#endif
    if (i < n_particles) {
        double3_t f = make_d3((double)force[i].x, (double)force[i].y, (double)force[i].z);
        double3_t v = vel[i];
        double3_t t = make_d3((double)torque[i].x, (double)torque[i].y, (double)torque[i].z);
        double3_t w = angVel[i];
        e_damp[i] += (fabs(v.x) * fabs(f.x) + fabs(v.y) * fabs(f.y) + fabs(v.z) * fabs(f.z)) * cundall * dt_val;
        e_damp[i] += (fabs(w.x) * fabs(t.x) + fabs(w.y) * fabs(t.y) + fabs(w.z) * fabs(t.z)) * cundall * dt_val;
        const unsigned char blk = blocked[i];
        // [FIX-2] YADE NewtonIntegrator::cundallDamp2nd: a_i *= 1 - d*sign(a_i*(v_i + 0.5*dt*a_i))
        double inv_m = 1.0 / mass[i];
        double3_t acc = scale_d3(f, inv_m);
        if (blk & 1) acc.x = 0.0;
        if (blk & 2) acc.y = 0.0;
        if (blk & 4) acc.z = 0.0;
        acc.x *= 1.0 - cundall * ((acc.x * (v.x + 0.5 * dt_val * acc.x) > 0.0) ? 1.0 : ((acc.x * (v.x + 0.5 * dt_val * acc.x) < 0.0) ? -1.0 : 0.0));
        acc.y *= 1.0 - cundall * ((acc.y * (v.y + 0.5 * dt_val * acc.y) > 0.0) ? 1.0 : ((acc.y * (v.y + 0.5 * dt_val * acc.y) < 0.0) ? -1.0 : 0.0));
        acc.z *= 1.0 - cundall * ((acc.z * (v.z + 0.5 * dt_val * acc.z) > 0.0) ? 1.0 : ((acc.z * (v.z + 0.5 * dt_val * acc.z) < 0.0) ? -1.0 : 0.0));
        v = add_d3(v, scale_d3(acc, dt_val));
        vel[i] = v;
        pos[i] = add_d3(pos[i], scale_d3(v, dt_val));

        double inv_I = 1.0 / inertia[i];
        double3_t angAcc = scale_d3(t, inv_I);
        if (blk & 8) angAcc.x = 0.0;
        if (blk & 16) angAcc.y = 0.0;
        if (blk & 32) angAcc.z = 0.0;
        angAcc.x *= 1.0 - cundall * ((angAcc.x * (w.x + 0.5 * dt_val * angAcc.x) > 0.0) ? 1.0 : ((angAcc.x * (w.x + 0.5 * dt_val * angAcc.x) < 0.0) ? -1.0 : 0.0));
        angAcc.y *= 1.0 - cundall * ((angAcc.y * (w.y + 0.5 * dt_val * angAcc.y) > 0.0) ? 1.0 : ((angAcc.y * (w.y + 0.5 * dt_val * angAcc.y) < 0.0) ? -1.0 : 0.0));
        angAcc.z *= 1.0 - cundall * ((angAcc.z * (w.z + 0.5 * dt_val * angAcc.z) > 0.0) ? 1.0 : ((angAcc.z * (w.z + 0.5 * dt_val * angAcc.z) < 0.0) ? -1.0 : 0.0));
        angVel[i] = add_d3(w, scale_d3(angAcc, dt_val));
        force[i] = make_d3(0.0, 0.0, 0.0);
        torque[i] = make_d3(0.0, 0.0, 0.0);
    }

    if (i < n_facets) {
        // las facetas (no dinámicas, masa 0) también pasan por updateEnergy en YADE
        const double3_t fv = f_vel[i], ff = f_force[i];
        f_edamp[i] += (fabs(fv.x) * fabs(ff.x) + fabs(fv.y) * fabs(ff.y) + fabs(fv.z) * fabs(ff.z)) * cundall * dt_val;
        double3_t dp = scale_d3(fv, dt_val);
        f_pos[i] = add_d3(f_pos[i], dp);
        f_v0[i] = add_d3(f_v0[i], dp);
        f_v1[i] = add_d3(f_v1[i], dp);
        f_v2[i] = add_d3(f_v2[i], dp);
    }
}

// [ESTADISTICAS] por bloque (orden fijo): suma |F| de las esferas y energía cinética (velocidad previa, como updateEnergy)
__global__ void kernel_body_stats(
    int n, const double* __restrict__ mass, const double* __restrict__ inertia,
    const double3_t* __restrict__ vel, const double3_t* __restrict__ angVel, const double3_t* __restrict__ force,
    double* __restrict__ partial
) {
    __shared__ double sF[256], sE[256];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    double fn = 0.0, ek = 0.0;
    if (i < n) {
        double fx = force[i].x, fy = force[i].y, fz = force[i].z;
        fn = sqrt(fx * fx + fy * fy + fz * fz);
        double3_t v = vel[i], w = angVel[i];
        double I = inertia[i];
        ek = 0.5 * mass[i] * (v.x * v.x + v.y * v.y + v.z * v.z) + 0.5 * (w.x * (I * w.x) + w.y * (I * w.y) + w.z * (I * w.z));
    }
    sF[tid] = fn; sE[tid] = ek;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) { sF[tid] += sF[tid + s]; sE[tid] += sE[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { partial[2 * blockIdx.x] = sF[0]; partial[2 * blockIdx.x + 1] = sE[0]; }
}

// [ESTADISTICAS] sumas por bloque sobre los pares esfera-esfera reales: 9 energías, |shearForce|, número de pares
__global__ void kernel_reduce_ss(
    int n, const unsigned char* __restrict__ st, const double* __restrict__ en,
    const double* __restrict__ sfx, const double* __restrict__ sfy, const double* __restrict__ sfz, double* __restrict__ partial
) {
    __shared__ double sh[11][256];
    int tid = threadIdx.x, k = blockIdx.x * blockDim.x + tid;
    bool a = (k < n) && (st[k] & 1);
    for (int c = 0; c < 9; ++c) sh[c][tid] = a ? en[9 * (size_t)k + c] : 0.0;
    sh[9][tid] = a ? sqrt(sfx[k] * sfx[k] + sfy[k] * sfy[k] + sfz[k] * sfz[k]) : 0.0;
    sh[10][tid] = a ? 1.0 : 0.0;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) for (int c = 0; c < 11; ++c) sh[c][tid] += sh[c][tid + s];
        __syncthreads();
    }
    if (tid < 11) partial[11 * blockIdx.x + tid] = sh[tid][0];
}

// [ESTADISTICAS] ídem para los pares esfera-faceta reales: 4 energías, |shearForce|, número de pares
__global__ void kernel_reduce_fs(
    int n, const int* __restrict__ act, const double* __restrict__ en, const double3_t* __restrict__ sf, double* __restrict__ partial
) {
    __shared__ double sh[6][256];
    int tid = threadIdx.x, k = blockIdx.x * blockDim.x + tid;
    bool a = (k < n) && act[k];
    for (int c = 0; c < 4; ++c) sh[c][tid] = a ? en[4 * (size_t)k + c] : 0.0;
    sh[4][tid] = a ? sqrt(sf[k].x * sf[k].x + sf[k].y * sf[k].y + sf[k].z * sf[k].z) : 0.0;
    sh[5][tid] = a ? 1.0 : 0.0;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) for (int c = 0; c < 6; ++c) sh[c][tid] += sh[c][tid + s];
        __syncthreads();
    }
    if (tid < 6) partial[6 * blockIdx.x + tid] = sh[tid][0];
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
    double3_t* __restrict__ force, double3_t* __restrict__ torque
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
    force[i] = make_d3(fx, fy, fz);
    torque[i] = make_d3(tx, ty, tz);
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

} // extern "C"
)CUDA";

struct Vec3 {
    double x, y, z;
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

// =====================================================================================
// Opciones de ejecución (clave=valor tras los 16 argumentos). La física del ensayo la fijan los argumentos y el script.
// =====================================================================================
struct CfgKey { const char* key; const char* def; const char* env; const char* help; };
static const CfgKey CFG_KEYS[] = {
    {"salida",           "",           nullptr,      "carpeta de salida (vacío = output_tribo_<identifier>, como YADE)"},
    {"sobrescribir",     "0",          nullptr,      "1 = permite escribir sobre una salida que ya tiene facet_force.txt"},
    {"pasos_max",        "50000000",   nullptr,      "tope de iteraciones (YADE no tiene tope: para por fractura)"},
    {"z_vel",            "",           nullptr,      "velocidad de carga [m/s] (vacío = la del script: 1e-5 BTS/UCS, 5e-6 UTS)"},
    {"dt_safety",        "",           nullptr,      "dt = dt_safety·PWaveTimeStep (vacío = la del script: 0.5 BTS/UCS, 0.2 UTS)"},
    {"filas_duplicadas", "1",          nullptr,      "1 = como YADE, extract_Data() se llama dos veces (filas repetidas) mientras O.iter < 20000"},
    {"semilla_weibull",  "0",          nullptr,      "semilla de la sección de Weibull (0 = aleatoria, como YADE)"},
    {"vtk",              "1",          nullptr,      "1 = exportar VTK como exp_vtk() del script (bondCreation, periodo del script y fractura)"},
    {"vtk_ascii",        "0",          "DEM_VTK_ASCII", "1 = VTK en ASCII"},
    {"vtk_intrs",        "1",          "DEM_VTK_INTRS", "0 = no exportar interacciones"},
    {"gpu",              "0",          "DEM_GPU",    "índice de la GPU"},
    {"arch",             "auto",       "DEM_ARCH",   "arquitectura NVRTC (auto = la de la GPU)"},
    {"fp32",             "0",          "DEM_FP32",   "1 = ley de contacto en FP32 (por defecto FP64)"},
    {"fmad",             "1",          nullptr,      "0 = NVRTC sin contracción a FMA (--fmad=false)"},
    {"skin_factor",      "0.3",        nullptr,      "skin de Verlet = factor·r_min"},
    {"contact_verify",   "0",          nullptr,      "cada N pasos cuenta contactos perdidos (0 = no)"},
};

struct Config {
    std::map<std::string, std::string> v;
    std::map<std::string, std::string> src;
    Config() { for (auto& k : CFG_KEYS) { v[k.key] = k.def; src[k.key] = "defecto"; } }
    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? "" : s.substr(a, b - a + 1);
    }
    bool set(const std::string& k, const std::string& val, const std::string& from) {
        if (!v.count(k)) { std::cerr << "Opción desconocida '" << k << "' (" << from << "). Use --ayuda para ver la lista.\n"; return false; }
        v[k] = val; src[k] = from; return true;
    }
    bool set_kv(const std::string& kv, const std::string& from) {
        size_t e = kv.find('=');
        if (e == std::string::npos) { std::cerr << "Se esperaba clave=valor: '" << kv << "'\n"; return false; }
        return set(trim(kv.substr(0, e)), trim(kv.substr(e + 1)), from);
    }
    void load_env() { for (auto& k : CFG_KEYS) if (k.env && getenv(k.env)) set(k.key, getenv(k.env), std::string("env ") + k.env); }
    const std::string& s(const char* k) const { return v.at(k); }
    double d(const char* k) const {
        char* e = nullptr; double x = strtod(v.at(k).c_str(), &e);
        if (v.at(k).empty() || *e) { std::cerr << "Valor no numérico para " << k << ": '" << v.at(k) << "'\n"; exit(1); }
        return x;
    }
    long long ll(const char* k) const { return std::llround(d(k)); }
    int i(const char* k) const { return (int)std::lround(d(k)); }
    static void help() {
        std::cout << "Uso: ./ensayos_cuda_native <bts|ucs|uts> young kn_ks_ratio tensile cohesion tgPhi density identifier intRadius\n"
                     "                            input_file bendingIn rollFric viscAlpha weibShape weibScale mu coupledShearBending [clave=valor]...\n"
                     "  bts = test-1-BTS.py, ucs = test-2-UCS.py, uts = test-3-UTS.py (mismos 16 argumentos y en el mismo orden).\n"
                     "  Ejemplo: ./ensayos_cuda_native ucs 6.671e+10 25 9.62e+06 6.704e+07 0.466 1e8 ucs01 1.3 inputs/probeta_1.geo 1 0.1 0.1 0 1 0.466 1\n\n"
                     "Opciones clave=valor:\n";
        for (auto& k : CFG_KEYS) std::cout << "  " << std::left << std::setw(17) << k.key << " = " << std::setw(10) << (*k.def ? k.def : "\"\"")
                                           << (k.env ? std::string(" [env ") + k.env + "]" : std::string()) << "  " << k.help << "\n";
    }
};

// repr() de Python para un float (lo que escriben los f-strings de los scripts): dígitos mínimos que reproducen el valor,
// notación fija si -4 <= exponente < 16 y si no científica con exponente de al menos 2 cifras ('1e-05', '1.5e+16').
static std::string py_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    std::string out;
    if (s[0] == '-') { out = "-"; s = s.substr(1); }
    size_t e = s.find('e');
    std::string digits;
    for (size_t q = 0; q < e; ++q) if (s[q] != '.') digits += s[q];
    int ex = std::atoi(s.c_str() + e + 1);
    int decpt = ex + 1, nd = (int)digits.size();
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) out += "0." + std::string(-decpt, '0') + digits;
        else if (decpt >= nd) out += digits + std::string(decpt - nd, '0') + ".0";
        else out += digits.substr(0, decpt) + "." + digits.substr(decpt);
    } else {
        out += digits.substr(0, 1);
        if (nd > 1) out += "." + digits.substr(1);
        char eb[16]; snprintf(eb, sizeof(eb), "e%c%02d", ex < 0 ? '-' : '+', std::abs(ex));
        out += eb;
    }
    return out;
}

// pendiente de mínimos cuadrados (scipy.stats.linregress / numpy.polyfit de grado 1)
static double ls_slope(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    double mx = 0, my = 0;
    for (size_t q = 0; q < n; ++q) { mx += x[q]; my += y[q]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0;
    for (size_t q = 0; q < n; ++q) { sxy += (x[q] - mx) * (y[q] - my); sxx += (x[q] - mx) * (x[q] - mx); }
    return sxx != 0.0 ? sxy / sxx : 0.0;
}
static double vmean(const std::vector<double>& a) { double s = 0; for (double x : a) s += x; return a.empty() ? 0.0 : s / a.size(); }


int main(int argc, char** argv) {
    Config cfg;
    cfg.load_env();
    std::vector<std::string> pos_args;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "--ayuda" || s == "-h" || s == "--help") { Config::help(); return 0; }
        if (pos_args.size() < 17) { pos_args.push_back(s); continue; }
        if (!cfg.set_kv(s, "línea de comandos")) return 1;
    }
    if (pos_args.size() != 17) { Config::help(); std::cerr << "\nSe esperan el ensayo (bts|ucs|uts) y los 16 argumentos de los scripts YADE.\n"; return 1; }
    std::string test = pos_args[0];
    for (auto& c : test) c = (char)std::tolower(c);
    const int T = test == "bts" ? 0 : test == "ucs" ? 1 : test == "uts" ? 2 : -1;
    if (T < 0) { std::cerr << "Ensayo desconocido '" << pos_args[0] << "' (bts, ucs o uts)\n"; return 1; }
    const bool BTS = T == 0, UCS = T == 1, UTS = T == 2;
    const char* TNAME = BTS ? "BTS" : UCS ? "UCS" : "UTS";
    auto num = [&](int k) {
        char* e = nullptr; double x = strtod(pos_args[k].c_str(), &e);
        if (pos_args[k].empty() || *e) { std::cerr << "Argumento " << k << " no numérico: '" << pos_args[k] << "'\n"; exit(1); }
        return x;
    };
    // mismos nombres y orden que sys.argv[1..16] de test-1-BTS.py / test-2-UCS.py / test-3-UTS.py
    const double young = num(1), kn_ks_ratio = num(2), tensile = num(3), cohesion = num(4), tgPhi = num(5), density = num(6);
    const std::string identifier = pos_args[7];
    const double intRadius = num(8);
    const std::string input_file = pos_args[9];
    const double bendingIn = num(10), rollFric = num(11), viscAlpha = num(12), weibShape = num(13), weibScale = num(14), mu = num(15);
    const bool coupledShearBending = std::stoi(pos_args[16]) != 0;   // bool(int(sys.argv[16]))
    const double young_wall = 1e10, kn_ks_wall_ratio = 2.5;         // mat_2 de los scripts
    const double PI = 3.14159265358979323846;

    // constantes de cada script
    const double dtSafety = cfg.s("dt_safety").empty() ? (UTS ? 0.2 : 0.5) : cfg.d("dt_safety");
    const double z_vel = cfg.s("z_vel").empty() ? (UTS ? 5.0e-6 : 1.0e-5) : cfg.d("z_vel");
    const int vtk_period = BTS ? 5000 : UCS ? 10000 : 2000;
    const double stop_factor = BTS ? 1.05 : 1.01;          // O.time >= fracture_time·factor
    const double fzmax_min = BTS ? 1000.0 : UCS ? 2000.0 : 1.0;
    const long long pasos_max = cfg.ll("pasos_max");
    const bool vtk_on = cfg.i("vtk") == 1;
    const int dup_rows = cfg.i("filas_duplicadas") == 1 ? 2 : 1;

    std::string out_dir = cfg.s("salida").empty() ? "output_tribo_" + identifier : cfg.s("salida");
    if (std::ifstream(out_dir + "/facet_force.txt").good() && cfg.i("sobrescribir") != 1) {
        std::cerr << "La salida " << out_dir << " ya contiene facet_force.txt. Use otro identifier, salida=<carpeta> o sobrescribir=1.\n";
        return 1;
    }
    { std::string acc; std::istringstream ps(out_dir); std::string part;
      if (!out_dir.empty() && out_dir[0] == '/') acc = "/";
      while (std::getline(ps, part, '/')) { if (part.empty()) continue; acc += part; mkdir(acc.c_str(), 0777); acc += "/"; } }

    std::cout << "============== INITIAL CONFIGURATION (" << TNAME << ", CUDA) ==============\n"
              << "Young: " << young << "  Kn/Ks: " << kn_ks_ratio << "  Tensile: " << tensile << "  Cohesion: " << cohesion
              << "  tgPhi: " << tgPhi << "  mu: " << mu << "  CoupledShearBend: " << (coupledShearBending ? "True" : "False") << "\n"
              << "Density: " << density << "  Identifier: " << identifier << "  intRadius: " << intRadius << "  InputFile: " << input_file << "\n"
              << "BendingInfluence: " << bendingIn << "  RollingFriction: " << rollFric << "  ViscousAlpha: " << viscAlpha
              << "  Weibull: " << weibShape << " / " << weibScale << "\n"
              << "z_vel: " << z_vel << " m/s  dtSafety: " << dtSafety << "  salida: " << out_dir << "\n";

    // 1. NVRTC y driver CUDA
    std::string nvrtc_used;
    void* h_nvrtc = load_nvrtc(nvrtc_used);
    if (!h_nvrtc) { std::cerr << "No se encontró libnvrtc (indique DEM_NVRTC=/ruta/a/lib o CUDA_HOME).\n"; return 1; }
    void* h_cuda = load_libcuda();
    if (!h_cuda) { std::cerr << "No se pudo cargar libcuda.so.1 (driver NVIDIA): " << dlerror() << std::endl; return 1; }
    std::cout << "[CUDA] NVRTC: " << nvrtc_used << "\n";
    auto nvrtcCreateProgram = (nvrtcResult(*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*))dlsym(h_nvrtc, "nvrtcCreateProgram");
    auto nvrtcCompileProgram = (nvrtcResult(*)(nvrtcProgram, int, const char* const*))dlsym(h_nvrtc, "nvrtcCompileProgram");
    auto nvrtcGetPTXSize = (nvrtcResult(*)(nvrtcProgram, size_t*))dlsym(h_nvrtc, "nvrtcGetPTXSize");
    auto nvrtcGetPTX = (nvrtcResult(*)(nvrtcProgram, char*))dlsym(h_nvrtc, "nvrtcGetPTX");
    auto nvrtcGetProgramLogSize = (nvrtcResult(*)(nvrtcProgram, size_t*))dlsym(h_nvrtc, "nvrtcGetProgramLogSize");
    auto nvrtcGetProgramLog = (nvrtcResult(*)(nvrtcProgram, char*))dlsym(h_nvrtc, "nvrtcGetProgramLog");
    auto nvrtcGetSupportedArchs = (nvrtcResult(*)(int*))dlsym(h_nvrtc, "nvrtcGetSupportedArchs");
    auto nvrtcGetNumSupportedArchs = (nvrtcResult(*)(int*))dlsym(h_nvrtc, "nvrtcGetNumSupportedArchs");
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

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, cfg.i("gpu")));
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));
    int arch = 0;
    {
        char name[256] = {0}; cuDeviceGetName(name, 255, device);
        int cc_major = 0, cc_minor = 0;
        CHECK_CU(cuDeviceGetAttribute(&cc_major, 75, device));
        CHECK_CU(cuDeviceGetAttribute(&cc_minor, 76, device));
        arch = cfg.s("arch") == "auto" ? 10 * cc_major + cc_minor : cfg.i("arch");
        if (cfg.s("arch") == "auto" && nvrtcGetNumSupportedArchs && nvrtcGetSupportedArchs) {
            int na = 0; nvrtcGetNumSupportedArchs(&na);
            std::vector<int> archs(na); if (na > 0) nvrtcGetSupportedArchs(archs.data());
            int best = 0; for (int a : archs) if (a <= arch) best = std::max(best, a);
            if (best > 0) arch = best;
        }
        std::cout << "[CUDA] GPU " << cfg.i("gpu") << ": " << name << " (sm_" << cc_major << cc_minor << "), compilación para compute_" << arch << "\n";
    }

    // 2. Probeta: ymport.gengeoFile / ymport.text con orientation (BTS: Quaternion((0,1,0), pi/2)) y shift = 0, scale = 1
    std::vector<Vec3> coords0;
    std::vector<double> radii;
    {
        std::ifstream gin(input_file);
        if (!gin) { std::cerr << "No se encontró la probeta " << input_file << "\n"; return 1; }
        std::vector<std::string> lines; std::string l;
        while (std::getline(gin, l)) lines.push_back(l);
        auto ends = [&](const std::string& s, const std::string& x) { return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0; };
        std::vector<std::array<double, 4>> raw;
        if (ends(input_file, ".geo")) {
            if (lines.size() < 7) { std::cerr << "Archivo .geo demasiado corto\n"; return 1; }
            int n = std::stoi(lines[6]);
            for (int q = 7; q < 7 + n && q < (int)lines.size(); ++q) {
                std::istringstream ss(lines[q]); std::array<double, 4> a;
                if (!(ss >> a[0] >> a[1] >> a[2] >> a[3])) { std::cerr << "Línea " << q + 1 << " de " << input_file << " no legible\n"; return 1; }
                raw.push_back(a);
            }
        } else if (ends(input_file, ".spheres")) {
            for (auto& ln : lines) {
                std::istringstream ss(ln); std::string t0; if (!(ss >> t0) || t0[0] == '#') continue;
                std::istringstream s2(ln); std::array<double, 4> a;
                if (s2 >> a[0] >> a[1] >> a[2] >> a[3]) raw.push_back(a);
            }
        } else { std::cerr << "Unknown file format: " << input_file << "\n"; return 1; }
        // Quaternion(AngleAxis(angle, axis)) de Eigen: w = cos(angle/2), vec = sin(angle/2)·axis; q*v = v + w·uv + vec×uv, uv = 2·(vec×v)
        double qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;
        if (BTS) { const double ha = 0.5 * (PI / 2); qw = std::cos(ha); const double s = std::sin(ha); qx = s * 0.0; qy = s * 1.0; qz = s * 0.0; }
        for (auto& a : raw) {
            double vx = a[0], vy = a[1], vz = a[2];
            double ux = qy * vz - qz * vy, uy = qz * vx - qx * vz, uz = qx * vy - qy * vx;
            ux += ux; uy += uy; uz += uz;
            double cx = qy * uz - qz * uy, cy = qz * ux - qx * uz, cz = qx * uy - qy * ux;
            double px = vx + qw * ux + cx, py = vy + qw * uy + cy, pz = vz + qw * uz + cz;
            coords0.push_back({0.0 + 1.0 * px, 0.0 + 1.0 * py, 0.0 + 1.0 * pz});
            radii.push_back(1.0 * a[3]);
        }
    }
    const int n_particles = radii.size();
    if (n_particles == 0) { std::cerr << "La probeta " << input_file << " no tiene partículas\n"; return 1; }
    std::cout << "[Entrada] " << n_particles << " esferas de " << input_file << (BTS ? " (rotadas 90° sobre y)" : "") << "\n";

    // aabbExtrema() (Shop::aabbExtrema: centros ± radio)
    double xMin = INFINITY, xMax = -INFINITY, yMin = INFINITY, yMax = -INFINITY, zMin = INFINITY, zMax = -INFINITY;
    for (int i = 0; i < n_particles; ++i) {
        xMin = std::min(xMin, coords0[i].x - radii[i]); xMax = std::max(xMax, coords0[i].x + radii[i]);
        yMin = std::min(yMin, coords0[i].y - radii[i]); yMax = std::max(yMax, coords0[i].y + radii[i]);
        zMin = std::min(zMin, coords0[i].z - radii[i]); zMax = std::max(zMax, coords0[i].z + radii[i]);
    }

    // utils.sphere: V = 4/3·pi·r³, masa = V·densidad, inercia = (2/5·V·r²)·densidad
    std::vector<double> mass(n_particles), inertia(n_particles);
    double min_r = INFINITY, r_max = 0.0, dt_pwave = INFINITY;
    for (int i = 0; i < n_particles; ++i) {
        const double r = radii[i];
        const double V = (4. / 3) * PI * std::pow(r, 3);
        mass[i] = V * density;
        inertia[i] = ((2. / 5.) * V * std::pow(r, 2)) * density;
        min_r = std::min(min_r, r); r_max = std::max(r_max, r);
        const double dens = mass[i] / ((4. / 3.) * PI * std::pow(r, 3));   // Shop::PWaveTimeStep
        dt_pwave = std::min(dt_pwave, r / std::sqrt(young / dens));
    }
    const double dt = dtSafety * dt_pwave;

    // 3. Enlaces: Ig2_Sphere_Sphere_ScGeom con interactionDetectionFactor = intRadius en la iteración 0 (d <= intRadius·(ri+rj))
    std::vector<std::pair<int,int>> raw_b_pre;
    {
        double h = 2.0 * intRadius * r_max;
        double lo[3] = {1e30, 1e30, 1e30};
        for (auto& p : coords0) { lo[0] = std::min(lo[0], p.x); lo[1] = std::min(lo[1], p.y); lo[2] = std::min(lo[2], p.z); }
        std::map<std::tuple<int,int,int>, std::vector<int>> cells;
        auto ci = [&](const Vec3& p) { return std::make_tuple((int)std::floor((p.x - lo[0]) / h), (int)std::floor((p.y - lo[1]) / h), (int)std::floor((p.z - lo[2]) / h)); };
        for (int i = 0; i < n_particles; ++i) cells[ci(coords0[i])].push_back(i);
        for (int i = 0; i < n_particles; ++i) {
            auto c = ci(coords0[i]);
            for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                auto it_c = cells.find(std::make_tuple(std::get<0>(c) + dx, std::get<1>(c) + dy, std::get<2>(c) + dz));
                if (it_c == cells.end()) continue;
                for (int j : it_c->second) {
                    if (j <= i) continue;
                    double ddx = coords0[j].x - coords0[i].x, ddy = coords0[j].y - coords0[i].y, ddz = coords0[j].z - coords0[i].z;
                    double lim = intRadius * (radii[i] + radii[j]);
                    if (lim * lim - (ddx * ddx + ddy * ddy + ddz * ddz) >= 0.0) raw_b_pre.push_back({i, j});
                }
            }
        }
        std::sort(raw_b_pre.begin(), raw_b_pre.end());
    }
    // Ip2_IBMat_IBMat_IBPhys: cohesivo solo si min(normalStrength, shearStrength) > 0
    const bool cohesive_bonds = std::min(tensile, cohesion) > 0;
    std::cout << "[Entrada] enlaces (d <= " << intRadius << "·(ri+rj)): " << raw_b_pre.size() << (cohesive_bonds ? " cohesivos" : " NO cohesivos (tensile o cohesion = 0)") << "\n";
    struct BondInit { int i, j; unsigned char st; };
    std::vector<BondInit> raw_b;
    std::set<std::pair<int,int>> bonded_set;
    for (auto& pr : raw_b_pre) { raw_b.push_back({pr.first, pr.second, (unsigned char)(cohesive_bonds ? 3 : 1)}); bonded_set.insert(pr); }
    // mismo orden de memoria que triaxial_cuda_native (z, y y transposición por bloques de 256)
    std::sort(raw_b.begin(), raw_b.end(), [&](const BondInit& a, const BondInit& b) {
        double za = coords0[a.i].z + coords0[a.j].z, zb = coords0[b.i].z + coords0[b.j].z;
        if (std::abs(za - zb) > 2e-3) return za < zb;
        return coords0[a.i].y + coords0[a.j].y < coords0[b.i].y + coords0[b.j].y;
    });
    std::vector<int> b_p1, b_p2;
    std::vector<unsigned char> b_state;
    {
        int nb_raw = raw_b.size();
        for (int base = 0; base < nb_raw; base += 256) {
            int chunk = std::min(256, nb_raw - base);
            for (int t = 0; t < chunk; ++t) {
                int idx_p = chunk == 256 ? base + (t % 16) * 16 + (t / 16) : base + t;
                b_p1.push_back(raw_b[idx_p].i); b_p2.push_back(raw_b[idx_p].j); b_state.push_back(raw_b[idx_p].st);
            }
        }
    }
    const int n_bonds = b_p1.size();
    std::vector<double> b_kn(n_bonds), b_ks(n_bonds), b_kb(n_bonds), b_cn(n_bonds), b_FnMax(n_bonds), b_FsMax(n_bonds), b_kseff(n_bonds), b_initD(n_bonds), b_area(n_bonds);
    std::vector<Vec3> b_prevNormal(n_bonds);
    std::mt19937_64 wrng(cfg.ll("semilla_weibull") > 0 ? (uint64_t)cfg.ll("semilla_weibull") : (uint64_t)std::random_device{}());
    std::weibull_distribution<double> wdist(weibShape > 0 ? weibShape : 1.0, weibScale);
    for (int k = 0; k < n_bonds; ++k) {
        int i = b_p1[k], j = b_p2[k];
        double R1 = radii[i], R2 = radii[j], minR = std::min(R1, R2);
        if (weibShape > 0) {   // distributeCrossSectionsWeibull (corte en [0, 10])
            double corr = wdist(wrng);
            if (corr < 0.0) corr = 0.0; else if (corr > 10.0) corr = 10.0;
            double ir = corr * minR;
            b_area[k] = PI * ir * ir;
        } else b_area[k] = PI * minR * minR;
        double Kn1 = 2 * young * R1, Kn2 = 2 * young * R2;
        b_kn[k] = (Kn1 * Kn2) / (Kn1 + Kn2);
        double Ks1 = Kn1 / kn_ks_ratio, Ks2 = Kn2 / kn_ks_ratio;
        b_ks[k] = (Ks1 * Ks2) / (Ks1 + Ks2);
        double kr1 = 0.5 * young * (R1 * R1 * R1), kr2 = 0.5 * young * (R2 * R2 * R2);
        double kr_base = (kr1 * kr2) / (kr1 + kr2);
        b_kb[k] = bendingIn > 0 ? kr_base * bendingIn : 0.0;
        double meff = (mass[i] * mass[j]) / (mass[i] + mass[j]);
        b_cn[k] = viscAlpha * 2.0 * std::sqrt(b_kn[k] * meff);
        b_FnMax[k] = cohesive_bonds ? tensile * b_area[k] : 0.0;
        b_FsMax[k] = cohesive_bonds ? cohesion * b_area[k] : 0.0;
        double dx = coords0[j].x - coords0[i].x, dy = coords0[j].y - coords0[i].y, dz = coords0[j].z - coords0[i].z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        b_initD[k] = (R1 + R2) - dist;
        double bl = R1 + R2 - b_initD[k];
        b_kseff[k] = coupledShearBending ? b_ks[k] + (12.0 * b_kb[k]) / (bl * bl) : b_ks[k];
        b_prevNormal[k] = {dx / dist, dy / dist, dz / dist};
    }

    // 4. Contorno
    std::vector<Vec3> f_v0, f_v1, f_v2;
    int n_plate = 0;
    if (BTS || UCS) {
        const int nW = 20;
        double z_Tol = 1.0, w_Tol, L_Tol, ori_x, ori_y = -0.5, base_x, base_y;
        int nL;
        if (BTS) {   // platos planos en zMax/zMin, x de 0 a spec_L, franja |y| <= 0.1·spec_R (spec_R = media de las extensiones en z e y)
            const double spec_R = .5 * (zMax - zMin + yMax - yMin), spec_L = xMax - xMin;
            w_Tol = 0.2; L_Tol = 1.0; ori_x = 0.0;
            nL = (int)(spec_R * w_Tol / (spec_L * L_Tol / nW));
            base_x = spec_L; base_y = spec_R;
            std::cout << "[Contorno] BTS: spec_R=" << spec_R << " m, spec_L=" << spec_L << " m, nW=" << nW << ", nL=" << nL << "\n";
        } else {     // platos de 1,1·spec_D centrados en el eje
            const double spec_D = .5 * (xMax - xMin + yMax - yMin), spec_L = zMax - zMin;
            w_Tol = 1.1; L_Tol = 1.1; ori_x = -0.5;
            nL = (int)(spec_L * L_Tol / (spec_D * w_Tol / nW));
            base_x = spec_D; base_y = spec_D;
            std::cout << "[Contorno] UCS: spec_D=" << spec_D << " m, spec_L=" << spec_L << " m, nW=" << nW << ", nL=" << nL << "\n";
        }
        if (nL < 1) { std::cerr << "nL = 0: el script YADE no generaría facetas con esta probeta\n"; return 1; }
        const double xs = BTS ? L_Tol : w_Tol;
        auto X = [&](int i) { return (ori_x + i / (double)nW) * xs * base_x; };
        auto Y = [&](int j) { return (ori_y + j / (double)nL) * w_Tol * base_y; };
        std::vector<Vec3> s0, s1, s2, i0, i1, i2;
        for (int i = 0; i < nW; ++i)
            for (int j = 0; j < nL; ++j) {
                for (int side = 0; side < 2; ++side) {
                    double zz = z_Tol * (side == 0 ? zMax : zMin);
                    Vec3 v1 = {X(i), Y(j), zz}, v2 = {X(i + 1), Y(j), zz}, v3 = {X(i), Y(j + 1), zz}, v4 = {X(i + 1), Y(j + 1), zz};
                    auto& a0 = side == 0 ? s0 : i0; auto& a1 = side == 0 ? s1 : i1; auto& a2 = side == 0 ? s2 : i2;
                    a0.push_back(v1); a1.push_back(v2); a2.push_back(v3);   // f1 = facet((v1, v2, v3))
                    a0.push_back(v2); a1.push_back(v3); a2.push_back(v4);   // f2 = facet((v2, v3, v4))
                }
            }
        n_plate = s0.size();
        f_v0 = s0; f_v1 = s1; f_v2 = s2;
        f_v0.insert(f_v0.end(), i0.begin(), i0.end()); f_v1.insert(f_v1.end(), i1.begin(), i1.end()); f_v2.insert(f_v2.end(), i2.begin(), i2.end());
    }
    const int n_facets = f_v0.size();
    std::vector<Vec3> f_pos(n_facets);
    {   // yade.utils.facet: centro = inscribedCircleCenter(v0, v1, v2)
        auto nrm = [](Vec3 u, Vec3 v) { return std::sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y) + (u.z - v.z) * (u.z - v.z)); };
        for (int k = 0; k < n_facets; ++k) {
            Vec3 a = f_v0[k], b = f_v1[k], c = f_v2[k];
            double l01 = nrm(b, a), l02 = nrm(c, a), l12 = nrm(c, b), s = l01 + l12 + l02;
            f_pos[k] = {a.x + ((c.x - a.x) * l01 + (b.x - a.x) * l02) / s, a.y + ((c.y - a.y) * l01 + (b.y - a.y) * l02) / s, a.z + ((c.z - a.z) * l01 + (b.z - a.z) * l02) / s};
        }
    }
    const std::vector<Vec3> f_pos_init = f_pos, f_v0_init = f_v0;
    if (n_facets) std::cout << "[Contorno] " << n_plate << " facetas por plato (" << n_facets << " en total)\n";

    // UTS: partículas de agarre (7,5 % superior e inferior de la longitud), blockedDOFs = 'z' y velocidad ±z_vel
    std::vector<int> top_ids, bot_ids;
    std::vector<unsigned char> blocked(n_particles, 0);
    std::vector<Vec3> vel0(n_particles, {0.0, 0.0, 0.0});
    double uts_spec_D = 0.0, uts_spec_L = 0.0;
    if (UTS) {
        uts_spec_D = .5 * (xMax - xMin + yMax - yMin); uts_spec_L = zMax - zMin;
        const double z_min_cutoff = zMin + 0.075 * uts_spec_L, z_max_cutoff = zMax - 0.075 * uts_spec_L;
        for (int i = 0; i < n_particles; ++i) {
            if (coords0[i].z >= z_max_cutoff) { top_ids.push_back(i); blocked[i] = 4; vel0[i] = {0.0, 0.0, z_vel}; }
            else if (coords0[i].z <= z_min_cutoff) { bot_ids.push_back(i); blocked[i] = 4; vel0[i] = {0.0, 0.0, -z_vel}; }
        }
        std::cout << "[Contorno] UTS: L=" << uts_spec_L << " m, D=" << uts_spec_D << " m, partículas de agarre: arriba=" << top_ids.size() << ", abajo=" << bot_ids.size() << "\n";
    }
    std::cout << "[Tiempo] dt = " << dtSafety << "·PWaveTimeStep = " << dt << " s, z_vel = " << z_vel << " m/s\n";

    // 5. NVRTC
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, CUDA_KERNEL_SOURCE, "ensayos_kernels.cu", 0, nullptr, nullptr));
    bool fp64_law = cfg.i("fp32") != 1;
    std::vector<std::string> opt_s = {"--gpu-architecture=compute_" + std::to_string(arch)};
    if (fp64_law) opt_s.push_back("-DDEM_FP64");
    opt_s.push_back("-DN_PLATE=" + std::to_string(n_plate));
    opt_s.push_back("-DDEM_DAMPING=0.01");
    if (cfg.i("fmad") == 0) opt_s.push_back("--fmad=false");
    std::vector<const char*> opts; for (auto& o : opt_s) opts.push_back(o.c_str());
    if (nvrtcCompileProgram(prog, (int)opts.size(), opts.data()) != 0) {
        size_t logSize; nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0'); nvrtcGetProgramLog(prog, &log[0]);
        std::cerr << "NVRTC Compile Log:\n" << log << std::endl; return 1;
    }
    size_t ptxSize; CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize); CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
    CUmodule module; CHECK_CU(cuModuleLoadData(&module, ptx.data()));
    CUfunction fn_ss, fn_fs, fn_newton, fn_bstats, fn_rss, fn_rfs, fn_maxdisp, fn_gather_p, fn_gather_f, fn_rotori;
    CHECK_CU(cuModuleGetFunction(&fn_ss, module, "kernel_sphere_sphere"));
    CHECK_CU(cuModuleGetFunction(&fn_fs, module, "kernel_facet_sphere"));
    CHECK_CU(cuModuleGetFunction(&fn_newton, module, "kernel_integrate_newton"));
    CHECK_CU(cuModuleGetFunction(&fn_bstats, module, "kernel_body_stats"));
    CHECK_CU(cuModuleGetFunction(&fn_rss, module, "kernel_reduce_ss"));
    CHECK_CU(cuModuleGetFunction(&fn_rfs, module, "kernel_reduce_fs"));
    CHECK_CU(cuModuleGetFunction(&fn_maxdisp, module, "kernel_max_disp"));
    CHECK_CU(cuModuleGetFunction(&fn_gather_p, module, "kernel_gather_particles"));
    CHECK_CU(cuModuleGetFunction(&fn_gather_f, module, "kernel_gather_facets"));
    CHECK_CU(cuModuleGetFunction(&fn_rotori, module, "kernel_rotate_ori"));
    std::cout << (fp64_law ? "[Precisión] ley de contacto en FP64\n" : "[Precisión] ley de contacto en FP32 (fp32=1)\n");

    // 6. Memoria en GPU
    auto dalloc = [&](size_t bytes) -> CUdeviceptr {
        CUdeviceptr d; size_t b = std::max<size_t>(bytes, 16);
        CHECK_CU(cuMemAlloc(&d, b)); CHECK_CU(cuMemsetD8(d, 0, b)); return d;
    };
    auto dcopy = [&](const void* src, size_t bytes) -> CUdeviceptr {
        CUdeviceptr d = dalloc(bytes);
        if (src && bytes) CHECK_CU(cuMemcpyHtoD(d, src, bytes));
        return d;
    };
    auto alloc_cap = [&](const void* src, size_t used_bytes, size_t cap_bytes) -> CUdeviceptr { CUdeviceptr d = dalloc(cap_bytes); if (src && used_bytes) CHECK_CU(cuMemcpyHtoD(d, src, used_bytes)); return d; };
    CUdeviceptr d_pos = dcopy(coords0.data(), n_particles * sizeof(Vec3));
    CUdeviceptr d_vel = dcopy(vel0.data(), n_particles * sizeof(Vec3));
    CUdeviceptr d_angVel = dalloc(n_particles * sizeof(Vec3));
    CUdeviceptr d_force = dalloc(n_particles * sizeof(Vec3));   // [FP64] fuerzas y momentos en double (YADE: ForceContainer en Real)
    CUdeviceptr d_torque = dalloc(n_particles * sizeof(Vec3));
    CUdeviceptr d_radius = dcopy(radii.data(), n_particles * sizeof(double));
    CUdeviceptr d_mass = dcopy(mass.data(), n_particles * sizeof(double));
    CUdeviceptr d_inertia = dcopy(inertia.data(), n_particles * sizeof(double));
    CUdeviceptr d_blocked = dcopy(blocked.data(), n_particles);
    CUdeviceptr d_edamp = dalloc(n_particles * sizeof(double));

    size_t cap_ss = (size_t)n_bonds + std::max<size_t>(200000, 2 * (size_t)n_bonds);
    size_t cap_fs = std::max<size_t>(50000, 8 * (size_t)n_facets);
    CUdeviceptr d_bp1 = alloc_cap(b_p1.data(), n_bonds * 4, cap_ss * 4);
    CUdeviceptr d_bp2 = alloc_cap(b_p2.data(), n_bonds * 4, cap_ss * 4);
    CUdeviceptr d_bkn = alloc_cap(b_kn.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bks = alloc_cap(b_ks.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bkb = alloc_cap(b_kb.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bcn = alloc_cap(b_cn.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bFnMax = alloc_cap(b_FnMax.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bFsMax = alloc_cap(b_FsMax.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bkseff = alloc_cap(b_kseff.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_binitD = alloc_cap(b_initD.data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bstate = alloc_cap(b_state.data(), n_bonds, cap_ss);
    CUdeviceptr d_bsfx = alloc_cap(nullptr, 0, cap_ss * 8), d_bsfy = alloc_cap(nullptr, 0, cap_ss * 8), d_bsfz = alloc_cap(nullptr, 0, cap_ss * 8);
    CUdeviceptr d_bmbx = alloc_cap(nullptr, 0, cap_ss * 8), d_bmby = alloc_cap(nullptr, 0, cap_ss * 8), d_bmbz = alloc_cap(nullptr, 0, cap_ss * 8);
    std::vector<double> pn0[3];
    for (int c = 0; c < 3; ++c) pn0[c].resize(n_bonds);
    for (int k = 0; k < n_bonds; ++k) { pn0[0][k] = b_prevNormal[k].x; pn0[1][k] = b_prevNormal[k].y; pn0[2][k] = b_prevNormal[k].z; }
    CUdeviceptr d_bpnx = alloc_cap(pn0[0].data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bpny = alloc_cap(pn0[1].data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_bpnz = alloc_cap(pn0[2].data(), n_bonds * 8, cap_ss * 8);
    CUdeviceptr d_ben = alloc_cap(nullptr, 0, cap_ss * 72);

    CUdeviceptr d_fv0 = dcopy(f_v0.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fv1 = dcopy(f_v1.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fv2 = dcopy(f_v2.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fpos = dcopy(f_pos.data(), n_facets * sizeof(Vec3));
    std::vector<Vec3> f_vel_h(n_facets, {0.0, 0.0, 0.0});
    if (UCS) for (int k = 0; k < n_facets; ++k) f_vel_h[k] = {0.0, 0.0, k < n_plate ? -z_vel : z_vel};   // velocidad impuesta desde el inicio
    CUdeviceptr d_fvel = dcopy(f_vel_h.data(), n_facets * sizeof(Vec3));
    CUdeviceptr d_fforce = dalloc(n_facets * sizeof(Vec3));
    CUdeviceptr d_fedamp = dalloc(n_facets * sizeof(double));

    int n_cand_fs = 0;
    CUdeviceptr d_fsf = dalloc(cap_fs * 4), d_fss = dalloc(cap_fs * 4), d_fskn = dalloc(cap_fs * 8), d_fsks = dalloc(cap_fs * 8), d_fscn = dalloc(cap_fs * 8);
    CUdeviceptr d_fsactive = dalloc(cap_fs * 4), d_fsinitD = dalloc(cap_fs * 8), d_fsshear = dalloc(cap_fs * sizeof(Vec3)), d_fsprevN = dalloc(cap_fs * sizeof(Vec3));
    CUdeviceptr d_fsen = dalloc(cap_fs * 32);

    CUdeviceptr d_nb_tens = dalloc(4), d_nb_shear = dalloc(4);
    const int CRK_CAP = 1 << 18;
    CUdeviceptr d_crk = dalloc((size_t)CRK_CAP * 11 * 8), d_crk_n = dalloc(4);
    int crk_cap = CRK_CAP;

    // =====================================================================================
    // [CONTACTOS] Detección dinámica (lista de Verlet en el host), como triaxial_cuda_native; la historia
    // (initD, estado, corte, momento, normal previa y energías) se transfiere por clave de par.
    // =====================================================================================
    int n_ss = n_bonds;
    const int nA = n_bonds;
    CUdeviceptr d_pfout = 0, d_fsout = 0, d_ss_start = 0, d_ss_idx = 0, d_fsp_start = 0, d_fsp_idx = 0, d_ff_start = 0, d_ff_idx = 0;
    size_t pf_cap = 0, fsout_cap = 0, ssidx_cap = 0, fspidx_cap = 0, ffidx_cap = 0;
    const double skin = cfg.d("skin_factor") * min_r;
    std::vector<int> h_bp1 = b_p1, h_bp2 = b_p2;
    std::vector<int> h_fsf, h_fss;
    auto pkey = [](int a, int b) -> uint64_t { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    std::unordered_set<uint64_t> bonded_keys;
    for (auto& pr : bonded_set) bonded_keys.insert(pkey(pr.first, pr.second));
    CUdeviceptr d_pos_ref = dalloc(n_particles * sizeof(Vec3));
    CUdeviceptr d_fv0_ref = dalloc(n_facets * sizeof(Vec3));
    CUdeviceptr d_maxdisp = dalloc(2 * sizeof(unsigned long long));
    const int verify_every = cfg.i("contact_verify");
    long long n_verifications = 0, n_rebuilds = 0, missed_ss = 0, missed_fs = 0;
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

    auto grow = [&](std::vector<std::pair<CUdeviceptr*, size_t>> arrs, size_t keep, size_t newcap) {
        for (auto& a : arrs) {
            CUdeviceptr nd = dalloc(newcap * a.second);
            if (keep) CHECK_CU(cuMemcpyDtoD(nd, *a.first, keep * a.second));
            CHECK_CU(cuMemFree(*a.first));
            *a.first = nd;
        }
    };
    auto build_csr = [&]() {
        auto realloc_if = [&](CUdeviceptr& p, size_t& cap, size_t need) {
            if (need <= cap && p) return;
            if (p) CHECK_CU(cuMemFree(p));
            cap = std::max<size_t>(need, 16);
            CHECK_CU(cuMemAlloc(&p, cap));
        };
        realloc_if(d_pfout, pf_cap, cap_ss * 9 * sizeof(double));
        realloc_if(d_fsout, fsout_cap, cap_fs * 6 * sizeof(double));
        std::vector<int> st(n_particles + 1, 0), idx(2 * (size_t)n_ss);
        for (int k = 0; k < n_ss; ++k) { st[h_bp1[k] + 1]++; st[h_bp2[k] + 1]++; }
        for (int i = 0; i < n_particles; ++i) st[i + 1] += st[i];
        { std::vector<int> f(st.begin(), st.end() - 1);
          for (int k = 0; k < n_ss; ++k) { idx[f[h_bp1[k]]++] = 2 * k; idx[f[h_bp2[k]]++] = 2 * k + 1; } }
        std::vector<int> pst(n_particles + 1, 0), pidx(n_cand_fs), fst(n_facets + 1, 0), fidx(n_cand_fs);
        for (int k = 0; k < n_cand_fs; ++k) { pst[h_fss[k] + 1]++; fst[h_fsf[k] + 1]++; }
        for (int i = 0; i < n_particles; ++i) pst[i + 1] += pst[i];
        for (int f = 0; f < n_facets; ++f) fst[f + 1] += fst[f];
        { std::vector<int> a(pst.begin(), pst.end() - 1), b(fst.begin(), fst.end() - 1);
          for (int k = 0; k < n_cand_fs; ++k) { pidx[a[h_fss[k]]++] = k; fidx[b[h_fsf[k]]++] = k; } }
        if (!d_ss_start) { d_ss_start = dalloc((n_particles + 1) * 4); d_fsp_start = dalloc((n_particles + 1) * 4); d_ff_start = dalloc((n_facets + 1) * 4); }
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
        if (n_facets) {
            CHECK_CU(cuMemcpyDtoH(V0.data(), d_fv0, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(V1.data(), d_fv1, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(V2.data(), d_fv2, n_facets * sizeof(Vec3)));
        }
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
        std::vector<double> o_initD(nB_old), o_sf[3], o_mb[3], o_pn[3], o_en((size_t)nB_old * 9);
        CUdeviceptr* pnp[3] = {&d_bpnx, &d_bpny, &d_bpnz};
        CUdeviceptr* sfp[3] = {&d_bsfx, &d_bsfy, &d_bsfz};
        CUdeviceptr* mbp[3] = {&d_bmbx, &d_bmby, &d_bmbz};
        std::vector<unsigned char> o_st(nB_old);
        if (nB_old > 0) {
            CHECK_CU(cuMemcpyDtoH(o_initD.data(), d_binitD + nA * 8, nB_old * 8));
            CHECK_CU(cuMemcpyDtoH(o_st.data(), d_bstate + nA, nB_old));
            CHECK_CU(cuMemcpyDtoH(o_en.data(), d_ben + (size_t)nA * 72, (size_t)nB_old * 72));
            for (int c = 0; c < 3; ++c) {
                o_sf[c].resize(nB_old); o_mb[c].resize(nB_old); o_pn[c].resize(nB_old);
                CHECK_CU(cuMemcpyDtoH(o_sf[c].data(), *sfp[c] + nA * 8, nB_old * 8));
                CHECK_CU(cuMemcpyDtoH(o_mb[c].data(), *mbp[c] + nA * 8, nB_old * 8));
                CHECK_CU(cuMemcpyDtoH(o_pn[c].data(), *pnp[c] + nA * 8, nB_old * 8));
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
                    np1.push_back(i); np2.push_back(j);
                }
            }
        }
        int nB = np1.size();
        if (apply) {
            if ((size_t)(nA + nB) > cap_ss) {
                size_t nc_ = (size_t)(nA + nB) * 3 / 2;
                grow({{&d_bp1, 4}, {&d_bp2, 4}, {&d_bkn, 8}, {&d_bks, 8}, {&d_bkb, 8}, {&d_bcn, 8}, {&d_bFnMax, 8}, {&d_bFsMax, 8},
                      {&d_bpnx, 8}, {&d_bpny, 8}, {&d_bpnz, 8}, {&d_bkseff, 8}, {&d_binitD, 8}, {&d_bstate, 1}, {&d_bsfx, 8}, {&d_bsfy, 8}, {&d_bsfz, 8},
                      {&d_bmbx, 8}, {&d_bmby, 8}, {&d_bmbz, 8}, {&d_ben, 72}}, nA, nc_);
                cap_ss = nc_;
            }
            std::vector<double> nkn(nB), nks(nB), nkb(nB), ncn(nB), nFn(nB, 0.0), nFs(nB, 0.0), nke(nB), npn[3], ninitD(nB, 0.0), nsf[3], nmb[3], nen((size_t)nB * 9, 0.0);
            std::vector<unsigned char> nst(nB, 0);
            for (int c = 0; c < 3; ++c) { nsf[c].assign(nB, 0.0); nmb[c].assign(nB, 0.0); npn[c].assign(nB, 0.0); }
            for (int b = 0; b < nB; ++b) {
                int i = np1[b], j = np2[b];
                double R1 = radii[i], R2 = radii[j];
                double Kn1 = 2 * young * R1, Kn2 = 2 * young * R2, kn_val = (Kn1 * Kn2) / (Kn1 + Kn2);
                double Ks1 = Kn1 / kn_ks_ratio, Ks2 = Kn2 / kn_ks_ratio, ks_val = (Ks1 * Ks2) / (Ks1 + Ks2);
                double kr1 = 0.5 * young * (R1 * R1 * R1), kr2 = 0.5 * young * (R2 * R2 * R2), kb_val = bendingIn > 0 ? (kr1 * kr2) / (kr1 + kr2) * bendingIn : 0.0;
                double meff = (mass[i] * mass[j]) / (mass[i] + mass[j]);
                nkn[b] = kn_val; nks[b] = ks_val; nkb[b] = kb_val; ncn[b] = viscAlpha * 2.0 * std::sqrt(kn_val * meff);
                nke[b] = ks_val;   // contacto nuevo: nunca cohesivo (FnMax = FsMax = 0)
                auto it_o = old_ss.find(pkey(i, j));
                if (it_o != old_ss.end()) {
                    int o = it_o->second;
                    ninitD[b] = o_initD[o]; nst[b] = o_st[o];
                    for (int c = 0; c < 3; ++c) { nsf[c][b] = o_sf[c][o]; nmb[c][b] = o_mb[c][o]; npn[c][b] = o_pn[c][o]; }
                    for (int c = 0; c < 9; ++c) nen[(size_t)b * 9 + c] = o_en[(size_t)o * 9 + c];
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
                CHECK_CU(cuMemcpyHtoD(d_bkseff + nA * 8, nke.data(), nB * 8));
                CHECK_CU(cuMemcpyHtoD(d_binitD + nA * 8, ninitD.data(), nB * 8));
                CHECK_CU(cuMemcpyHtoD(d_bstate + nA, nst.data(), nB));
                CHECK_CU(cuMemcpyHtoD(d_ben + (size_t)nA * 72, nen.data(), (size_t)nB * 72));
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
        std::vector<double> o_fid(nF_old), o_fen((size_t)nF_old * 4);
        std::vector<Vec3> o_fsh(nF_old), o_fpn(nF_old);
        if (nF_old > 0) {
            CHECK_CU(cuMemcpyDtoH(o_act.data(), d_fsactive, nF_old * 4));
            CHECK_CU(cuMemcpyDtoH(o_fid.data(), d_fsinitD, nF_old * 8));
            CHECK_CU(cuMemcpyDtoH(o_fsh.data(), d_fsshear, nF_old * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(o_fpn.data(), d_fsprevN, nF_old * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(o_fen.data(), d_fsen, (size_t)nF_old * 32));
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
                  {&d_fsshear, sizeof(Vec3)}, {&d_fsprevN, sizeof(Vec3)}, {&d_fsen, 32}}, 0, nc_);
            cap_fs = nc_;
        }
        std::vector<double> fkn(nF), fks(nF), fcn(nF), fid(nF, 0.0), fen((size_t)nF * 4, 0.0);
        std::vector<int> fact(nF, 0);
        std::vector<Vec3> fsh(nF, {0.0, 0.0, 0.0}), fpn(nF, {0.0, 0.0, 1.0});
        for (int k = 0; k < nF; ++k) {
            int si = nfs[k];
            double r2 = radii[si], r1 = 2.0 * r2;     // Ig2_Facet_Sphere_ScGeom: radius1 = 2·r
            double kn1 = 2.0 * young_wall * r1, kn2 = 2.0 * young * r2, kn_val = kn1 * kn2 / (kn1 + kn2);
            double ks1 = kn1 / kn_ks_wall_ratio, ks2 = kn2 / kn_ks_ratio;
            fkn[k] = kn_val; fks[k] = ks1 * ks2 / (ks1 + ks2); fcn[k] = viscAlpha * 2.0 * std::sqrt(kn_val * mass[si]);
            auto it_o = old_fs.find(pkey(nff[k], si));
            if (it_o != old_fs.end()) {
                int o = it_o->second; fact[k] = o_act[o]; fid[k] = o_fid[o]; fsh[k] = o_fsh[o]; fpn[k] = o_fpn[o];
                for (int c = 0; c < 4; ++c) fen[(size_t)k * 4 + c] = o_fen[(size_t)o * 4 + c];
            }
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
            CHECK_CU(cuMemcpyHtoD(d_fsen, fen.data(), (size_t)nF * 32));
        }
        n_cand_fs = nF;
        h_fsf = nff; h_fss = nfs;
        CHECK_CU(cuMemcpyDtoD(d_pos_ref, d_pos, n_particles * sizeof(Vec3)));
        if (n_facets) CHECK_CU(cuMemcpyDtoD(d_fv0_ref, d_fv0, n_facets * sizeof(Vec3)));
        n_rebuilds++;
        build_csr();
    };
    double last_disp = 0.0;
    auto check_rebuild = [&]() {
        CHECK_CU(cuMemsetD8(d_maxdisp, 0, 2 * sizeof(unsigned long long)));
        int nf = n_facets, np = n_particles;
        void* a[] = {&np, &nf, &d_pos, &d_pos_ref, &d_fv0, &d_fv0_ref, &d_maxdisp};
        CHECK_CU(cuLaunchKernel(fn_maxdisp, (std::max(n_particles, n_facets) + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        double md[2];
        CHECK_CU(cuMemcpyDtoH(md, d_maxdisp, sizeof(md)));
        double need = std::max(2.0 * md[0], md[0] + md[1]);
        double growth = std::max(0.0, need - last_disp);
        last_disp = need;
        if (need + 2.0 * growth > skin) { rebuild_contacts(true, true); last_disp = 0.0; }
    };
    rebuild_contacts(false, true);
    std::cout << "[Contactos] detección dinámica: skin=" << skin * 1e3 << " mm, pares esfera-esfera no enlazados=" << (n_ss - nA)
              << ", pares esfera-faceta=" << n_cand_fs << "\n";


    // Paso 0 (O.step() con dt=0): registra los contactos iniciales esfera-faceta con su initD
    double dt_zero = 0.0;
    void* args_fs[] = {&n_cand_fs, (void*)&dt, (void*)&mu, &d_pos, &d_vel, &d_angVel, &d_radius,
                       &d_fv0, &d_fv1, &d_fv2, &d_fvel, &d_fsf, &d_fss, &d_fskn, &d_fsks, &d_fscn,
                       &d_fsactive, &d_fsinitD, &d_fsshear, &d_fsprevN, &d_fsout, &d_fsen};
    if (n_cand_fs > 0) {
        void* args_fs0[22]; std::copy(args_fs, args_fs + 22, args_fs0); args_fs0[1] = &dt_zero;
        CHECK_CU(cuLaunchKernel(fn_fs, (n_cand_fs + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_fs0, nullptr));
    }
    CHECK_CU(cuCtxSynchronize());

    int iter_now = 0, init_pass = 1;
    double time_now = 0.0;
    void* args_ss[] = {&n_ss, (void*)&dt, (void*)&tgPhi, (void*)&mu, (void*)&rollFric,
                       &d_pos, &d_vel, &d_angVel, &d_radius, &d_mass, &d_inertia, &d_bp1, &d_bp2, &d_bkn, &d_bks, &d_bkb, &d_bcn,
                       &d_bFnMax, &d_bFsMax, &d_bkseff, &d_binitD, &d_bstate,
                       &d_bsfx, &d_bsfy, &d_bsfz, &d_bmbx, &d_bmby, &d_bmbz, &d_bpnx, &d_bpny, &d_bpnz,
                       &d_nb_tens, &d_nb_shear, &d_pfout, &d_ben, &d_crk, &d_crk_n, &crk_cap, &iter_now, &time_now, &init_pass};
    // Iteración 0 (isFresh): initD y normal inicial de los enlaces calculados en la GPU
    CHECK_CU(cuLaunchKernel(fn_ss, (n_ss + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_ss, nullptr));
    CHECK_CU(cuCtxSynchronize());
    init_pass = 0;
    int np_ = n_particles, nf_ = n_facets;
    void* args_newton[] = {&np_, &nf_, (void*)&dt, &d_mass, &d_inertia, &d_force, &d_torque,
                           &d_pos, &d_vel, &d_angVel, &d_fvel, &d_fpos, &d_fv0, &d_fv1, &d_fv2, &d_blocked, &d_fforce, &d_edamp, &d_fedamp};

    // ---- archivos de salida (mismos nombres y columnas que los scripts) ----
    FILE* f_facet = fopen((out_dir + "/facet_force.txt").c_str(), "w");
    FILE* f_energy = fopen((out_dir + "/sim_energy.txt").c_str(), "w");
    FILE* f_strain = fopen((out_dir + "/strain_data.txt").c_str(), "w");
    if (!f_facet || !f_energy || !f_strain) { std::cerr << "No se pudo escribir en " << out_dir << "\n"; return 1; }
    {
        std::ofstream m(out_dir + "/micro_data.txt");
        m << "============== INITIAL CONFIGURATION ==============\n"
          << "Young:           " << py_repr(young) << "\n" << "KnKsRatio:       " << py_repr(kn_ks_ratio) << "\n"
          << "TensileStrength: " << py_repr(tensile) << "\n" << "Cohesion:        " << py_repr(cohesion) << "\n"
          << "tgPhi [cohesive]:" << py_repr(tgPhi) << "\n" << "mu [post-break]: " << py_repr(mu) << "\n"
          << "CoupledShearBend:" << (coupledShearBending ? "True" : "False") << "\n" << "Density:         " << py_repr(density) << "\n"
          << "Identifier:      " << identifier << "\n" << "InteractionRadius:" << py_repr(intRadius) << "\n"
          << "InputFile:       " << input_file << "\n" << "BendingInfluence:" << py_repr(bendingIn) << "\n"
          << "RollingFriction: " << py_repr(rollFric) << "\n" << "ViscousAlpha:    " << py_repr(viscAlpha) << "\n"
          << "WeibullShape:    " << py_repr(weibShape) << "\n" << "WeibullScale:    " << py_repr(weibScale) << "\n"
          << "==================================================\n";
    }
    if (BTS || UCS) {   // bondCreation(): bonds_JCFP.txt con las interacciones esfera-esfera
        std::ofstream b(out_dir + "/bonds_JCFP.txt");
        for (auto& pr : raw_b_pre) b << pr.first << " " << pr.second << "\n";
    }
    if (UCS || UTS) std::ofstream(out_dir + "/bond_data.txt");
    if (UTS) {
        std::ofstream(out_dir + "/broken_bonds.txt");
        // exportGeometryToGeo(cN) (YADE lo escribe en la carpeta de trabajo; aquí va a la salida)
        int cN = (int)std::nearbyint(2.0 * n_bonds / n_particles);
        std::ofstream g(out_dir + "/prob_cn_" + std::to_string(cN) + ".geo");
        g << "LSMGeometry 1.2\nBoundingBox -0.1 -0.1 -0.1 0.1 0.1 0.1\nPeriodicBoundaries 0 0 0\nDimension 3D\nBeginParticles\nSimple\n" << n_particles << "\n";
        char buf[160];
        for (int i = 0; i < n_particles; ++i) { snprintf(buf, sizeof(buf), "%.8f %.8f %.8f %.8f 0 0\n", coords0[i].x, coords0[i].y, coords0[i].z, radii[i]); g << buf; }
        g << "EndParticles\nBeginBonds\n" << raw_b_pre.size() << "\n";
        for (auto& pr : raw_b_pre) g << pr.first << " " << pr.second << "\n";
        g << "EndBonds\n";
    }
    {
        std::ofstream c(out_dir + "/caso_usado.cfg");
        c << "# " << TNAME << " (CUDA). Línea de comandos equivalente:\n# ./ensayos_cuda_native";
        for (auto& a : pos_args) c << " " << a;
        c << "\n";
        for (auto& k : CFG_KEYS) c << std::left << std::setw(17) << k.key << " = " << cfg.v.at(k.key) << "    # " << cfg.src.at(k.key) << "\n";
        c << "# efectivos: z_vel = " << z_vel << ", dt_safety = " << dtSafety << ", dt = " << std::setprecision(17) << dt << "\n";
    }
    // recordCracks: cracks_<identifier>.txt (YADE lo crea en la carpeta de trabajo y lo mueve a la salida al detectar la fractura;
    // aquí todas las fisuras van directamente a la carpeta de salida)
    std::ofstream f_crk(out_dir + "/cracks_" + identifier + ".txt");
    bool crk_header = false;
    long long n_crk_written = 0, n_crk_lost = 0;
    auto flush_cracks = [&]() {
        int n = 0;
        CHECK_CU(cuMemcpyDtoH(&n, d_crk_n, 4));
        if (n == 0) return;
        int m = std::min(n, crk_cap);
        if (n > crk_cap) n_crk_lost += n - crk_cap;
        std::vector<double> c((size_t)m * 11);
        CHECK_CU(cuMemcpyDtoH(c.data(), d_crk, c.size() * 8));
        CHECK_CU(cuMemsetD8(d_crk_n, 0, 4));
        std::vector<int> ord(m);
        for (int q = 0; q < m; ++q) ord[q] = q;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return c[11 * a] != c[11 * b] ? c[11 * a] < c[11 * b] : c[11 * a + 1] < c[11 * b + 1]; });
        if (!crk_header) { f_crk << "iter time x y z failureMode energy area nx ny nz" << std::endl; crk_header = true; }
        for (int q : ord) {
            const double* e = &c[11 * (size_t)q];
            int k = (int)e[1];
            f_crk << (long long)e[0] << " " << e[10] << " " << e[4] << " " << e[5] << " " << e[6] << " " << (int)e[2] << " " << e[3] << " "
                  << (k < nA ? b_area[k] : 0.0) << " " << e[7] << " " << e[8] << " " << e[9] << "\n";
        }
        n_crk_written += m;
    };

    // =====================================================================================
    // [VTK] exp_vtk() del script (VTKExporter('output_tribo_<id>/data'): data-spheres/-intrs/-facets-NNNNNNNN.vtk)
    // =====================================================================================
    const bool vtk_ascii = cfg.i("vtk_ascii") == 1;
    const bool vtk_intrs = cfg.i("vtk_intrs") != 0;
    int vtk_index = 0;
    CUdeviceptr d_ori = 0;
    if (vtk_on) {
        std::vector<double> q0(4 * (size_t)n_particles, 0.0);
        for (int i = 0; i < n_particles; ++i) q0[4 * i] = 1.0;
        d_ori = dcopy(q0.data(), q0.size() * sizeof(double));
    }
    double vtk_seconds = 0.0;
    auto export_vtk = [&]() {
        if (!vtk_on) return;
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CU(cuCtxSynchronize());
        char num[16]; snprintf(num, sizeof(num), "%08d", vtk_index++);
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
        {
            W w{fopen((out_dir + "/data-spheres-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
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
            fprintf(w.f, "SCALARS rotAngle %s 1\nLOOKUP_TABLE default\n", w.tp());
            for (int i = 0; i < n_particles; ++i) w.flt(2.0 * std::atan2(std::sqrt(q[4*i+1]*q[4*i+1] + q[4*i+2]*q[4*i+2] + q[4*i+3]*q[4*i+3]), std::fabs(q[4*i])));
            w.end();
            fclose(w.f);
        }
        std::vector<Vec3> FV0(n_facets), FV1(n_facets), FV2(n_facets), FF(n_facets), FVel(n_facets);
        if (n_facets) {
            CHECK_CU(cuMemcpyDtoH(FV0.data(), d_fv0, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(FV1.data(), d_fv1, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(FV2.data(), d_fv2, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(FF.data(), d_fforce, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(FVel.data(), d_fvel, n_facets * sizeof(Vec3)));
            W w{fopen((out_dir + "/data-facets-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
            w.hdr("comment");
            fprintf(w.f, "POINTS %d %s\n", 3 * n_facets, w.tp());
            for (int k = 0; k < n_facets; ++k) { w.vec(FV0[k].x, FV0[k].y, FV0[k].z); w.vec(FV1[k].x, FV1[k].y, FV1[k].z); w.vec(FV2[k].x, FV2[k].y, FV2[k].z); }
            w.end();
            fprintf(w.f, "POLYGONS %d %d\n", n_facets, 4 * n_facets);
            for (int k = 0; k < n_facets; ++k) { w.i32(3); w.i32(3 * k); w.i32(3 * k + 1); w.i32(3 * k + 2); w.nl(); }
            w.end();
            fprintf(w.f, "CELL_DATA %d\nVECTORS pos %s\n", n_facets, w.tp());
            for (int k = 0; k < n_facets; ++k) w.vec(f_pos_init[k].x + FV0[k].x - f_v0_init[k].x, f_pos_init[k].y + FV0[k].y - f_v0_init[k].y, f_pos_init[k].z + FV0[k].z - f_v0_init[k].z);
            w.end();
            fprintf(w.f, "VECTORS displ %s\n", w.tp());
            for (int k = 0; k < n_facets; ++k) w.vec(FV0[k].x - f_v0_init[k].x, FV0[k].y - f_v0_init[k].y, FV0[k].z - f_v0_init[k].z);
            w.end();
            fprintf(w.f, "VECTORS forceVec %s\n", w.tp()); for (auto& v : FF) w.vec(v.x, v.y, v.z); w.end();
            fclose(w.f);
        }
        if (vtk_intrs) {   // interacciones reales: esfera-esfera y faceta-esfera (centro de la faceta)
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
            std::vector<int> fact(n_cand_fs); std::vector<double> fkn(n_cand_fs), fcn(n_cand_fs), fid(n_cand_fs); std::vector<Vec3> fsh(n_cand_fs);
            if (n_cand_fs > 0) {
                CHECK_CU(cuMemcpyDtoH(fact.data(), d_fsactive, n_cand_fs * 4)); CHECK_CU(cuMemcpyDtoH(fkn.data(), d_fskn, n_cand_fs * 8));
                CHECK_CU(cuMemcpyDtoH(fcn.data(), d_fscn, n_cand_fs * 8)); CHECK_CU(cuMemcpyDtoH(fid.data(), d_fsinitD, n_cand_fs * 8));
                CHECK_CU(cuMemcpyDtoH(fsh.data(), d_fsshear, n_cand_fs * sizeof(Vec3)));
            }
            std::vector<int> fids;
            for (int k = 0; k < n_cand_fs; ++k) if (fact[k]) fids.push_back(k);
            int n = nssi + (int)fids.size();
            auto fcen = [&](int fi) -> Vec3 { return {f_pos_init[fi].x + FV0[fi].x - f_v0_init[fi].x, f_pos_init[fi].y + FV0[fi].y - f_v0_init[fi].y, f_pos_init[fi].z + FV0[fi].z - f_v0_init[fi].z}; };
            W w{fopen((out_dir + "/data-intrs-" + num + ".vtk").c_str(), "wb"), vtk_ascii};
            w.hdr("comment");
            fprintf(w.f, "POINTS %d %s\n", 2 * n, w.tp());
            for (int k : ids) { const Vec3& a = P[h_bp1[k]]; const Vec3& b = P[h_bp2[k]]; w.vec(a.x, a.y, a.z); w.vec(b.x, b.y, b.z); }
            for (int k : fids) { Vec3 a = fcen(h_fsf[k]); const Vec3& b = P[h_fss[k]]; w.vec(a.x, a.y, a.z); w.vec(b.x, b.y, b.z); }
            w.end();
            fprintf(w.f, "LINES %d %d\n", n, 3 * n);
            for (int q = 0; q < n; ++q) { w.i32(2); w.i32(2 * q); w.i32(2 * q + 1); w.nl(); }
            w.end();
            std::vector<double> Fn(n), Ft(n), Mm(n), Fvm(n), ov(n), coh(n), brk(n); std::vector<Vec3> Fv(n);
            for (int q = nssi; q < n; ++q) {
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
                Fn[q] = -kn[k] * (pen - initD[k]);
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

    // =====================================================================================
    // extract_Data() del script: facet_force.txt, sim_energy.txt y strain_data.txt (una fila por llamada)
    // =====================================================================================
    double bs_sumF = 0.0, bs_ek = 0.0;          // estadísticas de cuerpos del último paso medido (antes de Newton)
    std::vector<double> h_force(3 * (size_t)n_particles);
    CUdeviceptr d_part_b = 0, d_part_ss = 0, d_part_fs = 0; size_t cap_pb = 0, cap_pss = 0, cap_pfs = 0;
    auto ensure = [&](CUdeviceptr& p, size_t& cap, size_t need) { if (need <= cap && p) return; if (p) CHECK_CU(cuMemFree(p)); cap = std::max<size_t>(need, 64); p = dalloc(cap); };
    auto body_stats = [&]() {
        int nb = (n_particles + 255) / 256;
        ensure(d_part_b, cap_pb, (size_t)nb * 16);
        void* a[] = {&np_, &d_mass, &d_inertia, &d_vel, &d_angVel, &d_force, &d_part_b};
        CHECK_CU(cuLaunchKernel(fn_bstats, nb, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        std::vector<double> p(2 * (size_t)nb);
        CHECK_CU(cuMemcpyDtoH(p.data(), d_part_b, p.size() * 8));
        bs_sumF = 0.0; bs_ek = 0.0;
        for (int b = 0; b < nb; ++b) { bs_sumF += p[2 * b]; bs_ek += p[2 * b + 1]; }
        if (UTS) CHECK_CU(cuMemcpyDtoH(h_force.data(), d_force, h_force.size() * 8));
    };
    double fz_Max = 1.0, fz_Max_stress = 0.0; int fz_Max_iter = 0; double last_unb = 0.0;
    std::vector<Vec3> P(n_particles);
    const double bts_R = .5 * (zMax - zMin + yMax - yMin), bts_L = xMax - xMin;   // specimenRadius / specimenLength de extract_Data (BTS y UCS)
    auto extract = [&](long long iter_lbl, double t_lbl, int reps) -> double {
        CHECK_CU(cuCtxSynchronize());
        // interacciones reales
        double ss[11] = {0}, fs[6] = {0};
        { int nb = (n_ss + 255) / 256; ensure(d_part_ss, cap_pss, (size_t)nb * 88);
          void* a[] = {&n_ss, &d_bstate, &d_ben, &d_bsfx, &d_bsfy, &d_bsfz, &d_part_ss};
          if (nb > 0) { CHECK_CU(cuLaunchKernel(fn_rss, nb, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
            std::vector<double> p(11 * (size_t)nb); CHECK_CU(cuMemcpyDtoH(p.data(), d_part_ss, p.size() * 8));
            for (int b = 0; b < nb; ++b) for (int c = 0; c < 11; ++c) ss[c] += p[11 * (size_t)b + c]; } }
        std::vector<int> fact(n_cand_fs);
        if (n_cand_fs > 0) {
            int nb = (n_cand_fs + 255) / 256; ensure(d_part_fs, cap_pfs, (size_t)nb * 48);
            void* a[] = {&n_cand_fs, &d_fsactive, &d_fsen, &d_fsshear, &d_part_fs};
            CHECK_CU(cuLaunchKernel(fn_rfs, nb, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
            std::vector<double> p(6 * (size_t)nb); CHECK_CU(cuMemcpyDtoH(p.data(), d_part_fs, p.size() * 8));
            for (int b = 0; b < nb; ++b) for (int c = 0; c < 6; ++c) fs[c] += p[6 * (size_t)b + c];
            CHECK_CU(cuMemcpyDtoH(fact.data(), d_fsactive, n_cand_fs * 4));
        }
        const double un_force = (bs_sumF / n_particles) / ((ss[9] + fs[4]) / (ss[10] + fs[5]));
        last_unb = un_force;
        int n_tens = 0, n_shear = 0;
        CHECK_CU(cuMemcpyDtoH(&n_tens, d_nb_tens, 4));
        CHECK_CU(cuMemcpyDtoH(&n_shear, d_nb_shear, 4));
        CHECK_CU(cuMemcpyDtoH(P.data(), d_pos, n_particles * sizeof(Vec3)));
        double e_damp = 0.0;
        { std::vector<double> ed(n_particles); CHECK_CU(cuMemcpyDtoH(ed.data(), d_edamp, n_particles * 8)); for (double x : ed) e_damp += x;
          if (n_facets) { std::vector<double> fe(n_facets); CHECK_CU(cuMemcpyDtoH(fe.data(), d_fedamp, n_facets * 8)); for (double x : fe) e_damp += x; } }
        std::vector<Vec3> FF(n_facets), FP(n_facets);
        if (n_facets) { CHECK_CU(cuMemcpyDtoH(FF.data(), d_fforce, n_facets * sizeof(Vec3))); CHECK_CU(cuMemcpyDtoH(FP.data(), d_fpos, n_facets * sizeof(Vec3))); }

        char line[1024];
        double fz;
        std::string facet_line;
        if (BTS || UCS) {
            double f1x = 0, f2x = 0, f1y = 0, f2y = 0, f1z = 0, f2z = 0;
            for (int k = 0; k < n_plate; ++k) { f1x += FF[k].x; f1y += FF[k].y; f1z += FF[k].z; }
            for (int k = n_plate; k < 2 * n_plate; ++k) { f2x += FF[k].x; f2y += FF[k].y; f2z += FF[k].z; }
            f2z *= -1;
            fz = .5 * (f1z + f2z);
            const double wall_dspl = -2 * (FP[0].z - f_pos_init[0].z);
            std::vector<double> dmin, dmax;
            for (int k = 0; k < n_cand_fs; ++k) if (fact[k]) {
                const double d = P[h_fss[k]].z - coords0[h_fss[k]].z;
                if (d < 0.0) dmin.push_back(d); else if (d > 0.0) dmax.push_back(d);
            }
            const double s_min_disp = (dmin.empty() ? 0.0 : vmean(dmin)) * -1.0, s_max_disp = dmax.empty() ? 0.0 : vmean(dmax);
            const double stress = 2.0 * fz / (PI * bts_R * bts_L);
            snprintf(line, sizeof(line), "%g %g %g %g %g %g %g %g %g %g %g %g %g %d %d\n", t_lbl, (double)iter_lbl, wall_dspl, s_min_disp, s_max_disp, stress,
                     f1x, f2x, f1y, f2y, f1z, f2z, un_force, n_tens, n_shear);
            facet_line = line;
            if (fz > fz_Max) { fz_Max = fz; fz_Max_stress = stress; fz_Max_iter = (int)iter_lbl; }
        } else {
            double f1x = 0, f2x = 0, f1y = 0, f2y = 0, f1z = 0, f2z = 0, dt_ = 0, db = 0;
            for (int i : top_ids) { f1x += h_force[3 * i]; f1y += h_force[3 * i + 1]; f1z += h_force[3 * i + 2]; dt_ += P[i].z - coords0[i].z; }
            for (int i : bot_ids) { f2x += h_force[3 * i]; f2y += h_force[3 * i + 1]; f2z += h_force[3 * i + 2]; db += P[i].z - coords0[i].z; }
            fz = 0.5 * (std::fabs(f1z) + std::fabs(f2z));
            const double area = PI * std::pow(uts_spec_D / 2.0, 2);
            const double sigma = fz / area;
            if (fz > fz_Max) { fz_Max = fz; fz_Max_stress = sigma; fz_Max_iter = (int)iter_lbl; }
            const double disp_top = top_ids.empty() ? 0.0 : dt_ / top_ids.size(), disp_bot = bot_ids.empty() ? 0.0 : db / bot_ids.size();
            facet_line = py_repr(t_lbl) + " " + std::to_string(iter_lbl) + " 0.0 " + py_repr(std::fabs(disp_bot)) + " " + py_repr(std::fabs(disp_top)) + " " + py_repr(sigma)
                       + " " + py_repr(f1x) + " " + py_repr(f2x) + " " + py_repr(f1y) + " " + py_repr(f2y) + " " + py_repr(f1z) + " " + py_repr(f2z)
                       + " " + py_repr(un_force) + " " + std::to_string(n_tens) + " " + std::to_string(n_shear) + "\n";
        }
        // Format: Time Iter El_N El_S El_B Visc Plas_Tot Plas_Fric Plas_Roll Fracture KinFail Damp Kin Grav
        snprintf(line, sizeof(line), "%g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", t_lbl, (double)iter_lbl,
                 ss[0] + fs[0], ss[1] + fs[1], ss[2], ss[3] + fs[2], ss[4] + fs[3], ss[5] + fs[3], ss[6], ss[7], ss[8], e_damp, bs_ek, 0.0);
        std::string energy_line = line;

        // compute_poisson_inline()
        std::vector<Vec3> p0(n_particles), d(n_particles);
        for (int i = 0; i < n_particles; ++i) {
            d[i] = {P[i].x - coords0[i].x, P[i].y - coords0[i].y, P[i].z - coords0[i].z};
            p0[i] = {P[i].x - d[i].x, P[i].y - d[i].y, P[i].z - d[i].z};
        }
        std::string strain_line = py_repr(t_lbl) + " " + std::to_string(iter_lbl) + " ";
        if (BTS) {
            double cx = 0, cy = 0, cz = 0;
            for (auto& q : p0) { cx += q.x; cy += q.y; cz += q.z; }
            cx /= n_particles; cy /= n_particles; cz /= n_particles;
            double xmn = INFINITY, xmx = -INFINITY, zmn = INFINITY, zmx = -INFINITY;
            for (auto& q : p0) { xmn = std::min(xmn, q.x); xmx = std::max(xmx, q.x); zmn = std::min(zmn, q.z); zmx = std::max(zmx, q.z); }
            const double half = 0.10 * (xmx - xmn);
            std::vector<int> sl;
            for (int i = 0; i < n_particles; ++i) if (std::fabs(p0[i].x - cx) < half) sl.push_back(i);
            if (sl.empty()) strain_line += "None None None None None None 0 error:'nu_prev' None None None\n";
            else {
                const double D = zmx - zmn, pt_rad = 0.03 * D;
                const double locs[4] = {-0.250 * D, -0.125 * D, 0.125 * D, 0.250 * D};
                std::vector<double> vy, vuy, vz, vuz;
                for (double t : locs) {
                    std::vector<double> u;
                    for (int i : sl) if (std::fabs(p0[i].z - cz) < (0.05 * D) && std::fabs((p0[i].y - cy) - t) < pt_rad) u.push_back(d[i].y);
                    if (!u.empty()) { vy.push_back(t); vuy.push_back(vmean(u)); }
                }
                for (double t : locs) {
                    std::vector<double> u;
                    for (int i : sl) if (std::fabs(p0[i].y - cy) < (0.05 * D) && std::fabs((p0[i].z - cz) - t) < pt_rad) u.push_back(d[i].z);
                    if (!u.empty()) { vz.push_back(t); vuz.push_back(vmean(u)); }
                }
                const double eps_yy = vy.size() >= 3 ? ls_slope(vy, vuy) : 0.0;
                const double eps_axial = vz.size() >= 4 ? ls_slope(vz, vuz) : 0.0;
                const double eps_diam = eps_yy;
                double eps_d_p = 0.0, eps_ax_p = 0.0;
                auto pair_eps = [](const std::vector<double>& x, const std::vector<double>& u) {
                    int imn = std::min_element(x.begin(), x.end()) - x.begin(), imx = std::max_element(x.begin(), x.end()) - x.begin();
                    double L0 = x[imx] - x[imn];
                    if (L0 <= 0) return 0.0;
                    double L1 = (x[imx] + u[imx]) - (x[imn] + u[imn]);
                    return (L1 - L0) / L0;
                };
                if (vy.size() >= 2) eps_d_p = pair_eps(vy, vuy);
                if (vz.size() >= 2) eps_ax_p = pair_eps(vz, vuz);
                const double denom_p = (3 * eps_ax_p + eps_d_p);
                const double nu_p = std::fabs(denom_p) > 1e-20 ? -(3 * eps_d_p + eps_ax_p) / denom_p : 0.0;
                const double denom = (3 * eps_axial + eps_diam);
                const double nu = std::fabs(denom) > 1e-20 ? -(3 * eps_diam + eps_axial) / denom : 0.0;
                strain_line += py_repr(nu) + " " + py_repr(nu) + " " + py_repr(eps_diam) + " " + py_repr(eps_axial) + " 0.0 " + py_repr(eps_yy)
                             + " 0 ok " + py_repr(nu_p) + " " + py_repr(eps_d_p) + " " + py_repr(eps_ax_p) + "\n";
            }
        } else {
            double z0min = INFINITY, z0max = -INFINITY, R0 = 0.0;
            for (int i = 0; i < n_particles; ++i) {
                z0min = std::min(z0min, p0[i].z - radii[i]); z0max = std::max(z0max, p0[i].z + radii[i]);
                R0 = std::max(R0, std::sqrt(p0[i].x * p0[i].x + p0[i].y * p0[i].y) + radii[i]);
            }
            const double H0 = z0max - z0min;
            std::vector<int> mrad, max_;
            if (UCS) {   // test-2: corte radial 10 % central, axial 95 % central
                const double mid = (z0min + z0max) / 2.0;
                for (int i = 0; i < n_particles; ++i) {
                    const double r_eff0 = std::sqrt(p0[i].x * p0[i].x + p0[i].y * p0[i].y) + radii[i];
                    if (p0[i].z > mid - 0.05 * H0 && p0[i].z < mid + 0.05 * H0 && r_eff0 > 0.0 * R0) mrad.push_back(i);
                    if (p0[i].z > mid - 0.475 * H0 && p0[i].z < mid + 0.475 * H0) max_.push_back(i);
                }
            } else {     // test-3: corte central de h_ratio = 0,10 para ambas deformaciones
                const double zc_min = z0min + 0.5 * (1.0 - 0.10) * H0, zc_max = z0max - 0.5 * (1.0 - 0.10) * H0;
                for (int i = 0; i < n_particles; ++i) {
                    const double r_eff0 = std::sqrt(p0[i].x * p0[i].x + p0[i].y * p0[i].y) + radii[i];
                    if (p0[i].z > zc_min && p0[i].z < zc_max && r_eff0 > 0.0 * R0) mrad.push_back(i);
                }
                max_ = mrad;
            }
            auto radial = [&](const std::vector<int>& m, double rlim_ref, bool rel_slice, int& npairs) {
                npairs = 0;
                double cx = 0, cy = 0;
                for (int i : m) { cx += p0[i].x; cy += p0[i].y; }
                cx /= m.size(); cy /= m.size();
                std::vector<double> rv(m.size()), ur(m.size());
                double rmax = 0.0;
                for (size_t q = 0; q < m.size(); ++q) {
                    int i = m[q];
                    double dx = p0[i].x - cx, dy = p0[i].y - cy, r = std::sqrt(dx * dx + dy * dy);
                    rv[q] = r; ur[q] = (d[i].x * dx + d[i].y * dy) / (r == 0 ? 1.0 : r);
                    rmax = std::max(rmax, r);
                }
                const double lim = 0.1 * (rel_slice ? rmax : rlim_ref);
                std::vector<double> xr, yr;
                for (size_t q = 0; q < m.size(); ++q) if (rv[q] > lim) { xr.push_back(rv[q]); yr.push_back(ur[q]); }
                if (xr.size() < 10) return 0.0;
                npairs = xr.size();
                return ls_slope(xr, yr);
            };
            auto axial = [&](const std::vector<int>& m) {
                std::vector<double> x, y;
                for (int i : m) { x.push_back(p0[i].z); y.push_back(d[i].z); }
                return ls_slope(x, y);
            };
            if (UCS) {
                if (max_.size() < 10) strain_line += "0.0 0.0 0.0 0 no_slice_ax 0.0 0.0 0.0\n";
                else {
                    const double eps_axial = axial(max_);
                    int npairs = 0;
                    const double eps_diam = mrad.size() < 10 ? 0.0 : radial(mrad, R0, false, npairs);
                    const double nu = std::fabs(eps_axial) > 1e-16 ? -(eps_diam / eps_axial) : 0.0;
                    strain_line += py_repr(nu) + " " + py_repr(eps_diam) + " " + py_repr(eps_axial) + " " + std::to_string(npairs) + " ok 0.0 0.0 0.0\n";
                }
            } else {
                if (mrad.empty()) {   // 'no_slice': eps_axial de la regresión robusta (10-90 % de los centros)
                    double zmn = INFINITY, zmx = -INFINITY;
                    for (auto& q : p0) { zmn = std::min(zmn, q.z); zmx = std::max(zmx, q.z); }
                    std::vector<double> x, y;
                    for (int i = 0; i < n_particles; ++i) if (p0[i].z > zmn + 0.1 * (zmx - zmn) && p0[i].z < zmx - 0.1 * (zmx - zmn)) { x.push_back(p0[i].z); y.push_back(d[i].z); }
                    strain_line += "0.0 0.0 " + py_repr(ls_slope(x, y)) + " 0 no_slice 0.0 0.0 0.0\n";
                } else {
                    int npairs = 0;
                    const double eps_diam = radial(mrad, 0.0, true, npairs);
                    const double eps_axial = axial(mrad);
                    const double nu = std::fabs(eps_axial) > 1e-16 ? -(eps_diam / eps_axial) : 0.0;
                    // n_pairs = número de partículas con r > 0,1·r_max del corte (aunque sean menos de 10)
                    { double cx = 0, cy = 0; for (int i : mrad) { cx += p0[i].x; cy += p0[i].y; } cx /= mrad.size(); cy /= mrad.size();
                      double rmax = 0; std::vector<double> rv; for (int i : mrad) { double r = std::sqrt((p0[i].x - cx) * (p0[i].x - cx) + (p0[i].y - cy) * (p0[i].y - cy)); rv.push_back(r); rmax = std::max(rmax, r); }
                      npairs = 0; for (double r : rv) if (r > 0.1 * rmax) npairs++; }
                    strain_line += py_repr(nu) + " " + py_repr(eps_diam) + " " + py_repr(eps_axial) + " " + std::to_string(npairs) + " ok 0.0 0.0 0.0\n";
                }
            }
        }
        for (int r = 0; r < reps; ++r) { fputs(facet_line.c_str(), f_facet); fputs(energy_line.c_str(), f_energy); fputs(strain_line.c_str(), f_strain); }
        return fz;
    };

    // =====================================================================================
    // Bucle principal. Calendario de YADE: O.step() con dt=0 en las iteraciones 0 y 1 (bondCreation fija dt en la 1);
    // el paso dinámico de la iteración 'it' (scene->iter) empieza en it = 2. Los PyRunner corren dentro del paso, tras Newton,
    // con O.iter = it y O.time = tiempo al inicio del paso; el bucle 'while True' del script lee O.iter = it + 1 tras el paso.
    // =====================================================================================
    export_vtk();   // bondCreation() -> exp_vtk()
    double scene_time = dt;   // tras la iteración 1 (time += dt con el dt recién fijado)
    double fracture_time = -1.0; long long it_fracture = -1;
    std::string stop_reason = "pasos_max";
    long long it_last = 1;
    const long long trace_n = getenv("DEM_TRACE") ? atoll(getenv("DEM_TRACE")) : 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "Starts " << TNAME << " test (CUDA) ....  Load vel: " << z_vel << "  Dt: " << dt << "\n";
    for (long long it = 2; ; ++it) {
        if (it > pasos_max) { stop_reason = "pasos_max"; break; }
        const bool meas = ((it + 1) % 100 == 0) || (UTS && it % 1000 == 0);
        if (it % 10 == 0) { check_rebuild(); flush_cracks(); }
        if (verify_every > 0 && it % verify_every == 0) { rebuild_contacts(true, false); n_verifications++; }
        iter_now = (int)it; time_now = scene_time;
        CHECK_CU(cuLaunchKernel(fn_ss, (n_ss + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_ss, nullptr));
        if (n_cand_fs > 0) CHECK_CU(cuLaunchKernel(fn_fs, (n_cand_fs + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_fs, nullptr));
        { void* ag[] = {&np_, &d_ss_start, &d_ss_idx, &d_pfout, &d_fsp_start, &d_fsp_idx, &d_fsout, &d_force, &d_torque};
          CHECK_CU(cuLaunchKernel(fn_gather_p, (n_particles + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, ag, nullptr)); }
        if (n_facets > 0) { void* af[] = {&nf_, &d_ff_start, &d_ff_idx, &d_fsout, &d_fforce};
          CHECK_CU(cuLaunchKernel(fn_gather_f, (n_facets + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, af, nullptr)); }
        if (meas) body_stats();
        if (trace_n > 0 && it + 1 <= trace_n) { CHECK_CU(cuCtxSynchronize()); CHECK_CU(cuMemcpyDtoH(h_force.data(), d_force, h_force.size() * 8)); }
        CHECK_CU(cuLaunchKernel(fn_newton, (std::max(n_particles, n_facets) + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args_newton, nullptr));
        if (vtk_on) { void* ar[] = {&np_, (void*)&dt, &d_angVel, &d_ori}; CHECK_CU(cuLaunchKernel(fn_rotori, (n_particles + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, ar, nullptr)); }
        if (trace_n > 0 && it + 1 <= trace_n) {   // DEM_TRACE=N: traza por paso (depuración frente a YADE)
            CHECK_CU(cuCtxSynchronize());
            std::vector<Vec3> FF(n_facets), Pt(n_particles);
            if (n_facets) CHECK_CU(cuMemcpyDtoH(FF.data(), d_fforce, n_facets * sizeof(Vec3)));
            CHECK_CU(cuMemcpyDtoH(Pt.data(), d_pos, n_particles * sizeof(Vec3)));
            double f1z = 0, f2z = 0;
            for (int k = 0; k < n_plate; ++k) f1z += FF[k].z;
            for (int k = n_plate; k < 2 * n_plate; ++k) f2z += FF[k].z;
            for (int i : top_ids) f1z += h_force[3 * i + 2];
            for (int i : bot_ids) f2z += h_force[3 * i + 2];
            std::vector<Vec3> Vt(n_particles);
            CHECK_CU(cuMemcpyDtoH(Vt.data(), d_vel, n_particles * sizeof(Vec3)));
            printf("TRACE %lld %.17g %.17g", it + 1, f1z, f2z);
            std::vector<int> tids = {0, 100, 300, 700};
            if (UTS) tids = {top_ids[0], top_ids[5], 300, 700};
            for (int b : tids) if (b < n_particles) printf(" %.17g %.17g %.17g %.17g%s", Pt[b].x, Pt[b].z, (double)h_force[3 * b], (double)h_force[3 * b + 2],
                                                            UTS ? (" " + std::to_string(Vt[b].x)).c_str() : "");
            printf("\n");
            if (it + 1 == 3 && getenv("DEM_TRACE_PAIR")) {
                int pid = atoi(getenv("DEM_TRACE_PAIR"));
                std::vector<double> po((size_t)n_ss * 9), sx(n_ss), sz(n_ss);
                CHECK_CU(cuMemcpyDtoH(po.data(), d_pfout, po.size() * 8));
                CHECK_CU(cuMemcpyDtoH(sx.data(), d_bsfx, n_ss * 8)); CHECK_CU(cuMemcpyDtoH(sz.data(), d_bsfz, n_ss * 8));
                for (int k = 0; k < n_ss; ++k) if (h_bp1[k] == pid || h_bp2[k] == pid)
                    printf("PAIR %d %d %.17g %.17g %.17g %.17g %.17g\n", h_bp1[k], h_bp2[k], po[9 * (size_t)k], po[9 * (size_t)k + 1], po[9 * (size_t)k + 2], sx[k], sz[k]);
            }
            if (UTS && it + 1 == 3 && getenv("DEM_TRACE_TOP")) for (int i : top_ids) printf("TOP %d %.17g %.17g %.17g\n", i, (double)h_force[3 * i], (double)h_force[3 * i + 2], Vt[i].z);
        }
        const double t_in = scene_time;   // O.time dentro del paso
        scene_time += dt;                 // O.time tras el paso
        it_last = it;

        // PyRunner (dentro del paso, tras Newton)
        if (BTS && it == 2) {   // load(): velocidad de los platos
            for (int k = 0; k < n_facets; ++k) f_vel_h[k] = {0.0, 0.0, k < n_plate ? -z_vel : z_vel};
            CHECK_CU(cuMemcpyHtoD(d_fvel, f_vel_h.data(), n_facets * sizeof(Vec3)));
        }
        if (it % vtk_period == 0) export_vtk();
        if (UTS && it % 1000 == 0) extract(it, t_in, 1);   // PyRunner(iterPeriod=1000, 'extract_Data()')

        // bucle 'while True' del script, tras O.step()
        const long long oi = it + 1;
        if (oi % 100 == 0) {
            const double fz = extract(oi, scene_time, (oi < 20000) ? dup_rows : 1);
            if (oi % 1000 == 0) {
                double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
                printf("%s - Iter %lld Tiempo: %.4f Fz %.1f Fz Max %.2f Un_Force %.3f | %.0f it/s\n", TNAME, oi, scene_time, fz, fz_Max, last_unb, (it - 1) / el);
                fflush(stdout);
                if (oi % 10000 == 0) { fflush(f_facet); fflush(f_energy); fflush(f_strain); f_crk.flush(); }
                if (fracture_time >= 0.0) {
                    if (scene_time >= fracture_time * stop_factor) { printf("5%% extra time after fracture reached. Stopping.\n"); stop_reason = "fractura"; break; }
                } else if (fz / fz_Max < 0.7 && fz_Max > fzmax_min) {
                    export_vtk();
                    fracture_time = scene_time; it_fracture = oi;
                    printf("Fracture detected at t = %.4f -> Continuing 5%% more in simulation time...   (parada en t >= %.2f·t_fractura)\n", fracture_time, stop_factor);
                }
            }
        }
    }
    CHECK_CU(cuCtxSynchronize());
    flush_cracks();
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    fclose(f_facet); fclose(f_energy); fclose(f_strain); f_crk.close();
    int nt = 0, ns = 0;
    CHECK_CU(cuMemcpyDtoH(&nt, d_nb_tens, 4)); CHECK_CU(cuMemcpyDtoH(&ns, d_nb_shear, 4));
    if (n_crk_lost) std::cerr << "AVISO: " << n_crk_lost << " fisuras no se registraron en el archivo (búfer lleno)\n";
    if (vtk_on) std::cout << "[VTK] " << vtk_index << " instantáneas, " << vtk_seconds << " s de escritura\n";
    std::cout << "\nCompletado en " << elapsed << " s (" << (it_last - 1) / elapsed << " it/s), parada: " << stop_reason << "\n";
    {
        std::ofstream r(out_dir + "/resumen.txt");
        r << std::setprecision(10);
        r << "ensayo = " << TNAME << "\nestado = completado\nparada = " << stop_reason << "\niter_final = " << it_last + 1
          << "\ntiempo_final = " << scene_time << "\nsegundos = " << elapsed << "\nit_por_s = " << (it_last - 1) / elapsed
          << "\nfz_max = " << fz_Max << "\niter_fz_max = " << fz_Max_iter;
        if (BTS) r << "\nsigma_t_bts_pico = " << fz_Max_stress << "    # 2·Fz/(pi·D·L), columna 6 de facet_force.txt";
        if (UCS) r << "\nsigma_c_pico = " << fz_Max / (PI * std::pow(.5 * (xMax - xMin + yMax - yMin) / 2.0, 2)) << "    # Fz/(pi·(D/2)^2)"
                   << "\nstress_columna_pico = " << fz_Max_stress << "    # columna 6 de facet_force.txt (fórmula del BTS, como el script)";
        if (UTS) r << "\nsigma_t_pico = " << fz_Max_stress << "    # Fz/(pi·(D/2)^2), columna 6 de facet_force.txt";
        r << "\niter_fractura = " << it_fracture << "\ntiempo_fractura = " << fracture_time
          << "\nfisuras_traccion = " << nt << "\nfisuras_corte = " << ns << "\nfisuras_en_archivo = " << n_crk_written
          << "\nparticulas = " << n_particles << "\nenlaces = " << nA << "\nfacetas = " << n_facets << "\ndt = " << dt
          << "\nz_vel = " << z_vel << "\nreconstrucciones_contactos = " << n_rebuilds << "\ncontactos_perdidos_ss_fs = " << missed_ss << "/" << missed_fs
          << " (verificaciones " << n_verifications << ")\n";
    }
    if (getenv("DEM_DUMP_STATE")) {   // regresión: estado final binario (pos, vel, angVel)
        std::vector<Vec3> Pp(n_particles), V(n_particles), Wd(n_particles);
        CHECK_CU(cuMemcpyDtoH(Pp.data(), d_pos, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(V.data(), d_vel, n_particles * sizeof(Vec3)));
        CHECK_CU(cuMemcpyDtoH(Wd.data(), d_angVel, n_particles * sizeof(Vec3)));
        FILE* fd = fopen(getenv("DEM_DUMP_STATE"), "wb");
        fwrite(Pp.data(), sizeof(Vec3), n_particles, fd); fwrite(V.data(), sizeof(Vec3), n_particles, fd); fwrite(Wd.data(), sizeof(Vec3), n_particles, fd); fclose(fd);
    }
    return 0;
}
