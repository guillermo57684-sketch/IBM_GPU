#!/bin/bash
# Compila ensayos_cuda_native (BTS, UCS y UTS). No toca triaxial_cuda_native. El binario busca NVRTC en tiempo de ejecución (DEM_NVRTC, CUDA_HOME, rutas estándar,
# paquetes pip nvidia-*); además, este script guarda como pista la carpeta de NVRTC que encuentre en este equipo.
# Uso: ./compile.sh            (o NVRTC_DIR=/ruta/a/lib ./compile.sh para fijar la pista)
set -e
cd "$(dirname "$0")"
find_nvrtc() {
    [ -n "$NVRTC_DIR" ] && { echo "$NVRTC_DIR"; return; }
    for d in "$CUDA_HOME/lib64" "$CUDA_PATH/lib64" /usr/local/cuda/lib64 /usr/local/cuda-*/lib64; do
        ls "$d"/libnvrtc.so* >/dev/null 2>&1 && { echo "$d"; return; }
    done
    f=$(ldconfig -p 2>/dev/null | grep -m1 -o '/[^ ]*libnvrtc\.so[^ ]*') && [ -n "$f" ] && { dirname "$f"; return; }
    f=$(python3 -c "import glob, os, nvidia; fs = sorted(f for p in nvidia.__path__ for f in glob.glob(os.path.join(p, '*', 'lib', 'libnvrtc.so.*')) if 'alt' not in f); print(fs[-1] if fs else '')" 2>/dev/null) && [ -n "$f" ] && { dirname "$f"; return; }
    f=$(find "$HOME" -maxdepth 12 -name 'libnvrtc.so.*' -not -name '*alt*' 2>/dev/null | sort -r | head -1) && [ -n "$f" ] && { dirname "$f"; return; }
}
HINT=$(find_nvrtc || true)
if [ -n "$HINT" ]; then echo "NVRTC encontrado en: $HINT"; else echo "AVISO: no se encontró NVRTC; en ejecución use DEM_NVRTC=/ruta/a/lib"; fi
echo "Compilando ensayos_cuda_native..."
g++ -O3 ensayos_cuda_native.cpp -ldl -DNVRTC_HINT="\"$HINT\"" -o ensayos_cuda_native
echo "Compilación exitosa -> ./ensayos_cuda_native"
