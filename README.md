# DEM-GPU: modelo IBM en CUDA

Motor DEM en GPU que replica en CUDA los ensayos de YADE con la ley de contacto Interlocking Bond Model (IBM).

- `triaxial_cuda_native.cpp`: ensayo triaxial (réplica de `test-4-triaxial-opt.py`).
- `ensayos_cuda_native.cpp`: ensayos BTS, UCS y UTS (réplica de `test-1-BTS.py`, `test-2-UCS.py` y `test-3-UTS.py`).

## Requisitos

- Linux con GPU NVIDIA y su driver (`libcuda.so`).
- `g++` con soporte de C++17.
- NVRTC (`libnvrtc.so`). No hace falta el CUDA Toolkit completo: basta `pip install nvidia-cuda-nvrtc-cu12`
  (o la variante cu13). Los kernels se compilan en tiempo de ejecución para la GPU detectada.

## Compilar

```bash
./compile.sh            # -> triaxial_cuda_native
./compile_ensayos.sh    # -> ensayos_cuda_native
```

Los scripts buscan NVRTC (CUDA_HOME, /usr/local/cuda, ldconfig, paquetes pip `nvidia-*`) y guardan su ruta en el binario.
Si no lo encuentran: `NVRTC_DIR=/ruta/a/lib ./compile.sh`, o en ejecución `DEM_NVRTC=/ruta/a/lib`.

## Ejecutar

Triaxial (los valores por defecto son el caso de referencia, probeta `inputs/probeta_1.geo` a 40 MPa):

```bash
./triaxial_cuda_native --ayuda                                   # parámetros y valores por defecto
./triaxial_cuda_native 20000000 corridas/ref                     # pasos (tope) y carpeta de salida
./triaxial_cuda_native --caso casos/referencia.cfg confining=2e7 # archivo de caso + clave=valor
```

- La corrida se detiene sola tras la fractura con el criterio `checkProgress()` de YADE; `pasos` es solo un tope.
- Cada corrida deja `caso_usado.cfg`, `resumen.txt` (q pico, ε1 en el pico, fisuras) y `facet_force_exact.txt`.
- No sobrescribe una carpeta con resultados (`sobrescribir=1` para forzarlo).

BTS / UCS / UTS (mismos 16 argumentos y orden que los scripts de YADE):

```bash
./ensayos_cuda_native --ayuda
./ensayos_cuda_native ucs 6.671e+10 25 9.62e+06 6.704e+07 0.466 1e8 ucs01 1.3 inputs/probeta_1.geo 1 0.1 0.1 0 1 0.466 1
```

La salida queda en `output_tribo_<identificador>/`.
