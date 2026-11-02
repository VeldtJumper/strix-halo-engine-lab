# Primeros pasos: requisitos, compilación y ejecución

[English](README.md) · [Español](README.es.md) · [README principal](../../README.es.md)

El [README principal](../../README.es.md#inicio-rápido) muestra el camino más corto: compilar en un nodo Linux Strix Halo y
ejecutar Qwen con los valores predeterminados de producción. Esta página contiene los requisitos de cada motor, la
compilación completa y dónde está el comando de lanzamiento de cada motor. Todas las opciones de CMake, la línea de
comandos completa y las variables de entorno están en la [referencia del motor](engine-reference.md);
los comandos de benchmark, la procedencia de los modelos y las interfaces de máquina, en [reproducción](../reproduce.md).

<a id="what-you-need"></a>

## Qué se necesita

| | Qwen3.8-Flash-Next | DeepSeek V4.1 Flash |
|---|---|---|
| **Máquinas** | un Strix Halo (`gfx1151`) con **128 GB** | dos Strix Halo con 128 GB cada uno, dos cables USB4 directos, la [serie de parches del driver USB4](https://github.com/hyukudan/strix-halo-usb4-driver); opcionalmente un PC con CUDA y una RTX PRO 6000 |
| **SO / pila de GPU** | Linux, ROCm/HIP (probado con ROCm core 10.1 en `/opt/rocm/core-10.1`) | igual; CUDA para las partes RTX opcionales |
| **Ficheros del modelo** | GGUF UD-Q4_K_XL (~111 GB) + sidecar MTP Q8_0 | el GGUF de expertos de 2 bits, la superposición densa de 6.5 bits, los paquetes trellis y el drafter DSpark ([docs/models/deepseek-v4.1/production-defaults.md](../models/deepseek-v4.1/production-defaults.md#production-identity-environment-of-both-ranks)) |
| **Herramientas de compilación** | CMake >= 3.24, Ninja, un compilador C++20, opcionalmente OpenSSL para hashing rápido | igual |

MiniMax necesita su directorio EXL3 de 3.06 bpw admitido y assets de draft locales compatibles para DSpark;
GLM necesita su directorio EXL3 de 4.05 bpw y los assets MTP nativos. Ambos usan dos ranks Linux Strix Halo
con el driver USB4 y la biblioteca trellis. La RTX independiente necesita Windows WDDM, CUDA, el modelo
Q2/Q8 admitido y los assets de recibo/estado/referencia. Los manifiestos de modelos, la procedencia de las conversiones y los avisos
obligatorios se enumeran en [modelos y procedencia](../reproduce.md#models-and-provenance).

La configuración del kernel/IOMMU corresponde al propietario de la máquina. Mantenga traducidos los controladores USB4; nunca desactive
el IOMMU para ganar velocidad. Registre el kernel en ejecución y `amdgpu.iommu_perfopt` en los recibos de tiempos y compare
PerfOpt activado/desactivado solo con el mismo kernel.

Las herramientas de línea de comandos reciben IDs de token; los servidores HTTP incorporan los tokenizadores y las plantillas de chat propios de los modelos.

<a id="build"></a>

## Compilación

```sh
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON \
  -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token                         # Qwen
ninja -C build strix-deepseek41-rank strix-deepseek41-serve   # DeepSeek (family module, STRIX_FAMILY_DEEPSEEK41=ON by default)
```

MiniMax y GLM también necesitan las cabeceras y el archivo de trellis adoptados. Tras la configuración HIP común,
use un checkout relativo `TRELLIS` que contenga `include/strix_trellis.h` y
`build-qualification-v03/libstrix_trellis.a`:

```sh
cmake -S . -B build -DSTRIX_FAMILY_MINIMAX=ON -DSTRIX_MINIMAX_TRELLIS_DIR=TRELLIS
ninja -C build strix-minimax-rank strix-minimax-serve
cmake -S . -B build -DSTRIX_FAMILY_GLM5=ON -DSTRIX_GLM5_TRELLIS_DIR=TRELLIS
ninja -C build strix-glm5-rank
```

Los ajustes de API/biblioteca trellis de producción de DeepSeek y las instrucciones completas de compilación por motor están en
[Compilar cada motor](../reproduce.md#build-each-engine). Configure y compile ambos ranks de forma idéntica.
Los objetivos de las familias MiniMax y GLM requieren estas opciones de compilación aunque sus valores predeterminados de ejecución sean rápidos.

Las partes CUDA para la RTX PRO se compilan por separado con `-DSTRIX_ENABLE_CUDA_DS41=ON`. `ctest --test-dir build` ejecuta
las pruebas de host y de dispositivo (muchas pruebas de dispositivo necesitan la GPU y los ficheros del modelo). Todas las opciones de CMake se enumeran en
[docs/getting-started/engine-reference.md](engine-reference.md#build-options).

<a id="run"></a>

## Ejecución

Empiece con el comando simple del motor seleccionado: las optimizaciones de producción están activadas por defecto
cuando se superan sus protecciones de modelo, forma, recursos y referencia, y `--print-effective-config` (RTX
independiente: `--print-config`) muestra la configuración resuelta antes de cualquier trabajo en el dispositivo
([valores predeterminados e interruptores de desactivación](../methodology.es.md#defaults-and-off-switches)). El
contrato de lanzamiento, los perfiles y las opciones de contexto largo de cada motor están en su página:

- [Qwen3.8-Flash-Next](../models/qwen3.8/README.es.md#qwen-recommended-command): un nodo; `--profile`,
  [contexto largo y YaRN](../models/qwen3.8/README.es.md#qwen-long-context), caché de prefijo, servidor HTTP.
- [Par DeepSeek V4.1 Flash](../models/deepseek-v4.1/README.es.md#deepseek-pair) y
  [RTX independiente](../models/deepseek-v4.1/README.es.md#deepseek-standalone-rtx); RTX + par:
  [etapas del pipeline](../models/deepseek-v4.1/pipeline-stages.md).
- [Par MiniMax M3.1 Flash](../models/minimax-m3.1/README.es.md#minimax-pair).
- [Par GLM-5.3 Flash](../models/glm-5.3/README.es.md#glm-pair) y [GLM en una RTX PRO 6000](../../src/glm/rtxpf/README.md).
- [Nemotron 3.5 Bolt E4B](../models/nemotron-3.5-bolt-e4b/README.es.md#compilación-y-ejecución) (compilación opcional).
- Exposición de los servidores, autenticación y límites: [servicio](../serving.es.md#exposición-de-los-servidores).
