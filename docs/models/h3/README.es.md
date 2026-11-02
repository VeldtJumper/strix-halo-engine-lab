# MiniMax H3 (audio + vídeo) en Strix Halo

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

MiniMax H3 es un transformer de difusión de audio + vídeo: 50 bloques densos (hidden 5376, 56 cabezas x 128, SwiGLU
14336) con atención 3D completa no causal sobre la secuencia empaquetada `[text | cond | audio | video]`, una torre de
texto Qwen3-VL-32B de 50 capas como codificador de texto, un VAE de vídeo ViT3D de 36 capas y un VAE de audio. El motor
ejecuta el clip completo en un nodo Strix Halo o en el par USB4. Es una **compilación opcional**
(`-DSTRIX_FAMILY_H3=ON`); con la opción desactivada no se configura ni se compila nada. Cómo se leen las tablas y qué
cubre "exacto": [metodología](../../methodology.md).

## Exactitud

- **Ruta densa = referencia.** Todos los kernels son deterministas e invariantes al tamaño del lote: el orden de las
  reducciones no depende del número de filas ni del nodo que calcula una fila. El clip en dos nodos es **idéntico byte
  a byte** al clip en un nodo (latentes y todos los fotogramas, comprobado con sha256).
- Cada paquete tiene una referencia de host en C++ simple y pruebas de host que se compilan y ejecutan sin GPU; las
  pruebas de GPU son ejecutables aparte (etiqueta CTest `gpu`).
- Las palancas de velocidad que cambian bits (clase fast) llevan cada una su propia identidad, un interruptor de
  desactivación y un control de calidad frente a la referencia densa.

## Paquetes

| Paquete | Directorio | Qué es |
|---|---|---|
| scaffold | `cmake/h3_00_common.cmake`, `src/h3/README.md` | opción `STRIX_FAMILY_H3`, objetivo compartido `strix_h3_common` |
| gemm | `src/h3/gemm` | proyecciones int8 W8A8 tensorwise + rotadas y el cuantizador de activaciones, SwiGLU fusionado; 37-45 TOPS de extremo a extremo con 19K-38K filas |
| attn | `src/h3/attn` | atención flash densa no causal (dimensión de cabeza 128), parciales split-KV + fusión canónica; ~33-35 TF/s |
| te | `src/h3/te`, `tools/h3/te` | tokenizador, torres de texto y de visión a partir de un te pack (códigos nvfp4, escalas de bloque sin swizzle, manifiesto sha256) |
| vae | `src/h3/vae` | decodificador de vídeo de 36 capas, decodificador de audio, codificador de vídeo; reparto de la decodificación en dos nodos con los mismos fotogramas |

El ejecutor del clip y la planificación en dos nodos llegan en cambios posteriores; los tiempos de abajo se midieron
con ellos sobre los mismos kernels.

## Compilación y pruebas (solo host)

```sh
cmake -S . -B build-h3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_FAMILY_H3=ON -DBUILD_TESTING=ON
cmake --build build-h3
ctest --test-dir build-h3 -R h3 -LE "gpu|qualification"
```

Las pruebas que dependen de datos indican SKIPPED si no tienen sus fixtures o pesos; las pruebas con la etiqueta
`qualification` fallan sin ellos a propósito (ahí un skip nunca cuenta como aprobado). Las compilaciones de GPU
(gfx1151) añaden `-DSTRIX_ENABLE_HIP_RUNTIME=ON`; las pruebas de GPU se registran en CTest con
`-DSTRIX_H3_GPU_CTEST=ON` donde lo indica el archivo CMake de cada paquete.

<a id="h3-results"></a>

## Resultados medidos

Clip: texto + audio a vídeo, 1344x768, 124 fotogramas (5.17 s a 24 fps) con audio, 8 pasos de muestreo, semilla 42,
37,734 tokens en la secuencia empaquetada. Los tiempos son trabajo de GPU con la caché de páginas caliente; pesos: DiT
int8 rotado, te pack nvfp4 del codificador de texto, VAE en fp16.

| Configuración | Codificador de texto | DiT, 8 pasos | VAE de vídeo | Total | Exactitud |
|---|---:|---:|---:|---:|---|
| Denso, 1 nodo | 10 s (carga 9.7 s, codificación 0.74 s) | **860 s** (105.5-113.2 s/paso) | 87 s (con carga) | **~16 min** | referencia |
| Denso, 2 nodos por USB4 | 10 s | **435 s** (52-55 s/paso, 1.98x) | **40.6 s** (+5.3 s de carga) | **~8 min** | idéntico byte a byte a 1 nodo |

VAE de audio: 0.3-0.6 s por clip. En cada ejecución, cada rango carga además los pesos del DiT (14-17 s, en paralelo
en el par).

Cifras por kernel (gfx1151, un nodo):

| Kernel | Resultado |
|---|---|
| proyecciones int8, cuantización + GEMM, 19K filas | qkv 45.2, out 38.4, fc1 44.6, fc1+SwiGLU 44.9, fc2 38.6 TOPS |
| proyecciones int8, 38K filas | qkv 43.3, out 39.2, fc1 42.6, fc1+SwiGLU 42.9, fc2 37.2 TOPS |
| atención, forma por bloque en 2 nodos | 34.8 TF/s (planificación de 16 waves, predeterminada) |
| atención, fusionada 37910 x 37910 | 32.9 TF/s |
| forward del codificador de texto, 300 / 2500 / 5000 tokens | 1.42 / 9.56 / 22.3 s (clase densa) |
| codificador de vídeo, clip de 17 fotogramas 768x1344 | 11.8 s |

## Te pack del codificador de texto

El codificador de texto lee un te pack generado por `tools/h3/te/te_pack.py` a partir del checkpoint nvfp4 del
codificador de texto:

```sh
python tools/h3/te/te_pack.py --src <checkpoint.safetensors> --out <pack dir> --verify all
```

El pack conserva exactamente los códigos de 4 bits y guarda las escalas de bloque e4m3 por filas; el cargador comprueba
`te-pack.json` frente a la geometría del motor y el sha256 de cada archivo que abre (`STRIX_H3_TE_VERIFY=0` omite solo
los hashes).

## Interruptores

| Variable | Efecto |
|---|---|
| `STRIX_H3_GEMM_QUANT_FAST=0/1` | fuerza una variante del cuantizador (bits idénticos; por defecto se elige según K) |
| `STRIX_H3_ATTN_WAVES=8` | planificación de atención de 8 waves (bits idénticos; por defecto 16) |
| `STRIX_H3_TE_SPLIT=0` | codificador de texto de clase fast `h3-te-lm-fast-f16-v1` (por defecto el denso `h3-te-lm-dense-split-v1`) |
| `STRIX_H3_TE_VERIFY=0` | omite la comprobación sha256 del pack al cargar |
