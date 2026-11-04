# MiniMax H3 (audio + vídeo) en el par USB4

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

**Un clip de vídeo de 5 s con audio estéreo en 405.5 s (6.8 min) en dos nodos Strix Halo.**

Texto a vídeo + audio, 1344x768, 124 fotogramas (5.17 s a 24 fps) y audio de 2 canales a 32 kHz, 6 pasos de muestreo
(Euler, shift 12 / 3), semilla 42, prompt *"A red fox trots through fresh snow in a quiet birch forest at dawn, soft
light, its breath visible."* (el prompt por defecto del runner), 37.734 tokens en la secuencia empaquetada.

MiniMax H3 es un transformer de difusión de audio + vídeo: 50 bloques densos (hidden 5376, 56 cabezas x 128, SwiGLU
14336) con atención 3D completa no causal sobre la secuencia `[texto | cond | audio | vídeo]`, una torre de texto
Qwen3-VL-32B de 50 capas como codificador de texto, un VAE de vídeo y un VAE de audio. Es una compilación opcional
(`-DSTRIX_FAMILY_H3=ON`); con la opción desactivada no se configura ni se compila nada. Cómo se leen las tablas y qué
cubre "exacto": [metodología](../../methodology.es.md).

<a id="h3-result"></a>

## El clip de 405.5 s

Configuración: dos nodos por USB4 (intercambio de K/V en paralelo de secuencia), el LoRA destilado de 6 pasos
`h3-lora-fasth3-6step-v1`, atención dispersa `h3-attn-sparse-thr-v1` con tau 0.75 (las 3 primeras llamadas y las capas
0-1 y 39-42 densas, las otras 44 capas dispersas desde la llamada 4), el VAE de vídeo rápido de 24 capas
`h3-vae-fast24-v1`, sobre la GEMM int8 W8A8 exacta y la atención flash de 16 waves.

**Exactitud:** para esta identidad, un nodo, el esquema de dos nodos ejecutado en un nodo (world 1) y los dos nodos dan
los mismos bits (latentes, fotogramas y audio, la velocidad de cada paso). El LoRA de pasos, la atención dispersa y el
VAE rápido son de clase FAST: cada uno tiene su propia línea de identidad en el registro de la ejecución y su propio
interruptor; sin ellos la ejecución es la referencia densa.

| Etapa | Tiempo |
|---|---:|
| Codificador de texto | 0.8 s (+ 5.0 s de carga) |
| DiT, 6 pasos | **300.2 s**: pasos 1-3 (ventana densa) 59.2 / 54.3 / 55.0 s, pasos 4-6 (dispersos) 46.3 / 42.8 / 42.2 s |
| VAE de vídeo (rápido, 24 capas) | 42.4 s (+ 12.1 s de carga) |
| VAE de audio | 0.3 s |
| Arranque, subida de pesos, hash de contenido de ficheros nuevos (una vez) | ~45 s |
| **Total** | **405.5 s** |

Referencia densa del mismo clip (8 pasos, VAE de vídeo de 36 capas): ~16 min en un nodo (DiT 860 s, VAE de vídeo 87 s,
codificador de texto 10 s) y ~8 min en el par (52-55 s por paso, VAE de vídeo 40.6 s), el par idéntico byte a byte a
un nodo.

<a id="h3-install"></a>

## Instalación y ejecución

1. **Compilar** en cada nodo (ROCm, gfx1151):

   ```sh
   cmake -S . -B build-h3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_FAMILY_H3=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON \
     -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DSTRIX_HIP_ARCHITECTURES=gfx1151 -DBUILD_TESTING=OFF
   cmake --build build-h3 --target strix-h3-clip
   ```

   Los dos nodos deben ejecutar el mismo binario (los ranks comparan binario e identificadores de los ficheros al
   arrancar).
2. **Ficheros del modelo**, en un directorio por nodo (`H3_MODELS`):

   | Fichero | Uso |
   |---|---|
   | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | DiT (int8 rotado) |
   | `te-pack/` (52 ficheros + `te-pack.json`) | pack del codificador de texto, escrito por `tools/h3/te/te_pack.py` a partir del checkpoint nvfp4 |
   | `fasth3_dense_comfy_6step.safetensors` | LoRA de 6 pasos |
   | `fast_h3_vae_noncausal_24l.safetensors` | VAE de vídeo rápido |
   | `minimax_h3_video_vae_fp16.safetensors` | VAE de vídeo de 36 capas (referencia densa) |
   | `minimax_h3_audio_vae_fp32.safetensors` | VAE de audio |

   y los ficheros del tokenizer Qwen (`vocab.json`, `merges.txt`, `tokenizer_config.json`) en `H3_TOKENIZER_DIR`.
3. **Un comando por clip**: arrancar el rank 1 en el segundo nodo y después el rank 0 en el primero (mismos argumentos;
   el rank 0 escribe las salidas):

   ```sh
   export H3_MODELS=/ruta/a/modelos H3_TOKENIZER_DIR=/ruta/al/tokenizer HSA_DISABLE_FRAGMENT_ALLOCATOR=1
   ARGS="--width 1344 --height 768 --length 124 --seed 42 --lora h3-lora-fasth3-6step-v1 \
     --sparse 0.75 --sparse-dense-layers 39-42 --sparse-dense-calls 3 --fast-vae"
   ./build-h3/strix-h3-clip --world 2 --rank 1 --transport usb4auto $ARGS             # nodo 2
   ./build-h3/strix-h3-clip --world 2 --rank 0 --transport usb4auto --out clips/fox $ARGS   # nodo 1
   ```

   `--prompt "..."` fija el prompt. `tools/h3/block/clip_pair.sh` lanza los dos ranks desde el nodo 1 por ssh
   (`H3_BLOCK_DIR`, `H3_PEER`, `NAME`, `ARGS`). Un nodo: `./build-h3/strix-h3-clip --out clips/fox $ARGS`.
   Salidas: `frames_rgb24.raw` (todos los fotogramas), cuatro fotogramas PNG de muestra + `preview_strip.png`,
   `audio.wav`, `meta.json` (tiempos por etapa), `hashes.txt` y `run_identity.txt`.

**Vuelta atrás:** cada palanca de clase FAST tiene su propio flag: quitar `--lora` (muestreo denso de 8 pasos),
`--sparse` (atención densa), `--fast-vae` (VAE de 36 capas); sin ninguno la ejecución es la referencia densa. Compilar
sin `-DSTRIX_FAMILY_H3=ON` elimina la familia por completo.

## Paquetes y interruptores

La lista de paquetes, las pruebas de host y la tabla de interruptores están en la [página en inglés](README.md#packages).
