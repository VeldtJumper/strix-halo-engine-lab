# Uso de una GPU discreta (opcional)

[English](discrete-gpu.md) · [Español](discrete-gpu.es.md) · [README principal](../README.es.md)

Todos los motores funcionan solo con Strix Halo; ese es el valor por defecto y nada de esta página lo cambia. Una GPU
discreta de NVIDIA añade rutas aparte y opcionales. Cada ruta es su propia compilación y su propio comando, con su
propia identidad declarada y sus propios controles. Si no la compila ni la inicia, los valores por defecto de Strix
Halo no cambian.

## Qué se ha probado

| | Probado |
|---|---|
| Tarjeta | RTX PRO 6000 Blackwell Workstation Edition, 96 GB (`sm_120`) |
| SO / controlador | Windows 11 (WDDM), controlador NVIDIA 617.14 |
| Herramientas | CUDA 13.0, Visual Studio 2022 (MSVC x64), CMake 3.24 o posterior, Ninja |
| Presupuestos menores | DeepSeek independiente y GLM funcionan exactos con `--vram-total-gib 24 --ram-total-gib 32`; DeepSeek también con 32 + 32 |
| No probado | Otras tarjetas, hosts CUDA con Linux, presupuestos por debajo de los indicados |

El planificador de DeepSeek independiente rechaza presupuestos por debajo de unos 18.8 GiB de VRAM con
`--max-positions 2560` e indica el mínimo que necesita.

## Rutas de un vistazo

| Ruta | Qué se ejecuta dónde | Medido | Cuándo no ayuda |
|---|---|---|---|
| DeepSeek V4.1 Flash, RTX independiente | El modelo completo en la tarjeta: expertos enrutados Q2, densos Q8_0, caché de expertos en VRAM + memoria fijada del host, objetivo y borrador DSpark en la misma GPU | Tarjeta completa: 82.3 / 87.7 / 94.4 tok/s (código / inglés / español). 24 GiB de VRAM + 32 GiB de RAM, un NVMe, prompt de 96 tokens: 6.55 tok/s sin borrador, 6.75 tok/s con DSpark, con 4 lecturas de archivo en vuelo (5.45 tok/s con 1) | Con 24 GiB de VRAM + 32 GiB de RAM la mayoría de los expertos se leen del disco: 6.55 tok/s frente a 82-94 tok/s con la tarjeta completa |
| DeepSeek V4.1 Flash, RTX + par | La RTX ejecuta las capas 0-21 con CUDA; los dos nodos Strix Halo ejecutan las capas 22-39 y la cabeza, por 1 GbE | Decode por lotes 60.1 / 97.5 / 137.2 tok/s con 2 / 4 / 8 filas; lectura del prompt en la RTX 3.64 / 4.39 / 12.20 s con 2K / 8K / 32K | La ubicación de expertos calientes empeora las ventanas DSpark de 4 o más |
| Qwen3.8-Flash-Next, descarga del prefill | La RTX lee el prompt (prefill CUDA exacto) y envía el estado de atención y recurrente al nodo capa a capa; el nodo decodifica | TTFT 1.36 / 1.92 / 2.94 / 4.95 / 8.99 s con 1K / 4K / 8K / 16K / 32K, frente a 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s en el mismo servidor sin la RTX | Por debajo de 1K tokens el enrutador mantiene los prompts en el nodo; con 1K el prefill de un nodo (1.12 s) es más rápido que la descarga (1.36 s) |
| GLM-5.3 Flash, una RTX (borrador) | Tronco, capas densas y expertos compartidos en la tarjeta; expertos enrutados servidos por la caché de [tierstream](https://github.com/hyukudan/tierstream) entre VRAM, RAM fijada y disco | Tarjeta completa: el solapamiento del prefill deja el TTFT en 0.56× con 8K y 0.66× con 32K; decode 6.0 / 6.2 tok/s en caliente con 1K / 8K. 24 GiB de VRAM + 32 GiB de RAM: TTFT 60.3 s con 1K, 62.4 s con 8K; decode 0.67 tok/s en caliente | Con presupuestos pequeños la mayoría de los expertos se leen del disco, así que el decode es lento |

Cada fila es exacta frente a su referencia declarada: los ids de token (y, donde se indica, los logits y el estado)
son iguales a los de la ejecución de referencia. Detalles y condiciones:
[DeepSeek independiente](models/deepseek-v4.1/README.es.md#deepseek-standalone-rtx-results),
[DeepSeek RTX + par](models/deepseek-v4.1/README.es.md#deepseek-rtx-pipeline),
[descarga del prefill de Qwen](models/qwen3.8/README.es.md#qwen-prefill-offload),
[GLM en una RTX](models/glm-5.3/README.es.md#glm-rtx-results).

## Fije la GPU correcta

Si el equipo tiene más de una GPU NVIDIA, seleccione la tarjeta por UUID, nunca por índice (los índices pueden diferir
entre `nvidia-smi` y CUDA):

```powershell
nvidia-smi -L                      # lists each GPU with its UUID
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "GPU-<uuid of the card to use>"
```

Establézcalo en la misma consola antes de cada comando de esta página, incluidas las impresiones de configuración y
las ejecuciones en seco.

## DeepSeek V4.1 Flash, RTX independiente

Compile primero la biblioteca CUDA de tierstream y después, desde la raíz del repositorio con una copia de tierstream
en `TIERSTREAM`:

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON `
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
cmake --build build-rtx --target strix-cuda-ds41-replica
```

Ejecución, el proceso persistente, presupuestos menores y todas las opciones:
[guía de la RTX independiente](models/deepseek-v4.1/rtx-standalone.md).

## DeepSeek V4.1 Flash, RTX + par

Requiere el par de dos nodos, una compilación CUDA (`STRIX_ENABLE_CUDA_DS41`) en el equipo de la GPU y un enlace de
1-2.5 GbE entre ellos. El puerto TCP de la canalización no tiene autenticación: manténgalo en una red privada.
Contrato de lanzamiento de las etapas: [etapas de la canalización](models/deepseek-v4.1/pipeline-stages.md).

## Qwen3.8-Flash-Next, descarga del prefill

El lado del nodo usa `--rtx-state-import` con `--http-serve` (sin `--http-batch`); con la opción desactivada nada
cambia. Compilación y protocolo del servicio: [servicio de prefill CUDA de Qwen](engine/serving/qwen-cuda-prefill.md) e
[importación de estado de la RTX](rtx-state-import.md).

## GLM-5.3 Flash en una RTX (borrador)

Compilación y después la ejecución medida (exactitud y velocidad; un servidor HTTP para esta ruta aún no se ha probado
de extremo a extremo):

```powershell
pwsh src/glm/rtxpf/producer/build.ps1 -BuildDir BUILD_DIR -Fresh -Parallel 2 `
    -TierstreamSource TIERSTREAM_DIR [-CutlassDir CUTLASS_DIR]
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR `
    --model MODEL_DIR --prompt-dir PROMPT_DIR --unit-file OUT_DIR/glm-units.bin `
    --unit-manifest-sha256 MANIFEST_SHA256 --full-card --allow-gpu
```

Para una tarjeta menor sustituya `--full-card` por `--vram-total-gib 24 --ram-total-gib 32` (o sus propios límites).
El archivo de unidades, todas las opciones y las filas medidas: [guía de GLM RTX](../src/glm/rtxpf/README.md).

El servicio de GLM (`glm_native_service`) acepta además `--dgpu auto|<uuid>`: comprueba mediante NVML, antes de
cualquier llamada CUDA, que la tarjeta es una RTX PRO `sm_120` (`auto` elige la tarjeta válida con más memoria) y la
fija; `--dgpu off` se rechaza (esta ruta no tiene camino por CPU). El arnés de la ejecución medida de arriba selecciona
la tarjeta con `CUDA_VISIBLE_DEVICES`, como se describe en [Fije la GPU correcta](#fije-la-gpu-correcta).

## Solución de problemas

- **Un plan no cabe.** Los planificadores de DeepSeek y GLM lo rechazan antes de cargar e indican la memoria que
  necesitan; reduzca el contexto (`--max-positions`) o aumente los límites. GLM: `--plan-budget` muestra el reparto
  sin CUDA.
- **Compruebe primero la configuración.** DeepSeek independiente: `--print-config` imprime la configuración resuelta y
  termina antes de cualquier llamada CUDA. GLM: `--print-effective-config` imprime cada opción, su valor y su origen
  sin cargar el modelo.
- **Se usa la tarjeta equivocada.** Establezca `CUDA_DEVICE_ORDER=PCI_BUS_ID` y el UUID como arriba.
- **Las velocidades son menores que en la tabla.** Otros programas que usan la misma GPU (grabación de pantalla,
  captura u otros trabajos de cálculo) reducen todas las cifras; las mediciones de arriba se tomaron con la tarjeta sin
  otra carga.
- **Para quedarse con los valores por defecto de Strix Halo,** basta con no compilar ni iniciar estas rutas. Cada ruta
  tiene además sus propios interruptores de desactivación en su guía.
