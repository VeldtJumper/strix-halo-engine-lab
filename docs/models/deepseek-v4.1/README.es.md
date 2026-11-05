# DeepSeek V4.1 Flash: el par USB4, la división en pipeline con la RTX y la RTX independiente

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

DeepSeek V4.1 Flash (checkpoint 1017) se ejecuta en tres configuraciones Production: dos nodos Strix Halo por USB4
(decode con paralelismo tensorial de expertos, prefill en dispositivo con paralelismo de expertos, decode especulativo
DSpark, decode por batch, servidor HTTP), el mismo par con una RTX PRO 6000 que ejecuta las capas 0-21 por 1 GbE, y todo
el modelo en una RTX PRO 6000 bajo Windows. Esta página conserva todas las mediciones publicadas con su compilación y
sus condiciones, los comandos de lanzamiento y los perfiles. Cómo se leen las tablas y qué cubre "exacto":
[metodología](../../methodology.es.md). Instrucciones de compilación: [primeros pasos](../../getting-started/README.es.md);
el par en sí: [dos nodos por USB4](../../pair.es.md).

<a id="deepseek-pair"></a>

## Ejecución: el par

**Pesos:** [hyukudan/deepseek-v4.1-flash-gguf-mirror](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror) (el GGUF),
[hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) (capa densa) y
[hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3)
(paquetes trellis; producción usa `ours-K4` más la cabeza `ours-K5`, según `mixB.txt`). Estructura de directorios:
[reproducción](../../reproduce.md).

**Hardware:** dos nodos Strix Halo de 128 GB cada uno, unidos por dos cables USB4 (también funciona con uno), con el
módulo de [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver) compilado para el kernel en uso.

Desde un checkout limpio, [`scripts/pair/ds41-pair`](../../pair-setup.es.md) configura los dos nodos (comprobación
previa, controlador USB4, compilación, pesos verificados, arranque y prueba exacta). A mano: ejecute un rank en cada
nodo con assets y entorno coincidentes; establezca `R` en 0 o 1 localmente:

```sh
./build/strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto
```

El resolutor descubre los paquetes de producción y el drafter, activa la atención TP y el prefill en dispositivo,
y descubre los paquetes de expertos preparados offline que coinciden. Si falta un paquete preparado, se usa el cargador original
con un aviso; la ejecución nunca prepara paquetes por sí misma. La retención acotada de prefijos de agente está activada por defecto.
Véanse los [valores predeterminados de producción](production-defaults.md),
la [preparación única de expertos](prepared-load.md),
la [caché de prefijos](../../engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache),
los [contextos largos](long-context.md) y
la [colocación de expertos calientes](hot-placement.md).

Todas las aplicaciones del par (`rank`, `batch`, `prefill`, `serve`) aceptan `--profile default|fast|quality`;
use el mismo perfil en los dos ranks. `default` es la configuración de producción exacta descrita arriba y no
cambia nada. `fast` añade el epílogo MoE relajado (`STRIX_DS41_FAST_MOE=1`, cambia bits; su cualificación está en curso) e
indica la atención dispersa FP16, aún pendiente, en el recibo y en una línea de stderr. `quality` no tiene paquete para este modelo y ejecuta `default` con
un aviso. Una variable de entorno u opción explícita siempre tiene prioridad; `--allow-pending` ejecuta las opciones
pendientes cuyo código está en el build, solo para benchmarks, marcadas como `unqualified`.
`--print-effective-config` muestra el perfil y todas las opciones.

**RTX + par:** use el [contrato de lanzamiento de etapas del pipeline](pipeline-stages.md).
La RTX ejecuta las capas 0–21; el par ejecuta las capas 22–39 y la cabeza. Conserve la identidad compuesta declarada
y sus assets al comparar referencias.

<a id="deepseek-standalone-rtx"></a>

## Ejecución: RTX independiente

Es un ejecutor de estado importado en Windows WDDM. El recibo del modelo, el estado de atención exportado,
la referencia greedy correspondiente, el drafter y el fichero de chat tokenizado son assets externos obligatorios:

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin --ref GREEDY.log `
  --draft DRAFTER.gguf --draft-device 0 `
  --phases exact0,pressure,exact8,wforce0,spec0,exact0,chats0,schats0,chats0 `
  --steps 64 --max-rows 6 --max-positions 4096 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --qual --json RESULTS.jsonl
```

Las opciones simples usan el resolutor rápido, incluida una arena dimensionada automáticamente. La posición importada,
el trabajo solicitado y el margen de seguridad deben caber en la capacidad. Añada `--print-config` para inspeccionar sin CUDA.
Para la ejecución persistente con `--serve SPOOL` y la preparación de los assets necesarios, siga el
[comando independiente](rtx-standalone.md#run-and-persistent-worker).
Las sesiones de conversación exactas se documentan en la [caché de sesiones de la RTX](rtx-session-cache.md).

Tierstream admite franjas parciales verificadas en varios discos mediante `--weight-dirs`;
sin un directorio secundario conserva la ruta del fichero del modelo. Si falla una franja, se recurre al fichero
del modelo. Los costes de preparación, verificación, elegibilidad de lectura y hashing en el arranque están en
[streaming con dos unidades](rtx-two-drives.md). Esto es configuración de almacenamiento,
no un perfil numérico distinto.

En una tarjeta o un host demasiado pequeños para todos los expertos (por ejemplo 24 GB de VRAM y 32 GB de RAM), la réplica
mantiene los expertos más rápidos en VRAM, los siguientes en RAM fijada y lee el resto del fichero del modelo (ubicación en
tres niveles; `--vram-total-gib` / `--ram-total-gib` dimensionan cada nivel a partir de límites de dispositivo completo).
Las lecturas de fichero van varias a la vez por defecto (`--disk-queue auto`: 4 con una unidad, 6 con dos, 8 con tres o
más; `--disk-queue 1` o `--baseline` leen de una en una). Medido con 4 a la vez en una RTX PRO 6000 con límites de dispositivo completo `--vram-total-gib 24 --ram-total-gib 32`, los expertos que no caben en ningún nivel se leen de una unidad NVMe, 2 chats x 64 tokens tras un prompt de 96 tokens, en caliente, mismo binario, 6.55 / 6.75 tok/s normal / DSpark frente a 5.45 / 5.37 de una en una (+20 % / +26 %),
16 a la vez igual que 4, tokens idénticos. Opciones y niveles: [RTX independiente](rtx-standalone.md).

Con esos presupuestos el drafter DSpark suele costar más de lo que aporta: sus ventanas leen del fichero más expertos que
la decodificación normal. `--drafter auto` (por defecto) no lo carga cuando los expertos se leerán seguro de una sola
unidad, y sus 4.47 GB guardan más expertos; las peticiones DSpark se ejecutan entonces como decodificación normal con los
mismos tokens. En una RTX PRO 6000 con una unidad NVMe: 24+32 GiB 8.2 tok/s sin drafter frente a 6-7.8 con DSpark, 48+32
32.2 frente a 21.8, 64+32 52.2 frente a 51.8; con los expertos repartidos en dos unidades DSpark gana en 24+32 (9.3 frente
a 7.8) y `auto` lo mantiene mientras el espacio de expertos en el dispositivo sea inferior a 18 GiB, igual que sin nivel
de fichero (tarjeta completa: DSpark 105-108 tok/s); con 48+32 y dos unidades DSpark es más lento (unos 21 frente a 31.5
tok/s), así que `auto` lo ejecuta sin drafter, y 32+32 con dos unidades (sin medir en ese caso) también se ejecuta sin él.
`--drafter on` / `off` lo fuerzan.

<a id="how-fast"></a>

## Resultados medidos: el par

Fuentes: [rendimiento publicado](../../performance.md) y [valores predeterminados actuales del par](production-defaults.md).

Filas solo del par: DeepSeek V4.1 Flash (checkpoint 1017), expertos enrutados en el GGUF publicado de 2 bits (Q2), pesos densos en
una superposición de 6.5 bits con pesos de atención, expertos compartidos y cabeza codificados con trellis. 2 nodos, paralelismo tensorial de expertos
(TP2) sobre USB4.

| Modo | Hardware | Prompt / contexto | Concurrencia | Mejor | Típico | Build |
|---|---|---|---:|---:|---:|---|
| Decode greedy | 2 nodos | benchmark de decode corto, R1 | 1 | **28.333 ms/token** (35.3 tok/s; 59.5 % del techo calculado, suelo de 16.87 ms) | 2 pares intercalados; tokens idénticos | `6a5604bf` |
| Decode especulativo DSpark | 2 nodos | 8 prompts por tipo, 256 tokens cada uno | 1 | **70.6 tok/s** (código) | 41.9 (inglés) / 45.3 (español); 50.0 sobre los 24 | `cab238d6` |
| Decode por batch, sin draft | 2 nodos | benchmark de decode N8, todas las filas iguales a la referencia | 8 | **70.24 ms/step** (113.9 tok/s agregados, calculado; 66.2 % del techo calculado, suelo de 46.48 ms) | | `6a5604bf` |
| Prefill en dispositivo (paralelismo de expertos) | 2 nodos | prompt de benchmark de 1K, tipo no especificado | 1 | TTFT **1.954 s** | mediana de 4; logits idénticos bit a bit en ambos ranks | `6a5604bf` |
| Prefill en dispositivo | 2 nodos | prompt de benchmark de 8K, tipo no especificado | 1 | TTFT **9.757 s** | mediana de 3; logits idénticos bit a bit en ambos ranks | `6a5604bf` |
| Prefill en dispositivo | 2 nodos | 16K | 1 | TTFT **20.40 s** (803 tok/s) | | `f9332f3e` |
| Prefill en dispositivo | 2 nodos | 32K | 1 | TTFT **39.6 s** (828 tok/s), 2 ejecuciones | | `43086d05` + valores predeterminados de prefill |
| Servidor, un cliente, chat | 2 nodos | prompt de chat | 1 | **66.9 tok/s** (código) | 34.9 tok/s (prosa) | versión publicada |
| Servidor, 8 chats × 128 tokens | 2 nodos | prompts de chat | 1 / 2 / 4 / 8 | **41.4 / 54.0 / 59.7 / 67.9 tok/s** agregados | | versión publicada |
| Servidor, conjunto corto de 8 categorías | 2 nodos | prompts de 36-303 tokens | 1 / 2 / 4 / 8 | | **39.88 / 61.66 / 66.53 / 74.25 tok/s** agregados | `cab238d6` |
| Servidor precalentado, conjunto corto de 8 categorías | 2 nodos | el mismo conjunto corto que arriba | 1 / 2 / 4 / 8 | **48.00 / 63.97 / 69.20 / 77.00 tok/s** agregados | 64/64 IDs iguales; ejecución C8 limpia (véase abajo) | `6a5604bf` |
| TTFT del servidor | 2 nodos | 8K, servidor caliente inactivo | 1 | **11.80 s** | **26.40 s** mientras otros 4 flujos hacen decode; primera petición a servidor recién arrancado 12.72 s | `cab238d6` |

**Activado por defecto** en la compilación del par (`6a5604bf`): fusión de shared-down
(`STRIX_DS41_SHARED_DOWN_FUSED=0` para desactivarla), kernel de prefill MoE con gate/up separados "forma 7"
(`STRIX_DS41_MOE_PREFILL=0`), selector exacto de prefill denso por sitio (`STRIX_DS41_DENSE_SELECT=0`;
las formas FP16 siguen desactivadas), plegado de ventana de atención / fragmentos vivos / tile f16
(`STRIX_DS41_ATTN_WINDOW_FOLD=0` / `STRIX_DS41_ATTN_LIVE_CHUNKS=0` / `STRIX_DS41_ATTN_TILE_F16=0`),
y precalentamiento del servidor + calibración de DSpark (`STRIX_DS41_SERVE_WARM=0` / `STRIX_DS41_DSPARK_CALIB=0`).
Mejoras medidas frente a `69c6bdab`, con los mismos turnos de GPU, el perfil de expertos Q2 + densos de 6.5 bits/trellis,
y dos nodos Strix Halo con paralelismo tensorial de expertos sobre USB4:

| Medición | `69c6bdab` → `6a5604bf` | Cambio | Verificación / condiciones |
|---|---:|---:|---|
| TTFT de prefill en dispositivo, 1K | 2.098 → **1.954 s** | −6.9 % | mediana de 4 |
| TTFT de prefill en dispositivo, 8K | 11.090 → **9.757 s** | −12.0 % | mediana de 3 |
| Decode greedy, R1 | 28.873 → **28.333 ms/token** | −1.9 % | 2 pares intercalados; tokens idénticos; 35.3 tok/s, 59.5 % del techo calculado (suelo de 16.87 ms) |
| Ventana de verificación de seis filas, R6 | 54.01 → **52.70 ms/window** | −2.4 % | idéntico |
| Ocho flujos, sin draft, N8 | 71.84 → **70.24 ms/step** | −2.2 % | todas las filas iguales a la referencia; 113.9 tok/s agregados (calculado), 66.2 % del techo calculado (suelo de 46.48 ms) |
| Servidor precalentado, conjunto corto de 8 categorías, C1 / C2 / C4 / C8 | 43.58 / 61.13 / 66.53 / 75.72 → **48.00 / 63.97 / 69.20 / 77.00 tok/s** agregados | +9.9 / +3.0 / +2.9 / +1.7 % | IDs de token 64/64 iguales en todas las ejecuciones |
| Streaming, un solo cliente | 54.67 → **58.77 tok/s** | +7.4 % | misma campaña de servidor precalentado |

El prefill usa prompts de benchmark de tipo no especificado; los logits son idénticos bit a bit a la referencia archivada
en todas las ejecuciones y en ambos ranks. El tipo de prompt y el contexto no se especifican en los benchmarks de decode cortos.
Servidor: dos ejecuciones por brazo en ambos órdenes; una de las dos ejecuciones de la nueva compilación tuvo un único bloqueo de una unidad de prefill
de 3.8 s (causa en investigación), por lo que C8 usa la ejecución limpia. La compilación del par medida sustituye
a la regresión anterior en el servicio con un solo cliente: sirve más rápido que `69c6bdab`.

**Activados por defecto** (exactos, mismos logits e ids generados):
kernel de atención por fragmentos «lean» (`STRIX_DS41_ATTN_LEAN=0` para desactivarlo), kernels MoE de prefill
separados v2 (`STRIX_DS41_PREFILL_MOE_V2=0`) y lectura anticipada de Engram entre piezas de prefill del servidor
(`STRIX_DS41_ENGRAM_LOOKAHEAD=0`). TTFT del servidor, un cliente, caché desactivada, prompts en inglés en frío,
2 nodos; la columna del valor predeterminado actual incluye además los cambios predeterminados posteriores hasta el PR #184:

| Prompt | Antes (main #148) | Estas tres palancas | Predeterminado actual (tras #184) | Cambio frente a antes |
|---|---:|---:|---:|---:|
| 8,399 tokens | 10.44 s | 9.70 s (−7.1 %) | **8.29 s** | −20.6 % |
| 32,499 tokens | 36.85 s | 33.79 s (−8.3 %) | **29.77 s** | −19.2 % |
| 44,699 tokens | 53.62 s | 48.27 s (−10.0 %) | **42.72 s** | −20.3 % |

Los logits y los 32 ids generados son iguales a las referencias publicadas en las tres longitudes. Por palanca (mismo
turno de GPU o turnos separados, una carga por brazo): MoE v2 −5.9 / −6.9 / −8.0 %, lectura anticipada de Engram
−3.3 % con 32,499 y −2.8 % con 44,699 tokens, atención «lean» −1.4 a −2.4 %.

También **activado por defecto**: el cono del decodificador diferido de las piezas de prefill no finales
(`STRIX_DS41_DEFER_DECODER=0` para desactivarlo). Las piezas anteriores de un prompt largo ya no ejecutan el cono de
las capas del decodificador; la última pieza construye todo lo que necesita el decode. Mismos logits e ids generados.
TTFT del servidor, un cliente, caché desactivada, prompts en inglés en frío, 2 nodos, perfil predeterminado, media de
dos turnos de GPU por brazo:

| Prompt | Palanca desactivada | Ahora | Cambio |
|---|---:|---:|---:|
| 8,399 tokens (una pieza) | 8.30 s | **8.19 s** | - |
| 32,499 tokens (dos piezas) | 29.90 s | **28.92 s** | −3.3 % |
| 44,699 tokens (tres piezas) | 43.07 s | **40.51 s** | −5.9 % |

`--profile fast` activa la palanca MoE de orden relajado `STRIX_DS41_FAST_MOE=1`: **7.56 / 26.33 / 36.77 s** con los
mismos prompts, frente a 8.36 / 29.38 / 41.55 s de `default` en la misma ejecución (−9.6 / −10.4 / −11.5 %; compilación
con el cono del decodificador diferido, anterior al down FAST de 32 filas descrito abajo). Cambia los logits; su cualificación de calidad está en curso, y `default` no cambia.
Dentro de `--profile fast`, la proyección down del MoE relajado trabaja ahora por defecto en grupos de 32 filas
(`STRIX_DS41_FAST_MOE_DOWN_ROWS=64` restablece los grupos anteriores; mismos bits de salida FAST). La forma de 64 filas
desbordaba registros. Prefill del par dentro de la aplicación, cuatro ejecuciones alternas: 8.4K 7.863 → 7.533 s
(−4.2 %), 32.5K 27.509 → 27.077 s (−1.6 %); el kernel down por sí solo es un 13-25 % más rápido.
En `default`, los bloques de prefill denso de 896 filas o más ejecutan ahora las dos mitades K de o_b en un solo
lanzamiento (`STRIX_DS41_PREFILL_O_B_ONE=0` restablece los dos lanzamientos; toda salida que no sea NaN es idéntica bit a bit).
TTFT del servidor en el par, ambos órdenes de ejecución, mismos logits e ids generados: 8.4K 8.069 → 7.935 s (−1.7 %),
32.5K 27.41 → 27.18 s (−0.8 %). La tabla anterior es previa a este cambio y no se volvió a medir con él.

`--prefill-rows auto` se detiene ahora en piezas de 16384 filas: desde el cono del decodificador diferido, las piezas de
32768 filas son más lentas en el par (8.01 / 28.30 / 39.94 s frente a 7.94 / 27.44 / 38.98 s con 8,399 / 32,499 / 44,699
tokens, mismo binario, una carga por brazo; 25.03 frente a 24.44 s con 32,499 en `--profile fast`); un
`--prefill-rows 32768` explícito sigue funcionando. Los flujos residuales guardan ahora solo las filas de este rank
(80 KiB menos por fila, 1.25 GiB con piezas de 16384 filas; sin efecto en la velocidad). Mismos logits e ids generados.
Con ambos, TTFT del servidor, un cliente, caché desactivada: **7.94 / 27.73 / 38.61 s** (1,057 / 1,172 / 1,158 tok/s)
en `default` y 7.05 / 24.32 / 32.95 s (1,192 / 1,336 / 1,357 tok/s) con `--profile fast` en la misma ejecución
(−11.3 / −12.3 / −14.7 %).

<a id="deepseek-rtx-pipeline"></a>

## Resultados medidos: RTX PRO + par (división en pipeline)

**DeepSeek con la RTX PRO, división del pipeline** (los mismos ficheros de modelo; la RTX ejecuta las capas 0-21 con CUDA, los dos nodos ejecutan
las capas 22-39 y la cabeza):

| Modo | Hardware | Prompt / contexto | Concurrencia | Mejor | Típico | Build |
|---|---|---|---:|---:|---:|---|
| Decode por batch, sin draft | RTX PRO + 2 nodos | 8 prompts de benchmark, 48 tokens; 2 grupos en curso | 2 / 4 / 8 | **60.1 / 97.5 / 137.2 tok/s** agregados | igual bit a bit a la referencia solo del par | par `6525d5e8`, línea CUDA `463c08ec` |
| Decode DSpark por batch | RTX PRO + 2 nodos | mismo benchmark, ventanas del controlador | 1 / 2 / 4 | **52.4 / 91.3 / 156.8 tok/s** agregados de reloj | gates de referencia bit a bit | misma línea |
| Servidor, conjunto corto de 8 categorías | RTX PRO + 2 nodos | prompts cortos, DSpark; 2 grupos en curso | 1 / 2 / 4 / 8 | **43.2 / 65.8 / 77.9 / 91.7 tok/s** agregados | IDs de token iguales a solo el par en 32/32 peticiones | par `6525d5e8`, CUDA `463c08ec` |
| Lectura del prompt en la RTX PRO (perfil exacto) | RTX PRO | 2K / 8K / 32K | 1 | TTFT **3.64 / 4.39 / 12.20 s** | | versión publicada |

El pipeline de batch/servicio solapa dos grupos de peticiones: la RTX procesa uno mientras el par procesa
el otro (un solo grupo para una única petición). Estas filas de batch/servicio usan expertos Q2, la superposición densa de 6.5 bits y la identidad expert-TP
con gate propio; no usan el perfil compuesto con trellis de petición única.

La división RTX de petición única
es igual bit a bit al par sin los sitios trellis; con los sitios trellis es una identidad compuesta
declarada, comprobada frente al par (KL 1.7e-3), no bit a bit. TTFT de la RTX: servicio residente caliente, carga del modelo
excluida, hasta el estado de atención exportado entregado al endpoint de importación. Un perfil de prefill CUDA más rápido no supera
el gate de calidad en prompts largos y no se cita.

<a id="deepseek-standalone-rtx-results"></a>

## Resultados medidos: RTX independiente

**DeepSeek en RTX independiente — Production.** Modelo objetivo DeepSeek V4.1 Flash completo en una
RTX PRO 6000 (96 GB, Windows WDDM): GGUF Q2 con gate/up enrutados **IQ2_XXS**, down enrutado **Q2_K**, densos **Q8_0**.
Los expertos usan VRAM exclusiva + una caché en memoria de host fijada; el modelo objetivo y el draft de DSpark comparten la GPU. El draft conserva
los 64 expertos más usados por etapa. Se han adoptado las escrituras diferidas/acotadas (write-backs), la corrección de instanciación de grafos CUDA y
el controlador caliente por defecto; la poda keep-64 midió **+16 %** en código con una ventana de cuatro filas.

| Modo | Código | Inglés | Español | Proporción del techo calculado |
|---|---:|---:|---:|---|
| Decode DSpark exacto, controlador caliente + write-backs diferidos, misma carga | **82.3 tok/s** | **87.7 tok/s** | **94.4 tok/s** | **51–58 %** del suelo de 6.18 ms/token con todos los aciertos |
| Decode DSpark exacto, mejores ejecuciones entre cargas | **94–95 tok/s** | **97 tok/s** | **100–101 tok/s** | **58–63 %** del mismo techo |

Cite el rendimiento en código como **82–95 tok/s**: las cargas varían hasta un **~10 %**, por lo que las comparaciones son válidas dentro de una misma carga.
Los gates exactos se superan para este perfil RTX; no se afirma igualdad con el perfil trellis del par.
Las longitudes de prompt y los recuentos de tokens generados no se especifican para estas últimas mediciones de decode.

| Trabajo de carga / servicio | Resultado medido | Estado |
|---|---|---|
| Carga de expertos en pipeline | Carga completa **108.7 → 61.8 s**; llenado de expertos **83.1 → 38.0 s**, **2.25 → 6.73 GB/s**, limitado por el disco | **Publicado, activado por defecto** ([PR #52](https://github.com/hyukudan/strix-halo-engine-lab/pull/52)); colocación y salidas idénticas bit a bit |
| Worker persistente cargado | **8–18 s por petición** en cargas de prueba sin recarga; reinicio de caché con carga nueva **3.4 s** | **Publicado** (`--serve`, PR #52); cada petición es idéntica bit a bit a un proceso nuevo |
| Prefetch de ventana | Sin ganancia repetible en comparaciones emparejadas en caliente | Exacto; desactivado por defecto |

**Arranque en frío, mismo perfil de RTX independiente.** Worker persistente,
binario probado `1569bce5`; los interruptores de write-back/flush en tiempo de ejecución preservan los tokens, logits y estado de un proceso nuevo
(6/6 comparaciones offline). Son tiempos por paso, no rendimiento de chat ni TTFT:

| Configuración | Paso 0 | Media pasos 0–15 | Media pasos 16–63 |
|---|---:|---:|---:|
| Carga nueva | 332.6 ms | 56.05 ms | 28.15 ms |
| Carga nueva con semilla de residencia guardada | 156.0 ms | 44.67 ms | 25.66 ms |
| Worker con precaptura de grafos | 11.0 ms | 33.99 ms | 27.26 ms |
| Worker, caché de expertos caliente | 10.9 ms | 11.63 ms | 11.49 ms |

La precaptura de grafos añade **1.5 s antes de estar listo**. El tipo y la longitud del prompt no se especifican para este benchmark
de la fase exacta. El worker, la semilla de residencia y la precaptura de grafos se publican con
[PR #52](https://github.com/hyukudan/strix-halo-engine-lab/pull/52): los ajustes exactos adoptados (keep-64, write-backs
diferidos, controlador caliente, paquete de kernels de decode y fusiones) están activados por defecto con interruptores de desactivación, y una ejecución sin flags
coincidió con la configuración explícita (todos los gates 64/64, trazas de especulación idénticas, 21/21 tokens de chat) en una caché
fijada reducida, por lo que ese gate no aporta cifras de velocidad. El rango medido de **82–95 tok/s** en código sigue siendo el resultado citado.
Compilación, ejecución y valores predeterminados: [docs/models/deepseek-v4.1/rtx-standalone.md](rtx-standalone.md).

## Todos los valores predeterminados incorporados de DeepSeek

El registro de cambios de cada palanca de DeepSeek activada por defecto (par y RTX), con su ganancia medida, su
evidencia de exactitud y su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el
catálogo completo es [docs/optimizations.md](../../optimizations.md#deepseek-two-node) y
[docs/optimizations.md#deepseek-rtx-pro-standalone](../../optimizations.md#deepseek-rtx-pro-standalone). Valores
predeterminados de producción del par: [production-defaults.md](production-defaults.md).
Servicio: [servicio](../../serving.es.md).
