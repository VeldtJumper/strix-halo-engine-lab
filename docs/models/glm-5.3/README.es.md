# GLM-5.3 Flash: el par USB4 y una RTX PRO 6000

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

GLM-5.3 Flash, EXL3 4.05 bpw. En dos nodos Strix Halo (TP2 sobre USB4) está en Production: identidad de decode R1b por
defecto, autoespeculación MTP exacta por defecto, prefill exacto R1 por chunks. En una RTX PRO 6000 (Windows) es un
borrador opcional (opt-in): prefill exacto y decode greedy con los expertos enrutados servidos a través de la caché
por niveles de tierstream. Esta página conserva todas las mediciones publicadas con su compilación y sus condiciones y
los comandos de lanzamiento. Cómo se leen las tablas y qué cubre "exacto": [metodología](../../methodology.es.md).
Instrucciones de compilación (biblioteca trellis incluida): [primeros pasos](../../getting-started/README.es.md); el par
en sí: [dos nodos por USB4](../../pair.es.md).

<a id="glm-pair"></a>

## Ejecución: el par

Ejecute una vez por nodo con `R` establecido en 0 o 1 y el mismo prompt tokenizado en ambos:

```sh
./build/strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

El decode R1b predeterminado usa autoespeculación MTP exacta y prefill de prompt R1; `--mtp 0` ejecuta decode simple
con los mismos tokens. `--identity r1` selecciona una identidad de referencia distinta. El prefill usa automáticamente
8192 filas si ambos ranks las admiten; en caso contrario, 4096. Véanse el [resolutor](../../../apps/glm_rank_defaults.hpp),
las [familias de modelos](../model-families.md) y la [cualificación/rendimiento](../../performance.md).

## Ejecución: una RTX PRO 6000

Solo Windows, opcional (opt-in): compile con `pwsh src/glm/rtxpf/producer/build.ps1` contra una exportación del código
de tierstream, genere una vez el archivo de unidades con `glm_unit_file.exe` y ejecute
`decode_qualification_window.py --full-card --allow-gpu`. El propio servicio acepta también `--dgpu auto|<uuid>`
para elegir la RTX PRO antes de iniciar CUDA (`off` se rechaza: no hay ruta en CPU). Todos los pasos, el commit fijado de tierstream y las
opciones (`--profile`, `--print-effective-config`, `--vram-total-gib`) están en la
[guía de GLM RTX](../../../src/glm/rtxpf/README.md).

<a id="how-fast"></a>

## Resultados medidos: el par

Fuentes: [registros de rendimiento](../../performance.md) y [valores predeterminados/pruebas actuales de GLM](../../optimizations.md#glm).

EXL3 4.05 bpw, dos nodos Strix Halo TP2 sobre USB4. Motor en Production. R1b es la identidad de decode predeterminada; los prompts usan prefill por fragmentos R1 exacto;
el decode usa autoespeculación MTP exacta por defecto (`--mtp 0` = decode simple).
`--identity r1` restaura R1 en todo el proceso. Las mediciones históricas de R1 siguientes conservan su identidad original.

El prefill del prompt en las ejecuciones de decode con dos ranks usa por defecto 8192 filas cuando ambos ranks admiten la carga calibrada
y todos los propietarios del prefill; en caso contrario, ambos usan 4096. Esto se aplica al decode de producción con autoespeculación MTP
y al prefill por fragmentos simple dentro de ejecuciones `--decode`. La selección usa el margen de RAM
y GTT de la admisión de carga, el staging, las reservas, las lecturas de tensores completos y el búfer de rebote, más el crecimiento de prefill y de comunicación
y la protección de memoria libre del dispositivo.
Se conserva el mayor entre el margen de carga, el margen de prefill y 8 GiB. El decode Comm normal con dos ranks reutiliza
su intercambio de configuración existente y su voto de discrepancia para acordar el modo y el mínimo de filas propuestas.
Los modos fijos conservan sus bytes de carga útil y el orden de etiquetas; los modos mixtos auto/fijo se rechazan en ambos ranks.
Los pools de comunicación automáticos empiezan en 4096 y solo crecen tras acordar 8192. Una nueva comprobación de admisión
tras el crecimiento usa un único voto mínimo solo automático y reduce ambos pools en caso de volver a 4096.
Las ejecuciones Comm con dos ranks sin ese handshake (solo prefill, gates de referencia y solo batch) usan 4096 en modo
auto, con origen `auto-4096-no-handshake`, sin intercambio de control. Los recuentos de filas explícitos deben coincidir.
`--pf-ttft` y los arneses de gates mantienen 4096 salvo que se indique explícitamente `--max-rows 8192`.
Ahí no se pueden comprobar discrepancias de filas fijas al arrancar sin
cambiar el protocolo existente. `--xch xfer` externo en auto usa 4096 sin intercambio de acuerdo.
El prefill de referencia (`--prefill ref`, prefill automático con `r1b-all` o `--idcheck` con decode) usa
4096 en modo auto (`auto-4096-ref`) sin intercambio de acuerdo del recuento de filas.
La ruta de prefill exacta conserva su aritmética entre tamaños de fragmento; el TTFT medido en la ruta de prefill por fragmentos
con 8192 filas mejora un 4 % a 8K y de 116.3 a 109.9 s (5.5 %) a 32K, usando unos 7.3 GiB más de
memoria de GPU por nodo.
`STRIX_GLM_MAX_ROWS_AUTO=0` restaura el valor fijo 4096; `--baseline` mantiene 1024, y un `--max-rows` explícito mantiene su valor.
`--print-effective-config` informa del candidato independiente del host y de la política de arranque; stderr informa
del margen de admisión y del valor final con su origen. Véase el [catálogo](../../optimizations.md#glm-max-rows-auto).

| Modo | Prompt / contexto | Medido | Build / exactitud |
|---|---|---:|---|
| Decode greedy, R1 | prompt de chat de 31 tokens + 64 tokens generados | **45.29 ms/token = 22.1 tok/s**; **51 %** del suelo calculado de 23.2 ms | `1836c99c`; tokens iguales a la referencia |
| Decode por batch, R1 | Ocho filas independientes, contexto del benchmark no especificado | **122.0 ms/step = 65.5 tok/s agregados** | `1836c99c`; tokens iguales a la referencia anterior del par real |
| Prefill por fragmentos exacto | prompts de benchmark de 1K / 8K / 32K, tipo no especificado | TTFT **4.268 / 34.70 / 144.75 s** (240 / 236 / 226 tok/s) | `875ca821`, integrado en `41d8cfdc`; los logits de ambos ranks son iguales bit a bit al prefill exacto anterior |
| R1b-fast, ahora la identidad de decode predeterminada | Mismo benchmark de chat, comparación alterna de 96 tokens | **41.93–41.96 ms/token**, frente a R1 47.33–47.48 (**−11.5 %**, **55 %** del suelo de 23.2 ms) | `8877de1d`; estable dentro de cada identidad, no igual bit a bit a R1 |
| Autoespeculación MTP (predeterminada) | Prompts de chat / código / JSON, 64–128 tokens generados, rank 0 | **26.98 / 22.46 / 24.98 ms/token = 37.1 / 44.5 / 40.0 tok/s** (1.50× / 1.79× / 1.62× el simple de 40.3–40.6 ms/token); TTFT 8K **31.13 s** (+2.9 %), 32K **140.9 s** (+8.7 %: llenado de la caché MTP de la capa adicional) | Vectores de tokens completos y logits del prompt iguales a `--mtp 0` en 7 prompts, ambos ranks; TTFT simple sin cambios |
| Prefill por fragmentos exacto, núcleo lineal X2 (predeterminado) | prompts de benchmark de 1K / 8K / 32K | TTFT **3.78 / 27.93 / 120.26 s** (271 / 293 / 272 tok/s); misma ejecución con `STRIX_GLM_PF_EXACT_X=0`: 1K 3.84 s, 8K 29.59 s (**−5.6 %** a 8K) | logits iguales bit a bit a la ruta X=0 en ambos ranks; anclas de decode y oráculo especulativo sin cambios |
| Prefill por fragmentos exacto, núcleo lineal X2 + MoE auto (predeterminado) | prompts de benchmark de 1K / 8K / 32K | TTFT **3.74 / 26.85 / 113.73 s** (274 / 305 / 288 tok/s); misma ejecución con `STRIX_GLM_PF_EXACTMOE_V=3`: 1K 3.76 s, 8K 28.17 s (**−4.7 %** a 8K) | salidas iguales bit a bit a la ruta v3 en ambos ranks (las cargas útiles NaN FP32 internas solo pueden diferir en el caso de desbordamiento FP16); anclas de decode y oráculo especulativo sin cambios |
| Prefill por fragmentos exacto, núcleo MoE X3 y cubetas de filas (PR #183) | prompts de benchmark de 1K / 8K, fragmentos de 8192 filas | TTFT **3.18 / 20.32 s**; misma ejecución con `STRIX_GLM_PF_EXACTMOE_V=7`: 1K 3.75 s, 8K 25.24 s (**−15 % / −19.5 %**) | dos pares por brazo; R1 bit a bit (mismas cadenas DOT2, reduce4 y orden de división) |
| Prefill por fragmentos exacto, cubetas de cola X3 + X3 denso + plan MoE en el dispositivo (PR #215) | prompts de benchmark de 8K / 32K, fragmentos de 8192 filas | TTFT **17.77 / 76.59 s** (461 / 428 tok/s); misma ejecución con los predeterminados anteriores (PR #183): 20.22 / 86.66 s (**−12.1 % / −11.6 %**) | un par por brazo; los logits de cada rank son iguales a las anclas públicas con 8K y 32K en ambos brazos (mismas cadenas DOT2, reduce4 y orden de división) |
| Prefill por fragmentos exacto, kernels de atención: MLA scores v5 / o-bar v7 / GEMM absorb-out v2, indexer scores v3 / select v2 (medido con los predeterminados de PR #183, antes de PR #215) | prompts de benchmark de 8K / 32K, fragmentos de 8192 filas | TTFT 32K **−6.8 %** (−7.2 s, media de ambos órdenes: 105.44 → 103.66 s con los kernels anteriores primero, 110.31 → 97.66 s con los nuevos primero); kernels de atención a 32K en el rango 0 **18.6 → 10.3 s**; 8K **25.23 → 23.62 s (−6.4 %)**, dos pares, sin la GEMM v2 | logits bit a bit iguales a los kernels anteriores en ambos rangos; mismo binario, kernels anteriores con sus interruptores; medido antes del núcleo MoE X3 (PR #183) |
| Prefill por fragmentos exacto, KDA scan v3 + MLA scores v6 (medido en f7e063b7e; código de prefill del par sin cambios hasta fb66457e4) | prompts de benchmark de 8K / 32K, fragmentos de 8192 filas | TTFT 32K **−1.5 %** (−1.04 s, media de ambos órdenes: 68.81 → 67.26 s con los kernels nuevos primero, 68.71 → 68.17 s con los anteriores primero); kernels del rango 0 a 32K: KDA scan **2.18 → 1.43 s**, MLA scores **2.96 → 2.62 s**; 8K solo con KDA scan 16.63 → 16.34 s (−1.7 %) | logits bit a bit iguales a los kernels anteriores en ambos rangos; mismo binario, kernels anteriores con `STRIX_GLM_PF_KDA_SCAN=1` `STRIX_GLM_PF_SCORES=5`; el rango 0 corrió bajo el trazador de kernels en todos los brazos |
| Concordancia R1b con teacher forcing | 32 prompts, 256 posiciones cada uno; chat, código, JSON, matemáticas, razonamiento, traducción | **8,126/8,192 = 99.19 %** top-1; KL media **~1.8e-3 nats** | `4dad4c19`; las ejecuciones libres divergen en **28/32** prompts entre 19 y 227 tokens |

La corrección del búfer R≥3 está verificada en el par real. La integración más reciente usa intercambio directo para los forwards
simples y MoE v3 exacto para el prefill; el oráculo especulativo también se supera con conmutación de intercambio drenado.
Las filas de prefill son ejecuciones únicas. Sustituyen al resultado anterior, solo sintético.

R1b cambia el orden de la suma FP32 y no es idéntico bit a bit a R1; qué identidad está más cerca de float64 a profundidad completa
sigue sin resolverse. Los grafos HIP **no están implementados**: el perfil más reciente (`41d8cfdc`) mantiene la GPU ocupada el **96.3 %** de cada forward (antes 80.8 %; 909 lanzamientos en lugar de 1,679) y deja solo **2.10 ms de inactividad por forward**
y un hueco recuperable calculado de **≤1.8 ms** (una cota, no una aceleración medida). El avance del host a través de
los límites de intercambio requiere captura segmentada; los kernels de dispositivo y el desfase entre ranks tienen prioridad.

<a id="glm-rtx-results"></a>

## Resultados medidos: una RTX PRO 6000

**GLM-5.3 Flash en una RTX PRO 6000 (borrador, opcional, Windows).** El tronco, las capas densas y los expertos
compartidos se quedan en la tarjeta; los 43 x 288 expertos enrutados los sirve la caché con compuerta en el dispositivo
de [tierstream](https://github.com/hyukudan/tierstream), y el prefill los lee a través de la misma caché, así que la
tarjeta guarda una sola copia de cada experto. El prefill y el decode son exactos (digests bit a bit de logits y
estado; estado del prefill igual con tiles de 2048 / 4096 / 8192 filas). Tarjeta completa, tile de 8192: TTFT **19.7 s** con 8K y
**41.7 s** con 32K (exacto; la restauración de la residencia del decode se ejecuta ahora después del prefill, fuera del
tiempo hasta el primer token, y solo cuando sigue un decode); decode de **6.00 / 6.15 tok/s** en caliente
con 1K / 8K (3.75 / 3.70 en frío) con los kernels de decode de pocas filas activados por defecto, frente a 2.43 / 2.36
tok/s con ellos desactivados en la misma carga del servicio (tarjeta completa; la RAM del host dejó 1,098 unidades de
expertos en el archivo de unidades, así que se incluyen los fallos de archivo por token; prueba de 24 tokens). Los
tokens, todos los digests de logits y el estado final de la sesión son iguales bit a bit con ellos activados y
desactivados, y la cualificación de 1K/8K pasa con ellos activados. Los kernels decodifican cada tesela de pesos
directamente en fragmentos de tensor cores y mantienen el orden de acumulación de los GEMM fijados; desactívelos con
`--decode-row-kernels 0` o `STRIX_GLM_RTX_ROW_KERNELS=0`.
El solapamiento del prefill está activado por defecto: mientras corren los kernels de una capa, los expertos de la
siguiente se hacen residentes. El estado del prefill, los tokens y los logits son idénticos bit a bit con él desactivado.
En comparaciones dentro de una misma ventana deja el TTFT en 0.56× (8K) y 0.66× (32K) del tiempo sin solapamiento con la
tarjeta completa. Se desactiva con `--prefill-overlap 0`.
Con 24 GiB de VRAM + 32 GiB de
RAM del host (la mayoría de los expertos leídos del archivo de unidades en disco) también funciona de forma exacta: TTFT
**60.3 s** con 1K y **62.4 s** con 8K con el solapamiento del prefill (63.4 / 73.7 s sin él en la misma ventana); decode
de **0.67 tok/s** en caliente (0.63 en frío). Compilación, archivo de unidades y opciones: [guía de GLM RTX](../../../src/glm/rtxpf/README.md).

## Todos los valores predeterminados incorporados de GLM

El registro de cambios de cada palanca de GLM activada por defecto, con su ganancia medida, su evidencia de exactitud y
su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el catálogo completo es
[docs/optimizations.md](../../optimizations.md#glm). Compilación, fichero de unidades y opciones de GLM RTX:
[guía de GLM RTX](../../../src/glm/rtxpf/README.md); notas del prefill nativo en RTX:
[docs/models/glm/rtx-prefill-native.md](../glm/rtx-prefill-native.md).
