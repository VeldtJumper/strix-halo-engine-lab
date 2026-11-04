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
con los mismos tokens. Ambos rangos arrancan el temporizador del prefill tras una barrera del par, de modo que los dos
informan el mismo tiempo de prefill y ninguno incluye el tiempo de carga del otro (se imprime la espera en la barrera;
`STRIX_GLM_PF_TIMER_BARRIER=0` restaura el arranque anterior). `strix-tokenize MODEL_DIR encode|plain|decode FILE` ejecuta el tokenizador del motor (el
mismo `MODEL_DIR/tokenizer.json` que carga el servidor) en la línea de comandos. `--identity r1` selecciona una identidad de referencia distinta. El prefill usa automáticamente
8192 filas si ambos ranks las admiten; en caso contrario, 4096. Véanse el [resolutor](../../../apps/glm_rank_defaults.hpp),
las [familias de modelos](../model-families.md) y la [cualificación/rendimiento](../../performance.md).

<a id="glm-serve"></a>

## Servicio: el par

`strix-glm5-serve` es el servidor HTTP (`/v1/chat/completions` y `/v1/completions` de OpenAI con streaming SSE,
`/v1/models`, `/health`). Arranque primero el rank 1 y después el rank 0; ambos reciben las mismas opciones del motor
(el saludo inicial rechaza una discrepancia) y leen el tokenizador de `MODEL_DIR/tokenizer.json`:

```sh
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 1 --transport usb4-auto
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 0 --transport usb4-auto --host 0.0.0.0 --port 8080 --model-id glm-5.3-flash
```

Valores predeterminados: 8 slots de petición (`--slots`, 8 como máximo), 8192 posiciones por slot (`--capacity`),
autoespeculación MTP exacta con una sola petición en curso, verificación MTP por lotes con varias
(`STRIX_GLM_SERVE_G2=0` = pasos simples por lotes) y prefill por chunks de 512 tokens con prioridad al decode mientras
otras peticiones decodifican (`--prefill-chunk 0` = prompts enteros). `--serial` admite una petición cada vez;
`--api-key-file F` exige una clave bearer. `--selftest N` en el rank 0 (sin HTTP) ejecuta N prompts integrados juntos
y cada uno por separado e informa EXACT/FAIL con tok/s por nivel de concurrencia.

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
| Prefill por fragmentos exacto, KDA scan v5 (sobre los predeterminados de PR #254) | prompt de benchmark de 8K, fragmentos de 8192 filas | TTFT **16.60 frente a 16.60 s** (media de 8 brazos, ambos órdenes: neutro); scan del rango 0 **0.355 → 0.340 s**; scan por capa 1.23–1.25 → 1.18–1.20 µs/fila (−3.7 %) | logits bit a bit iguales a v3 en ambos rangos; mismo binario, v3 con `STRIX_GLM_PF_KDA_SCAN=3`; el rango 0 corrió bajo el trazador de kernels; el cuarto brazo de cada par fue ~5 % más lento en todos los kernels en ambos órdenes |
| Prefill por fragmentos exacto, palancas del par: expansión mHC + norma fusionadas e intercambio de prefill por staging (predeterminados actuales) | prompts de benchmark de 1K / 8K, fragmentos de 8192 filas, en caliente | TTFT **2.83 / 15.27 s** con todas las palancas del par activadas; intercambio por staging, ABBA: 1K 2.856 / 2.860 → 2.832 / 2.838 s, 8K 15.43 / 15.41 → 15.28 / 15.25 s (**−0.8 % / −0.9 %**, todos los brazos con staging más rápidos); expansión mHC + norma fusionadas, ABBA ×2: **−1.1 % / −1.3 %** | anclas de prefill exactas 16/16; logits bit a bit iguales a cada palanca desactivada (`STRIX_GLM_PF_STAGED=0`, `STRIX_GLM_PF_HC_FUSE=0`) |
| Norma HC fusionada en el decode (predeterminada) | decode C1 con MTP y lote R8, ABA ×2 | **−2.3 %** C1 MTP, **−3.1 %** R8 | tokens y R8 iguales a la referencia; `STRIX_GLM_HC_NORM=0` restaura el cierre sin fusionar |
| Sitio mHC del decode en un lanzamiento (predeterminado) | decode C1 `--mtp 0` y lote R8, ABBA con proceso nuevo | **−1.45 %** C1, R8 sin cambio | tokens y R8 iguales a la referencia en ambos rangos; `STRIX_GLM_HC_SITE=0` restaura los tres lanzamientos |
| Indicador del paso comprobado con la cabeza (predeterminado) | decode C1 `--mtp 0` y lote R8, ABBA con proceso nuevo | **−0.2 %** C1, R8 dentro del ruido | tokens y R8 iguales a la referencia en ambos rangos; `STRIX_GLM_STEP_FLAG=0` restaura la sincronización de host a mitad de paso |
| Servido, `strix-glm5-serve --selftest 8` (predeterminados) | Flujos concurrentes C1 / C2 / C4 / C6 / C8, régimen estable (todos los flujos en decode); 8 prompts de chat cortos integrados con la plantilla de chat de GLM, 22–59 tokens de prompt (media ~33), hasta 128 tokens greedy cada uno, arranque con barrera, caché desactivada, el servidor no tiene caché de prompts | **C1 40.0 / C2 53.0 (G2 desactivado 37.8) / C4 59.9 (53.6) / C6 62.5 (62.4) / C8 69.5 (69.4) tok/s agregados**; por flujo 40.1 / 26.5 / 15.0 / 10.4 / 8.7; C1 sin MTP 25.30; CLI del motor `--batch` R8 69–71 | RESULT EXACT: los tokens de cada flujo son iguales a los del mismo prompt ejecutado solo (referencia C=1 simple) en todos los niveles, con G2 activado y desactivado |
| Servido, prefill por chunks con prioridad al decode (`--prefill-chunk 512`, predeterminado) | 4 prompts de ~1,830 tokens que llegan cada 16 rondas de decode a 4 flujos en curso, 128 tokens generados cada uno, caché desactivada | Mayor parada del decode **6.65 → 2.67 s (−60 %)** frente a `--prefill-chunk 0`; una petición sola se procesa entera | exacto |
| Prefill por fragmentos exacto, MLA scores v7 (sobre los predeterminados de PR #279) | prompts de benchmark de 8K / 32K, fragmentos de 8192 filas | kernel de scores del rango 0 **2.60 → 2.27 s** a 32K (−12 %) y **0.533 → 0.485 s** a 8K (−9 %); TTFT no más lento: 32K 65.68 → 65.54 s (−0.2 %, media de ambos órdenes: 65.574 → 65.524 con el nuevo primero, 65.778 → 65.549 con el anterior primero), 8K 16.23 → 16.12 s (−0.6 %, 4 brazos) | logits bit a bit iguales a v6 en ambos rangos; mismo binario, v6 con `STRIX_GLM_PF_SCORES=6`; el rango 0 corrió bajo el trazador de kernels; el cambio de TTFT está dentro de la dispersión de ~1 s entre ejecuciones a 32K |
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
estado; estado del prefill igual con tiles de 2048 / 4096 / 8192 filas). Tarjeta completa, tile de 8192: TTFT **11.4 s** con 8K y
**30.5 s** con 32K (primera petición, media de dos cargas; 30.5 s con 32K frente a 43.7 s con tile de 2048, -30 %).
Con 32K el prefill está limitado por el cómputo de la GPU: la preparación de expertos (unos 128 GB de host a
dispositivo por prompt, 8-10 s de make-resident, unos 4 GB/s de media, muy por debajo de PCIe 5 x16) corre detrás
de los kernels con 0.0-0.2 s de GPU ociosa, así que el -30 % viene de menos decodificaciones de pesos de expertos
por prompt y filas de GEMM más largas por experto con el tile mayor, no de E/S; con 8K quedan expuestos 2-4 s de
preparación. (Exacto; la restauración de la residencia del decode se ejecuta ahora después del prefill, fuera del
tiempo hasta el primer token, y solo cuando sigue un decode); decode de **13.25 / 13.52 tok/s** en caliente
con 1K / 8K (3.63 / 4.09 en frío) con los kernels de decode actuales (bench de 24 tokens, medias ABBA de una carga).
Cuando se añadieron los kernels de decode de pocas filas, el decode midió 6.00 / 6.15 tok/s en caliente (3.75 / 3.70 en
frío) con ellos activados por defecto, frente a 2.43 / 2.36
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
La expulsión limpia en el prefill está activada por defecto: un residente de decode que el prefill desplaza sin
bloque fijado libre vuelve al fichero de unidades en lugar de sacar una unidad fijada. En la tarjeta completa reduce
el TTFT de 8K un 15 % / 12 % (primera / segunda petición), empata a 32K y lee un 60 % menos del fichero de unidades,
idéntico bit a bit. Se desactiva con `--prefill-clean-evict 0`.
Por defecto se asigna el mayor tile admitido hasta 8192, con crecimiento de activaciones + workspace
sobre 2048 <= 5% de la arena de expertos a 2048. Un tile mayor solo puede añadir unidades al fichero
si el plan de 2048 tiene <= 15% de las unidades enrutadas en fichero y el aumento es <= 5% de las
unidades enrutadas. Se admite 8192 en el intervalo medido con tarjeta completa (con 2048, ~1.5-4.6%
de las unidades enrutadas en fichero, ~2.4% más con 8192) y en planes dentro de estos límites si
caben el plan y el crecimiento; fuera se
conserva el veto anterior a unidades adicionales en fichero. Con tarjeta completa, 8192 frente a 2048
redujo el TTFT un 30% a 32K y un 9% / 5% a 8K, idéntico bit a bit. No establece el mejor tile a 8K:
en un par de cargas 4096 fue 1.3 / 2.0 s más rápido (primera / segunda petición), en el otro la
diferencia fue menor de 0.3 s. `--row-tile` explícito omite las reglas
automáticas de crecimiento y fichero (el plan debe caber), así como `row_tiles_tried`; los candidatos
automáticos indican los motivos de admisión y `file_units_over_2048`.
Las lecturas de fichero simultáneas usan por defecto auto según el régimen: se admite todo plan inicial
de tres niveles con cola 1 y al menos el 70 % de unidades enrutadas en fichero si staging >= 2 y caben
los lectores. Con aproximadamente el 76 % (24 GiB de VRAM + 32 GiB de RAM), auto nunca fue más lento
en las últimas medias de decode por celda; una celda anterior dio -3.4 %, atribuido a fases de lectura
de la unidad. Con aproximadamente el 61 % (48 GiB + 32 GiB), una noche anterior se observó un -32 %
en decode 1K en frío frente a la media con cola 1; las otras celdas dieron +3 % a 1K en caliente,
+7 % a 8K en frío y +1 % a 8K en caliente. Un conjunto posterior con el mismo presupuesto invirtió
el resultado de 1K en frío: auto +8.4 % agregado, todos los brazos auto más rápidos. El corte del
70 % sigue siendo un umbral empírico conservador; el 61 % conserva la cola 1.
Con tarjeta completa (aproximadamente el 1-4 % con tile 2048), el TTFT empató: 8K +0.6..+2 %, 32K igual;
se conserva la cola 1. Staging < 2 o lectores que no caben también vuelven a 1.
El `auto` explícito sigue siendo válido con cualquier proporción; `--tier-disk-queue 1` desactiva auto.
Con 24 GiB de VRAM + 32 GiB de
RAM del host (la mayoría de los expertos leídos del archivo de unidades en disco) también funciona de forma exacta: TTFT
**60.3 s** con 1K y **62.4 s** con 8K con el solapamiento del prefill (63.4 / 73.7 s sin él en la misma ventana); decode
de **0.67 tok/s** en caliente (0.63 en frío) con una lectura de fichero cada vez; con las lecturas de fichero
simultáneas por defecto con este presupuesto (auto, 76 % de los expertos en el fichero de unidades) el decode es de
**0.77 / 0.75 tok/s** en caliente / en frío con 1K (media de dos cargas, bench de 24 tokens). Con el prefetch de archivo en dos etapas opcional
(`--tier-landing-blocks 16`, tres unidades en franjas) el decode fue un 20-40 % más rápido que sin él en las mismas
ventanas, con salida idéntica (las repeticiones variaron entre un 15 y un 55 %). Compilación, archivo de unidades y
opciones: [guía de GLM RTX](../../../src/glm/rtxpf/README.md).

**Prefill en RTX hacia el par (FAST, desactivado por defecto).** `strix-glm5-serve --rtx-prefill HOST:PUERTO
--import-listen IP:PUERTO --import-peer IP:PUERTO --import-key FICHERO --import-model-sha SHA` (`--import-listen` en cada
rango; solo motores con su capa MTP) deja que el productor RTX haga el prefill de los prompts largos y entregue a cada
rango su estado. Ambos rangos preparan la entrega y votan (la vinculación autenticada y después ambos restaurados) antes
de ejecutar el último token del prompt; cualquier fallo antes de ese voto vuelve al prefill propio del par en los dos
rangos, y un fallo dentro de las operaciones colectivas del último token detiene el servidor (fatal). `--import-dsa-once`
(con `dsa_once` del productor) envía el estado DSA solo al rango 0, que lo retransmite al rango 1 por el enlace del par
(8K: de 480 a 316 MB en el enlace del productor, con tokens iguales a los de la entrega de la versión 1). Los prompts importados llevan la identidad FAST
`glm-pf-rtx-native-v1` (no son bit a bit el prefill propio del par; el `prefill_route` de cada respuesta la nombra).
`--import-gate` falla si algún brazo enrutado no importó de verdad (y una batería sin prompts también falla).
`--import-model-sha` es una atestación del operador: el receptor la compara con la cabecera de la entrega del
productor; no se calcula a partir de los tensores que cargó el par. Los ajustes de importación (receptor activo,
`--import-dsa-once`, versión del protocolo y SHA del modelo atestado) deben coincidir en todos los rangos: el hello
rechaza una discrepancia al arrancar. El bloque de control es la versión 6: un rango de esta compilación rechaza en el
hello a un rango de una compilación anterior.

Medido con la versión publicada (greedy, 128 tokens, caché desactivada, 1 GbE entre el PC con RTX y los nodos, productor
arrancado en cada ejecución): el tiempo hasta el primer token con la importación fue de 39.4 s a 1K y 45.6 s a 8K, frente a
2.83 s y 15.7 s con el prefill exacto propio del par (con MTP: 38.6 frente a 2.91 s y 44.5 frente a 16.3 s), por lo que la
ruta no es más rápida en esta configuración. Casi todo el tiempo importado es el prefill en la RTX (38-44 s): esa
ejecución arrancó el productor sin su caché de tierstream, de modo que cargaba capas de expertos desde los fragmentos
del modelo en cada petición (el README del productor muestra el arranque con caché); el coste de una importación en el par es de unos 0.4 s (cola
de la transferencia, restauración, prefill del último token y commit). Que compense depende del tiempo de prefill en la
RTX para prompts largos. Con el productor arrancado con su caché (primera petición, una ejecución, misma versión publicada, 1 GbE): 8K importado en 21.0 s
(prefill en la RTX 19.7 s) frente a 15.7 s con el prefill propio del par, todavía no más rápido.

## Todos los valores predeterminados incorporados de GLM

El registro de cambios de cada palanca de GLM activada por defecto, con su ganancia medida, su evidencia de exactitud y
su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el catálogo completo es
[docs/optimizations.md](../../optimizations.md#glm). Compilación, fichero de unidades y opciones de GLM RTX:
[guía de GLM RTX](../../../src/glm/rtxpf/README.md); notas del prefill nativo en RTX:
[docs/models/glm/rtx-prefill-native.md](../glm/rtx-prefill-native.md).
