# Qwen3.8-Flash-Next en un nodo Strix Halo

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

Qwen3.8-Flash-Next, GGUF UD-Q4_K_XL (111 GB) más una cabeza de draft MTP destilada en Q8_0 (2.8 GB), en un nodo Strix
Halo (`gfx1151`, 128 GB): decodificación especulativa MTP exacta, prefill FP16 por tiles automático, contexto nativo
hasta 262K y YaRN explícito hasta 1M, servicio HTTP con slots por batch y caché de estado de prefijo, y dos réplicas tras
un router. Esta página conserva todas las mediciones publicadas de Qwen con su compilación y sus condiciones, los
comandos de lanzamiento, los perfiles y los interruptores de ajuste. Cómo se leen las tablas y qué cubre "exacto":
[metodología](../../methodology.es.md). Instrucciones de compilación: [primeros pasos](../../getting-started/README.es.md).

<a id="qwen-recommended-command"></a>

## Ejecución

Ejecución recomendada: el comando simple selecciona los valores predeterminados de producción, 256 tokens.

```sh
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

Añada `--print-effective-config` para inspeccionar antes de la inicialización de HIP. Véase [Valores predeterminados e interruptores de desactivación](../../methodology.es.md#defaults-and-off-switches) para el alcance de la línea base.

Los prompts de CLI simple que lo admiten seleccionan automáticamente el prefill por tiles con prefijo conocido (T8192 para prompts largos,
T4096 a 4K), la unión de expertos, las filas del router y `fast-prefill-f16wmma-v1`. El contexto se dimensiona automáticamente;
por encima de 8192 el resolutor añade `--unqualified-context`. La residencia registrada (`--residency register`)
es la predeterminada en UMA; las páginas registradas del modelo se bloquean en memoria por defecto cuando `RLIMIT_MEMLOCK`
las cubre (`STRIX_MODEL_REGISTER_MLOCK=0`), véase [Bloqueo de páginas registradas](../../operations/model-weight-residency.md#registered-page-lock). El prefill FP16 tiene su propia referencia exacta y no es idéntico bit a bit al prefill serial A8 denso.
El MTP nativo simple conserva su contrato de prefill dependiente de los assets. El servicio HTTP ahora admite automáticamente
T8192 (recurriendo a tiles más pequeños bajo presión de memoria), con su aritmética existente y
checkpoints de 1024 filas para una caché de prefijos configurada. Esto es independiente del prefill FP16 automático de la CLI.
Véase [prefill HTTP](../../engine/serving/serving.md#default-prompt-prefill). Véase la
[ruta automática y los interruptores de desactivación](../../getting-started/engine-reference.md#command-line).

`prompt.ids` contiene el prompt como IDs de token del tokenizador del modelo; el motor imprime un recibo JSON con
`generated_tokens` y los tiempos. Para las velocidades anteriores, añada las opciones especulativas (sidecar MTP, ventanas de verificación
2/4/8, vocabulario de draft): el comando completo está en **[docs/getting-started/engine-reference.md → Línea de comandos](../../getting-started/engine-reference.md#command-line)**
y en [docs/engine/speculative/mtp-native-cli.md](../../engine/speculative/mtp-native-cli.md). El servidor HTTP, los slots por batch (`--http-batch`) y la
caché de prefijos están en [docs/engine/serving/serving.md](../../engine/serving/serving.md); varias peticiones en un proceso, en
[docs/engine/scheduling/multi-request-cohort.md](../../engine/scheduling/multi-request-cohort.md); el muestreo, en
[docs/engine/decode/sampled-decoding.md](../../engine/decode/sampled-decoding.md).

La [caché de estado de prefijos](../../engine/serving/prefix-state-cache.md) de Qwen restaura el estado completo de atención/GDN
en los límites admitidos y solo lee el sufijo. Configure `--prefix-cache-host-bytes N` y,
opcionalmente, `--prefix-cache-dir DIR`; la retención equitativa es la predeterminada (`--prefix-cache-retention lru` restaura
LRU). Las capturas reservan memoria antes de las copias fijadas. `"state_cache":{"retain":false}` permite lecturas
sin almacenar prefijos nuevos; `GET /v1/cache/stats` informa de la retención y de los motivos de expulsión.
`--prefix-cache off` / `STRIX_PREFIX_CACHE=0` la desactiva. La caché certificada solo está activada por defecto
en HTTP MTP en tiles con un único ejecutor (`--http-batch 0`, o capacidad automática que resuelve a un solo
ejecutor) con T=2048, el perfil y los assets por defecto y contexto de hasta 16,384; ahí el presupuesto automático
tras la residencia nativa es `min(4 GiB, A/16, A - reserve)` de la memoria disponible A, y los resultados son
idénticos bit a bit a los de sin caché. Cualquier otra configuración automática, incluido el comando de servicio
simple (que resuelve T=8192), la rechaza por nombre y se ejecuta exactamente como sin caché; los bytes de host
explícitos mantienen su comportamiento. El arranque, `--print-effective-config`, `/v1/capabilities` y
`/v1/cache/stats` informan de la política solicitada y efectiva y de su motivo. La unión a prefijos en curso
(una petición espera el checkpoint certificado que otro slot está produciendo en lugar de recalcularlo) está
activada por defecto allí donde la caché está activada y solo actúa en rutas batch concurrentes con certificación
L0 estricta; `STRIX_PREFIX_CACHE_INFLIGHT=0` / `--prefix-cache-inflight off` la desactiva. La restauración MTP solo acepta planes de sufijo equivalentes a la ejecución sin caché (en la ruta batch la restauración de prefijo necesita una unidad de sufijo en solitario, así que con la agrupación de prompts suele recalcular; los rechazos se nombran `batch-suffix-not-solo`);
las copias de checkpoints siguen los límites de bloque existentes.

<a id="qwen-profile"></a>

### `--profile default|fast|quality`

Un solo interruptor selecciona un perfil de producto para Qwen (CLI, `--http-serve` y el ejecutor MTP):

- `default`: el valor exacto predeterminado descrito arriba; no cambia nada.
- `fast`: aplica por ahora el almacenamiento BF16 de las rutas MoE (`STRIX_QPF2K_MOE_ROUTE_BF16=1`) donde la ruta lo admite,
  antes de su recibo de cualificación, sin `--allow-pending`. Cambia bits y modifica los logits; la cualificación está en curso.
  El recibo indica `qualification in progress: moe_route_bf16` y marca la fila como `provisional`; una línea de
  stderr informa del cambio de bits y del estado de la cualificación. En las demás rutas, incluidas HTTP y MTP,
  la fila queda `pending` con el motivo del rechazo, mantiene el almacenamiento FP32 y no se exporta ni se marca como aplicada.
  El presupuesto de selección de atención, la recurrencia por fragmentos FP16 y la cabeza de salida FP16 siguen pendientes
  y mantienen sus rutas internas, con un aviso cada una; la fila de la recurrencia por fragmentos FP16 ya es ejecutable
  (`--allow-pending` la ejecuta, marcada `unqualified`).
- `quality`: no hay paquete de calidad para este modelo, así que ejecuta `default` y lo indica en el recibo.

`--allow-pending` (o `STRIX_QWEN_PROFILE_ALLOW_PENDING=1`) ejecuta las opciones realmente pendientes cuyo código está
en el build, solo para benchmarks, y las marca como `unqualified: <rows>`. Las tres filas pendientes no tienen todavía
una implementación ejecutable, así que este indicador no cambia la resolución de fast. En YaRN, una fila rechazada
por la admisión rotatoria queda pendiente con su motivo y no se marca como aplicada. Una variable de entorno u opción
de CLI explícita siempre tiene prioridad sobre el perfil, incluido el apagado explícito. `--print-effective-config`
muestra el perfil seleccionado y el estado de cada opción.

<a id="qwen-long-context"></a>

## Contexto largo

La capacidad nativa es de 262,144 posiciones; el prompt más la generación deben caber. El resolutor marca automáticamente
como no cualificados los contextos nativos por encima de la cualificación de fidelidad de 8192 tokens. Superar una prueba de capacidad o de
exactitud no establece la precisión en tareas con esa longitud.

YaRN es una identidad rotatoria opcional (opt-in), solo en la CLI. Con la
[receta de bundle distill-g](../../../recipes/qwen-distill-g-yarn.mtp-bundle.json) verificada y el sidecar/vocabulario correspondientes
junto al modelo, un comando YaRN sin más opciones selecciona MTP exacto automáticamente:

```sh
./build/strix-hip-decode-token MODEL.gguf --prompt-ids prompt.ids --rope-yarn 2 --context 524288 --generate 256
```

Use el factor 4 con `--context 1048576` para una capacidad de 1M. `--prompt-ids` evita los límites de argumentos del shell.
El pool MTP confirmado de QSA y el PLE rodante están activados por defecto en las rutas admitidas. La admisión de memoria prueba
la escalera de almacenamiento antes que tiles de prompt más pequeños e informa de `storage_ladder_step` y `yarn_mtp_storage`.
Un bundle ausente o no válido antes de la mutación recurre a YaRN serial con un aviso; un `--mtp` explícito
conserva su contrato de assets fijados. `STRIX_QWEN_YARN_MTP=0` selecciona YaRN serial.

Se publican comprobaciones de igualdad MTP completas a 262K/1M y de alcance objetivo a 524K; son relativas a YaRN
serial, no a la rotatoria nativa, y no cualifican la fidelidad general del modelo a 1M ni cargas de trabajo completas.
Medido en un nodo Strix Halo, ruta por defecto (código `a199c8e0` más la selección de bundle de #137), prosa real
diversa (50 libros de dominio público en seis idiomas), decodificación greedy y escalera de almacenamiento automática (sin
ajustes manuales de almacenamiento). El TTFT va desde el arranque del proceso hasta el primer token e incluye la carga del modelo.
La celda de prefill a 262K procede de la rama main pública `669bcb83` (prompt de 262,144 tokens, tiempo de prefill sin la
carga del modelo, `--profile default` y `--profile fast` intercalados, dos pares: 179.07 s y 174.18 s); las celdas de decode a
262K proceden del build `7b2453fe`; las demás celdas de la fila proceden del build indicado arriba.

| Contexto (factor YaRN) | Gate exacto (MTP == serial == pool desactivado) | TTFT serial / MTP | Prefill, serial | Decode serial / MTP | Aceptación MTP |
|---|---|---|---|---|---|
| 262,400 (2) | Alcance completo | 219 s / 224 s | **1,464 tok/s** `--profile default`; 1,505 tok/s `--profile fast` (`669bcb83`) | 20.55 / **59.5 tok/s** (2.9x, `7b2453fe`) | 75 % |
| 524,288 (2) | Alcance objetivo | 470 s / 508 s | 1,215 tok/s | 19.7 / **60.6 tok/s** (3.1x) | 85 % |
| 1,048,576 (4) | Alcance completo, más el recibo de la ruta por defecto | 1,005 s / 1,096 s | 1,081 tok/s (ruta MTP unos 970) | 18.7 / **55.9 tok/s** (3.0x) | 83 % |

Véanse [MTP con YaRN y bundles](../../engine/speculative/mtp-native-cli.md#explicit-yarn-mtp-cli),
la [matriz de rutas de pool/PLE](../../engine/qsa-committed-pool.md) y
las [comprobaciones de fidelidad en contexto largo](../../../tools/qualification/longctx_fidelity/README.md).

<a id="beyond-english"></a>

## Más allá del inglés

- **Paquetes de vocabulario de draft por idioma (Qwen).** La cabeza de draft propone a partir de un subconjunto del vocabulario. El
  repositorio incluye un núcleo de 32k más paquetes de 16k para alemán, español, francés, italiano, portugués, ruso, japonés y
  chino en [`data/draft-vocab/`](../../../data/draft-vocab); `--mtp-draft-vocab auto:data/draft-vocab` elige el paquete
  que corresponde al prompt. En un prompt de chat en español esto elevó el decode de 32.9 a 38.9 tok/s (+18 %). El
  DSpark de DeepSeek usa la misma idea (`--dspark-vocab auto:`). El paquete solo cambia cuántas suposiciones se aceptan, nunca el
  texto.
- **Calidad por idioma y categoría.** El arnés informa de KL y top-1 para texto multilingüe, prosa en español,
  prosa en inglés, código, tareas de agente, matemáticas y chat, cada uno con un margen. Una elección que perjudica a una categoría se rechaza
  aunque la media parezca correcta.

<a id="how-fast"></a>

## Resultados medidos

Fuentes: [mediciones de prefill/valores predeterminados](../../performance.md#qwen-prefill-and-defaults) y [resultados de decode/servicio](../../performance.md).

Todas las filas: Qwen3.8-Flash-Next, GGUF UD-Q4_K_XL (111 GB) más una cabeza de draft MTP Q8_0 destilada (2.8 GB), decodificación
especulativa MTP con ventanas 2/4/8, perfil A8 denso, thinking desactivado, 256 tokens generados salvo que se indique otra cosa.

| Modo | Hardware | Prompt / contexto | Concurrencia | Mejor | Típico | Build |
|---|---|---|---:|---:|---:|---|
| Decode MTP | 1 nodo | 1K, código, ventanas automáticas | 1 | **59.59 tok/s** (69 % del techo de bytes de 86.8 tok/s) | prosa: 55.7 tok/s (`0117a140`) | `702bc074` |
| Decode MTP | 1 nodo | 8K | 1 | **54.00 tok/s** (código, ventanas automáticas) | prosa: 49.58 tok/s (`226bb691`, atención de draft exacta) | código `702bc074`; prosa `226bb691` |
| Decode MTP | 1 nodo | 32K | 1 | **47.9 tok/s** (código; rango observado 44.5–52.6) | mediana de 4 | `0117a140` |
| MTP + ventanas de copia (opt-in), edición larga con copia | 1 nodo | documento Markdown (~10 KB) a corregir, respuesta de 2,047 tokens | 1 | **183.5 tok/s** | | `e1b115a2` |
| MTP + ventanas de copia (opt-in), edición larga con copia | 1 nodo | fichero Python + diff a aplicar, respuesta de 1,417 tokens | 1 | **163.9 tok/s** | | `e1b115a2` |
| MTP + ventanas de copia (opt-in), edición larga con copia | 1 nodo | fichero C++ a reescribir con renombrados, respuesta de 1,711 tokens | 1 | **155.1 tok/s** | | `e1b115a2` |
| MTP + ventanas de copia (opt-in), 7 tareas de edición con copia | 1 nodo | entradas de 5-10 KB, respuestas de 1,114-2,047 tokens | 1 | | **125.2 tok/s** (90.2 sin ventanas de copia) | `dd65dbd5` |
| MTP + ventanas de copia (opt-in), sesiones de agente | 1 nodo | 6 sesiones de agente de programación | 1 | | 59.9 tok/s (sin cambios con ventanas de copia) | `dd65dbd5` |

Las ventanas de copia (`--mtp-ngram-draft=wide --mtp-copy-wide-rows 32`) verifican hasta 32 filas cuando la respuesta copia texto
del prompt; los tokens son idénticos a los predeterminados en todas las ejecuciones. Ayudan cuando la respuesta vuelve a emitir tramos largos
de la entrada (reescrituras, parches, correcciones de documentos), no hacen nada con texto nuevo y actualmente funcionan para una petición por
proceso (todavía no con el servidor por batch).

**Dos réplicas** (un modelo completo por nodo tras el router, `--http-batch 8`, servidores reiniciados antes de cada
una de las 3 rondas, sin precalentamiento; los ids de token son iguales a la ejecución en un solo nodo en 570/570 respuestas). Las tasas de reloj más recientes
de chat de programación usan la compilación `0117a140`, servidores recién arrancados con una petición de precalentamiento excluida, 3 rondas ABBAAB; 90/90
respuestas iguales a la referencia de un solo nodo:

| Modo | Hardware | Prompt / contexto | Concurrencia | Mejor: chat de programación | Típico: conjunto corto de 8 categorías | Build |
|---|---|---|---:|---:|---:|---|
| Servidor, dos réplicas | 2 nodos | chat de ~70 tokens / prompts de 36-303 tokens | 1 | **68.2 tok/s** de reloj | 42.8 tok/s de reloj, TTFT p50 0.89 s | `0117a140` programación; conjunto corto de la versión publicada |
| Servidor, dos réplicas | 2 nodos | igual | 2 | **136.4 tok/s** de reloj | 85.8 de reloj, TTFT p50 0.85 s | `0117a140` programación; conjunto corto de la versión publicada |
| Servidor, dos réplicas | 2 nodos | igual | 4 | **146.8 tok/s** de reloj | 102.9 de reloj, TTFT p50 1.32 s | `0117a140` programación; conjunto corto de la versión publicada |
| Servidor, dos réplicas | 2 nodos | igual | 8 | **158.1 tok/s** de reloj | 120.4 de reloj, TTFT p50 2.55 s | `0117a140` programación; conjunto corto de la versión publicada |

*De reloj* (wall) es el total de tokens de completion sobre el tiempo de reloj, incluidos el primer token y la espera en cola. La campaña
anterior de chat de programación midió **90.7 / 181.5 / 201.6 / 255.0 tok/s** de decode sumado por petición
con concurrencia 1 / 2 / 4 / 8; el decode sumado excluye el tiempo de prompt y de cola de cada petición.

<a id="qwen-prefill-offload"></a>

### Lectura del prompt, un nodo y con la RTX PRO

**Lectura del prompt, un nodo** (perfiles y condiciones más abajo) y **con la RTX PRO** (prefill CUDA exacto
en la RTX, estado de atención y recurrente enviado al nodo sobre 1 GbE a medida que termina cada capa; después el
nodo hace el decode; 64 tokens generados idénticos a la ejecución solo en el nodo para cada longitud; una ejecución por longitud):

| Modo | Hardware | 1K | 4K | 8K | 16K | 32K | Build |
|---|---|---:|---:|---:|---:|---:|---|
| Prefill, TTFT | 1 nodo | **1.12 s** | **3.02 s** | | **14.64 s** | **22.95 s** | `0117a140` (1K/16K); perfil combinado más abajo (activado por defecto) |
| Descarga de prefill, TTFT | RTX PRO + 1 nodo | **1.36 s** | **1.92 s** | **2.94 s** | **4.95 s** | **8.99 s** | `70b93af3` |
| Mismo servidor sin la RTX, TTFT | 1 nodo | 1.80 s | 5.20 s | 10.46 s | 21.43 s | 44.02 s | misma compilación del servidor |

La descarga se compara con el mismo servidor de producción al que reenvía (tiles de prompt de 2,048 filas), que lee
los prompts más despacio que la fila de prefill de un nodo anterior (tiles de 8,192 filas). A 32K la ruta RTX es 4.9× más rápida que
ese servidor. Con la descarga activada, el router envía los prompts de 1K tokens o más a la RTX PRO; los más cortos
se leen en el nodo.

**Build `0117a140`** (medida frente a la compilación publicada en la misma sesión, ejecuciones intercaladas,
todas exactas): TTFT un 2.3 % (1K), un 3.4 % (4K) y un 5.1 % (16K, 32K) menor; decode igual o hasta un 0.4 % más rápido;
dos réplicas con concurrencia 4 y 8: un 3.9 % y un 3.6 % más de rendimiento de reloj, y un TTFT un 6.9 % menor con
concurrencia 4. Detalles en [docs/performance.md](../../performance.md#qwen38-flash-next-1).

### Valores predeterminados y candidatos del prefill exacto

**Prefill combinado exacto.** Un nodo Strix Halo, Qwen3.8-Flash-Next UD-Q4_K_XL,
decode A8 denso con una identidad de prefill FP16 rápida separada; compilación `8b99cf58`, valores predeterminados sin variables de entorno que los sustituyan. Comparaciones en el mismo nodo
frente a `4a44683b`, excepto la comparación con contexto máximo y desactivado en la misma compilación. Los logits FP32 finales y los tokens
coinciden con la referencia en todas las longitudes; los recibos de cualificación code-1K y prose-8K coinciden con la línea base archivada.
El tipo de prompt no se especifica para el barrido de longitudes.

| Tokens del prompt | TTFT de línea base | TTFT predeterminado | Cambio | tok/s predeterminado | Proporción del techo del contrato |
|---|---:|---:|---:|---:|---:|
| 4,096 | 3.53 s | **3.00 s** | −15.0 % | 1,363 | 43.4 % de 3,139 tok/s |
| 17,700 | 14.97 s | **12.81 s** | −14.4 % | 1,382 | 44.7 % de 3,089 tok/s |
| 32,768 | 25.95 s | **23.49 s** | −9.5 % | 1,395 | 45.3 % de 3,078 tok/s |
| 65,536 | 55.37 s | **48.18 s** | −13.0 % | 1,360 | 44.9 % de 3,030 tok/s |
| 131,072 | 120.96 s | **105.84 s** | −12.5 % | 1,238 | 42.6 % de 2,907 tok/s |
| 262,143 | 291.82 s (misma compilación, desactivado) | **240.62 s** | −17.5 % | 1,089 | 40.5 % de 2,687 tok/s |

Dos ejecuciones por brazo a 4K / 17,700; 32K tiene una ejecución de línea base y dos predeterminadas; las filas más largas son ejecuciones únicas.
Tiles de prompt: 4,096 a 4K / 128K, 8,192 a 17,700 / 32K / 64K, 2,048 con colas de 256/16 filas a 262,143.
La dispersión entre ejecuciones y entre nodos llega a ~5 %; las comparaciones únicas son orientativas.

[PR #49](https://github.com/hyukudan/strix-halo-engine-lab/pull/49) incorpora como valores predeterminados el prefill combinado, el batching/core16 de los gates GDN
y el staging de selección/activación del router. [PR #50](https://github.com/hyukudan/strix-halo-engine-lab/pull/50)
añade la salida HC v3 con banda 16 (`5011b842`): TTFT a 32K **−0.52 s (−2.2 %)** en tres pares alternos,
4K **−54 ms (−1.8 %)** en dos pares; los logits y los tokens siguen siendo iguales. Una ejecución única posterior a 32K con esa compilación
midió **22.03 s, 1,487 tok/s (48.3 % del techo del contrato de 3,078 tok/s)**.
`STRIX_QPF2K_DEFAULTS=0` suprime el conjunto de valores predeterminados de prefill y rechaza el tiling automático; los interruptores de desactivación específicos de HC son
`STRIX_QPF2K_HC_OUTPUT3=0` / `STRIX_QPF2K_HC_OUTPUT3_BAND=0`.

**Batching de etapas QSA — activado por defecto, [PR #51](https://github.com/hyukudan/strix-halo-engine-lab/pull/51).**
TTFT exacto a 32K **22.21 / 22.66 / 22.76 → 21.57 / 22.32 / 22.58 s** (tres pares, media **−0.39 s, −1.7 %**);
en la compilación publicada, 32K en cuatro pares ABBA **−0.17 s** de media (todos los pares más rápidos) y 64K **46.76 → 46.01 s (−1.6 %)**;
17,700 **12.54 → 12.46 s** (un par). Los tokens, los logits finales y los recibos de cualificación code-1K / prose-8K coinciden.
`STRIX_QPF2K_QSA_MASK=0x57` restaura las etapas anteriores (predeterminado `0x2257`; `0x257` mantiene todo excepto la variante de ocupación de attend).
Estos tiempos son comparaciones separadas, no un nuevo barrido de longitud completa.

La CLI simple ahora se resuelve a la ruta de producción medida. En un nodo Strix Halo, la identidad de prefill FP16
rápida midió **22.03 s / 1,487 tok/s a 32K**, **100.3 s a 131,072 tokens** y
**202.8 s / 1,293 tok/s a 262,143 tokens** (T8192 en las longitudes largas, recuperación de páginas PLE admitida).
Los digests coinciden con la referencia de esa identidad a 32K/64K/128K/262K; esto no es igualdad de bits con el serial A8 denso
ni una nueva cualificación de fidelidad del modelo en contexto largo. Véase el [catálogo](../../optimizations.md#qwen-tiled).

La atención de draft MTP exacta y los valores predeterminados del router están activados en la línea pública de Qwen.
El drafting MTP por batch sigue siendo opt-in. El servidor HTTP dimensiona sus slots de petición a partir de la memoria (`--http-batch N` pone el límite, `0` = de uno en uno), atiende las peticiones en decode en los límites de capa dentro de bloques de prompt largos (`STRIX_BATCH_SERVICE_MS=250`), ejecuta primero los prompts cortos y los anida en un prompt largo en pausa (`STRIX_BATCH_NEST_ROWS=2048`), y cancela en esos límites; cada petición conserva los IDs de token y el estado confirmado que tendría sola. Con un prompt de 32K llegando entre tres flujos en decode (comando predeterminado), el hueco de decode p90 pasó de 14.3 s a 0.54 s y el primer token de una petición corta de 42-52 s a 2.3-3.8 s ([servicio](../../engine/serving/serving.md#batched-decoding)).

**Almacenamiento BF16 de las rutas MoE: candidato opcional (opt-in), pendiente de cualificación.** `STRIX_QPF2K_MOE_ROUTE_BF16=1`
(o `--prefill-moe-route-bf16 on`) guarda en BF16 los productos escalares MoE de cada ruta y los amplía de forma exacta en
la fusión ordenada. Cambia bits (identidad de almacenamiento del prefill `fp16-v1+moeroute16`), está desactivado por
defecto y solo `--profile fast` lo aplica, como fila provisional, hasta que supere la cualificación. Misma compilación, CLI simple, tres
ejecuciones emparejadas cada una: TTFT a 32K **21.83 → 21.15 s (−3.2 %)**, a 16K **10.38 → 10.11 s (−2.3 %)**, todos
los pares más rápidos. A 262K (YaRN factor 2, ruta en serie, `--profile default` frente a `--profile fast`, dos pares
intercalados) el prefill pasó de **191.2 a 185.7 s (−2.9 %, 1,371 → 1,412 tok/s)**, cada ejecución fast más rápida
que cada ejecución default; el decode no cambia. Véase el [catálogo](../../optimizations.md#qwen-moe-route-bf16).

**GDN por bloques en WMMA FP16 en el prefill: candidato opcional (opt-in), pendiente de cualificación.**
`STRIX_QPF2K_GDN_FP16_CHUNK=1` (o `--gdn-fp16-chunk 1`) ejecuta la convolución, la recurrencia y la norma con puerta de
las capas GDN en bloques de 32 filas sobre WMMA FP16 con acumulación FP32 y estado FP32 (identidad
`qwen-gdn-fp16-chunk rev 2`; la proyección de entrada escribe entonces sus valores, ya redondeados a BF16, como filas
BF16). Cambia bits y está desactivado por defecto; `--profile fast` lo lista como pendiente hasta que supere la
cualificación de fidelidad (`--allow-pending` lo ejecuta, marcado `unqualified`). Misma compilación, CLI simple, pares
intercalados: 4K **2.42 → 2.36 s (−2.6 %)**, 32K **19.61 → 18.81 s (−4.1 %)**, 262K **177.9 → 170.0 s (−4.5 %, 1,471 → 1,539 tok/s; un par)**; los primeros tokens
generados coinciden con la ruta exacta. Sin la variable o con `=0` se usa la ruta exacta. Véase el
[catálogo](../../optimizations.md#qwen-gdn-chunk).

GSQ, más abajo, es un perfil separado que prioriza la velocidad sobre la calidad.

<a id="gsq-opt-in-speed-over-quality-profile"></a>

### GSQ: perfil opcional (opt-in) de velocidad sobre calidad

GSQ HYB-Q2 sustituye los expertos enrutados de la receta de calidad por pesos Q2_0; **nunca es el valor predeterminado**.
En la puntuación por fichero de pesos sobre el conjunto de calidad bloqueado de 64 prompts (23,552 posiciones puntuadas, contextos de 2K–32K), la concordancia top-1 con la
referencia Q8_0 es del **76.6 %**, frente al **88.1 %** de UD-Q4_K_XL: una pérdida de **11.5 puntos porcentuales**. Este perfil
no supera el criterio de aceptación de la receta de calidad. Estas puntuaciones de calidad no vuelven a puntuar la ruta de prefill FP16 del motor.

Medido en un nodo Strix Halo, Qwen3.8-Flash-Next, una sola petición, 256 tokens generados,
`702bc074`; el tiempo de decode excluye el prompt. Cada ejecución coincide con los tokens de referencia (golden) de su propia receta.
Las dos recetas tienen pesos distintos y pueden producir respuestas distintas.

| Prompt de código | Ventana MTP | GSQ HYB-Q2, tok/s | Receta de calidad, tok/s | GSQ / calidad |
|---|---|---:|---:|---:|
| 1K | auto | **73.18** | **59.59** | 1.23x |
| 1K | ruta {2} | **49.16** | **41.85** | 1.17x |
| 1K | R2 forzada | **49.17** | **41.82** | 1.18x |
| 1K | R4 forzada | **64.42** | **51.56** | 1.25x |
| 1K | R8 forzada | **66.56** | **52.07** | 1.28x |
| 8K | auto | **65.96** | **54.00** | 1.22x |
| 8K | ruta {2} | **45.51** | **40.05** | 1.14x |
| 8K | R2 forzada | **45.42** | **40.27** | 1.13x |
| 8K | R4 forzada | **59.61** | **51.44** | 1.16x |
| 8K | R8 forzada | **56.60** | **50.05** | 1.13x |

Auto usa la política MTP de producción. La ruta {2} es el ajuste MTP "R1" del arnés, no el decode greedy serial.
Aceptación MTP en auto: 1K 0.72 (GSQ) frente a 0.66 (calidad); 8K 0.66 frente a 0.73. Procesamiento del prompt en las mismas ejecuciones
(en caliente): 1K 1.03 s frente a 1.17 s; 8K 7.7-7.8 s frente a 7.9 s. Respecto al techo de bytes de cada receta para código 1K con MTP,
GSQ auto alcanza alrededor del 64 % (techo de 113.8 tok/s) y la receta de calidad alrededor del 69 % (techo de 86.8 tok/s).

### Cómo se lee un prompt largo

**Lectura de un prompt largo (prefill por tiles de Qwen).** El prompt se corta en tiles admitidos (hasta 8192 filas en la ruta automática); dentro de un bloque cada
peso se lee una sola vez para todas las filas, y las capas recurrentes (GDN) siguen actualizando las filas en orden, así que el resultado es
exacto respecto a la referencia de prefill declarada. El FP16 rápido de la CLI y el prefill HTTP conservan contratos aritméticos separados.

## Valores predeterminados exactos de arranque y teselado

Valores predeterminados exactos de arranque y teselado de Qwen: la preparación estática independiente del prompt precede al
temporizador de la petición; la carga anticipada del código HIP respeta un ajuste HIP proporcionado por el usuario; el T16384
automático empieza en 32,768 tokens de prompt, con alternativa por memoria a T8192; el procesamiento por lotes exacto de la proyección
del indexador es independiente de las opciones numéricas. Los interruptores de desactivación por elemento son
`STRIX_QPF2K_EARLY_SETUP=0`, `STRIX_EAGER_CODE_LOAD=0`, `STRIX_QPF2K_T16K=0` y
`STRIX_QPF2K_INDEX_BATCH=0`. La preparación y la carga trasladan trabajo al arranque; no demuestran
un menor tiempo total del proceso. Véanse el [catálogo](../../optimizations.md#early-tiled-prefill-geometry-setup)
y las [puertas nativas del paquete](../../engine/prefill/hc-output-wmma-qualification.md#bundle-native-gates).

<a id="bit-changing-prefill-options"></a>

## Opciones de prefill que cambian bits

El registro de Qwen es [prefill_identity.hpp](../../../include/strix/prefill_identity.hpp). Cada opción está
**activada por defecto una vez en READY**, con un interruptor de desactivación individual y un interruptor maestro FP16 exacto.
El diseño HC de banda baja usa por defecto el modo 3 a través de los valores predeterminados qpf2k del decodificador;
`STRIX_QPF2K_HC_LOWBAND=0` restaura el diseño anterior. Solo está activo con `hc_output_wmma` desactivado (la ruta de salida FP16 exacta),
no con la identidad predeterminada de salida WMMA. Solo lo usa la forma 4 de proyección
qpf2k a salida FP32 v3 banda 16. La salida HC WMMA, el modo de comprobación, normalized16,
la inversa y las demás rutas conservan su diseño anterior. Las identidades aritméticas y numéricas
no cambian; reinicie para este control en caché. La CI local nativa superó 132/132 casos
sin escrituras espurias, incluidos ocho casos T8224 con redondeo del pool a 512 filas, y
los logits de la última fila a 32K coincidieron bit a bit con el modo 0 con `hc_output_wmma` desactivado. Exacto,
comprobado bit a bit; sin afirmación de tiempos porque la banda baja está inactiva en la ruta predeterminada.
Véase la [CI local](../../engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).

Las [imágenes compactas de salida QSA](../../engine/prefill/compact-stage-images.md) exactas están
activadas por defecto para los propietarios elegibles del anillo de puntuaciones de T completo;
`STRIX_QPF2K_COMPACT_IMAGES=0` restaura los contenedores FP32 tras reiniciar.
El ahorro de memoria calculado es de 96 MiB por propietario T8192. La CI local nativa superó los
fixtures de host/HIP y la comparación bit a bit de logits de la última fila a 32K. El A/B intercalado a 32K
fue neutro dentro del ruido (desactivado 19.95 s, n=2, frente a 20.09 s por defecto); no se afirma ninguna
ganancia de velocidad. Los bits BF16 ya redondeados se amplían de forma idéntica. Las identidades numéricas,
rotary, del plan de kernels y numéricas del estado guardado/caché no cambian.
La identidad física de ejecución debe cambiar porque incluye en su hash los tamaños, desplazamientos y strides
más pequeños del arena; no es una identidad numérica. La configuración efectiva
informa de la política solicitada; la organización compilada informa de los ahorros admitidos.

La aceptación del staged-gather de PLE ahora registra el consumo independiente correcto de la fila 0
y acepta la cola diferida ordenada completa solo después de comparar cada ID de fila
preparada, incluida la fila 0. Esta corrección es incondicional, sin interruptor; la semántica de copia,
las protecciones de preparación, la alternativa ante errores del worker, los eventos de slot y las identidades no
cambian. La CI local nativa superó los 11 fixtures de dispositivo/host, incluidas las pruebas del lanzador
de PLE y del staged-gather. La línea base y `STRIX_QPF2K_DEFAULTS=0` desactivan cada uno
todos los elementos de organización exacta controlados. A 32K, los logits de la última fila coinciden bit a bit entre
la configuración predeterminada y cada brazo desactivado (preparación temprana, carga anticipada, T16384, lotes de índice, imágenes
compactas). Las medianas intercaladas de tres rondas fueron 20.09 s por defecto, 20.48 s con la preparación temprana
desactivada, 20.34 s con T16384 desactivado y 20.07 s con la carga anticipada desactivada. La preparación temprana y T16384 muestran
ganancias medidas; la carga anticipada es neutra dentro del ruido. Estas ganancias no son sumables.
Con 131,072 y 262,143 tokens de texto real, la configuración predeterminada admite T16384 automático y, con
`STRIX_QPF2K_T16K=0`, admite T8192 como antes, sin alternativa. Las puertas abiertas y no bloqueantes
son las siete cargas útiles de estado y T16384 a 262K con la alternativa por presión; véase la
[CI local](../../engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).
La corrección de la aceptación del staged-gather se midió con un mismo binario, con preparación frente a la regla de
aceptación anterior: logits completos iguales con 4K y 32K, tokens iguales con 4K, 32K y 262K; prefill 4K -1.1 %, 32K
-1.9 % (-0.40 s) y 262K (YaRN factor 2, bloques T8192) 198.19 -> 185.49 s (-6.4 %, media de dos pares), con el tiempo
inactivo de la GPU en ese prompt reducido de 10.2 s a 1.56 s.
En conjunto, con un mismo binario en la ruta CLI predeterminada (ejecuciones alternas), los valores predeterminados
anteriores frente a todos sus interruptores de desactivación: 4K 2.548 -> 2.397 s (-5.9 %), 32K 20.57 -> 19.05 s (-7.4 %;
T16384 frente a T8192). La rama desactivada reproduce byte a byte los logits anteriores y los tokens generados son
iguales en todas las ramas.

La máscara READY contiene `hc_output_wmma`. La candidata implementada `index_proj_wmma` queda fuera de READY y está desactivada por defecto a la espera de cualificación. La identidad de prefill predeterminada sigue siendo
`fp16-v1+hcout16` (`prefill_identity_hash` ee2cd64684149069); `--prefill-exact-fp16` da la identidad
simple `fp16-v1` (a0094634843e16ee).

| Opción | Qué hace | Ganancia | Cambia bits frente a la identidad FP16 | Calidad medida | Desactivación (CLI / config / API / env) |
|---|---|---|---|---|---|
| `hc_output_wmma` (activada) | Salida HC: valores bajos FP16 y coeficientes Q8, acumulación WMMA en orden K16 ascendente, mismo sigmoide y misma mezcla de flujos | Superó un conjunto reservado de confirmación declarado de antemano | sí | `hc_output_wmma` superó un conjunto reservado de confirmación declarado de antemano; su criterio de NLL se mide frente a un control equivalente en redondeo cuyo movimiento por caso está limitado a +-0.02 nats/token (una tolerancia de control acotada, no una estimación insesgada de la diferencia sin límite); el cambio absoluto de NLL frente a la referencia exacta también se superó (-0.37 % NLL/token, límite superior +0.38 %). El interruptor de desactivación restaura la ruta de salida exacta. | `--prefill-disable hc_output_wmma` / `prefill_disable = ["hc_output_wmma"]` / solo a nivel de servidor / `STRIX_PREFILL_HC_OUTPUT_WMMA=0` |
| `index_proj_wmma` (desactivada; candidata, fuera de READY) | WMMA BF16 de q/k del indexador con paneles K16 independientes y suma ordenada de paneles FP32; etiqueta `idxproj16` | Pendiente de cualificación local | sí | Puertas de compilador, ejecución nativa, fidelidad del modelo y carga de trabajo completa pendientes | Activación opt-in: `STRIX_PREFILL_INDEX_PROJ_WMMA=1`; desactivación: `STRIX_PREFILL_INDEX_PROJ_WMMA=0`, `--prefill-disable index_proj_wmma`, config `prefill_disable = ["index_proj_wmma"]`; el interruptor maestro exacto también la desactiva |

El hash de la identidad de prefill es FNV-1a 64 del nombre de la identidad. La etiqueta del conjunto activado no es
prueba de la ruta ejecutada: OUT16 requiere el componente HC rápido, la forma ancha cualificada
(hidden 2560, cuatro flujos, rango 320), la admisión HC de QPF2K, filas normalizadas FP32 y una
selección de salida compatible. Los sufijos serie, las formas de normalización incompatibles y
los perfiles no rápidos pueden conservar `hcout16` sin ejecutar OUT16. El testigo de stderr
emitido una vez por proceso no demuestra la ejecución en cada petición.

Use `--prefill-exact-fp16`, `prefill_exact_fp16 = true` o `STRIX_PREFILL_EXACT_FP16=1`
para desactivar todas las opciones aritméticas de este registro. `--prefill-disable` acepta nombres separados por comas y puede repetirse;
los nombres desconocidos o vacíos se rechazan. Las restricciones de CLI/config se combinan con las restricciones del entorno;
un `false` o una matriz vacía en la configuración no vuelve a activar una opción desactivada por el entorno.
Estos controles aritméticos conservan la selección automática de ruta.

La aplicación decode-token y su modo `--http-serve` comparten estos controles de arranque.
`--config FILE` lee las claves aritméticas anteriores y los booleanos de arranque exacto `qpf2k_early_setup` y `eager_code_load` como asignaciones `key = value`,
una por línea, con booleanos en minúsculas, matrices de una sola línea con nombres entre comillas dobles y comentarios `#`.
Se rechazan las claves desconocidas o duplicadas y los valores mal formados. Por ejemplo:

```text
prefill_exact_fp16 = false
prefill_disable = ["hc_output_wmma"]
```

Los recibos, `--print-effective-config`, los recibos `strix` de las respuestas HTTP y `GET /v1/capabilities`
informan de `prefill_identity`, `prefill_identity_hash` (16 dígitos hexadecimales en minúscula) y
`prefill_options_active` (nombres en el orden del registro). Esto identifica el conjunto de opciones; el par de
perfiles numéricos existente sigue identificando la aritmética circundante. Cada conjunto activo modificado
debe volver a cualificarse frente a su propia referencia. El conjunto se declara en todas las rutas, pero
`hc_output_wmma` solo se ejecuta cuando el prefill usa un perfil de prefill rápido (testigo de stderr
`qpf2k HC out16 active`); en otras rutas, incluidas `--baseline` y las rutas dense-A8 explícitas,
es inerte. La equivalencia nunca se basa solo en el conjunto declarado: las claves de la caché de estado de prefijo incluyen la
línea de comandos efectiva de prefill (ambos perfiles numéricos y `--baseline`), y el estado de sesión
lleva la identidad de ejecución de prefill compilada, de modo que las rutas con aritmética distinta nunca se consideran iguales.

**Solo a nivel de servidor:** reinicie el servidor para cambiar estas opciones. Los planes de prefill compilados y
las cachés de estado de prefijo se comparten entre peticiones; el conjunto activado queda vinculado a las claves de ejecución compilada,
de linaje y de caché persistente. La anulación a nivel de proceso no tiene sincronización por petición.
Un getter por llamada no basta por sí solo para aislar peticiones concurrentes o reanudadas. Los campos de petición `prefill_exact_fp16` y `prefill_disable` se rechazan con HTTP 400.

La desactivación debe preservar los logits de public-main y los siete componentes de estado en las rutas CLI, MTP, HTTP, anidada, GSQ y de línea base. Los bytes de identidad de ejecución compilada, de linaje y de caché pueden diferir aunque los bytes numéricos coincidan; las puertas de restauración nativas deben comparar las cargas útiles numéricas por separado. La restauración nativa frente a public main sigue siendo una puerta de publicación obligatoria.

Los [comandos de la puerta de publicación nativa](../../engine/prefill/hc-output-wmma-qualification.md) cubren la restauración public-main/desactivado, la repetibilidad con activación, las exportaciones completas de estado, MTP, HTTP y la planificación anidada.

## Todos los valores predeterminados incorporados de Qwen

El registro de cambios de cada palanca de Qwen activada por defecto, con su ganancia medida, su evidencia de exactitud y
su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el catálogo completo con
condiciones previas y avisos es [docs/optimizations.md](../../optimizations.md#qwen). Servicio (rutas HTTP, batching,
caché de prefijo, seguridad): [servicio](../../serving.es.md).
