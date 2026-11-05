# Nemotron 3.5 Bolt E4B (familia gemma4) en un nodo Strix Halo

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

NVIDIA Nemotron 3.5 Bolt E4B es un Gemma 4 E4B post-entrenado (`model_type: gemma4_text`): 42 capas, hidden 2560,
FFN GeGLU 10240, 35 capas de ventana deslizante (ventana 512, dimensión de cabeza 256) y 7 capas globales (dimensión
de cabeza 512, RoPE parcial proporcional), embeddings por capa (PLE), KV compartida en las últimas 18 capas, softcap de
logits 30, cabeza atada, vocabulario 262144, contexto de 64K. El motor lo ejecuta de forma nativa como la familia
`gemma4` (`STRIX_FAMILY_GEMMA4=ON`, compilación opcional). Esta página conserva los perfiles, los assets, los comandos
de lanzamiento, todas las mediciones publicadas con sus condiciones y los interruptores de desactivación. Cómo se leen
las tablas y qué cubre "exacto": [metodología](../../methodology.es.md).

<a id="profiles"></a>

## Perfiles

Un solo interruptor lo selecciona todo: `--profile default|fast|quality` (predeterminado: `default`). Una opción
explícita o una variable de entorno siempre prevalece sobre el perfil. Cada recibo registra el perfil, el valor y el
origen de cada palanca y cualquier fila pendiente. Una fila pendiente (aún no cualificada) nunca se aplica: el perfil
ejecuta todas las filas listas, la fila pendiente conserva el comportamiento exacto, el recibo la lista con su objetivo
declarado y una línea de stderr lo indica. Un perfil solo rechaza por un asset obligatorio ausente (`quality` sin
`--exl3-arm`) o por una petición explícita de algo que esta compilación no contiene.

| Perfil | Decode | Prefill | Garantía |
|---|---|---|---|
| `default` | `g4-v1` (activaciones FP32, pesos Q4_K_M) + drafter MTP en C1 | Declarado: `g4-wmma-pf-v2` en paquetes GGUF Q4_K_M (otros tipos de paquete: `prefill-exact`); cualificación contra los umbrales de `default` pendiente, hasta entonces se ejecuta `prefill-exact` y el recibo lista `pending g4-wmma-pf-v2 (exact runs)`. `STRIX_G4_PREFILL_IDENTITY=exact` mantiene el prefill exacto | El decode sigue siendo exacto: los bytes de salida son iguales a la ruta serial declarada para cualquier tamaño de batch (C4 == C1) y con el drafter activado o desactivado; el prefill de prompts `g4-wmma-pf-v2` solo se ejecuta en `default` cuando se nombra su recibo contra los umbrales de `default` (no es idéntico bit a bit al prefill exacto) |
| `fast` | `g4-a8-v1` (activaciones int8, cualificado) + drafter MTP; atención de decode en profundidad `g4-dattn-f2` (copias int8 de la KV de las capas globales): cualificación pendiente, hasta entonces se ejecuta la atención exacta | prefill de prompts `a8`: lineales `g4-a8-v1` en bloques de 512 filas; las filas del prompt son iguales al decode a8 de las mismas filas, así que la cualificación de `g4-a8-v1` las cubre (no es idéntico bit a bit al prefill exacto; `STRIX_G4_PREFILL_IDENTITY=exact` restaura el prefill exacto) | Perfil con nombre, cualificado contra bandas de fidelidad congeladas |
| `quality` | Paquete EXL3 (`--exl3-arm DIR`), identidad elegida una vez por paquete y sesión (`g4-exl3-v1`/`v3`) | `prefill-exact` | Exacto dentro de su identidad EXL3; mayor fidelidad que Q4_K_M con un tamaño similar |

La identidad de la ruta EXL3 se fija al arrancar a partir del paquete y de `--streams`, nunca por paso, así que la
salida de una sesión no depende de la carga. `STRIX_G4_EXL3_IDENTITY=v1|v2|v3` la fuerza.

## Assets

- Modelo: el directorio safetensors BF16 (tokenizer, plantilla de chat) y, para `default`/`fast`, el GGUF Q4_K_M.
- Tablas rotatorias, generadas una vez fuera de línea (véase `src/gemma4/ref/README.md`):
  `python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6`
- Drafter: un paquete de asistente MTP en `<directorio del modelo>/drafter-mtp/` (config, pesos, LICENSE, NOTICE,
  LINEAGE, MANIFEST) se encuentra automáticamente; el drafter publicado es
  [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter).
  `--draft-mtp DIR` apunta a otro sitio; `--draft off` o `STRIX_G4_DRAFT=0` descarga el drafter (la búsqueda en el
  prompt, más abajo, sigue proponiendo tokens); `STRIX_G4_LOOKUP=0` desactiva la búsqueda en el prompt y `--dflash off`
  desactiva todo el drafting.
- Paquetes EXL3 (`quality`): un directorio por brazo del paquete, cargado con `--exl3-arm DIR` junto al modelo BF16; el
  paquete publicado es [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3) (M4).

## Compilación y ejecución

Compile con `-DSTRIX_FAMILY_GEMMA4=ON` (desactivado por defecto) y `ninja -C build strix-gemma4-serve`. Genere las
tablas rotatorias una vez y sirva en un nodo:

```sh
python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6
./build/strix-gemma4-serve MODEL_DIR --rope-tables rotary-tables --tokenizer MODEL_DIR/tokenizer.json --profile default
```

`--profile fast` selecciona activaciones int8; `--profile quality` requiere `--exl3-arm DIR`. Un paquete de drafter
MTP en `MODEL_DIR/drafter-mtp/` se usa automáticamente (`--draft off` lo desactiva). Descargue los paquetes
publicados:

```sh
hf download hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter --local-dir MODEL_DIR/drafter-mtp
hf download hyukudan/Nemotron-3.5-Bolt-E4B-EXL3 --local-dir EXL3_DIR
./build/strix-gemma4-serve MODEL_DIR --rope-tables rotary-tables --tokenizer MODEL_DIR/tokenizer.json --profile quality --exl3-arm EXL3_DIR
```

Sin la carpeta del drafter, el decode funciona exactamente igual que sin drafter. Véase la
[guía del modelo](README.md).

Con `MODEL/drafter-mtp/` presente, el decode de una petición propone tokens automáticamente; la salida es idéntica a la
de drafting desactivado. Con más de una petición activa el drafter se admite solo donde su coste medido compensa. El
decode de una petición también propone tokens por búsqueda en el prompt (la continuación de la coincidencia anterior
más larga de los últimos tokens en el prompt y la salida), verificada token a token, así que la salida es idéntica a
la de drafting desactivado. Junto al drafter, un selector por petición la usa solo con una coincidencia fuerte (igual al
drafter solo dentro del ruido); sin drafter, los prompts de tipo agente que copian de su contexto son los que más ganan
(C1 `default`: 115.6 frente a 48.0 tok/s). `STRIX_G4_LOOKUP=0` la desactiva.
`--print-effective-config` (decode) imprime el perfil resuelto y los interruptores sin cargar ninguna petición.

<a id="how-fast"></a>

## Resultados medidos

Un nodo Strix Halo (gfx1151, 128 GB), GGUF Q4_K_M oficial para `default` y `fast`, paquete EXL3 para `quality`;
decode de chat en inglés de 256 tokens, greedy. C4 y C8 son tok/s agregados entre peticiones concurrentes. Salvo que
una celda esté marcada, todos los números de esta sección proceden de una misma sesión de medida con una misma
compilación del motor, y cada valor de llama.cpp se midió en el mismo turno que la fila con la que se compara.

| Perfil | C1 publicado ([drafter MTP](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) + prompt lookup) | C1, sin draft | C4 | C8 | Prefill 512 / 2K / 8K |
|---|---:|---:|---:|---:|---:|
| `default` (`g4-v1`) | **91.9** | 55.8 | 169.8 | 237.9 | 532 / 535 / 509 exacto¹ |
| `fast` (`g4-a8-v1`, cualificado) | **99.5** | 54.0 | 206.5 | 340.6 | 551 / 538 / 509 exacto; prefill `a8`: ver abajo |
| `quality` ([paquete EXL3 M4](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3), `g4-exl3-v3`) | **96.4** | 50.2 | 184.8 | 303.0 | 794 / 815 / 746 |
| llama.cpp `74a4fb00`, mismo Q4_K_M | — | 55.5 | 193.3 | 279.9 | 2,077 / 2,062 / 1,881 |

C1 publicado es la configuración que se obtiene por defecto: el drafter MTP encontrado en `MODEL_DIR/drafter-mtp/` más
prompt lookup, medido en tres rondas intercaladas frente a la ejecución sin draft, con los ids de los tokens y el
estado confirmado iguales en todas las rondas. C1 sin draft, C4 y C8 son decode simple (sin draft). Las celdas de prefill
son prefill exacto salvo donde se marca. ¹ Identidad de prefill de `default` `g4-wmma-pf-v2` en Q4_K_M (cualificación pendiente;
hasta entonces se ejecuta el prefill exacto, el medido aquí; `STRIX_G4_PREFILL_IDENTITY=exact` lo mantiene). El
prefill por identidad está en la tabla de prefill WMMA de abajo.

Frente a la tabla publicada anteriormente (misma máquina): C1 publicado de `quality` 83.8 -> **96.4** tok/s,
decode simple 44.4 / 153.1 / 258.3 -> **50.2 / 184.8 / 303.0** (C1 / C4 / C8); prefill exacto con 2K tokens
466 / 458 / 769 -> **535 / 538 / 815** tok/s (`default` / `fast` / `quality`); el decode de `default`
y `fast` no cambia dentro del ruido entre ejecuciones.

Con una petición, el motor verifica solo los tokens propuestos de los que el drafter está seguro (una ventana de
probabilidad acumulada, con umbral guardado por drafter), lo que subió C1 con drafter de 85.1 / 92.2 / 72.1 a
92.9 / 100.6 / 83.8 tok/s (`default` / `fast` / `quality`, medido cuando se incorporó); `STRIX_G4_MTP_CONF=0` restaura
la política de ventana anterior.

En contextos largos cada profundidad propuesta cuesta más en relación con una pasada del modelo, así que la ventana sigue
los costes medidos del propio flujo: cuando una profundidad de draft cuesta al menos 0.04 de una pasada de una fila, propone
una profundidad cada vez, sigue mientras la probabilidad acumulada es >= 0.6 y verifica el prefijo con >= 0.15. Decode con
16K / 32K tokens (`default`, 512 tokens, media de dos ejecuciones por brazo): +0.3 / +1.6 % código, +0.8 / +1.2 % inglés,
+3.1 / +2.4 % español, +0.6 / +3.5 % japonés, mismos ids de tokens; código y japonés con 16K quedan dentro de la
variación entre ejecuciones; `STRIX_G4_MTP_CONF_LONG=0` mantiene la ventana de contexto corto en
cualquier longitud.

Las pasadas de decode devuelven el token greedy calculado en el dispositivo para cada fila y copian logits y taps de
estado oculto al host solo cuando una petición o el drafter los lee: +1.9 / +4.6 / +5.9 % en C1 / C4 / C8 sin drafter y
+3.3 / +8.3 / +9.3 % con el drafter MTP (`default`, prompts cortos, mismos bytes); `STRIX_G4_DECODE_EXPORT=0` restaura
las copias completas.

La atención del drafter MTP carga cada cabeza KV del objetivo una vez para las dos cabezas de consulta que la
comparten, y su top-k de clústeres es una ordenación bitónica: ms de GPU por profundidad del borrador 0.423 / 0.746 /
2.617 → 0.346 / 0.578 / 1.750 con contexto 512 / 4K / 32K (drafter q8_0, top-64, grafos HIP, un nodo), ids del
borrador idénticos bit a bit; `STRIX_G4_MTP_KERNELS=legacy` restaura los kernels anteriores.

El decode de una petición también propone tokens a partir del propio texto (prompt lookup): propone la continuación
de la coincidencia anterior más larga de los últimos tokens del prompt y de la salida, y el modelo principal verifica
cada token, así que los ids de los tokens son iguales a los obtenidos sin draft. Compensa cuando la salida copia del
contexto. Prompt de agente que edita código (un fichero Python de 2.2K tokens más una petición de edición, `default`,
256 tokens, ids de token iguales a los obtenidos sin draft en todos los casos):

| Prompt de agente que edita código, C1 | tok/s |
|---|---:|
| Sin draft | 48.0 |
| Solo prompt lookup (sin drafter) | **115.6** |
| Solo el drafter MTP (`STRIX_G4_LOOKUP=0`) | 149.0 |
| Configuración publicada: drafter MTP + prompt lookup | **149.1** |

Junto al drafter MTP, un selector por petición solo propone una ventana de lookup ante una coincidencia fuerte y se
retira tras los fallos, así que no estorba: igual que el drafter solo dentro del ruido (chat en inglés 91.9
frente a 91.9, agente con edición 149.1 frente a 149.0). Sin drafter gana mucho; en la sesión en que
se incorporó, un prompt de herramienta JSON (1.2K tokens) alcanzó 122.9 frente a 49.3 tok/s, un documento de 29.5K
tokens 35.0 frente a 28.1 y código 55.8 frente a 51.9. `STRIX_G4_LOOKUP=0` lo desactiva; `--dflash off` desactiva todo
el draft.

El servidor HTTP agrupa las peticiones de chat concurrentes: hasta ocho peticiones (`STRIX_G4_SERVE_SLOTS`) comparten
un planificador y se decodifican en la misma pasada del modelo, las peticiones nuevas entran entre pasos y los ids de
tokens de cada petición son iguales a los de esa misma petición servida sola (verificado con `default`, `fast` y
`quality` con ocho a once peticiones solapadas: chat greedy en cuatro idiomas, un documento largo, muestreo con
semilla, continuación del presupuesto de razonamiento, streaming y un cliente que se desconecta a mitad). Una sola
petición no cambia (C1 con el drafter, mismos ids y velocidad). `STRIX_G4_SERVE_BATCH=0` restaura un trabajo de chat
cada vez. Si no se indica `--streams`, la identidad del núcleo EXL3, fijada por sesión, se elige para el número de
ranuras (ocho), así que un paquete M5 selecciona `g4-exl3-v3` en lugar de `v1`; M4 y M6 usan siempre `v3`.

| `default`, streaming, 128 tokens por petición | 1 cliente | 4 clientes | 8 clientes |
|---|---:|---:|---:|
| Prompts mixtos (chat, código, un documento de ~1.3K tokens): batching activado, tok/s agregados | 72.6 | **76.7** | **76.1** |
| Igual, una petición cada vez (`STRIX_G4_SERVE_BATCH=0`) | — | 65.4 | 66.9 |
| Igual, tiempo hasta el primer token p50, activado / desactivado | 0.1 s | **0.3** / 5.1 s | **0.6** / 13.8 s |
| servidor de llama.cpp `74a4fb00` (`-np 8`, mismo Q4_K_M, mismo turno) | 52.5 | 123.0 | 227.1 |
| Prompts de chat cortos, 256 tokens: batching activado / servidor de llama.cpp | — | — | **160.4** / 244.1 |

La tabla anterior se midió antes de los dos cambios de servicio que siguen (fragmentos de prompt mientras otras peticiones decodifican y la caché de prefijos de prompt).
Después, en main `d7069831d` (prompts de chat cortos, 128 tokens, caché de prefijos desactivada, `default`): C1 / C4 / C8
**92.1 / 143.8 / 187.4 tok/s** agregados, tiempo hasta el primer token p50 0.10 / 0.19 / 0.37 s.

Mientras otras peticiones decodifican, un prompt se procesa en fragmentos de como mucho 64 filas
(`STRIX_G4_SERVE_PREFILL_ROWS`, `0` = fragmentos completos de 512 filas), así que un prompt largo las pausa durante
tramos más cortos: con los prompts mixtos de arriba, el mayor hueco entre dos tokens de una petición bajó de 1.05-1.12 s
a 0.41-0.42 s y el percentil 99 de 0.44 / 1.08 s a 0.17 / 0.20 s (4 / 8 clientes); el rendimiento y los ids de tokens no
cambian. `fast` (las filas del prompt `a8` son iguales a sus filas de decode) acorta los fragmentos igual; con
`STRIX_G4_PREFILL_IDENTITY=exact` en `fast` (procesamiento exacto del prompt junto al decode con activaciones int8) los
fragmentos mantienen su tamaño completo, porque dónde termina un fragmento decide qué filas del prompt escribe una
pasada de decode. Una ruta WMMA del prompt (`g4-wmma-pf-v2` en `default` una vez cualificada, o `=v1` / `=v2`) también
mantiene los fragmentos completos; `default` con prefill exacto los acorta.

El servidor también guarda el estado confirmado al final de cada prompt y de cada petición terminada, y lo reutiliza
cuando un prompt posterior empieza exactamente con esos tokens, normalmente el siguiente turno de una conversación, que
vuelve a enviar la conversación hasta ese punto: solo se procesan las filas nuevas. Tiempo hasta el primer token del
turno 2 (`default`, seis conversaciones en paralelo): documento de 4K tokens **0.20 s** frente a 12.59 s sin caché,
documento de 1K tokens **0.19 s** frente a 3.48 s, chat 0.39-0.51 s frente a 1.31-1.66 s; los ids de tokens de ambos
turnos son iguales a los obtenidos sin caché. Hasta 16 estados dentro de 4 GB (`STRIX_G4_SERVE_PREFIX_MB`);
`STRIX_G4_SERVE_PREFIX=0` la desactiva. `fast` reutiliza cualquier estado, como `default` con prefill exacto (sus filas
de prompt `a8` son iguales a sus filas de decode); con `STRIX_G4_PREFILL_IDENTITY=exact` en `fast`, o con una ruta WMMA
del prompt (`g4-wmma-pf-v2` en `default` una vez cualificada, o `=v1` / `=v2`), solo se reutilizan estados en un
límite de 512 filas escritos por el procesamiento del prompt, nunca una respuesta. Dos conversaciones distintas que comparten un documento largo pero divergen
después todavía no lo comparten.

**Drafter MTP** - [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). Un modelo asistente pequeño (el asistente MTP
de Gemma-4-E4B-it de Google, 78.8M parámetros) propone los siguientes tokens y el modelo principal los verifica, así
que la salida es idéntica a la del decode sin él; solo cambia la velocidad. La aceleración procede de la arquitectura
del asistente MTP y del decode especulativo exacto del motor: el decode en C1 pasa de 51.8 a 91.9 tok/s en
inglés (`default`, con y sin draft intercalados en el mismo turno, x1.78); el [asistente de Google sin ajustar](https://huggingface.co/google/gemma-4-E4B-it-assistant) es casi igual de rápido
en este motor (véase la comparación más abajo). Lo ajustamos ligeramente (fine-tuning) con unos 6.3M tokens generados
por Nemotron 3.5 Bolt E4B a partir de 17,865 prompts en ocho idiomas, más código, llamadas a herramientas y
razonamiento; en un conjunto reservado de 697 prompts eso añade alrededor de un 1.5 % de tokens esperados por ciclo
(inglés 3.61 -> 3.65, japonés 2.52 -> 2.60); con la ventana de confianza es un 1.4 % más rápido que el asistente sin
ajustar (media geométrica en inglés, español, japonés y código). Descárguelo en
`MODEL_DIR/drafter-mtp/`; el motor lo usa automáticamente.

**Paquete EXL3 M4 (perfil `quality`)** - [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3). Una cuantización EXL3
(codificada con trellis, Hessiana escalada) del modelo BF16: MLP a 4 bits, atención q/o a 5, k/v y la cabeza de salida
a 6, embedding y tablas por capa en BF16; 4.65 bits por peso en las capas lineales cuantizadas, una descarga de 9.7 GB
(2.5 GB de pesos cuantizados) que lee 2.70 GB de pesos por token (alrededor de un 14 % menos que Q4_K_M). En un
conjunto reservado de 40 documentos en 8 idiomas, su salida está más cerca del modelo BF16 original que la del Q4_K_M
estándar: coincidencia top-1 del 92.94 % frente al 87.0 % y KL media de 0.0367 frente a 0.141. Elíjalo
(`--profile quality --exl3-arm DIR`) cuando quiera una salida más cercana a BF16; quédese en `default` o `fast` con
Q4_K_M para obtener la máxima velocidad. `quality` decodifica a 96.4 tok/s en la configuración publicada
(50.2 sin draft, mismos ids de token) y hace prefill a 815 tok/s con 2K tokens; su prefill y su decode se
siguen optimizando.

**Comparación con las versiones oficiales de NVIDIA.** NVIDIA publica el modelo principal en [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) y en
una sola cuantización, el [GGUF Q4_K_M](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), además de un [drafter DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash). Solo celdas medidas; una celda vacía
no se midió. Misma máquina, greedy, decode de chat de 256 tokens, tok/s.

| Modelo (perfil del motor) | Descarga GB | Bytes de pesos leídos por token | Top-1 frente a BF16 | KL media frente a BF16 | C1 publicado | C1 simple | C4 | C8 | Prefill 2K | Prefill 8K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 oficial, referencia ¹ | 16.06 | 9.29 GB | 100 % | 0 | 38.1 | 19.8 | 72.4 | 156.7 | 565 | 534 |
| Q4_K_M oficial, `default` | 6.18 | 3.13 GB | 87.0 % | 0.141 | 91.9 | 55.8 | 169.8 | 237.9 | 535 | 509 |
| Q4_K_M oficial, `fast` | 6.18 | 3.13 GB | 86.7 % | 0.150 | 99.5 | 54.0 | 206.5 | 340.6 | 538 | 509 |
| **EXL3 M4**, `quality` | 9.71 | **2.70 GB** | **92.94 %** | **0.0367** | 96.4 | 50.2 | 184.8 | 303.0 | 815 | 746 |
| llama.cpp `74a4fb00`, Q4_K_M oficial | 6.18 | 3.13 GB | | | | 55.5 | 193.3 | 279.9 | 2,062 | 1,881 |

Bytes de pesos por token: todas las capas lineales más una fila de las tablas de embedding y por capa (la descarga EXL3
es mayor porque lleva esas tablas en BF16). Fidelidad: el conjunto reservado de 40 documentos en 8 idiomas frente a los
logits del BF16 oficial; la fila `fast` procede de la evaluación de activaciones int8, que midió `default` en
86.95 % / 0.144 en paralelo. C1 publicado = drafter MTP + prompt lookup; C4 y C8 se ejecutan sin draft. ¹ Fila BF16
de una sesión anterior (drafter MTP anterior a la ventana de confianza, prefill anterior a los kernels actuales de
prefill exacto); todas las demás celdas de velocidad proceden de la sesión de las tablas anteriores.

Drafters sobre el Q4_K_M oficial (`default`, C1, ids de token idénticos a los obtenidos sin draft con todos los
drafters; drafters MTP con la ventana de confianza en su propio umbral, 0.4 el nuestro y 0.6 el de Google). Esta tabla
procede de una sesión anterior a las tablas de arriba (antes de prompt lookup); las columnas DFlash, de una sesión
anterior aún:

| Prompt | Nuestro drafter MTP: tokens / paso | tok/s | Asistente de Google sin ajustar: tokens / paso | tok/s | DFlash de NVIDIA (360M): tokens / paso | tok/s | Sin draft: tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Inglés | 2.39 | 93.8 | 2.59 | **97.6** | 2.21 | 69.0 | 52.1 |
| Español | 2.10 | **86.7** | 2.10 | 83.4 | 1.55 | 58.5 | 52.0 |
| Japonés | 2.13 | **85.9** | 2.19 | 85.0 | 1.56 | 59.9 | 52.0 |
| Código | 3.46 | **113.3** | 3.32 | 108.1 | 2.15 | 73.6 | 52.0 |

DFlash obtiene más tokens por paso en inglés, pero cada uno de sus pasos de draft cuesta unos 10 ms en esta máquina
frente a ~1 ms del asistente MTP, así que su aceleración es menor (x1.12-1.41 frente a x1.60-2.18).

Decode con contexto largo (`default`, una petición, Q4_K_M, tok/s con la profundidad de contexto indicada; sesión
anterior; la atención de las capas globales lee cada fragmento K/V una vez por pasada, idéntico byte a byte,
`STRIX_G4_ATTN_KV_SHARED=0` restaura el kernel anterior):

| Contexto | Antes | Ahora | llama.cpp `74a4fb00`, misma profundidad |
|---|---:|---:|---:|
| 8K | 33.9 | **38.7** | 49.4 |
| 32K | 9.8 | **26.7** | 42.6 |
| 64K | 5.7 | **18.4** | 36.2 |

Prefill (tok/s; prefill exacto en todas las columnas):

| Prompt | `default` | `fast` | `quality` | llama.cpp |
|---|---:|---:|---:|---:|
| 512 | 532 | 551 | 794 | 2,077 |
| 2K | 535 | 538 | 815 | 2,062 |
| 8K | 509 | 509 | 746 | 1,881 |
| 16K | 462 | 465 | 638 | 1,689 |
| 32K | 397 | 399 | 533 | 1,376 |

El prefill exacto usa aritmética FP32 sin contracción, así que los ids de token y el estado son idénticos en todas
las longitudes; `quality` usa el prefill exacto de muchas filas de EXL3. `STRIX_G4_PREFILL_EXACT_MANYROW=0` restaura
las pasadas de ocho filas con los mismos bytes.

Prefill de prompts en `default`: la identidad declarada es `g4-wmma-pf-v2` en paquetes GGUF Q4_K_M (los demás tipos de
paquete mantienen el prefill exacto). El contrato `default` (prefill de prompts `g4-wmma-pf-v2` + decode `g4-v1`) tiene
pendiente la cualificación contra los umbrales de `default` sobre un conjunto held-out nuevo; hasta que se nombre su
recibo, `default` usa el prefill exacto, el recibo lista `pending g4-wmma-pf-v2 (exact runs)` y una línea de stderr lo
indica. `STRIX_G4_PREFILL_IDENTITY=exact` es el interruptor de desactivación: prefill exacto, idéntico bit a bit a la
ruta serial, en cualquier momento. `=v1` selecciona `g4-wmma-pf-v1`, cuyo contrato con el decode `g4-v1` superó los
mismos umbrales (fidelidad del prefill con teacher forcing hasta ~8.4K tokens en 8 idiomas dentro de sus bandas;
Global-MMLU, 2,000 ítems pareados en cada uno de 8 idiomas: cota superior de la pérdida de acierto 0.15-0.32 puntos por
idioma, umbral 0.5; recibo `0b98d054`). Prefill por identidad (Q4_K_M, tok/s con 2K / 4K / 8K, sesión intercalada de
abajo): exacto 526 / 509 / 481, `g4-wmma-pf-v1` 2,359 / 2,249 / 1,957.

Prefill de prompts de `fast`: `a8` (`STRIX_G4_PREFILL_IDENTITY=a8`, válido solo con el decode `g4-a8-v1`; con otra
identidad de decode se ejecuta el prefill exacto y una línea en stderr lo indica). El prompt se procesa en bloques de
512 filas con los mismos lineales `g4-a8-v1` que el decode (IU8 WMMA en bloques largos, los kernels del decode hasta 8
filas) y atención canónica: el KV y los logits de cada fila del prompt son iguales, byte a byte, al decode a8 de las
mismas filas, así que la cualificación de `g4-a8-v1` lo cubre, y los ids servidos son iguales a los de los mismos
prompts procesados con filas de decode (C1 y C4). No es idéntico bit a bit al prefill exacto. El servidor imprime
`gemma4 prefill route a8-many-row (g4-a8-v1 linears, 512-row chunks)` cuando se usa. `STRIX_G4_PREFILL_IDENTITY=exact`
lo desactiva. Q4_K_M, un nodo, tok/s de prefill, el mismo binario para las tres rutas, un proceso nuevo por ejecución,
los dos órdenes de ejecución, decode `g4-a8-v1`:

| Prompt | exacto (muchas filas) | **`a8` (`fast`)** | `g4-wmma-pf-v1` + atención flash (no cualificado) |
|---|---:|---:|---:|
| 512 | 542 | **1,871** | 2,756 |
| 2K | 516 | **1,733** | 2,656 |
| 8K | 477 | **1,498** | 2,591 |

A través del motor (prompt de 8K tokens, un proceso nuevo cada uno, el mismo binario): exacto 16.19 s, **`a8` 5.13 s**,
v1 + flash 2.97 s.

Otras rutas de prefill de prompts seleccionables (paquetes no EXL3; `quality` mantiene el prefill exacto salvo que se
elija una): `STRIX_G4_PREFILL_IDENTITY=v1` usa `g4-wmma-pf-v1`: los lineales usan los GEMM WMMA del prefill con
operandos FP16 sobre paneles de pesos decodificados una vez al cargar; la atención sigue siendo canónica; el decode no
cambia. `=v2` usa `g4-wmma-pf-v2` (activaciones divididas hi/lo), el prefill de prompts declarado de `default`. Ambas rutas procesan el grueso del prompt: los
últimos <= 8 tokens del prompt y el ancla (la posición final del prompt, cuyos logits eligen el primer token) usan la
aritmética del decode (`g4-v1` en `default`), como las filas de decode. Ambas cambian los logits, así que no son exactas.
Prefill de prompts v1 para `fast`: cualificación del par prefill v1 + decode a8 pendiente; `fast` usa el prefill `a8` hasta entonces. Q4_K_M, un nodo, tok/s de prefill, ejecuciones intercaladas en una sesión:

| Prompt | exacto (`=exact`) | `g4-wmma-pf-v1` (`=v1`) |
|---|---:|---:|
| 2K | 526 | 2,359 |
| 4K | 509 | 2,249 |
| 8K | 481 | 1,957 |

Las rutas mantienen la atención canónica, idéntica en bytes a la atención de decodificación. Su rejilla de scores recorre primero las teselas de filas en grupos de 8 y sus valores D512 se calculan en bloques de 512 hilos que comparten una tesela de probabilidades en LDS: atención 8K 1,333 → 1,265 ms (−5.3 %), 32K 16.83 → 15.91 s (−5.5 %), mismos bytes; se apaga con `STRIX_G4_PREFILL_CANON_WIDE=0 STRIX_G4_PREFILL_CANON_GROUP=0` (`STRIX_G4_PREFILL_CANON_ROWS_MAJOR=0` vuelve a recorrer primero las teselas de claves).

Comandos de llama.cpp: C1 `llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 128 -p 0 -r 3`; C4/C8
`llama-batched-bench -m Q4_K_M.gguf -fa on -ngl 99 -c 2048 -npp 64 -ntg 32 -npl 1,4,8`; prefill
`llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 0 -r 2 -p 512,2048,8192` y `-r 1 -p 16384,32768`; el servidor de llama.cpp `74a4fb00` con
`-m Q4_K_M.gguf -ngl 99 -fa on -c 16384 -np 8 --jinja`.

Las cifras varían con la carga del nodo (ancho de banda LPDDR5X compartido); compare solo ejecuciones de la misma sesión.

## Interruptores de desactivación

Cada palanca de velocidad que mantiene los bytes idénticos está activada por defecto con un interruptor `STRIX_G4_*=0`
que restaura la ruta exacta anterior; los interruptores activos aparecen bajo `off_switches` en cada recibo. Los README
de los módulos los enumeran: `src/gemma4/kernels`, `src/gemma4/prefill`, `src/gemma4/exl3`, `src/gemma4/eagle`,
`src/gemma4/engine`. El registro de cambios de cada palanca de Nemotron activada por defecto con su ganancia medida está
en [ganancias medidas](../../measured-gains.es.md).
