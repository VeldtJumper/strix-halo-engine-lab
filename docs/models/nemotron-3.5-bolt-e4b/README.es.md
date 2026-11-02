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
| `default` | `g4-v1` (activaciones FP32, pesos Q4_K_M) + drafter MTP en C1 | `prefill-exact` | Exacto: los bytes de salida son iguales a la ruta serial declarada para cualquier tamaño de batch (C4 == C1) y con el drafter activado o desactivado |
| `fast` | `g4-a8-v1` (activaciones int8, cualificado) + drafter MTP | Prefill WMMA (pendiente `g4-wmma-pf-v2`; hasta entonces se ejecuta el prefill exacto) | Perfil con nombre, cualificado contra bandas de fidelidad congeladas |
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
(C1 `default`: 114.9 frente a 48.2 tok/s). `STRIX_G4_LOOKUP=0` la desactiva.
`--print-effective-config` (decode) imprime el perfil resuelto y los interruptores sin cargar ninguna petición.

<a id="how-fast"></a>

## Resultados medidos

Un nodo Strix Halo (gfx1151, 128 GB), GGUF Q4_K_M oficial para `default` y `fast`, paquete EXL3 para `quality`;
decode de chat en inglés de 256 tokens, greedy. C4 y C8 son tok/s agregados entre peticiones concurrentes. Cada valor
de llama.cpp procede de la misma sesión de medida que la fila con la que se compara.

| Perfil | C1 con el [drafter MTP](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) (configuración publicada) | C1, sin drafter | C4 | C8 |
|---|---:|---:|---:|---:|
| `default` (exacto, `g4-v1`) | **92.9** | 56.0 | 169.3 | 238.5 |
| `fast` (`g4-a8-v1`, cualificado) | **100.6** | 54.1 | 206.7 | 340.2 |
| `quality` ([paquete EXL3 M4](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3), `g4-exl3-v3`) | **83.8** | 44.4 | 153.1 | 258.3 |
| llama.cpp `74a4fb00`, mismo Q4_K_M | — | 55.5 | 193.8 | 283.4 |

Con el drafter, los ids de los tokens son iguales a los obtenidos sin draft (verificado en inglés, español, japonés y
código en C1, y en C4). Con una petición, el motor verifica solo los tokens propuestos de los que el drafter está
seguro (una ventana de probabilidad acumulada, con umbral guardado por drafter), lo que subió C1 con drafter de
85.1 / 92.2 / 72.1 a 92.9 / 100.6 / 83.8 tok/s (`default` / `fast` / `quality`, misma máquina);
`STRIX_G4_MTP_CONF=0` restaura la política de ventana anterior.

Las pasadas de decode devuelven el token greedy calculado en el dispositivo para cada fila y copian logits y taps de
estado oculto al host solo cuando una petición o el drafter los lee: +1.9 / +4.6 / +5.9 % en C1 / C4 / C8 sin drafter y
+3.3 / +8.3 / +9.3 % con el drafter MTP (`default`, prompts cortos, mismos bytes); `STRIX_G4_DECODE_EXPORT=0` restaura
las copias completas.

El decode de una petición también propone tokens a partir del propio texto (prompt lookup): propone la continuación
de la coincidencia anterior más larga de los últimos tokens del prompt y de la salida, y el modelo principal verifica
cada token, así que los ids de los tokens son iguales a los obtenidos sin draft. Sin drafter gana mucho (C1,
`default`, 256 tokens): prompt de agente con edición (2.2K tokens) **114.9** tok/s frente a 48.2 sin draft, prompt de
herramienta JSON (1.2K) **122.9** frente a 49.3, documento de 29.5K tokens **35.0** frente a 28.1, código **55.8**
frente a 51.9; chat en inglés, español y japonés +1 a 2 %. Junto al drafter MTP, un selector por petición solo propone
una ventana de lookup ante una coincidencia fuerte y se retira tras los fallos, así que no estorba: igual que el
drafter solo dentro del ruido en chat, código, prompts de agente y 32K (agente con edición 146.8 frente a 146.2,
herramienta JSON 172.2 frente a 171.6, código 112.6 frente a 112.5, 29.5K 50.5 frente a 50.4, chat en inglés 92.8 frente
a 92.8). `STRIX_G4_LOOKUP=0` lo desactiva.

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
| Prompts mixtos (chat, código, un documento de ~1.3K tokens): batching activado, tok/s agregados | 70.1 | **87.9** | **73.1** |
| Igual, una petición cada vez (`STRIX_G4_SERVE_BATCH=0`) | 70.0 | 62.8 | 63.0 |
| Igual, tiempo hasta el primer token p50, activado / desactivado | 0.10 / 0.10 s | **0.26** / 4.6 s | **0.60** / 13.8 s |
| servidor de llama.cpp `74a4fb00` (`-np 8`, mismo Q4_K_M, misma sesión) | 50.1 | 124.4 | 239.7 |
| Prompts de chat cortos, 256 tokens: batching activado / servidor de llama.cpp | — | — | **172.4** / 245.9 |

El paso de decode con ocho peticiones iguala el paso C8 del propio motor. Con prompts largos en la mezcla, el
procesamiento exacto del prompt (unos 435 tok/s) todavía pausa a las demás peticiones mientras corre un fragmento del
prompt.

Mientras otras peticiones decodifican, un prompt se procesa en fragmentos de como mucho 64 filas
(`STRIX_G4_SERVE_PREFILL_ROWS`, `0` = fragmentos completos de 512 filas), así que un prompt largo las pausa durante
tramos más cortos: con los prompts mixtos de arriba, el mayor hueco entre dos tokens de una petición bajó de 1.05-1.12 s
a 0.41-0.42 s y el percentil 99 de 0.44 / 1.08 s a 0.17 / 0.20 s (4 / 8 clientes); el rendimiento y los ids de tokens no
cambian. Con `fast` (decode con activaciones int8 junto al procesamiento exacto del prompt) los fragmentos mantienen
su tamaño completo, porque dónde termina un fragmento decide qué filas del prompt escribe una pasada de decode.

El servidor también guarda el estado confirmado al final de cada prompt y de cada petición terminada, y lo reutiliza
cuando un prompt posterior empieza exactamente con esos tokens, normalmente el siguiente turno de una conversación, que
vuelve a enviar la conversación hasta ese punto: solo se procesan las filas nuevas. Tiempo hasta el primer token del
turno 2 (`default`, seis conversaciones en paralelo): documento de 4K tokens **0.20 s** frente a 12.59 s sin caché,
documento de 1K tokens **0.19 s** frente a 3.48 s, chat 0.39-0.51 s frente a 1.31-1.66 s; los ids de tokens de ambos
turnos son iguales a los obtenidos sin caché. Hasta 16 estados dentro de 4 GB (`STRIX_G4_SERVE_PREFIX_MB`);
`STRIX_G4_SERVE_PREFIX=0` la desactiva. Con `fast`, cuyo decode con activaciones int8 difiere del procesamiento exacto
del prompt, solo se reutilizan estados en un límite de 512 filas escritos por el procesamiento del prompt, nunca una
respuesta. Dos conversaciones distintas que comparten un documento largo pero divergen
después todavía no lo comparten.

**Drafter MTP** - [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). Un modelo asistente pequeño (el asistente MTP
de Gemma-4-E4B-it de Google, 78.8M parámetros) propone los siguientes tokens y el modelo principal los verifica, así
que la salida es idéntica a la del decode sin él; solo cambia la velocidad. La aceleración procede de la arquitectura
del asistente MTP y del decode especulativo exacto del motor: el decode en C1 pasa de 52.0 a 92.9 tok/s en inglés
(`default`, con y sin drafter en la misma sesión, x1.79); el [asistente de Google sin ajustar](https://huggingface.co/google/gemma-4-E4B-it-assistant) es casi igual de rápido
en este motor (véase la comparación más abajo). Lo ajustamos ligeramente (fine-tuning) con unos 6.3M tokens generados
por Nemotron 3.5 Bolt E4B a partir de 17,865 prompts en ocho idiomas, más código, llamadas a herramientas y
razonamiento; en un conjunto reservado de 697 prompts eso añade alrededor de un 1.5 % de tokens esperados por ciclo
(inglés 3.61 -> 3.65, japonés 2.52 -> 2.60); con la ventana de confianza es más rápido que el asistente sin ajustar
en español, japonés y código y más lento en inglés (+1.4 % de media geométrica en los cuatro). Descárguelo en
`MODEL_DIR/drafter-mtp/`; el motor lo usa automáticamente.

**Paquete EXL3 M4 (perfil `quality`)** - [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3). Una cuantización EXL3
(codificada con trellis, Hessiana escalada) del modelo BF16: MLP a 4 bits, atención q/o a 5, k/v y la cabeza de salida
a 6, embedding y tablas por capa en BF16; 4.65 bits por peso en las capas lineales cuantizadas, una descarga de 9.7 GB
(2.5 GB de pesos cuantizados) que lee 2.70 GB de pesos por token (alrededor de un 14 % menos que Q4_K_M). En un
conjunto reservado de 40 documentos en 8 idiomas, su salida está más cerca del modelo BF16 original que la del Q4_K_M
estándar: coincidencia top-1 del 92.94 % frente al 87.0 % y KL media de 0.0367 frente a 0.141. Elíjalo
(`--profile quality --exl3-arm DIR`) cuando quiera una salida más cercana a BF16; quédese en `default` o `fast` con
Q4_K_M para obtener la máxima velocidad. `quality` decodifica a 83.8 tok/s con el drafter (44.4 sin él, mismos ids de
token); su prefill y su decode se siguen optimizando.

**Comparación con las versiones oficiales de NVIDIA.** NVIDIA publica el modelo principal en [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) y en
una sola cuantización, el [GGUF Q4_K_M](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), además de un [drafter DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash). Solo celdas medidas; una celda vacía
no se midió. Misma máquina, greedy, decode de chat de 256 tokens, tok/s.

| Modelo (perfil del motor) | Descarga GB | Bytes de pesos leídos por token | Top-1 frente a BF16 | KL media frente a BF16 | C1 con drafter MTP | C1 simple | C4 | C8 | Prefill 2K | Prefill 8K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 oficial, referencia | 16.06 | 9.29 GB | 100 % | 0 | 38.1 ¹ | 19.8 | 72.4 | 156.7 | 565 | 534 |
| Q4_K_M oficial, `default` | 6.18 | 3.13 GB | 87.0 % | 0.141 | 92.9 | 56.0 | 169.3 | 238.5 | 466 | 429 |
| Q4_K_M oficial, `fast` | 6.18 | 3.13 GB | 86.7 % | 0.150 | 100.6 | 54.1 | 206.7 | 340.2 | 458 | 414 |
| **EXL3 M4**, `quality` | 9.71 | **2.70 GB** | **92.94 %** | **0.0367** | 83.8 | 44.4 | 153.1 | 258.3 | 769 | 701 |
| llama.cpp `74a4fb00`, Q4_K_M oficial | 6.18 | 3.13 GB | | | | 55.5 | 193.8 | 283.4 | 2,052 | 1,873 |

Bytes de pesos por token: todas las capas lineales más una fila de las tablas de embedding y por capa (la descarga EXL3
es mayor porque lleva esas tablas en BF16). Fidelidad: el conjunto reservado de 40 documentos en 8 idiomas frente a los
logits del BF16 oficial; la fila `fast` procede de la evaluación de activaciones int8, que midió `default` en
86.95 % / 0.144 en paralelo. C4 y C8 se ejecutan sin draft. Las celdas de prefill de EXL3 proceden de una sesión
posterior, en la que llama.cpp midió 2,053 / 1,875 tok/s. ¹ Fila BF16 medida antes de la ventana de confianza; las celdas de decode de
Q4_K_M, EXL3 y llama.cpp proceden de una misma sesión posterior.

Drafters sobre el Q4_K_M oficial (`default`, C1, ids de token idénticos a los obtenidos sin draft con todos los drafters; drafters MTP con la ventana
de confianza en su propio umbral, 0.4 el nuestro y 0.6 el de Google; DFlash de una sesión anterior):

| Prompt | Nuestro drafter MTP: tokens / paso | tok/s | Asistente de Google sin ajustar: tokens / paso | tok/s | DFlash de NVIDIA (360M): tokens / paso | tok/s | Sin draft: tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Inglés | 2.39 | 93.8 | 2.59 | **97.6** | 2.21 | 69.0 | 52.1 |
| Español | 2.10 | **86.7** | 2.10 | 83.4 | 1.55 | 58.5 | 52.0 |
| Japonés | 2.13 | **85.9** | 2.19 | 85.0 | 1.56 | 59.9 | 52.0 |
| Código | 3.46 | **113.3** | 3.32 | 108.1 | 2.15 | 73.6 | 52.0 |

DFlash obtiene más tokens por paso en inglés, pero cada uno de sus pasos de draft cuesta unos 10 ms en esta máquina
frente a ~1 ms del asistente MTP, así que su aceleración es menor (x1.12-1.41 frente a x1.60-2.18).

Decode con contexto largo (`default`, una petición, Q4_K_M, tok/s con la profundidad de contexto indicada; la
atención de las capas globales lee cada fragmento K/V una vez por pasada, idéntico byte a byte,
`STRIX_G4_ATTN_KV_SHARED=0` restaura el kernel anterior):

| Contexto | Antes | Ahora | llama.cpp `74a4fb00`, misma profundidad |
|---|---:|---:|---:|
| 8K | 33.9 | **38.7** | 49.4 |
| 32K | 9.8 | **26.7** | 42.6 |
| 64K | 5.7 | **18.4** | 36.2 |

Prefill (tok/s; todos los perfiles usan prefill exacto):

| Prompt | `default` | `fast` | `quality` | llama.cpp |
|---|---:|---:|---:|---:|
| 512 | 471 | 474 | 768 | 2,065 |
| 2K | 466 | 458 | 769 | 2,052 |
| 8K | 429 | 414 | 701 | 1,873 |
| 16K | 404 | 403 | 618 | 1,684 |
| 32K | 358 | 355 | 520 | 1,373 |

El prefill exacto usa aritmética FP32 sin contracción; `quality` usa el prefill exacto de muchas filas de EXL3. La columna `quality` se midió en una sesión posterior, en la que llama.cpp dio
2,048 / 2,053 / 1,875 / 1,664 / 1,371 tok/s; `STRIX_G4_PREFILL_EXACT_MANYROW=0` restaura las pasadas de ocho filas
con los mismos bytes. El prefill WMMA del perfil `fast` está pendiente de cualificación y no se aplica.

Comandos de llama.cpp: C1 `llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 128 -p 0 -r 3`; C4/C8
`llama-batched-bench -m Q4_K_M.gguf -fa on -ngl 99 -c 2048 -npp 64 -ntg 32 -npl 1,4,8`; prefill
`llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 0 -r 2 -p 512,2048,8192` y `-r 1 -p 8192,16384,32768`.

Las cifras varían con la carga del nodo (ancho de banda LPDDR5X compartido); compare solo ejecuciones de la misma sesión.

## Interruptores de desactivación

Cada palanca de velocidad que mantiene los bytes idénticos está activada por defecto con un interruptor `STRIX_G4_*=0`
que restaura la ruta exacta anterior; los interruptores activos aparecen bajo `off_switches` en cada recibo. Los README
de los módulos los enumeran: `src/gemma4/kernels`, `src/gemma4/prefill`, `src/gemma4/exl3`, `src/gemma4/eagle`,
`src/gemma4/engine`. El registro de cambios de cada palanca de Nemotron activada por defecto con su ganancia medida está
en [ganancias medidas](../../measured-gains.es.md).
