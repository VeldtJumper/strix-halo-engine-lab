# Strix Halo engine lab

**Inferencia de LLM rápida y exacta escrita desde cero para AMD Strix Halo: un mini PC, dos enlazados por USB4 y una
GPU discreta cuando la hay.**

[English](README.md) · [Español](README.es.md)

Este motor de inferencia en C++/HIP está dirigido a **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`,
128 GB de memoria unificada) en Linux con ROCm. Los kernels, la planificación, la decodificación especulativa, el lector
GGUF, el planificador de memoria y el comunicador de dos nodos son código propio; el motor no envuelve otro runtime de
inferencia. Seis configuraciones están en **Production** y dos son opcionales (opt-in); véase [Modelos](#models); cada
una tiene su propio comando de lanzamiento, sus gates de verificación y su servidor.

**Exactitud significa idéntico token a token a la ruta de referencia declarada.** La decodificación especulativa y la
concurrencia preservan ese contrato; cambiar los pesos, la semántica rotatoria, la identidad numérica o la política de
peticiones cambia la referencia. Cada velocidad de este README se midió en el laboratorio con decodificación greedy
(temperatura 0) en la compilación que indica la fila; son resultados específicos de cada carga de trabajo, no garantías
de velocidad. Cómo medimos, qué cubre "exacto" y cómo una optimización pasa a ser predeterminada:
[metodología](docs/methodology.es.md).

<a id="models"></a>

## Modelos

| Modelo | Se ejecuta en | Estado | Aspectos destacados |
|---|---|---|---|
| **Qwen3.8-Flash-Next** (UD-Q4_K_XL, 111 GB) | 1 nodo; 2 réplicas tras un router | **Production** | MTP exacto con una cabeza de draft destilada; contexto nativo hasta 262K y YaRN explícito hasta 1M; prefill por tiles HTTP automático; servidor con tantos slots de petición como admita la memoria (construidos bajo demanda); el decode sigue emitiendo durante prompts largos y las peticiones cortas se admiten dentro de ellos; caché de estado de prefijos; [perfil GSQ opcional (opt-in)](docs/models/qwen3.8/README.es.md#gsq-opt-in-speed-over-quality-profile) |
| **DeepSeek V4.1 Flash** (1017; expertos de 2 bits, densos de 6.5 bits + trellis) | 2 nodos sobre USB4 | **Production** | Decode con paralelismo tensorial de expertos, prefill en dispositivo con paralelismo de expertos, decode especulativo DSpark, decode por batch, servidor; descubrimiento de expertos preparados y retención acotada de prefijos; gates bit a bit |
| DeepSeek V4.1 Flash + RTX PRO 6000 | RTX PRO (capas 0-21) + 2 nodos (22-39) | **Production** (D-064) | División del pipeline sobre 1 GbE, servicio de prefill CUDA, colocación de expertos calientes |
| DeepSeek V4.1 Flash, RTX independiente | RTX PRO 6000 (96 GB, Windows WDDM) | **Production** | Expertos GGUF Q2, densos Q8_0; VRAM exclusiva + caché en memoria de host fijada (pinned); DSpark en la GPU con el controlador caliente seleccionado como opción especulativa predeterminada |
| MiniMax M3.1 Flash (EXL3, 3.06 bpw) | 2 nodos TP2 sobre USB4 | **Production** | Gate de calidad de ocho textos superado; prefill/decode exactos; admisión automática de DSpark en el servicio y modo MSA 6; valores predeterminados de prefill/intercambio activados |
| GLM-5.3 Flash (EXL3, 4.05 bpw) | 2 nodos TP2 sobre USB4 | **Production** | Identidad de decode R1b por defecto; autoespeculación MTP exacta por defecto; prefill por fragmentos R1 exacto; `--identity r1` restaura R1; corrección R≥3 verificada en el par; servidor HTTP `strix-glm5-serve` (chat/completions de OpenAI + SSE, batching continuo) |
| GLM-5.3 Flash en una RTX PRO 6000 (borrador) | RTX PRO 6000 (96 GB, Windows) | **Opcional (opt-in)** | Prefill y decode greedy exactos; expertos enrutados servidos por la caché con compuerta en el dispositivo de [tierstream](https://github.com/hyukudan/tierstream) (dispositivo, RAM pinned, archivo de unidades); `--profile`; [guía](src/glm/rtxpf/README.md) |
| Nemotron 3.5 Bolt E4B (Q4_K_M; paquete EXL3 para `quality`) | 1 nodo | **Compilación opcional (opt-in)** (`STRIX_FAMILY_GEMMA4=ON`) | `--profile default\|fast\|quality`; decode exacto con cualquier tamaño de batch; drafter MTP exacto en C1; servidor con batching continuo |
| **MiniMax H3** (audio + vídeo; DiT int8, pack nvfp4 del codificador de texto) | 1 nodo; 2 nodos sobre USB4 | **Compilación opcional (opt-in)** (`STRIX_FAMILY_H3=ON`) | Clip de 5 s 1344x768 + audio estéreo en **405.5 s** en el par; dos nodos idénticos byte a byte a un nodo; referencia densa y palancas de clase FAST (LoRA de 6 pasos, atención dispersa, VAE rápido) cada una con su interruptor; [página](docs/models/minimax-h3/README.es.md) |

El núcleo no conoce ningún modelo: cada familia es un módulo con su propio contrato, plan, kernels y gates
([docs/models/model-families.md](docs/models/model-families.md)). Las rutas automáticas admiten los assets de modelo según sus contratos documentados; la procedencia y las licencias se registran en
[docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).

<a id="hardware"></a>

## Hardware

- **Nodo:** un mini PC AMD Strix Halo, Ryzen AI Max+ 395 con la iGPU Radeon 8060S (`gfx1151`), 128 GB de
  memoria unificada, Linux, ROCm.
- **Par:** dos nodos unidos por dos cables USB4 directos (un enlace de 40 Gb/s y otro de 20 Gb/s), gestionados por nuestra
  [serie de parches del driver USB4](https://github.com/hyukudan/strix-halo-usb4-driver): unos 37 Gb/s por sentido con ambos
  sentidos ocupados, 18.4 µs para un intercambio de 20 KB.
- **GPU discreta opcional:** una RTX PRO 6000 (96 GB, Windows WDDM), usada para DeepSeek independiente, para delegar el prompt de Qwen
  o para GLM en una sola tarjeta; la división del pipeline de DeepSeek llega al par sobre 1 GbE.

## Resultados principales

Primero los valores predeterminados optimizados: cada fila es la configuración incorporada del motor salvo que el modo
indique otra cosa, con la ganancia medida respecto a nuestro valor predeterminado anterior tal como quedó registrada en
la campaña de esa misma fila. **tok/s** es la velocidad de decode una vez leído el prompt, para una petición salvo que la
concurrencia indique otra cosa; **TTFT** es el tiempo hasta el primer token; C*n* son *n* peticiones concurrentes. Las
ganancias son específicas de cada carga de trabajo y no deben sumarse. Las páginas de cada modelo conservan todas las
filas con sus condiciones, número de repeticiones, evidencia de exactitud e interruptores de desactivación.

| Modelo | Hardware | Modo y carga de trabajo | Resultado | Ganancia respecto a nuestro valor predeterminado anterior | Build |
|---|---|---|---:|---|---|
| **Qwen3.8-Flash-Next** | 1 nodo | Decode MTP, prompts de código de 1K / 8K, ventanas automáticas, 256 tokens | **62.44 / 54.00 tok/s** (prosa 55.7 / 49.58) | código 1K 59.59 → 62.44 tok/s con la puntuación QSA agrupada (+3.0 a +5.7 %) y la fusión de dispatch (−0.3 a −3.2 % de tiempo de decode) | código 1K main `85353311b`; código 8K `702bc074`; prosa `0117a140` / `226bb691` |
| Qwen3.8-Flash-Next | 1 nodo | Decode simple, 1K / 32K / 262K (YaRN) | **41.59 / 43.92 / 45.91 ms/paso** (262K: 21.8 tok/s) | 43.95 / 46.75 / 49.46 ms/paso → −5.4 / −6.1 / −7.2 %, tokens idénticos | main, un binario |
| Qwen3.8-Flash-Next | 1 nodo | TTFT de prefill exacto, 4K / 32K, ruta predeterminada de la CLI simple | **2.397 / 19.05 s** | 2.548 / 20.57 s con todas las palancas predeterminadas desactivadas → −5.9 / −7.4 % (un binario, ejecuciones alternas; 32K T16384 frente a T8192) | main, un binario |
| Qwen3.8-Flash-Next | 1 nodo | Prefill exacto, comando simple, 1K / 4K / 32K / 128K / 262,143 tokens | **1,337 / 1,695 / 1,713 / 1,617 / 1,509 tok/s** (32K 19.13 s; 54–56 % de los techos del contrato) | 1,363 / 1,395 / 1,238 / 1,089 tok/s con 4K / 32K / 128K / 262,143 en `8b99cf58` | main `85353311b`, un binario |
| Qwen3.8-Flash-Next | 1 nodo | Prefill de 262,144 tokens (YaRN factor 2), ruta serial | **1,484 tok/s** `default` (una ejecución) | 1,464 tok/s en `669bcb83` | main `17b9c57b8` |
| Qwen3.8-Flash-Next | 1 nodo | Decode MTP con contexto de 262K / 524K / 1M (YaRN) | **59.5 / 60.6 / 60.2 tok/s** | decode YaRN serial 21.82 / 19.7 / 19.6 tok/s; con 1M el serial se midió con Linux 7.3 y el MTP con Linux 7.4, así que no se da un cociente | `7b2453fe` (MTP 262K), `17b9c57b8` (serial 262K), `a199c8e0` (524K), #288 `3b8b9ac0f` (1M) |
| Qwen3.8-Flash-Next | 2 nodos, una réplica en cada uno | Chats de programación por HTTP, C1 / C8, rendimiento de reloj (wall) | **68.2 / 158.1 tok/s** | +3.6 % wall en C8 y −6.9 % de TTFT en C4 respecto a la compilación release, misma sesión | `0117a140` |
| Qwen3.8-Flash-Next | 1 nodo | Runner HTTP único, conversación de 127.6K tokens, decode en frío | **52.0 tok/s** | 39.5 tok/s antes del pool MTP de QSA; turnos restaurados +31 % / +37 % | main |
| Qwen3.8-Flash-Next + RTX PRO | 1 nodo + RTX PRO | Prefill delegado (offload), TTFT con 32K (8K) | **8.99 s** (2.94 s) | 44.02 s (10.46 s) en el mismo servidor sin la RTX, 4.9× con 32K | `70b93af3` |
| **DeepSeek V4.1 Flash** | 2 nodos por USB4 | Decode especulativo DSpark, código / inglés / español, 256 tokens | **70.6** / 41.9 / 45.3 tok/s | draft cooperativo en ambos ranks: edición de agente 83.8 frente a 80.4 tok/s (+4.3 %), JSON de herramientas 93.5 frente a 89.0 (+5.0 %), prosa 36.1 frente a 34.1 (+5.7 %) | `cab238d6`; draft cooperativo en main |
| DeepSeek V4.1 Flash | 2 nodos | Decode por batch, 8 filas, sin draft | **70.24 ms/paso** (113.9 tok/s agregados, calculado) | 71.84 ms/paso en `69c6bdab` → −2.2 %; filas de expertos enrutados en registros: R6 −1.43 % | `6a5604bf` |
| DeepSeek V4.1 Flash | 2 nodos | TTFT del servidor, un cliente, caché desactivada, 8,399 / 32,499 / 44,699 tokens, `default` | **8.19 / 28.92 / 40.51 s** | 10.44 / 36.85 / 53.62 s en main #148 → 8.29 / 29.77 / 42.72 s (−20.6 / −19.2 / −20.3 %); el cono del decodificador diferido después −3.3 % / −5.9 % con 32.5K / 44.7K; fusión HC −1.1 / −0.9 % | compilación del cono diferido, media de dos turnos |
| DeepSeek V4.1 Flash | 2 nodos | Mismos prompts, `--profile fast` (cambia bits) | **7.56 / 26.33 / 36.77 s** | 8.36 / 29.38 / 41.55 s para `default` en la misma ejecución → −9.6 / −10.4 / −11.5 % | compilación del cono diferido |
| DeepSeek V4.1 Flash + RTX PRO | RTX PRO + 2 nodos | División en pipeline, decode por batch N2 / N4 / N8; servidor C8 | **60.1 / 97.5 / 137.2 tok/s** agregados; **91.7 tok/s** | idéntico bit a bit a la referencia solo del par (sin valor predeterminado anterior en esta ruta) | par `6525d5e8`, CUDA `463c08ec` |
| DeepSeek V4.1 Flash, independiente | RTX PRO 6000 (Windows) | Decode DSpark en la GPU, código / inglés / español, misma carga | **82.3 / 87.7 / 94.4 tok/s** (mejores entre cargas 94–95 / 97 / 100–101) | poda del draft keep-64 +16 % en código; conjunto de kernels de decode/caché con todo en caché 12.62 → 10.50 ms/token | línea medida del PR #52 |
| **MiniMax M3.1 Flash** | 2 nodos | Decode greedy, 1K / 8K, 256 tokens | **43.30 / 45.06 ms/token** (23.10 / 22.19 tok/s), en frío | 44.57 / 46.10 ms/token en `8cc30ac0`; valores predeterminados de frontera de intercambio: N1 48.22 → 43.92 ms/paso; palancas posteriores de malla MoE + o_proj en caliente −1.2 % y −1.3 % con 8K | árbol público del PR #59 |
| MiniMax M3.1 Flash | 2 nodos | Decode DSpark, JSON / Python / prosa | **59.96 / 36.93 / 27.05 tok/s** | 54.4 / 34.1 / 24.1 tok/s en `8cc30ac0`; admisión automática en el servicio: JSON −34.6 %, código −17.5 % de tiempo de reloj, la prosa sigue simple | PR #70 `fbd43423` |
| MiniMax M3.1 Flash | 2 nodos | TTFT del prefill combinado, 8K / 32K | **14.04 s** (583 tok/s) / **63.48 s** (516 tok/s) | 17.731 / 91.508 s en `e0bad3d1` → −21 / −31 % (gather + q/k/v, conjunto de prefill publicado, stream de cola, par de tuberías del prefill) | #334 `c225cdaa2`; 8K brazos alternos en un proceso (13.85–14.29 s), 32K proceso nuevo |
| **GLM-5.3 Flash** | 2 nodos | Decode con autoespeculación MTP, chat / código / JSON | **37.1 / 44.5 / 40.0 tok/s** | 1.50× / 1.79× / 1.62× el decode R1b simple (40.3–40.6 ms/token); identidad R1b −11.5 % frente a R1 | identidad predeterminada |
| GLM-5.3 Flash | 2 nodos | TTFT de prefill exacto por chunks, 1K / 8K / 32K, chunks de 8192 filas | **2.83 / 15.15 / 61.96 s** (541 / 529 tok/s con 8K / 32K), todo predeterminado | 3.18 / 17.77 / 76.59 s → 2.83 / 15.15 / 61.96 s (−11.0 / −14.7 / −19.1 %), logits bit a bit; último paso, tiles del resto MoE: 8K −1.3 % (pares alternos), 32K −1.2 % (ambos órdenes de ejecución) | #300 `32ec66bd3` |
| GLM-5.3 Flash | 2 nodos | Servido (`strix-glm5-serve`), flujos concurrentes C1 / C2 / C4 / C6 / C8, decode en régimen estable; 8 prompts de chat cortos integrados con la plantilla de chat de GLM, 22–59 tokens de prompt (media ~33), hasta 128 tokens greedy cada uno, arranque con barrera, caché desactivada, el servidor no tiene caché de prompts | **C1 40.0 / C2 53.0 (G2 desactivado 37.8) / C4 59.9 (53.6) / C6 62.5 (62.4) / C8 69.5 (69.4) tok/s agregados** | cada flujo == la misma petición sola (exacto); C1 con MTP (25.30 tok/s sin MTP); verificación MTP por lotes (G2) +40 / +12 % con C2 / C4, cede el paso a los pasos simples por lotes desde C6 | actual |
| GLM-5.3 Flash en una RTX PRO 6000 | RTX PRO (Windows) | TTFT de prefill exacto, 8K / 32K, tarjeta completa, tile de 8192 | **11.4 / 30.5 s** | primera petición, media de dos cargas; expulsión limpia en el prefill y el tile de 8192 de la regla en la tarjeta completa; limitado por el cómputo de la GPU con 32K | actual |
| GLM-5.3 Flash en una RTX PRO 6000 | RTX PRO (Windows) | Decode, 1K / 8K, en caliente (en frío) | **13.25 / 13.52 tok/s** (3.63 / 4.09) | tarjeta completa, bench de 24 tokens, medias ABBA de una carga; todas las comparaciones idénticas bit a bit | actual |
| GLM-5.3 Flash en una RTX PRO 6000 | 24 GiB de VRAM + 32 GiB de RAM del host | TTFT 1K / 8K; decode 1K en caliente (en frío) | **65.6 / 57.5 s**; **0.77 tok/s** (0.75) | lecturas de fichero simultáneas en auto por defecto con este presupuesto (76 % de los expertos en el fichero de unidades); TTFT una carga, decode media de dos cargas, bench de 24 tokens; cargas sueltas varían hasta 2x | actual |
| **Nemotron 3.5 Bolt E4B** | 1 nodo | `default`, chat en inglés, 256 tokens: C1 publicado (drafter MTP + prompt lookup) / C1 simple / C4 / C8 | **91.9** / 55.8 / 169.8 / 237.9 tok/s | prompt lookup en un prompt de agente que edita código: 48.0 → 115.6 tok/s sin drafter, **149.1** en la configuración publicada; decode de chat sin cambios dentro del ruido | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 nodo | `fast` (activaciones int8, cualificado), mismas columnas | **99.5** / 54.0 / 206.5 / **340.6** tok/s | prefill exacto 2K 458 → 538 tok/s; decode sin cambios dentro del ruido | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 nodo | `quality` (paquete EXL3 M4), mismas columnas; coincidencia top-1 con BF16 | **96.4** / 50.2 / 184.8 / 303.0 tok/s; **92.94 %** (Q4_K_M 87.0 %) | C1 publicado 83.8 → 96.4; C1 / C4 / C8 simple 44.4 / 153.1 / 258.3 → 50.2 / 184.8 / 303.0 | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 nodo | Ruta de prefill de prompts seleccionable `g4-wmma-pf-v1` (`STRIX_G4_PREFILL_IDENTITY=v1`; lineales WMMA FP16 en el grueso del prompt, los últimos <= 8 tokens del prompt y el ancla con la aritmética del decode; no exacto), 2K / 4K / 8K | **2,359 / 2,249 / 1,957** tok/s | prefill exacto 526 / 509 / 481, `g4-wmma-pf-v2` (`=v2`) 1,537 / 1,478 / 1,349 en la misma sesión; todos los perfiles mantienen el prefill exacto por defecto | actual |
| Nemotron 3.5 Bolt E4B | 1 nodo | Prefill exacto, 2K / 8K / 32K: `default` y `quality` | **535 / 509 / 397** y **815 / 746 / 533 tok/s** | 466 / 429 / 358 y 769 / 701 / 520 tok/s antes del tile fusionado gate/up, los tiles lineales, la decodificación de 24 bits y la de Q6_K compartida, y los pares LDS de EXL3 | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 nodo | Servidor HTTP, `default`, 8 clientes, prompts mixtos (chat, código, un documento de ~1.3K tokens), caché de prefijos desactivada, 128 tokens: agregado / TTFT p50 | **76.1 tok/s / 0.6 s** | una petición cada vez: 66.9 tok/s / 13.8 s | `f32aab24` |
| **MiniMax H3** | 2 nodos sobre USB4 | Texto a vídeo + audio, 1344x768, 124 fotogramas (5.17 s) + audio estéreo, semilla 42: LoRA de 6 pasos, atención dispersa tau 0.75, VAE rápido | **405.5 s** (DiT 300.2 s, VAE de vídeo 42.4 s) | referencia densa de 8 pasos: ~16 min en 1 nodo, ~8 min en el par | h3 block `9d9d523b` |
| llama.cpp `74a4fb00`, misma máquina, mismo Q4_K_M | 1 nodo | Referencia: C1 simple / C4 / C8; prefill 2K / 8K / 32K; servidor con 8 clientes | 55.5 / 193.3 / 279.9 tok/s; 2,062 / 1,881 / 1,376 tok/s; 227.1 tok/s | medido en la misma sesión que las filas de Nemotron | `74a4fb00` |

Páginas de los modelos: [Qwen](docs/models/qwen3.8/README.es.md) · [DeepSeek](docs/models/deepseek-v4.1/README.es.md) ·
[MiniMax](docs/models/minimax-m3.1/README.es.md) · [GLM](docs/models/glm-5.3/README.es.md) ·
[Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.es.md) · [MiniMax H3](docs/models/minimax-h3/README.es.md). El registro de cambios de cada valor predeterminado
incorporado con su ganancia medida y su interruptor de desactivación es [ganancias medidas](docs/measured-gains.es.md);
los registros fechados están en [rendimiento](docs/performance.md).

## Inicio rápido

Este ejemplo mínimo es para **un nodo Linux Strix Halo** con la cadena de herramientas ROCm necesaria, el GGUF del modelo y un fichero `prompt.ids` generado por el tokenizador. Ejecuta **la generación de Qwen con los valores predeterminados de producción**, incluido el prefill rápido automático para los prompts que lo admiten.

```bash
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
ctest --test-dir build --output-on-failure
```

Cada target HIP se compila para los objetivos de GPU AMD de `STRIX_HIP_ARCHITECTURES` (por defecto `gfx1151`); para añadir
una arquitectura, pase por ejemplo `-DSTRIX_HIP_ARCHITECTURES="gfx1151;gfx1160"`. El runtime sigue decidiendo qué conjuntos
de kernels puede ejecutar cada arquitectura.

Con el modelo Qwen correspondiente y los IDs de token disponibles en su directorio de trabajo:

```bash
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

Añada `--print-effective-config` para inspeccionar antes de la inicialización de HIP. Véase [Valores predeterminados e interruptores de desactivación](docs/methodology.es.md#defaults-and-off-switches) para el alcance de la línea base.

El comando produce un recibo JSON con los IDs de token y los tiempos. Los ficheros reales del modelo y los tokens del prompt **no se incluyen**. Para los assets del modelo, la aceleración opcional MTP/DSpark, el servicio y el transporte entre dos nodos, continúe con los [requisitos](docs/getting-started/README.es.md#what-you-need), la [compilación](docs/getting-started/README.es.md#build), la [ejecución](docs/getting-started/README.es.md#run) y la [referencia completa del motor](docs/getting-started/engine-reference.md). Algunos casos de CTest requieren la GPU y los assets del modelo.

Requisitos, la compilación completa de cada motor y el comando de lanzamiento de cada uno:
[primeros pasos](docs/getting-started/README.es.md). Rutas opcionales con GPU NVIDIA: [uso de una GPU discreta](#discrete-gpu).

## Configuración rápida del par DeepSeek

Dos máquinas Strix Halo con 128 GB cada una y dos cables USB4 entre ellas: un solo script comprueba cada nodo, compila
los módulos del controlador USB4 y el motor, descarga y verifica los pesos, arranca los dos ranks y ejecuta una prueba
exacta. Ejecútelo en ambos nodos (en el segundo, con `--rank 1` y la dirección del otro):

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
```

y después `scripts/pair/ds41-pair run --both` y `scripts/pair/ds41-pair smoke` en el nodo del rank 0. Los pasos que
requieren root se imprimen; nunca se ejecutan sin avisar. Todos los subcomandos: [docs/pair-setup.es.md](docs/pair-setup.es.md).

## Perfiles: `default`, `fast`, `quality`

Qwen (CLI, `--http-serve` y el runner MTP), todas las aplicaciones del par DeepSeek y Nemotron aceptan
`--profile default|fast|quality`; ejecute el mismo perfil en ambos ranks de un par. `default` es la configuración exacta
de producción y no establece nada. Una opción explícita o una variable de entorno siempre prevalece sobre el perfil,
incluida la desactivación explícita. `--print-effective-config` (RTX independiente: `--print-config`) muestra el perfil
seleccionado y el estado de cada opción antes de cualquier trabajo en el dispositivo. `--allow-pending` ejecuta las
opciones pendientes cuyo código está en la compilación, solo para benchmarks, marcadas `unqualified`.

| Motor | `fast` | `quality` |
|---|---|---|
| Qwen3.8-Flash-Next | Almacenamiento BF16 de la ruta MoE (`STRIX_QPF2K_MOE_ROUTE_BF16=1`) donde la ruta lo admite: cambia bits, cualificación en curso, el recibo marca la fila como `provisional`. Prefill de 262K 1,505 frente a 1,464 tok/s (`669bcb83`), TTFT de 32K 21.83 → 21.15 s (−3.2 %). El prefill GDN FP16 por chunks sigue pendiente (`--allow-pending` lo ejecuta: 32K −4.1 %, 262K −4.5 %). | No hay paquete de calidad para este modelo: ejecuta `default` y lo indica en el recibo. |
| Par DeepSeek V4.1 Flash | MoE de orden relajado (`STRIX_DS41_FAST_MOE=1`, proyección down en grupos de 32 filas): cambia bits, cualificación en curso. TTFT del servidor 7.56 / 26.33 / 36.77 s frente a 8.36 / 29.38 / 41.55 s de `default` en la misma ejecución. | Sin paquete: ejecuta `default` con un aviso. |
| Nemotron 3.5 Bolt E4B | Activaciones int8 (`g4-a8-v1`, cualificado contra bandas de fidelidad congeladas): C1 99.5 tok/s en la configuración publicada, C8 340.6 tok/s; top-1 frente a BF16 86.7 %. El prefill de prompts sigue siendo exacto; prefill de prompts v1 para `fast`: cualificación del par prefill v1 + decode a8 pendiente. | Paquete EXL3 M4 (`--profile quality --exl3-arm DIR`): coincidencia top-1 con BF16 92.94 % frente a 87.0 % de Q4_K_M, KL media 0.0367 frente a 0.141; 96.4 tok/s en C1 en la configuración publicada. |

Qué perfil de Qwen usar: empiece con el comando simple (decode A8 denso y prefill FP16 rápido de la CLI cuando sea
elegible, cada uno exacto respecto a su propia referencia). Use el perfil FP32 denso cuando necesite esa aritmética; los
dos eligen tokens distintos donde dos candidatos están casi empatados (2.8 % de las posiciones), y cada uno es
determinista y exacto respecto a sí mismo. Detalles de los perfiles por motor:
[Qwen](docs/models/qwen3.8/README.es.md#qwen-profile), [par DeepSeek](docs/models/deepseek-v4.1/README.es.md#deepseek-pair),
[Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.es.md#profiles); GLM en una RTX tiene su propio `--profile` en su
[guía](src/glm/rtxpf/README.md).

<a id="discrete-gpu"></a>

## Uso de una GPU discreta (opcional)

Todo lo anterior funciona solo con Strix Halo. Una RTX PRO 6000 (96 GB) añade cuatro rutas opcionales; ninguna cambia
los valores predeterminados de un nodo ni del par, y cada una tiene su propia identidad declarada y sus gates. Probado
con Windows 11, controlador NVIDIA 617.14 y CUDA 13.0; seleccione la tarjeta por UUID (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=GPU-<uuid>`). Requisitos, una compilación o un lanzamiento por ruta, qué se ejecuta dónde, cuándo
no ayuda una ruta y solución de problemas: [uso de una GPU discreta](docs/discrete-gpu.es.md).

- **DeepSeek V4.1 Flash, RTX independiente (Production, Windows WDDM).** Todo el modelo en la tarjeta: expertos
  enrutados del GGUF Q2, denso Q8_0, una caché de expertos en VRAM exclusiva + host pinned, el modelo objetivo y el
  draft DSpark en la misma GPU. 82.3 / 87.7 / 94.4 tok/s en código / inglés / español con la misma carga. Worker
  persistente con cola de ficheros (`--serve`), striping de expertos en dos discos.
  [Resultados](docs/models/deepseek-v4.1/README.es.md#deepseek-standalone-rtx-results) ·
  [compilación, ejecución y valores predeterminados](docs/models/deepseek-v4.1/rtx-standalone.md).
- **DeepSeek V4.1 Flash, RTX + par (Production, D-064).** La RTX ejecuta las capas 0-21 con CUDA, los dos nodos las
  capas 22-39 y la cabeza, por 1 GbE: 60.1 / 97.5 / 137.2 tok/s agregados con 2 / 4 / 8 filas; lectura exacta del
  prompt en la RTX 3.64 / 4.39 / 12.20 s con 2K / 8K / 32K.
  [Resultados](docs/models/deepseek-v4.1/README.es.md#deepseek-rtx-pipeline) ·
  [contrato de lanzamiento de las etapas del pipeline](docs/models/deepseek-v4.1/pipeline-stages.md).
- **Qwen3.8-Flash-Next, prefill delegado.** Prefill CUDA exacto en la RTX con el estado de atención y recurrente
  transmitido al nodo conforme termina cada capa; el nodo decodifica. TTFT de 32K 8.99 s frente a 44.02 s en el mismo
  servidor sin la RTX; 64 tokens generados idénticos en todas las longitudes.
  [Resultados](docs/models/qwen3.8/README.es.md#qwen-prefill-offload).
- **GLM-5.3 Flash en una RTX PRO 6000 (borrador, opt-in, Windows).** Tronco, capas densas y expertos compartidos en la
  tarjeta; los 43 × 288 expertos enrutados se sirven a través de la caché por niveles de
  [tierstream](https://github.com/hyukudan/tierstream) entre VRAM, RAM pinned y disco. Prefill y decode exactos; TTFT
  11.4 / 30.5 s con 8K / 32K en la tarjeta completa, y también funciona de forma exacta con 24 GiB de VRAM + 32 GiB de
  RAM del host. [Resultados](docs/models/glm-5.3/README.es.md#glm-rtx-results) · [guía](src/glm/rtxpf/README.md).

## MiniMax H3 audio + vídeo (compilación opcional)

**Un clip de vídeo de 5 s 1344x768 con audio estéreo en 405.5 s en el par USB4.** Texto a vídeo + audio, 124
fotogramas, semilla 42, LoRA de 6 pasos, atención dispersa, VAE rápido; un nodo, el esquema de dos nodos en un nodo y
los dos nodos dan los mismos bits para esa identidad. Se compila con `-DSTRIX_FAMILY_H3=ON` (sin la opción no se
compila nada); cada palanca de velocidad tiene su propio flag y sin ellas la ejecución es la referencia densa.
[Instalación, ficheros del modelo, un comando por clip y vuelta atrás](docs/models/minimax-h3/README.es.md#h3-install) ·
[desglose por etapas](docs/models/minimax-h3/README.es.md#h3-result).

## Documentación

- **Primeros pasos:** [requisitos, compilación y ejecución](docs/getting-started/README.es.md) ·
  [referencia del motor](docs/getting-started/engine-reference.md) (opciones de compilación, CLI completa, variables de
  entorno) · [kit de benchmark y reproducción](docs/reproduce.md).
- **Modelos:** [Qwen3.8-Flash-Next](docs/models/qwen3.8/README.es.md) · [DeepSeek V4.1 Flash](docs/models/deepseek-v4.1/README.es.md) ·
  [MiniMax M3.1 Flash](docs/models/minimax-m3.1/README.es.md) · [GLM-5.3 Flash](docs/models/glm-5.3/README.es.md) ·
  [Nemotron 3.5 Bolt E4B](docs/models/nemotron-3.5-bolt-e4b/README.es.md) · [familias de modelos](docs/models/model-families.md).
- **[Metodología](docs/methodology.es.md):** cómo se leen las tablas, exactitud, fidelidad del modelo, valores
  predeterminados e interruptores de desactivación, decisiones de diseño.
- **[Dos nodos por USB4](docs/pair.es.md):** el par, qué necesita y cómo DeepSeek, MiniMax y GLM se ejecutan en él;
  [configuración del par DeepSeek](docs/pair-setup.es.md) con un solo script.
- **[Servicio](docs/serving.es.md):** servidores HTTP, contratos de petición, batching, cachés de prefijo, exposición y
  seguridad.
- **[Ganancias medidas](docs/measured-gains.es.md):** cada valor predeterminado incorporado con su ganancia medida, su
  evidencia de exactitud y su interruptor de desactivación · [catálogo de optimizaciones](docs/optimizations.md)
  ([JSON](docs/optimizations.json)) · [registros de rendimiento](docs/performance.md).
- **Internos:** [guía de arquitectura](docs/architecture/README.md) · [registro de decisiones](docs/decisions/decision-register.md) ·
  [estado de la implementación](docs/reference/implementation-status.md) · [libro de cualificación](docs/reference/qualification-ledger.md) ·
  [fuentes y procedencia](docs/reference/sources-ledger.md) · [índice completo de la documentación](docs/README.md) ·
  [AGENTS.md](AGENTS.md) para agentes de programación.

<a id="related-repositories"></a>

## Repositorios relacionados

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): la serie de parches de Linux que convierte
  los controladores USB4 de Strix Halo en el enlace directo entre los dos nodos.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): la biblioteca de cuantización trellis (estilo EXL3/QTIP)
  usada para los pesos densos de DeepSeek y la inferencia de MiniMax.
- [TensorMeld](https://github.com/hyukudan/tensormeld): colocación y enrutamiento entre nodos y GPU; su router
  sirve las réplicas de Qwen.
- [tierstream](https://github.com/hyukudan/tierstream): streaming de expertos por niveles entre VRAM, memoria de host fijada
  y NVMe, usado por la ruta de DeepSeek en RTX independiente.

<a id="license"></a>

## Licencia

El código fuente del motor se distribuye bajo la **[Apache License 2.0](LICENSE)**, excepto el módulo del kernel de Linux
en `lab/direct_gpu_stream/kernel/`, que es **GPL-2.0-only** como debe serlo el código del kernel de Linux (sus textos
de licencia se conservan en ese directorio). El código adoptado de otros proyectos conserva su licencia y avisos originales. Los pesos
de los modelos se distribuyen bajo sus propias licencias. **Built with MiniMax M3.1-Flash**; la licencia de su modelo
es no comercial por defecto. Conserve los avisos del modelo y de la conversión y revise los términos aplicables. Las referencias de terceros usadas para los formatos y la semántica de los modelos
se enumeran con sus licencias en [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
