# Servicio por HTTP

[English](serving.md) · [Español](serving.es.md) · [README principal](../README.es.md)

Cada motor incluye un servidor. Esta página es el punto de entrada: qué servidor tiene cada motor, qué hace y cómo
exponerlo con seguridad. Los resultados medidos del servicio están en las páginas de los modelos; los contratos de
petición, las opciones y las puertas están en [docs/engine/serving/serving.md](engine/serving/serving.md).

| Motor | Servidor | Funciones de servicio | Resultados medidos |
|---|---|---|---|
| Qwen3.8-Flash-Next | `strix-hip-decode-token --http-serve` | rutas de chat, completions y Responses sin estado compatibles con OpenAI/Anthropic; slots de petición dimensionados según la memoria (`--http-batch N`); prefill por tiles automático (primero T8192, tiles menores bajo presión de memoria); el decode sigue transmitiendo dentro de los bloques de prompt largos y las peticiones cortas se anidan en un prompt largo en pausa; caché certificada de estado de prefijo; JSON restringido, presupuestos de razonamiento, logprobs, transformaciones de logits, progreso SSE; pool MTP de QSA en el runner único; dos réplicas tras el router TensorMeld | [página de Qwen](models/qwen3.8/README.es.md#how-fast) |
| Par DeepSeek V4.1 Flash | `strix-deepseek41-serve` (uno por rank) | rutas de chat, completions y Responses; calentamiento del servidor y calibración de DSpark; lookahead de Engram entre piezas de prefill; retención acotada de prefijos de agente; cono del decodificador diferido para prompts largos; `--profile default\|fast` | [página de DeepSeek](models/deepseek-v4.1/README.es.md#how-fast) |
| DeepSeek V4.1 Flash, RTX independiente | `strix-cuda-ds41-replica --serve SPOOL` (worker persistente con cola de ficheros) | cada petición idéntica bit a bit a un proceso nuevo; semilla de residencia y precaptura de grafos; caché de sesión | [página de DeepSeek](models/deepseek-v4.1/README.es.md#deepseek-standalone-rtx-results) |
| MiniMax M3.1 Flash | `strix-minimax-serve` | decodificación greedy; admisión automática estructural de DSpark; reanudaciones con plazo, admisión de cortas primero y anidada; sesiones con nombre (`context_length_exceeded` al desbordar) | [página de MiniMax](models/minimax-m3.1/README.es.md#how-fast) |
| Par GLM-5.3 Flash | `strix-glm5-serve` (uno por rank) | rutas de chat y completions con SSE; batching continuo de hasta ocho peticiones, los tokens de cada petición son iguales a los de la misma petición servida sola; MTP exacto con una petición, verificación MTP por lotes desde dos (`STRIX_GLM_SERVE_G2=0` la desactiva); prefill por chunks con prioridad al decode (`--prefill-chunk`, 0 lo desactiva); caché exacta de prefijos de prompt, 4 GiB por rank por defecto (`--prefix-cache-gib 0` la desactiva) (filas medidas: caché de prefijos desactivada; 8 prompts de chat cortos integrados con la plantilla de chat de GLM, 22–59 tokens de prompt (media ~33), hasta 128 tokens greedy cada uno, arranque con barrera) | [página de GLM](models/glm-5.3/README.es.md#glm-serve) |
| Nemotron 3.5 Bolt E4B | `strix-gemma4-serve` | batching continuo de hasta ocho peticiones de chat (`STRIX_G4_SERVE_SLOTS`), los tokens de cada petición son iguales a los de la misma petición servida sola; drafter MTP y búsqueda en el prompt con una petición; `--profile default\|fast\|quality` | [página de Nemotron](models/nemotron-3.5-bolt-e4b/README.es.md#how-fast) |

## Exposición de los servidores

Qwen y DeepSeek publican las funciones ejecutadas en `GET /v1/capabilities`; el objeto `strix` de cada respuesta
indica la política de muestreo solicitada frente a la efectiva. En todos los servidores (Qwen, DeepSeek, MiniMax) los
campos de generación no admitidos devuelven 400 indicando el campo. MiniMax decodifica de forma greedy: una temperatura
positiva se acepta y se indica como ignorada en `timings.sampling_note`. Qwen admite JSON restringido exacto respecto a su objetivo restringido, presupuestos de thinking, logprobs,
transformaciones de logits y progreso SSE. Los presupuestos/transformaciones son políticas declaradas que cambian tokens;
los logprobs/el progreso conservan los tokens. La selección de políticas usa ventanas MTP admitidas equivalentes al serial o
la alternativa serial; `--http-selection-mtp off` fuerza esta última. Véanse los
[contratos de peticiones de chat](engine/serving/serving.md#chat-requests) y la
[salida restringida](engine/serving/serving.md#constrained-json-output).

Los servidores HTTP (`strix-hip-decode-token --http-serve` y `strix-deepseek41-serve`) escuchan en loopback salvo que
`--http-bind` indique otra cosa. No tienen TLS propio. La autenticación es una única clave compartida opcional:
con `--http-api-key-file FILE` todas las rutas de API y de administración (completions, `/admin/drain`, `/admin/import-state`)
necesitan `Authorization: Bearer <key>` o `x-api-key: <key>`; sin ella, el servidor acepta cualquier cliente. `/health`,
`/metrics` y `/api/status` responden sin la clave. Los cuerpos, las peticiones en cola y las conexiones abiertas están acotados
(`--http-max-queue`, `--http-max-connections`). Los servidores HTTP cierran una petición que no se ha recibido
por completo en 60 segundos (408; más un segundo por cada MiB de cuerpo) y dan por desaparecido a un cliente que no
acepta datos durante 60 segundos, cancelando su generación. Las peticiones con un encuadre ambiguo (Content-Length en
conflicto, cualquier Transfer-Encoding, línea de petición o nombres de cabecera mal formados) se rechazan con 400/501.
Los cuerpos de petición que se están recibiendo o atendiendo comparten un presupuesto de 256 MiB (503 con Retry-After
cuando está lleno), y un cuerpo de petición puede contener como máximo 2,097,152 valores JSON. Mantenga los servidores en loopback o en una red privada y ponga
los clientes remotos tras un front end que autentique, como el router de [TensorMeld](https://github.com/hyukudan/tensormeld)
o su canal de control; los ranks DeepSeek de dos nodos intercambian tensores por el enlace USB4, y la división RTX
(puerto TCP de la etapa del pipeline) y los servidores de expertos usan TCP sin más: ninguno de ellos lleva autenticación, por lo que deben
permanecer en un cable directo o en una red privada.

## Lecturas adicionales

- [Servicio por HTTP](engine/serving/serving.md): contratos de petición, decode por batch, prefill de prompt
  predeterminado, salida restringida, políticas de planificación medidas.
- [Caché de estado de prefijo](engine/serving/prefix-state-cache.md) y
  [composición del prefill de servicio con la caché de prefijo](engine/serving/serving-prefill-prefix-cache.md).
- [Cualificación de la caché de prefijo de DeepSeek](engine/serving/ds41-prefix-cache-qualification.md),
  [admisión automática de DSpark en el servicio](engine/serving/dspark-auto-qualification.md),
  [caché de sesión de MiniMax](minimax/session-cache.md),
  [caché de sesión de DeepSeek en RTX](models/deepseek-v4.1/rtx-session-cache.md).
- [Varias peticiones en un bucle de decode](engine/scheduling/multi-request-cohort.md),
  [decodificación con muestreo](engine/decode/sampled-decoding.md),
  [assets de release](operations/release-assets.md), [preflight de arranque](operations/startup-preflight.md),
  [salud del servicio](operations/service-health.md).
