# MiniMax M3.1 Flash en el par USB4

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

MiniMax M3.1 Flash, pesos trellis EXL3 a 3.06 bpw, en dos nodos Strix Halo (TP2 sobre USB4): prefill y decode
exactos, servicio DSpark con admisión automática, modo 6 de prefill MSA, servicio con plazos y anidado. Production; la
puerta de calidad de ocho textos superada. Esta página conserva todas las mediciones publicadas con su compilación y sus
condiciones y el comando de lanzamiento. Cómo se leen las tablas y qué cubre "exacto":
[metodología](../../methodology.es.md). Instrucciones de compilación (biblioteca trellis incluida):
[primeros pasos](../../getting-started/README.es.md); el par en sí: [dos nodos por USB4](../../pair.es.md).

<a id="minimax-pair"></a>

## Ejecución

Ejecute una vez por nodo con `R` establecido en 0 o 1 y assets EXL3 coincidentes:

```sh
./build/strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR
```

El [resolutor](../../../src/minimax/engine/mm_defaults.cpp) activa el conjunto de producción de decode/prefill/intercambio,
incluido el modo de prefill MSA 6. `strix-minimax-serve` usa la admisión automática estructural de DSpark con assets
locales compatibles; las peticiones elegibles sin contención pueden usar draft, mientras que las peticiones en competencia/agrupadas o no admitidas
se quedan en simple. Véase [admisión y cualificación](../../engine/serving/dspark-auto-qualification.md).
Una petición de sesión con nombre que no cabe en el contexto devuelve HTTP 400 `context_length_exceeded` (la
sesión sigue aparcada); la generación se detiene al final del contexto con el motivo de fin `length`. Véase
[caché de sesiones](../../minimax/session-cache.md).

<a id="how-fast"></a>

## Resultados medidos

Fuentes: [registros de rendimiento](../../performance.md) y [pruebas actuales de optimización](../../optimizations.md#minimax).

Todas las filas: pesos trellis EXL3 a 3.06 bpw, dos nodos Strix Halo TP2 sobre USB4, referencia greedy.
El motor está en Production; los conjuntos de optimización cualificados siguientes son públicos y están activados por defecto.
Los [cambios de planificación del prefill agrupado en el host](../../minimax-grouped-prefill.md) están activados por defecto para
las llamadas agrupadas en las que cada miembro tiene al menos 2048 filas. Prompts equilibrados de 7940 tokens midieron
un tiempo de prefill un 2.3–2.6% menor con IDs de token completos idénticos; las colas pequeñas desequilibradas usan la ruta ordinaria.

| Modo | Prompt / contexto | Resultado medido | Build / exactitud |
|---|---|---|---|
| Decode greedy | prompts de benchmark de 1K / 8K, 256 tokens | **44.57 / 46.10 ms/token** (22.44 / 21.69 tok/s) | `8cc30ac0`; los ranks coinciden, IDs de referencia iguales |
| Prefill combinado nuevo | prompt de benchmark de 8K | **17.731 s**, 462 tok/s; línea base 21.874 s (**−19 %**); **36–39 %** del suelo de kernel calculado de 6.3–7.0 s | `e0bad3d1`; el estado y los logits finales de ambos ranks son iguales a la producción archivada |
| Prefill combinado nuevo | prompt de benchmark de 32K | **91.508 s**, 358 tok/s; **28–32 %** del suelo de kernel calculado de 26–29 s | `e0bad3d1`; referencia nueva, no una comparación de exactitud con una referencia archivada |
| Prefill combinado, mismo proceso | 1K / 8K, tres pares alternos | **2.73–2.77 / 20.3–20.8 s**, **0.848–0.853× / 0.783–0.797×** el tiempo de la línea base | `561de319`; comprobaciones de estado/fila final y tokens iguales en ambos ranks |
| Batch, proceso nuevo por brazo | N=1 / N=8, benchmark de 1K | **43.92 / 130.24 ms/step** (N8: **61.4 tok/s agregados**, calculado) | `8cc30ac0`; cada fila es igual a ejecutarse sola; la invariancia de batch se cumple |
| DSpark | JSON / Python / prosa, 256 tokens | **54.4 / 34.1 / 24.1 tok/s** | `8cc30ac0`; IDs iguales a greedy |

Los tipos de prompt no se especifican para los benchmarks de prefill y de batch. Las filas de prefill nuevo son ejecuciones únicas;
el resultado del primer uso a 1K fue de **3.509 s**. La ejecución combinada alcanzó después su límite de tiempo en otro brazo; solo se citan
las mediciones completadas. Los nuevos valores predeterminados de prefill en `561de319` incluyen proyecciones densas por tiles, atención/router por batch
y tiles de filas MoE de ajuste exacto. Preservan la identidad de producción.

Los valores predeterminados de límites de intercambio en `8cc30ac0` están cualificados a 1K/8K/16K (16K con repetición sintética): las repeticiones
emparejadas ahorran **4.20 / 2.82 / 3.09 ms/token**. En el A/B de batch con proceso nuevo, N1/N8 bajan de **48.22 / 134.03**
a **43.92 / 130.24 ms/step**; la ralentización anterior de N8 se debió a la limitación térmica/de potencia de la GPU bajo
carga sostenida (unos 96 °C, reloj de shaders −20 % en ~100 s), no al motor. Las filas anteriores son
cifras de arranque en frío con procesos nuevos; con servicio sostenido, el decode medido es alrededor de un 5 % y el prefill
alrededor de un 15 % más lento con esta refrigeración (29 peticiones secuenciales: 44.9 → 47.2 ms/step). Esto no justifica una afirmación general de velocidad de servicio. Los valores predeterminados selectivos posteriores de MoE/atención figuran en el [catálogo](../../optimizations.md#minimax).

**Gate de calidad: superado** en la evaluación prerregistrada de ocho textos, incluidas dos ejecuciones TP2 independientes
e idénticas byte a byte y todas las comprobaciones de prompt/régimen. La KL media respecto a la referencia exacta es **0.1449** frente a
**0.1425** de la aritmética de CPU de producción. La concordancia top-1 es **0.47 puntos porcentuales menor** que la referencia
de CPU, dentro del margen prerregistrado de 2 puntos; la precisión del siguiente token difiere en **+0.01 puntos**. Esta
cualificación está condicionada a esos ocho textos; sustituye al fallo anterior con dos prompts.

## Todos los valores predeterminados incorporados de MiniMax

El registro de cambios de cada palanca de MiniMax activada por defecto, con su ganancia medida, su evidencia de
exactitud y su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el catálogo
completo es [docs/optimizations.md](../../optimizations.md#minimax). Notas relacionadas:
[planificación agrupada del prefill en el host](../../minimax-grouped-prefill.md), [prefill MSA](../../minimax-msa.md),
[caché de sesión](../../minimax/session-cache.md),
[admisión y cualificación de DSpark](../../engine/serving/dspark-auto-qualification.md). Servicio:
[servicio](../../serving.es.md).
