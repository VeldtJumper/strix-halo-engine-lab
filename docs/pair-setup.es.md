# Configurar el par DeepSeek con `ds41-pair`

[English](pair-setup.md) · [Español](pair-setup.es.md) · [Dos nodos por USB4](pair.es.md) · [README principal](../README.es.md)

`scripts/pair/ds41-pair` lleva dos máquinas Strix Halo desde un checkout limpio hasta un par DeepSeek V4.1 Flash en
marcha y verificado: comprobación previa, controlador USB4, compilación, pesos, preparación opcional de expertos,
arranque y una prueba exacta. Ejecute los mismos comandos en los dos nodos. Cada subcomando es idempotente: si falla o
se interrumpe, vuelva a ejecutarlo y continúa donde se quedó. Necesita Python 3.10+ y solo la biblioteca estándar.

El script nunca usa root por sí mismo. Los pasos que requieren root (instalar los módulos USB4, la regla udev y el
servicio de streams) se imprimen para que usted los ejecute; `driver --yes-root` los ejecuta con `sudo` y muestra
cada comando antes de ejecutarlo.

## El recorrido completo

En **ambos** nodos (el nodo A será el rank 0 y servirá HTTP; el nodo B, el rank 1):

```sh
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models   # en el nodo B: --rank 1 --peer user@node-a
scripts/pair/ds41-pair check
scripts/pair/ds41-pair driver          # compila los módulos y luego imprime los pasos de root
scripts/pair/ds41-pair driver-verify
scripts/pair/ds41-pair build
scripts/pair/ds41-pair fetch           # el nodo B puede usar: fetch --from-peer
scripts/pair/ds41-pair prepare         # opcional, una vez
```

Después, en el nodo A:

```sh
scripts/pair/ds41-pair run --both      # arranca el rank 1 en el nodo B por SSH y luego el rank 0 aquí
scripts/pair/ds41-pair smoke
```

`smoke` termina con `PASS smoke ... token ids equal the published reference` y una línea de velocidad. El servidor
habla las API de OpenAI y Anthropic en `http://127.0.0.1:8080` del nodo A ([servicio](serving.es.md)).

## Antes de empezar

- Dos máquinas Strix Halo (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`) con 128 GB cada una, Linux y ROCm con un
  compilador HIP ([requisitos](getting-started/README.es.md#what-you-need)).
- Dos cables USB4 directos entre las máquinas (con uno también funciona, con aproximadamente la mitad del ancho de
  banda). Conéctelos con las dos máquinas encendidas.
- Unos **390 GB** libres para los ficheros del modelo en cada nodo, más unos 77 GB en `~/.cache` si usa `prepare`.
- SSH con clave de cada nodo al otro (`ssh -o BatchMode=yes user@node-b true` debe funcionar). El par no usa la red;
  SSH solo sirve para `check`, `fetch --from-peer`, `run --both`, `stop --both` y `driver-verify --link-test`.
- Mantenga el IOMMU activado. El controlador USB4 depende de él para aislar los controladores; `check` falla si está
  desactivado en la línea de comandos del kernel.

## Fichero de configuración

`init` escribe `~/.config/strix-pair/ds41.conf` (o el fichero de `--config FILE` / `$STRIX_PAIR_CONFIG`). Usa
sintaxis de shell, un `KEY=value` por línea, y puede editarlo. Cada nodo tiene su propio fichero; solo cambian
`RANK` y `PEER`.

| Clave | Por defecto | Significado |
|---|---|---|
| `RANK` | (obligatoria) | `0` sirve HTTP, `1` es la otra mitad |
| `PEER` | | destino SSH del otro nodo, por ejemplo `user@node-b` |
| `MODEL_DIR` | (obligatoria) | los ficheros de producción, con la estructura de abajo |
| `PROFILE` | `default` | `default` (la configuración exacta de producción), `fast` o `quality`; el mismo en ambos ranks |
| `BUILD_DIR` | `build-ds41` en el checkout | directorio de compilación del motor |
| `DEPS_DIR` | `~/.local/share/strix-pair` | biblioteca trellis, fuentes del controlador y del kernel, módulos compilados, manifiestos |
| `PEER_REPO`, `PEER_MODEL_DIR` | las mismas rutas que aquí | el checkout y el directorio de modelos en el otro nodo |
| `ROCM_PATH` | `hipconfig -R`, si no `/opt/rocm` | raíz de ROCm |
| `HTTP_BIND`, `HTTP_PORT` | `127.0.0.1`, `8080` | dirección HTTP del rank 0; consulte [exponer los servidores](serving.es.md#exposing-the-servers) antes de usar otra dirección |
| `API_KEY_FILE` | | fichero con la clave de la API HTTP |
| `CONTEXT` | valor por defecto del servidor | longitud de contexto |
| `EXTRA_ARGS` | | más opciones de `strix-deepseek41-serve`, las mismas en ambos ranks |

`ds41-pair config` imprime los valores efectivos.

## Subcomandos

### `init`

`init --rank 0|1 --model-dir DIR [--peer USER@HOST] [--profile P] [--build-dir DIR] [--deps-dir DIR]
[--http-bind ADDR] [--http-port N] [--api-key-file F] [--context N] [--peer-repo DIR] [--peer-model-dir DIR]
[--rocm DIR]` escribe o actualiza la configuración. Las opciones que no indique conservan su valor actual.

### `check`

Comprobación previa de solo lectura de este nodo, con una línea `PASS`, `WARN`, `FAIL` o `INFO` por elemento; el
estado de salida es 1 si algo falla. `--no-peer` omite la parte de SSH.

| Elemento | Comprueba |
|---|---|
| `cpu`, `gpu`, `kfd` | una CPU Ryzen AI Max, una GPU `gfx1151` en la topología de KFD, `/dev/kfd` con lectura y escritura para usted (si no, únase a los grupos `render` y `video`) |
| `memory` | un nodo de 128 GB; la GPU puede mapear unos 117 GiB de memoria del sistema (GTT, `ttm pages_limit`); `kho=off`; avisa si el firmware reserva una gran parte de VRAM o si otros procesos ocupan la GPU |
| `kernel`, `iommu` | el kernel en ejecución; el IOMMU está activado y los controladores USB4 se traducen (`DMA` / `DMA-FQ`); `iommu=pt` avisa, `amd_iommu=off` / `iommu=off` falla. Nunca sugiere desactivar el IOMMU |
| `usb4`, `link` | `thunderbolt_stream` cargado, `thunderbolt_net` sin cargar, velocidad y carriles de cada cable según sysfs: Gen3 x2 (20 Gb/s por carril, 40 Gb/s por sentido) pasa; Gen2 o un solo carril avisa con "re-seat that cable with both nodes on" (vuelva a conectar ese cable con los dos nodos encendidos) |
| `streams` | cuatro dispositivos `/dev/tbstream*` con lectura y escritura para usted, y el fichero de disposición de los streams |
| `disk`, `models` | espacio libre frente a lo que `fetch` aún debe transferir; los ficheros de producción por nombre y tamaño |
| `rocm`, `tools` | un compilador HIP, CMake 3.24+, Ninja, Git, Python 3.10+, las cabeceras de OpenSSL (hash más rápido) y las herramientas de compilación del kernel para `driver` |
| `build`, `peer` | el binario del motor; el otro nodo es accesible por SSH, tiene el otro rank y ejecuta el mismo binario del motor (SHA-256) |

### `driver` y `driver-verify`

El par usa la [serie del controlador USB4](https://github.com/hyukudan/strix-halo-usb4-driver): módulos `thunderbolt`
y `thunderbolt_stream` parcheados para un kernel upstream fijado (Linux 7.3-rc5), más un servicio de arranque que crea
dos streams por cable. `driver` (sin root) elige la vía para este nodo:

- **El módulo de streams de la serie fijada ya está cargado** (un kernel compilado a partir de la serie): no hay nada
  que compilar. Solo imprime los ajustes de memoria de abajo si faltan.
- **El kernel en ejecución es la versión de la serie pero no tiene el módulo:** obtiene el repositorio del controlador
  y el árbol upstream en sus commits fijados (un fetch superficial de Git, una vez), aplica la serie, configura el árbol
  con la configuración del kernel en ejecución más el fragmento del controlador, compila `thunderbolt`,
  `thunderbolt_stream` y `thunderbolt_net` contra el `Module.symvers` del kernel en ejecución (linux-headers; sin él se
  compila el árbol completo una vez, o puede forzarlo con `--full`), comprueba la version magic de cada módulo y los
  deja en `DEPS_DIR/usb4-modules/<kernel>/` con sus sumas SHA-256 y el srcversion de `thunderbolt_stream`.
- **Cualquier otro kernel, o USB4 integrado en el kernel:** los módulos cargables no pueden coincidir con él. `driver`
  lo indica y se detiene; `driver --kernel` compila entonces el kernel parcheado completo como un kernel nuevo junto al
  actual (LOCALVERSION bajo `-strix-gpu-bosgame-`, comprobado con el `check_kconfig.py` del controlador; paquetes
  Debian con `bindeb-pkg` donde exista `dpkg-deb`), unos 30-60 min de compilación. `--localversion` fija su sufijo.

Después imprime los pasos de root para este nodo: instalar los módulos o el paquete del kernel, una regla udev que da
a su grupo lectura y escritura en `/dev/tbstream*`, una regla de modprobe que mantiene `thunderbolt_net` desactivado
(tomaría un par de anillos por controlador de los streams), `usb4stream-describe` y `setup-usb4stream-lab` del
repositorio del controlador y el servicio de streams, y, si faltan, los dos ajustes de memoria de la preparación del
nodo del controlador: `options ttm pages_limit=30720000` (permite a la GPU mapear 117 GiB de memoria del sistema; el
par no cabe en la mitad de la RAM que se usa por defecto) y `kho=off` en la línea de comandos del kernel. Los módulos o
el kernel nuevos surten efecto en el siguiente arranque; no se recarga nada con el sistema en marcha. Arranque un
kernel nuevo una vez con `grub-reboot` antes de hacerlo el predeterminado, mantenga el kernel actual como entrada de
respaldo y no desactive nunca el IOMMU. Con Secure Boot activado, firme antes los módulos con una clave registrada
(`driver` avisa). `--dry-run` muestra la vía y el plan sin compilar.

`--yes-root` ejecuta esos pasos con `sudo`, imprimiendo cada uno antes de ejecutarlo y deteniéndose en el primer
fallo. `--rebuild` vuelve a compilar los módulos; `--jobs N` fija el paralelismo.

`driver-verify` (solo lectura) comprueba en este nodo: ambos módulos cargados (el módulo de streams de la serie fijada,
y la compilación preparada si `driver` compiló una), sin `thunderbolt_net`, ambos cables activos a Gen3 x2, y los cuatro
streams del fichero de disposición conectados y con lectura y escritura. `--link-test` además empareja los cuatro
streams con el otro nodo igual que lo hace el motor e intercambia datos comprobados (arranca el otro lado por SSH y usa
`strix-dist-host-exchange-bench` de la compilación). Ejecútelo sin ningún motor en marcha en ninguno de los dos nodos.

### `build`

Obtiene la biblioteca [strix-trellis](https://github.com/hyukudan/strix-trellis) en un commit fijado, la compila para
`gfx1151` con la API densa R1 TP y después configura y compila el motor con las opciones de producción de DeepSeek
(`STRIX_FAMILY_DEEPSEEK41=ON`, `STRIX_DS41_TRELLIS=ON`, `STRIX_DS41_TRELLIS_API=dense_r1`, la biblioteca y sus
cabeceras; hash con OpenSSL si están las cabeceras). Los objetivos son `strix-deepseek41-serve`,
`strix-deepseek41-rank`, `strix-tokenize`, `strix-ds41-prepare-load` y `strix-dist-host-exchange-bench`. Escribe
`BUILD_DIR/ds41-pair-build.json` con los commits del motor y de la biblioteca y el SHA-256 de cada binario. Compile el
mismo commit en ambos nodos; `check` compara los binarios. `--jobs N`, `--clean`.

### `fetch`

Descarga los ficheros de producción desde los tres repositorios espejo en las revisiones fijadas en
`scripts/bench/lib/fetch_models.py` y comprueba cada fichero frente al `SHA256SUMS` del espejo en esa revisión:

| Espejo | Ficheros |
|---|---|
| [deepseek-v4.1-flash-gguf-mirror](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror) | `DeepSeek-V4.1-Flash-1017-Q2.gguf` (365.7 GB), `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` (8.0 GB) |
| [DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) | `c-u6.gguf` (7.6 GB), `c-u6.json`, `aux/vocab/*.bin`, `aux/draft-keep-stats.jsonl` |
| [DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) | `ours-K4/` (3.6 GB), `ours-K5/` (4.5 GB), `mixB.txt` (producción: ours-K4 más la cabeza de ours-K5) |

El resultado es la estructura documentada ([reproducir](reproduce.md#models-and-provenance)); `vocab/` se rellena a
partir de `aux/vocab/` ya verificado:

```text
/path/to/models/
  DeepSeek-V4.1-Flash-1017-Q2.gguf
  DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
  c-u6.gguf  c-u6.json
  ours-K4/  ours-K5/  mixB.txt
  aux/  vocab/
  SHA256SUMS.<mirror>  LICENSE*  README*
```

Las descargas van a `NAME.part` y se reanudan tras una interrupción (vuelva a ejecutar `fetch`). Un fichero solo se
renombra a su sitio cuando su SHA-256 coincide; un fichero que no coincide se borra y se vuelve a descargar. Los hashes
de los ficheros verificados se recuerdan por tamaño y fecha de modificación en `MODEL_DIR/.ds41-pair-verified.json`,
así que una nueva ejecución no vuelve a leer 390 GB. Para una descarga con acceso restringido o con límite de
peticiones, exporte `HF_TOKEN` (solo se envía al espejo y nunca se imprime).

| Opción | Efecto |
|---|---|
| `--dry-run` | el plan: cada fichero que falta, está a medias o no coincide, con su tamaño, los bytes pendientes y el espacio libre; no transfiere nada |
| `--from-peer` | copia cada fichero desde el directorio de modelos del otro nodo con `rsync` por SSH en lugar de descargarlo dos veces, y después lo verifica aquí |
| `--verify` | vuelve a calcular el hash de cada fichero aunque el hash recordado siga siendo válido |
| `--verify-only` | calcula el hash de los ficheros ya presentes frente a los manifiestos; nunca transfiere |
| `--only GLOB` | limita a los ficheros que coinciden (repetible), por ejemplo `--only 'ours-K*'` |

`--from-peer` usa la dirección a la que resuelva `PEER`. Los enlaces USB4 llevan los streams del motor, no IP
(`thunderbolt_net` está en la lista negra), así que la copia va normalmente por su red local.

### `prepare` (opcional)

Ejecuta `strix-ds41-prepare-load` una vez para este rank con la identidad densa de producción: un pack local del rank
con los expertos enrutados en la disposición TP `dup` que usa el servidor, unos 72 GiB en
`~/.cache/strix/ds41-prepared`. El servidor lo encuentra y lo valida en cada arranque y carga sus expertos unos 20 s más
rápido; sin él, se usa el cargador original con una línea de aviso. Detalles:
[preparación única de expertos](models/deepseek-v4.1/prepared-load.md). Prepare en cada nodo, después de `fetch` y
`build`; vuelva a preparar tras cambiar los ficheros del modelo (borre el pack antiguo). `--dry-run` imprime el comando
y el espacio necesario; `--force` prepara aunque ya exista un pack.

### `run`, `stop`, `status`, `logs`, `unit`

`run` arranca el rank de este nodo en segundo plano mediante `scripts/pair/ds41-pair-launch` y espera su línea `READY`
(la carga del modelo y el calentamiento tardan de uno a unos pocos minutos; los dos ranks se esperan hasta un minuto).
`--both` arranca primero el rank del otro nodo por SSH. `--foreground` lo ejecuta en la terminal; `--no-wait` vuelve
enseguida; `--wait S` fija la espera. Los registros van a `~/.local/state/strix-pair/rank.log`.

`ds41-pair-launch` es el comando de producción tal cual, con su configuración:

```sh
STRIX_DS41_PACKS=/path/to/models \
  BUILD_DIR/strix-deepseek41-serve /path/to/models/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --rank R --transport usb4auto --profile default --http-bind 127.0.0.1 --http-port 8080
```

El servidor resuelve por sí mismo el overlay, los packs trellis, el drafter, la atención TP y el prefill en el
dispositivo ([valores de producción](models/deepseek-v4.1/production-defaults.md)). Los argumentos adicionales van al
servidor, por ejemplo `scripts/pair/ds41-pair-launch --print-effective-config`.

`stop [--both]` envía SIGTERM (el servidor termina las peticiones en curso; detener el rank 0 detiene también el rank
1). `status` muestra el proceso, su estado READY y `/health`. `logs [-n N]` imprime el final del registro.

`unit` escribe una unidad systemd de usuario `strix-ds41-pair.service` para este rank (ejecuta `ds41-pair-launch` y se
reinicia tras un fallo al cabo de 30 s) e imprime los comandos `systemctl --user`; `--install` además la habilita y la
arranca. Para arrancarla al iniciar el sistema sin iniciar sesión, ejecute una vez `sudo loginctl enable-linger USER`.
Use la unidad o `run`, no ambos.

### `smoke`

En el rank 0 (en el rank 1 se ejecuta en el otro nodo por SSH). Espera a `/health`, envía un prompt fijo con
decodificación greedy y `return_token_ids`, y compara los 32 IDs de token con la referencia publicada en
`scripts/pair/ds41-smoke-reference.json`. Después, una petición de 128 tokens imprime una línea de velocidad: tiempo
hasta el primer token y tokens por segundo de decode de un stream (DSpark está activado, así que la tasa depende del
texto). La referencia es para el perfil `default`; con `fast` se imprimen los IDs sin veredicto. Una discrepancia
significa que los dos nodos difieren en compilación, perfil o ficheros: ejecute `check` en ambos y `fetch --verify-only`.

## Cuando algo falla

| Síntoma | Qué hacer |
|---|---|
| `link ... trained at Gen2` o `single lane` | vuelva a conectar ese cable con los dos nodos encendidos y después `driver-verify` |
| `no USB4 link up` | cables conectados con los dos nodos encendidos; `driver-verify`; el servicio de streams se ejecutó (`systemctl status usb4stream-pair`) |
| `driver`: la serie es para otro kernel | `driver --kernel` (o el `BUILD.md` del repositorio del controlador a mano), arranque el kernel nuevo una vez, `driver-verify` |
| `memory ... GTT limit` | instale el paso `ttm.conf` de `driver` y reinicie |
| `streams ... not read/write for you` | instale la regla udev de `driver` y vuelva a iniciar sesión |
| `run`: sin READY, POISONED | `ds41-pair logs`; ambos ranks deben ejecutar la misma compilación y el mismo perfil; arranque el otro rank en menos de un minuto |
| `fetch`: SHA-256 no coincide | vuelva a ejecutar `fetch` (la copia mala se borró) |
| `smoke`: los IDs de token difieren | el mismo commit compilado en ambos nodos (`check`), `PROFILE=default`, `fetch --verify-only` en ambos |
