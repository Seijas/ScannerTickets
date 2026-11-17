# ScannerTickets

Lector automático de la **fecha de tickets de compra** usando una o varias cámaras y OCR.

Pones un ticket delante de la cámara, el programa detecta que hay un papel nuevo, le pasa el OCR
y te dice la fecha del ticket, marcándola con un rectángulo verde sobre la imagen en directo.

Funciona en **Windows** y **macOS** (también debería en Linux), con la webcam del portátil, una
webcam USB o un iPhone/iPad usado como cámara.

---

## Cómo funciona

```
 Cámaras ──► Detección de ticket ──► Recorte ──► OCR ──► Búsqueda de fechas ──► Fecha confirmada
```

1. **Captura.** Cada cámara se lee en su propio hilo, así siempre se tiene el último fotograma
   sin retrasos.
2. **Calibración del fondo.** Al arrancar se toman unos fotogramas *sin ticket* para saber cómo es
   la escena vacía. El fondo se va adaptando poco a poco a cambios de luz mientras no haya ticket.
3. **Detección del ticket.** En cada cámara se busca papel claro (con un balance de blancos previo,
   porque con luz cálida el ticket sale anaranjado) que **no estaba en el fondo**. La cámara con
   más "papel nuevo" es la que tiene el ticket. Si hay varias cámaras se elige sola.
4. **OCR.** Se recorta la zona del ticket, se amplía y se mejora el contraste (CLAHE), y se pasa
   por [RapidOCR](https://github.com/RapidAI/RapidOCR) (modelos ONNX, funciona sin GPU y sin
   conexión).
5. **Extracción de la fecha.** Sobre el texto leído se buscan fechas en formato español
   **día / mes / año**, tolerando los errores típicos del OCR:
   - `09/11/2026`, `09-11-26`, `09.11.2026`, `17 NOV 2026`, `12 de enero de 2024`
   - letras confundidas con números: `O9-11-2O26`, `09-1l-2026`, `Z2/09/2025`
   - separadores raros: `09–11–2026`, `09_11_2026`, `09,11,2026`
   - fecha pegada a la hora: `09-11-202614:35`
   - barra leída como `1`, `l`, `|` o `\`: `22109/2025`, `22l09l2025`
   - números pegados delante que no son fecha: `CAJA 23-  06/09/2025` → 06/09/2025
6. **Elección y confirmación.** Si aparecen varias fechas se prefiere la que está bien escrita, la
   que tiene la palabra *FECHA* cerca, la que no es futura, la que más se repite y, a igualdad, la
   más reciente. Una fecha se da por **confirmada** cuando sale igual en al menos 2 de las 3
   últimas lecturas.

## Interfaz

![colores](https://img.shields.io/badge/verde-fecha_confirmada-3cdc3c) ![colores](https://img.shields.io/badge/naranja-fecha_sin_confirmar-ffa500) ![colores](https://img.shields.io/badge/amarillo-texto_le%C3%ADdo_por_el_OCR-ffdc00) ![colores](https://img.shields.io/badge/azul-zona_del_ticket-28a0ff)

- **Vista principal:** la cámara activa con los recuadros superpuestos.
- **Fecha del ticket:** la fecha leída (verde si está confirmada, naranja si aún no) y el texto
  exacto que la ha producido.
- **Cámara:** modo automático o forzar una cámara concreta (sólo aparece si hay más de una).
- **Miniaturas** de cada cámara con su nivel de detección.
- **Recalibrar fondo:** úsalo si cambias la luz o mueves la cámara (sin ticket delante).
- **Registro** de eventos en la parte inferior.

## Instalación

Necesitas **Python 3.10 – 3.13** (con 3.14+ puede que `onnxruntime` todavía no tenga versión).

### macOS

```bash
git clone https://github.com/Seijas/ScannerTickets.git
cd ScannerTickets
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> Usa el Python de [python.org](https://www.python.org/downloads/macos/), que ya trae `tkinter`.
> Si usas el de Homebrew, instala también `brew install python-tk@3.13`.

**Permiso de cámara:** la app desde la que lanzas Python (Terminal, iTerm, VS Code…) necesita
permiso en *Ajustes del Sistema → Privacidad y seguridad → Cámara*. Tras activarlo, cierra y
vuelve a abrir esa app.

### Windows

```powershell
git clone https://github.com/Seijas/ScannerTickets.git
cd ScannerTickets
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Uso

Interfaz gráfica (recomendado):

```bash
.venv/bin/python app_gui.py              # macOS / Linux
.venv\Scripts\python app_gui.py          # Windows
```

Por defecto usa las cámaras que encuentre entre la 0 y la 1, así que funciona igual con una sola
cámara que con dos. Para elegir otras:

```bash
.venv/bin/python app_gui.py --cams 0 2
```

Modo terminal (sin ventana), útil para depurar:

```bash
.venv/bin/python ticket_scanner.py --list     # ver qué cámaras hay y su número
.venv/bin/python ticket_scanner.py --debug    # imprime todo el texto que lee el OCR
```

| Opción            | Por defecto   | Descripción                                                        |
|-------------------|---------------|--------------------------------------------------------------------|
| `--cams N [N…]`   | las que haya  | Índices de cámara a usar                                            |
| `--list`          |               | Lista las cámaras disponibles y sale                               |
| `--threshold F`   | `0.06`        | Fracción de imagen que debe ocupar el ticket para considerarlo presente |
| `--ocr-interval S`| `0.7`         | Segundos entre lecturas OCR                                        |
| `--debug`         |               | Muestra el texto OCR de cada lectura                               |

(En `app_gui.py` sólo existe `--cams`.)

### Consejos

- Arranca **sin ningún ticket delante**: los primeros segundos se calibra el fondo.
- Buena luz y el ticket lo más plano posible. Si la vista dice *BORROSA*, acerca o enfoca.
- **iPhone como cámara (macOS):** con *Continuity Camera* aparece como una cámara más.
- **iPad como cámara:** macOS no lo permite de serie (Sidecar sólo extiende la pantalla). Usa
  una app como [Camo](https://reincubate.com/camo/) y comprueba su número con `--list`.
- Si una fecha no se detecta, ejecuta con `--debug` y mira qué texto devuelve el OCR en esa línea.

## Estructura

```
ScannerTickets/
├── app_gui.py          # Interfaz gráfica (tkinter)
├── ticket_scanner.py   # Núcleo: cámaras, detección, OCR, fechas + modo terminal
└── requirements.txt    # opencv-python, numpy, rapidocr-onnxruntime, pillow
```

Piezas principales de `ticket_scanner.py`:

| Componente        | Qué hace                                                        |
|-------------------|-----------------------------------------------------------------|
| `Camera`          | Captura en un hilo; backend DirectShow (Windows) / AVFoundation (macOS) |
| `TicketDetector`  | Modelo de fondo y puntuación de "papel nuevo" por cámara        |
| `ticket_rect`     | Rectángulo del ticket para recortar antes del OCR               |
| `OCREngine`       | Preprocesado + RapidOCR                                         |
| `extract_dates`   | Busca fechas en el texto, corrigiendo errores del OCR           |
| `best_date`       | Elige la fecha más probable                                     |
| `Scanner`         | Hilo que lo coordina todo: elige cámara, lanza OCR, confirma    |
