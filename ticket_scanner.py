"""
ScannerTickets - núcleo: cámaras, detección de ticket, OCR y extracción de fechas.

Se puede usar de dos formas:
  python app_gui.py                        # interfaz gráfica (recomendado)
  python ticket_scanner.py [--debug]       # sólo terminal

Fechas: en España siempre van en orden DÍA / MES / AÑO (dd/mm/aaaa, dd-mm-aa,
dd.mm.aaaa, "17 NOV 2026", "12 de enero de 2024"...).
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime

import cv2
import numpy as np

try:  # silenciar avisos de OpenCV al probar índices de cámara inexistentes
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

# --------------------------------------------------------------------------------------
# Cámaras
# --------------------------------------------------------------------------------------

if sys.platform == "win32":
    CAP_BACKEND = cv2.CAP_DSHOW
elif sys.platform == "darwin":
    CAP_BACKEND = cv2.CAP_AVFOUNDATION
else:
    CAP_BACKEND = cv2.CAP_ANY

MAC_CAMERA_HINT = (
    "En macOS la app desde la que lanzas Python (Terminal, iTerm, VS Code...) necesita "
    "permiso de cámara: Ajustes del Sistema > Privacidad y seguridad > Cámara. "
    "Actívalo, cierra y vuelve a abrir esa app, y ejecuta de nuevo."
)


class Camera:
    """Captura en un hilo propio para tener siempre el último frame disponible."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080):
        self.index = index
        self.cap = cv2.VideoCapture(index, CAP_BACKEND)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


def list_cameras(max_index: int = 6) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, CAP_BACKEND)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  Cámara {i}: OK ({w}x{h})")
                found.append(i)
        cap.release()
    return found


def wait_cameras_ready(cams: dict[int, "Camera"], min_s: float = 3, max_s: float = 10) -> None:
    """Algunas webcams dan frames negros hasta que ajustan la exposición:
    esperamos a que el brillo medio de todas se estabilice."""
    t0 = time.time()
    prev: dict[int, float] = {}
    while time.time() - t0 < max_s:
        time.sleep(0.5)
        cur = {}
        for i, c in cams.items():
            f = c.read()
            if f is not None:
                cur[i] = float(cv2.cvtColor(cv2.resize(f, DETECT_SIZE), cv2.COLOR_BGR2GRAY).mean())
        stable = len(cur) == len(cams) and all(abs(cur[i] - prev.get(i, -99)) < 2 for i in cur)
        prev = cur
        if stable and time.time() - t0 > min_s:
            return


# --------------------------------------------------------------------------------------
# Detección de ticket
# --------------------------------------------------------------------------------------

DETECT_SIZE = (320, 180)  # resolución reducida para la detección (rápida)


def _small_gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, DETECT_SIZE), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


def paper_mask(frame: np.ndarray) -> np.ndarray:
    """Máscara booleana de zonas 'papel blanco'. Aplica balance de blancos antes,
    porque con luz cálida el ticket sale anaranjado y no parece blanco."""
    f = frame.astype(np.float32)
    flat = f.reshape(-1, 3)
    lum = flat.sum(1)
    bright = flat[lum >= np.percentile(lum, 90)]
    if bright.size == 0 or bright.mean() < 20:  # imagen negra: no hay papel
        return np.zeros(frame.shape[:2], bool)
    gain = np.clip(bright.mean() / np.maximum(bright.mean(0), 1), 0.5, 2.0)
    wb = np.clip(f * gain, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] > 120) & (hsv[..., 1] < 85)


def sharpness(frame: np.ndarray) -> float:
    """Varianza del laplaciano: valores bajos = imagen desenfocada."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def ticket_rect(frame: np.ndarray, background: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Rectángulo (x0, y0, x1, y1) del ticket en coordenadas del frame, o None."""
    small = cv2.resize(frame, DETECT_SIZE)
    paper = cv2.morphologyEx(
        paper_mask(small).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    changed = (
        np.abs(_small_gray(frame).astype(np.float32) - background) > 30
        if background is not None
        else np.ones(paper.shape, bool)
    )
    # Zonas de papel que contienen algo nuevo respecto al fondo = el ticket
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper)
    keep = [
        k for k in range(1, n)
        if stats[k, cv2.CC_STAT_AREA] > 0.01 * paper.size and np.any(changed[labels == k])
    ]
    if not keep:
        return None
    # Un ticket doblado puede salir en varios trozos: unirlos todos
    ys, xs = np.nonzero(np.isin(labels, keep))
    sx, sy = frame.shape[1] / DETECT_SIZE[0], frame.shape[0] / DETECT_SIZE[1]
    pad = 10
    x0, y0 = max(int(xs.min() * sx) - pad, 0), max(int(ys.min() * sy) - pad, 0)
    x1 = min(int((xs.max() + 1) * sx) + pad, frame.shape[1])
    y1 = min(int((ys.max() + 1) * sy) + pad, frame.shape[0])
    if x1 - x0 < 50 or y1 - y0 < 50:
        return None
    return x0, y0, x1, y1


@dataclass
class TicketDetector:
    """Puntúa cuánto 'ticket' hay delante de una cámara comparando con su fondo."""

    diff_threshold: int = 30
    background: np.ndarray | None = None

    def calibrate(self, frames: list[np.ndarray]) -> None:
        stack = np.stack([_small_gray(f) for f in frames]).astype(np.float32)
        self.background = np.median(stack, axis=0)

    def score(self, frame: np.ndarray) -> float:
        """Devuelve 0..1: fracción de imagen con papel claro que no estaba en el fondo."""
        gray = _small_gray(frame).astype(np.float32)
        if self.background is None:
            self.background = gray
        changed = np.abs(gray - self.background) > self.diff_threshold
        paper = paper_mask(cv2.resize(frame, DETECT_SIZE))
        return float(np.mean(changed & paper))

    def adapt(self, frame: np.ndarray, rate: float = 0.02) -> None:
        """Actualiza el fondo poco a poco (cambios de luz) cuando no hay ticket."""
        gray = _small_gray(frame).astype(np.float32)
        if self.background is None:
            self.background = gray
        else:
            cv2.accumulateWeighted(gray, self.background, rate)


# --------------------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------------------


@dataclass
class OCRLine:
    text: str
    box: np.ndarray  # 4x2 (arriba-izq, arriba-der, abajo-der, abajo-izq) en coords del frame
    conf: float


class OCREngine:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    @staticmethod
    def preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 1.0
        if max(h, w) < 1600:  # subir resolución ayuda con la letra pequeña del ticket
            scale = 1600 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), scale

    def read(self, frame: np.ndarray, rect: tuple[int, int, int, int] | None = None,
             min_conf: float = 0.4) -> list[OCRLine]:
        """OCR sobre el frame (o sólo sobre `rect`). Las cajas vuelven en coords del frame."""
        x0, y0 = 0, 0
        img = frame
        if rect is not None:
            x0, y0, x1, y1 = rect
            img = frame[y0:y1, x0:x1]
        pre, scale = self.preprocess(img)
        result, _ = self.engine(pre)
        lines = []
        for box, text, conf in result or []:
            if float(conf) < min_conf:
                continue
            b = np.array(box, dtype=np.float32) / scale + (x0, y0)
            lines.append(OCRLine(text, b, float(conf)))
        return lines


# --------------------------------------------------------------------------------------
# Extracción de fechas (siempre DÍA / MES / AÑO)
# --------------------------------------------------------------------------------------

MONTHS = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}

# Separadores que el OCR suele devolver en lugar de "-" o "/" (guiones Unicode, _, ~, ",")
_SEP_CHARS = "/\\\\\\-.\u2010\u2011\u2012\u2013\u2014\u2015\u2212_~,"
# Año de 4 cifras: no puede ir seguido de otra cifra, salvo que sea la hora pegada
# ("09-11-202614:35"). El de 2 cifras no puede ir seguido de cifra.
_YEAR = r"(?P<y>\d{4}(?:(?!\d)|(?=\d{1,2}[:.h]\d{2}))|\d{2}(?!\d))"
# Los dos separadores de una fecha deben ser el mismo (tras normalizar con _fix_ocr_digits
# quedan "/" o "-"): así en "23-  06/09/2025" (caja 23) no se lee 23/06/09.
def _dmy_regex(day: str) -> re.Pattern:
    return re.compile(
        rf"(?P<d>{day})\s*(?P<s>[{_SEP_CHARS}])\s*(?P<mo>\d{{1,2}})\s*(?P=s)\s*{_YEAR}"
    )


# dd/mm/aaaa  dd-mm-aa  dd.mm.aaaa  (también d/m/aa)
RE_DMY = _dmy_regex(r"(?<!\d)\d{1,2}")
# Igual pero permitiendo cifras delante ("T0109-11-2026"): sólo se usa si lo anterior no encuentra nada
RE_DMY_LOOSE = _dmy_regex(r"\d{2}")
# 17 NOV 2026 / 17-NOV-26 / 12 de enero de 2024
RE_DMONY = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:DE\s+|[/\-.\s])?\s*("
    + "|".join(MONTHS)
    + r")[A-Z]*\.?\s*(?:DE\s+|[/\-.\s])?\s*(\d{4}|\d{2})(?!\d)"
)
# dd/mm/aaaa con la barra mal leída por el OCR como 1, l, I, | o !  ("22109/2025", "22l09l2025").
# Exige día y mes de 2 cifras y año de 4 para no confundirlo con otros números.
_SLASH_LIKE = r"[/\\1lI|!]"
RE_DMY_SLASH = re.compile(
    rf"(?<![\dA-Z])([\dOo]{{2}})({_SLASH_LIKE})([\dOo]{{2}})({_SLASH_LIKE})(\d{{4}}|[\dOoZ]{{4}})"
    r"(?:(?!\d)|(?=\d{1,2}[:.h]\d{2}))"
)
KEYWORDS = ("FECHA", "FEC.", "FEC:", "DATA")

_OCR_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                          "i": "1", "!": "1", "Z": "2", "z": "2", "S": "5", "s": "5",
                          "B": "8", "g": "9", "q": "9"})
_OCR_CHARS = r"\dOoDQlIi|!ZzSsBgq"


def _fix_ocr_digits(text: str) -> str:
    """Corrige confusiones típicas del OCR (O->0, l/I->1, Z->2...) y unifica separadores
    sólo en trozos con forma de fecha. Mantiene la longitud del texto (sustituciones 1 a 1),
    así las posiciones siguen valiendo para dibujar."""
    def fix(m: re.Match) -> str:
        t = m.group(0).translate(_OCR_FIX)
        return re.sub(rf"[{_SEP_CHARS}]", lambda c: "/" if c.group(0) in "/\\" else "-", t)

    return re.sub(
        rf"[{_OCR_CHARS}]{{1,6}}(?:\s*[{_SEP_CHARS}]\s*[{_OCR_CHARS}]{{1,6}}){{2}}",
        fix,
        text,
    )


def _make_date(d: int, m: int, y: int) -> date | None:
    if y < 100:
        y += 2000
    if not (2000 <= y <= date.today().year + 1):
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


@dataclass
class DateMatch:
    value: date
    raw: str          # texto que ha casado
    line: int         # índice de la línea OCR
    start: int        # posición dentro del texto de la línea
    end: int
    keyword: bool     # junto a "FECHA"
    guess: bool = False  # sacada de una barra mal leída (menos fiable)


def extract_dates(texts: list[str]) -> list[DateMatch]:
    found: list[DateMatch] = []
    for i, raw in enumerate(texts):
        up = raw.upper()
        kw_here = any(k in up for k in KEYWORDS)
        has_kw = kw_here or (
            i > 0 and any(k in texts[i - 1].upper() for k in KEYWORDS)
        )
        fixed = _fix_ocr_digits(raw).upper()
        n_before = len(found)
        for rx in (RE_DMY, RE_DMY_LOOSE):
            for m in rx.finditer(fixed):
                dt = _make_date(int(m.group("d")), int(m.group("mo")), int(m.group("y")[:4]))
                if dt:
                    found.append(DateMatch(dt, m.group(0), i, m.start(), m.end(), has_kw))
            if len(found) > n_before:
                break
        if len(found) == n_before:
            for m in RE_DMY_SLASH.finditer(raw):
                # Si las dos "barras" son un 1 podría ser cualquier número: sólo con "FECHA" cerca
                if m.group(2) == m.group(4) == "1" and not kw_here:
                    continue
                d, mo, y = (int(m.group(k).translate(_OCR_FIX)) for k in (1, 3, 5))
                dt = _make_date(d, mo, y)
                if dt:
                    raw_txt = f"{m.group(1)}/{m.group(3)}/{m.group(5)}".translate(_OCR_FIX)
                    found.append(DateMatch(dt, raw_txt, i, m.start(), m.end(), has_kw, guess=True))
        for m in RE_DMONY.finditer(up):
            dt = _make_date(int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
            if dt:
                found.append(DateMatch(dt, m.group(0), i, m.start(), m.end(), has_kw))
    return found


def best_date(candidates: list[DateMatch]) -> DateMatch | None:
    if not candidates:
        return None
    today = date.today()
    counts = Counter(c.value for c in candidates)
    # Prioridad: fecha bien escrita (no adivinada), junto a "FECHA", no futura,
    # la más repetida y, a igualdad, la más reciente
    return max(candidates, key=lambda c: (not c.guess, c.keyword, c.value <= today,
                                          counts[c.value], c.value))


def match_box(line: OCRLine, m: DateMatch) -> np.ndarray:
    """Caja (4x2) que cubre sólo el trozo de la línea donde está la fecha."""
    n = max(len(line.text), 1)
    s, e = m.start / n, m.end / n
    tl, tr, br, bl = line.box
    return np.array([tl + (tr - tl) * s, tl + (tr - tl) * e,
                     bl + (br - bl) * e, bl + (br - bl) * s], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Escáner: hilo que detecta la cámara con ticket y pasa el OCR
# --------------------------------------------------------------------------------------


@dataclass
class ScanResult:
    cam: int
    time: float
    rect: tuple[int, int, int, int] | None
    lines: list[OCRLine]
    match: DateMatch | None
    match_box: np.ndarray | None
    sharp: float


class Scanner(threading.Thread):
    """Bucle en segundo plano. La interfaz (o la terminal) sólo lee su estado."""

    def __init__(self, cams: dict[int, Camera], ocr: OCREngine, threshold: float = 0.06,
                 ocr_interval: float = 0.7, on_event=None):
        super().__init__(daemon=True)
        self.cams = cams
        self.ocr = ocr
        self.threshold = threshold
        self.ocr_interval = ocr_interval
        self.on_event = on_event or (lambda msg: None)
        self.detectors = {i: TicketDetector() for i in cams}
        self.lock = threading.Lock()
        self.forced_cam: int | None = None   # None = automático
        self.scores: dict[int, float] = {}
        self.active_cam: int | None = None
        self.result: ScanResult | None = None
        self.confirmed: date | None = None   # fecha leída de forma estable
        self.status = "Iniciando..."
        self._recent: deque = deque(maxlen=3)
        self._since = 0.0
        self._recalibrate = True
        self._running = True

    # --- control desde fuera ---
    def recalibrate(self) -> None:
        self._recalibrate = True

    def stop(self) -> None:
        self._running = False

    def _set_status(self, s: str) -> None:
        with self.lock:
            self.status = s

    def _calibrate(self) -> None:
        self._set_status("Calibrando fondo: no pongas ningún ticket delante...")
        samples: dict[int, list[np.ndarray]] = {i: [] for i in self.cams}
        t0 = time.time()
        while time.time() - t0 < 2.0:
            for i, cam in self.cams.items():
                f = cam.read()
                if f is not None:
                    samples[i].append(f)
            time.sleep(0.1)
        for i, frames in samples.items():
            if frames:
                self.detectors[i].calibrate(frames)
        self._set_status("Listo. Pon un ticket delante de una cámara.")
        self.on_event("Fondo calibrado")

    def run(self) -> None:
        self._set_status("Esperando a que las cámaras ajusten la exposición...")
        wait_cameras_ready(self.cams)
        last_ocr = 0.0
        while self._running:
            if self._recalibrate:
                self._recalibrate = False
                self._calibrate()
                self._change_active(None, announce=False)

            frames = {i: c.read() for i, c in self.cams.items()}
            scores = {i: self.detectors[i].score(f) for i, f in frames.items() if f is not None}

            if self.forced_cam is not None:
                cam = self.forced_cam if frames.get(self.forced_cam) is not None else None
            else:
                cam = max(scores, key=scores.get) if scores else None
                if cam is not None and scores[cam] < self.threshold:
                    cam = None

            for i, f in frames.items():  # el fondo se adapta en las cámaras sin ticket
                if f is not None and i != cam and scores.get(i, 1) < self.threshold / 2:
                    self.detectors[i].adapt(f)

            with self.lock:
                self.scores = scores
            if cam != self.active_cam:
                self._change_active(cam)

            now = time.time()
            if cam is not None and now - last_ocr >= self.ocr_interval and now - self._since > 0.4:
                last_ocr = now
                self._do_ocr(cam, frames[cam])
            time.sleep(0.03)

    def _change_active(self, cam: int | None, announce: bool = True) -> None:
        if announce:
            if cam is None:
                self.on_event(f"Ticket retirado de la cámara {self.active_cam}")
            else:
                self.on_event(f"Ticket en la cámara {cam} - leyendo...")
        with self.lock:
            self.active_cam = cam
            self.result = None
            self.confirmed = None
            self.status = "Leyendo ticket..." if cam is not None else "Esperando ticket..."
        self._recent.clear()
        self._since = time.time()

    def _do_ocr(self, cam: int, frame: np.ndarray) -> None:
        rect = None if self.forced_cam is not None else ticket_rect(frame, self.detectors[cam].background)
        crop = frame if rect is None else frame[rect[1]:rect[3], rect[0]:rect[2]]
        lines = self.ocr.read(frame, rect)
        m = best_date(extract_dates([ln.text for ln in lines]))
        box = match_box(lines[m.line], m) if m else None
        res = ScanResult(cam, time.time(), rect, lines, m, box, sharpness(crop))
        confirmed_now = None
        if m:
            self._recent.append(m.value)
            common, n = Counter(self._recent).most_common(1)[0]
            if n >= 2:
                confirmed_now = common
        with self.lock:
            if cam != self.active_cam:
                return
            self.result = res
            if confirmed_now and confirmed_now != self.confirmed:
                self.confirmed = confirmed_now
                self.on_event(f"FECHA (cámara {cam}): {confirmed_now:%d/%m/%Y}  [leído: '{m.raw}']")
            self.status = (f"Fecha: {self.confirmed:%d/%m/%Y}" if self.confirmed
                           else "Leyendo... (fecha no encontrada todavía)")


def open_cameras(indices: list[int], log=print) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    for idx in indices:
        try:
            cams[idx] = Camera(idx)
            log(f"Cámara {idx} abierta")
        except RuntimeError as e:
            log(f"AVISO: {e}")
    if not cams and sys.platform == "darwin":
        log(MAC_CAMERA_HINT)
    return cams


# --------------------------------------------------------------------------------------
# Modo terminal
# --------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lee la fecha de tickets con OCR desde dos cámaras.")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1], help="índices de cámara (por defecto 0 1)")
    ap.add_argument("--list", action="store_true", help="listar cámaras disponibles y salir")
    ap.add_argument("--threshold", type=float, default=0.06,
                    help="fracción mínima de imagen ocupada por el ticket para considerarlo presente")
    ap.add_argument("--ocr-interval", type=float, default=0.7, help="segundos entre lecturas OCR")
    ap.add_argument("--debug", action="store_true", help="imprimir texto OCR de cada lectura")
    args = ap.parse_args()

    if args.list:
        print("Buscando cámaras...")
        if not list_cameras():
            print("  No se ha encontrado ninguna cámara.")
            if sys.platform == "darwin":
                print(MAC_CAMERA_HINT)
        return

    log("Cargando motor OCR...")
    ocr = OCREngine()
    cams = open_cameras(args.cams, log)
    if not cams:
        log("No hay cámaras disponibles. Prueba: python ticket_scanner.py --list")
        return

    scanner = Scanner(cams, ocr, args.threshold, args.ocr_interval, on_event=log)
    scanner.start()
    last_status, last_res = "", None
    try:
        while True:
            time.sleep(0.2)
            with scanner.lock:
                status, res = scanner.status, scanner.result
            if status != last_status and not status.startswith(("Fecha", "Leyendo")):
                log(status)
            last_status = status
            if args.debug and res is not None and res is not last_res:
                log(f"OCR cam{res.cam} (nitidez {res.sharp:.0f}): {[ln.text for ln in res.lines]}")
            last_res = res
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        for cam in cams.values():
            cam.release()
        log("Saliendo.")


if __name__ == "__main__":
    main()
