"""
ScannerTickets - interfaz gráfica.

Muestra la cámara en tiempo real, detecta en cuál hay un ticket, le pasa el OCR
y dibuja un rectángulo verde sobre la fecha que ha leído.

    python app_gui.py              # cámaras 0 y 1
    python app_gui.py --cams 0 1 2
"""

from __future__ import annotations

import argparse
import queue
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from ticket_scanner import OCREngine, Scanner, open_cameras

MAIN_W, MAIN_H = 960, 540
THUMB_W, THUMB_H = 300, 169

GREEN = (60, 220, 60)
ORANGE = (0, 165, 255)
BLUE = (255, 160, 40)
YELLOW = (0, 220, 255)


def to_tk(frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def fit(frame: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona manteniendo proporción dentro de w x h (con bandas negras)."""
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    canvas = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s, ox, oy


class App:
    def __init__(self, root: tk.Tk, cam_indices: list[int]):
        self.root = root
        self.events: queue.Queue[str] = queue.Queue()
        root.title("Scanner Tickets")
        root.configure(bg="#1e1e1e")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui(cam_indices)
        self.log("Cargando motor OCR...")
        root.update()

        self.ocr = OCREngine()
        self.cams = open_cameras(cam_indices, self.log)
        if not self.cams:
            self.log("No hay cámaras disponibles.")
            return
        self.scanner = Scanner(self.cams, self.ocr, on_event=self.events.put)
        self.scanner.start()
        self.last_cam = next(iter(self.cams))
        self.tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, cam_indices: list[int]) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        for w in ("TFrame", "TLabel", "TRadiobutton", "TCheckbutton", "TLabelframe", "TLabelframe.Label"):
            style.configure(w, background="#1e1e1e", foreground="#e0e0e0")
        style.configure("Date.TLabel", font=("Segoe UI", 34, "bold"), foreground="#777")
        style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#aaa")
        style.configure("Cam.TLabel", font=("Segoe UI", 9))

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        # Vista principal
        self.view = tk.Label(main, bg="black", width=MAIN_W, height=MAIN_H)
        self.view.grid(row=0, column=0, rowspan=2, sticky="nw")
        self.view_title = ttk.Label(main, text="", font=("Segoe UI", 11, "bold"))
        self.view_title.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Panel derecho
        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="n")

        box = ttk.LabelFrame(side, text=" Fecha del ticket ", padding=8)
        box.pack(fill="x")
        self.date_lbl = ttk.Label(box, text="--/--/----", style="Date.TLabel")
        self.date_lbl.pack()
        self.read_lbl = ttk.Label(box, text="", style="Status.TLabel")
        self.read_lbl.pack()
        self.status_lbl = ttk.Label(side, text="", style="Status.TLabel", wraplength=THUMB_W)
        self.status_lbl.pack(fill="x", pady=(6, 6))

        mode = ttk.LabelFrame(side, text=" Cámara ", padding=6)
        mode.pack(fill="x")
        self.mode = tk.StringVar(value="auto")
        ttk.Radiobutton(mode, text="Automática (la que tenga ticket)", value="auto",
                        variable=self.mode, command=self._mode_changed).pack(anchor="w")
        for i in cam_indices:
            ttk.Radiobutton(mode, text=f"Cámara {i}", value=str(i),
                            variable=self.mode, command=self._mode_changed).pack(anchor="w")

        self.thumbs: dict[int, tuple[tk.Label, ttk.Label]] = {}
        for i in cam_indices:
            img = tk.Label(side, bg="black", width=THUMB_W, height=THUMB_H)
            img.pack(pady=(8, 0))
            cap = ttk.Label(side, text=f"Cámara {i}", style="Cam.TLabel")
            cap.pack(anchor="w")
            self.thumbs[i] = (img, cap)

        opts = ttk.Frame(side)
        opts.pack(fill="x", pady=(8, 0))
        self.show_all = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Ver todo el texto que lee el OCR",
                        variable=self.show_all).pack(anchor="w")
        ttk.Button(opts, text="Recalibrar fondo (sin ticket delante)",
                   command=self._recalibrate).pack(fill="x", pady=(6, 0))

        # Registro
        self.logbox = tk.Text(main, height=8, bg="#111", fg="#ccc", insertbackground="#ccc",
                              font=("Consolas", 9), relief="flat")
        self.logbox.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def log(self, msg: str) -> None:
        self.logbox.insert("end", f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.logbox.see("end")

    def _mode_changed(self) -> None:
        v = self.mode.get()
        self.scanner.forced_cam = None if v == "auto" else int(v)
        self.log("Modo automático" if v == "auto" else f"Forzada la cámara {v}")

    def _recalibrate(self) -> None:
        self.scanner.recalibrate()

    # ------------------------------------------------------------------ bucle
    def tick(self) -> None:
        while not self.events.empty():
            self.log(self.events.get_nowait())

        with self.scanner.lock:
            scores = dict(self.scanner.scores)
            active = self.scanner.active_cam
            res = self.scanner.result
            confirmed = self.scanner.confirmed
            status = self.scanner.status

        if active is not None:
            self.last_cam = active
        cam_id = active if active is not None else self.last_cam
        frames = {i: c.read() for i, c in self.cams.items()}

        # Vista principal con superposiciones
        frame = frames.get(cam_id)
        if frame is not None:
            view, s, ox, oy = fit(frame, MAIN_W, MAIN_H)
            if res is not None and res.cam == cam_id:
                self._draw_result(view, res, confirmed, s, ox, oy)
            self._main_img = to_tk(view)
            self.view.configure(image=self._main_img)
        title = f"Cámara {cam_id}"
        title += "  ·  TICKET DETECTADO" if active is not None else "  ·  sin ticket"
        if res is not None and res.cam == cam_id:
            title += f"  ·  nitidez {res.sharp:.0f}" + ("  (BORROSA)" if res.sharp < 60 else "")
        self.view_title.configure(text=title)

        # Miniaturas
        self._thumb_imgs = {}
        for i, (img_lbl, cap_lbl) in self.thumbs.items():
            f = frames.get(i)
            if f is None:
                continue
            th, *_ = fit(f, THUMB_W, THUMB_H)
            if i == active:
                cv2.rectangle(th, (1, 1), (THUMB_W - 2, THUMB_H - 2), GREEN, 3)
            self._thumb_imgs[i] = to_tk(th)
            img_lbl.configure(image=self._thumb_imgs[i])
            tag = "  ← TICKET" if i == active else ""
            cap_lbl.configure(text=f"Cámara {i}   detección {scores.get(i, 0):.3f}{tag}")

        # Fecha
        if confirmed:
            self.date_lbl.configure(text=f"{confirmed:%d/%m/%Y}", foreground="#3cdc3c")
        elif res is not None and res.match:
            self.date_lbl.configure(text=f"{res.match.value:%d/%m/%Y}", foreground="#ffa500")
        else:
            self.date_lbl.configure(text="--/--/----", foreground="#777")
        if res is not None and res.match:
            self.read_lbl.configure(text=f"texto leído: «{res.match.raw}»")
        else:
            self.read_lbl.configure(text="")
        self.status_lbl.configure(text=status)
        self.root.after(33, self.tick)

    def _draw_result(self, view, res, confirmed, s, ox, oy) -> None:
        def pts(box):
            return (np.asarray(box) * s + (ox, oy)).astype(np.int32)

        if res.rect is not None:
            x0, y0, x1, y1 = res.rect
            cv2.rectangle(view, (int(x0 * s) + ox, int(y0 * s) + oy),
                          (int(x1 * s) + ox, int(y1 * s) + oy), BLUE, 1)
        if self.show_all.get():
            for ln in res.lines:
                cv2.polylines(view, [pts(ln.box)], True, YELLOW, 1)
        if res.match_box is not None:
            color = GREEN if confirmed == res.match.value else ORANGE
            p = pts(res.match_box)
            # margen para que el rectángulo no pise el texto
            c = p.mean(0)
            p = ((p - c) * np.array([1.2, 1.6]) + c).astype(np.int32)
            cv2.polylines(view, [p], True, color, 3)
            x, y = p[:, 0].min(), p[:, 1].min()
            label = f"{res.match.value:%d/%m/%Y}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(view, (x, y - th - 12), (x + tw + 10, y), color, -1)
            cv2.putText(view, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def close(self) -> None:
        if hasattr(self, "scanner"):
            self.scanner.stop()
        for c in getattr(self, "cams", {}).values():
            c.release()
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interfaz gráfica del lector de fechas de tickets.")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.cams)
    root.mainloop()


if __name__ == "__main__":
    main()
