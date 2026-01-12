#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NUT - GUI Tkinter (3 �crans)

1) Accueil : logo + bouton "D�marrer la reconnaissance"
2) Reconnaissance : flux cam�ra + infos (dimensions / salissure / statut)
3) Merci : message + retour auto � l'accueil

D�pendances (Raspberry / Linux):
  sudo apt-get install -y python3-tk
  pip install pillow

S'appuie sur vos modules existants :
  - camera.get_frame(), camera.resize_and_crop_frame(), camera.find_contours()
  - trapdoor.open()/close()/smart_close()/stop()
  - fork.dumpLeft()/center()
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from time import sleep
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
import cv2

import camera
from camera import containerErrorCode, containerInfo_t
import trapdoor
import fork



FILE_OBJECT_ORDER_NAME = "awaiting_object_list.txt"
WAITING_VALIDATION_DELAY_S = 0.3  # validation si d�tect� en continu X secondes


def waiting_list_read_file(file_path: str) -> list[int]:
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f.readlines() if line.strip()]


@dataclass
class UiDetectionState:
    status: str = "En attente�"
    expected_id: Optional[int] = None
    detected_id: Optional[int] = None
    size_label: str = "-"      # si tu as un mapping ID->XS/S/M/L tu le mets ici
    dim_cm: str = "-"
    dim_px: str = "-"
    dirt: str = "-"            # ton coef salissure (d�j� calcul�) => containerInfo.colorDelta
    error: str = "-"


class CameraWorker(threading.Thread):
    """
    Thread cam�ra/d�tection pour �viter de bloquer l'UI.
    Il produit :
      - latest_frame_bgr : frame analys�e (retour find_contours)
      - latest_info : containerInfo_t
      - ui_state : infos d�j� format�es pour l'IHM
    """
    def __init__(self, *, calibration: bool, enable_motor: bool):
        super().__init__(daemon=True)
        self.calibration = calibration
        self.enable_motor = enable_motor

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self.running = False
        self.latest_frame_bgr: Optional["cv2.Mat"] = None
        self.latest_info: Optional[containerInfo_t] = None
        self.ui_state = UiDetectionState()

        self._match_start_ts: float = 0.0
        self._update_disp_once: bool = True

    def stop(self) -> None:
        self._stop.set()


    def set_running(self, value: bool) -> None:
        with self._lock:
            self.running = value
            self._match_start_ts = 0.0
            self._update_disp_once = True

    def get_latest(self) -> Tuple[Optional["cv2.Mat"], Optional[containerInfo_t], UiDetectionState, bool]:
        with self._lock:
            state = UiDetectionState(**self.ui_state.__dict__)
            return self.latest_frame_bgr, self.latest_info, state, self.running

    def _dump_container(self) -> None:
        # Reprend votre s�quence main.py
        if trapdoor.smart_close():
            print("Error when closing the trapdoor")
        if fork.dumpLeft():
            print("Error while going to dump on the left")
        sleep(1)
        if fork.center():
            print("Error while going to the center")
        if trapdoor.open():
            print("Error when opening the trapdoor")

    def _err_to_str(self, err) -> str:
        if err == containerErrorCode.VALID:
            return "OK"
        if err == containerErrorCode.NOT_RECOGNIZED:
            return "Non reconnu"
        if err == containerErrorCode.MISSMATCH_SIZE:
            return "Taille invalide"
        if err == containerErrorCode.MISSMATCH_COLOR:
            return "Couleur / salissure invalide"
        return str(err)

    def _update_ui_state_from_info(self, info: containerInfo_t) -> None:
        exp = camera.s_waitingOrderArr[camera.s_waitingIndex] if camera.s_waitingOrderArr else None

        size_label = f"ID {info.id}" if info.id != -1 else "-"

        dim_cm = "-"
        dim_px = "-"
        if info.dimCm:
            dim_cm = f"{info.dimCm[0]:.1f} x {info.dimCm[1]:.1f} cm"
        if info.dimPx:
            dim_px = f"{info.dimPx[0]:.0f} x {info.dimPx[1]:.0f} px"

        dirt = f"{info.colorDelta:.1f}" if info.id != -1 else "-"

        self.ui_state.expected_id = exp
        self.ui_state.detected_id = None if info.id == -1 else info.id
        self.ui_state.size_label = size_label
        self.ui_state.dim_cm = dim_cm
        self.ui_state.dim_px = dim_px
        self.ui_state.dirt = dirt
        self.ui_state.error = self._err_to_str(info.error)

    def _process_waiting_logic(self, info: containerInfo_t) -> None:
        exp_id = camera.s_waitingOrderArr[camera.s_waitingIndex]

        # pas de contenant d�tect�
        if info.id == -1:
            if self._update_disp_once:
                self.ui_state.status = f"En attente du contenant {exp_id}�"
                self._update_disp_once = False
            self._match_start_ts = 0.0
            return

        # contenant d�tect�
        self._update_disp_once = True

        if (info.id == exp_id) and (info.error == containerErrorCode.VALID):
            if self._match_start_ts == 0.0:
                self._match_start_ts = time.time()
                self.ui_state.status = f"Contenant {info.id} d�tect� (validation�)"
                return

            if (time.time() - self._match_start_ts) >= WAITING_VALIDATION_DELAY_S:
                self.ui_state.status = "Validation OK � �jection�"

                # action m�canique (dans le thread, UI reste fluide)
                if self.enable_motor:
                    self._dump_container()
                else:
                    sleep(1.0)

                # next container
                camera.s_waitingIndex += 1
                if camera.s_waitingIndex >= len(camera.s_waitingOrderArr):
                    camera.s_waitingIndex = 0

                self.ui_state.status = "Termin�"
                self.running = False
                self._match_start_ts = 0.0
                return

        else:
            self.ui_state.status = f"Mauvais contenant: {info.id} (attendu {exp_id})"
            self._match_start_ts = 0.0

    def run(self) -> None:
        # Init s�curit�
        if self.enable_motor:
            try:
                fork.center()
                trapdoor.open()
            except Exception as e:
                print("Motor init error:", e)

        while not self._stop.is_set():
            try:
                bgr = camera.get_frame()
                resized = camera.resize_and_crop_frame(bgr)
                analyzed, info = camera.find_contours(resized, self.calibration)

                with self._lock:
                    self.latest_frame_bgr = analyzed
                    self.latest_info = info
                    self._update_ui_state_from_info(info)
                    is_running = self.running

                if (not self.calibration) and is_running:
                    with self._lock:
                        self._process_waiting_logic(info)

            except Exception as e:
                with self._lock:
                    self.ui_state.status = f"Erreur cam�ra: {e}"
                time.sleep(0.1)

        # Cleanup
        try:
            trapdoor.stop()
        except Exception:
            pass
        try:
            camera.uninit()
        except Exception:
            pass


class NutApp(tk.Tk):
    def __init__(self, *, worker: CameraWorker, return_delay_ms: int = 1800):
        super().__init__()
        self.worker = worker
        self.return_delay_ms = return_delay_ms

        self.title("NUT � Reconnaissance contenants")
        self.geometry("1100x700")
        self.minsize(900, 600)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomeFrame, ScanFrame, ThanksFrame):
            frame = F(parent=self.container, app=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show("HomeFrame")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show(self, name: str) -> None:
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def start_scan(self) -> None:
        if self.worker.enable_motor:
            try:
                trapdoor.open()
            except Exception:
                pass
        self.worker.set_running(True)
        self.show("ScanFrame")

    def stop_scan(self) -> None:
        self.worker.set_running(False)

    def scan_done(self) -> None:
        self.show("ThanksFrame")

    def back_home(self) -> None:
        if self.worker.enable_motor:
            try:
                trapdoor.close()
            except Exception:
                pass
        self.show("HomeFrame")

    def on_close(self) -> None:
        try:
            self.worker.stop()
        except Exception:
            pass
        self.destroy()


class HomeFrame(ttk.Frame):
    def __init__(self, parent, app: NutApp):
        super().__init__(parent)
        self.app = app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        wrap = ttk.Frame(self)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.columnconfigure(0, weight=1)

        ttk.Label(wrap, text="NUT", font=("Helvetica", 56, "bold")).grid(row=0, column=0, pady=(80, 10))
        ttk.Label(wrap, text="Interface de reconnaissance de contenants", font=("Helvetica", 16)).grid(row=1, column=0, pady=(0, 30))

        ttk.Button(wrap, text="D�marrer la reconnaissance", command=self.app.start_scan).grid(row=2, column=0, pady=10, ipadx=15, ipady=10)

        self.status = ttk.Label(wrap, text="", font=("Helvetica", 11))
        self.status.grid(row=3, column=0, pady=(30, 0))

    def on_show(self):
        self.status.configure(text="Pr�t. Placez un contenant puis cliquez sur D�marrer.")


class ScanFrame(ttk.Frame):
    def __init__(self, parent, app: NutApp):
        super().__init__(parent)
        self.app = app

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.video_label = ttk.Label(self)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

        side = ttk.Frame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        side.columnconfigure(0, weight=1)

        self.lbl_status = ttk.Label(side, text="�", font=("Helvetica", 16, "bold"), wraplength=380)
        self.lbl_status.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        self._kv(side, 1, "Contenant attendu", "expected")
        self._kv(side, 2, "D�tect�", "detected")
        self._kv(side, 3, "Taille", "size")
        self._kv(side, 4, "Dimensions (cm)", "dim_cm")
        self._kv(side, 5, "Dimensions (px)", "dim_px")
        self._kv(side, 6, "Coefficient salissure", "dirt")
        self._kv(side, 7, "Erreur", "error")

        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        bar.columnconfigure(0, weight=1)

        self.btn_pause = ttk.Button(bar, text="? Pause", command=self._pause)
        self.btn_pause.grid(row=0, column=0, sticky="w")

        ttk.Button(bar, text="? Retour accueil", command=self._cancel).grid(row=0, column=1, sticky="e")

        self._imgtk = None
        self._update_job = None

    def _kv(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="ew", pady=6)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text=label, font=("Helvetica", 11)).grid(row=0, column=0, sticky="w")
        val = ttk.Label(frm, text="�", font=("Helvetica", 12, "bold"))
        val.grid(row=0, column=1, sticky="e")
        setattr(self, f"val_{key}", val)

    def on_show(self):
        if self._update_job is None:
            self._tick()

    def _pause(self):
        self.app.stop_scan()
        self.lbl_status.configure(text="Pause")
        self.btn_pause.configure(text="? Reprendre", command=self._resume)

    def _resume(self):
        self.app.worker.set_running(True)
        self.btn_pause.configure(text="? Pause", command=self._pause)

    def _cancel(self):
        self.app.stop_scan()
        self._stop_tick()
        self.app.back_home()

    def _stop_tick(self):
        if self._update_job is not None:
            try:
                self.after_cancel(self._update_job)
            except Exception:
                pass
            self._update_job = None

    def _tick(self):
        frame_bgr, info, state, running = self.app.worker.get_latest()

        if frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            w = max(self.video_label.winfo_width(), 640)
            h = max(self.video_label.winfo_height(), 480)
            img.thumbnail((w, h))
            self._imgtk = ImageTk.PhotoImage(img)
            self.video_label.configure(image=self._imgtk)

        self.lbl_status.configure(text=state.status)

        self.val_expected.configure(text="�" if state.expected_id is None else str(state.expected_id))
        self.val_detected.configure(text="�" if state.detected_id is None else str(state.detected_id))
        self.val_size.configure(text=state.size_label)
        self.val_dim_cm.configure(text=state.dim_cm)
        self.val_dim_px.configure(text=state.dim_px)
        self.val_dirt.configure(text=state.dirt)
        self.val_error.configure(text=state.error)

        # transition vers Merci
        if (not running) and state.status == "Termin�":
            self._stop_tick()
            self.app.scan_done()
            return

        self._update_job = self.after(30, self._tick)


class ThanksFrame(ttk.Frame):
    def __init__(self, parent, app: NutApp):
        super().__init__(parent)
        self.app = app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        wrap = ttk.Frame(self)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.columnconfigure(0, weight=1)

        ttk.Label(wrap, text="Merci !", font=("Helvetica", 48, "bold")).grid(row=0, column=0, pady=(140, 10))
        ttk.Label(
            wrap,
            text="Votre contenant a �t� analys�.\nVous pouvez en pr�senter un nouveau.",
            font=("Helvetica", 16),
            justify="center",
        ).grid(row=1, column=0, pady=(0, 30))

        self.hint = ttk.Label(wrap, text="Retour � l'accueil�", font=("Helvetica", 12))
        self.hint.grid(row=2, column=0)

        self._job = None

    def on_show(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
        self._job = self.after(self.app.return_delay_ms, self._back)

    def _back(self):
        self._job = None
        self.app.back_home()


def main():
    ap = argparse.ArgumentParser(description="NUT GUI Tkinter")
    ap.add_argument("-c", "--calibration", action="store_true", help="Mode calibration (pas de dump).")
    ap.add_argument("--no-motor", action="store_true", help="D�sactive les moteurs (simulation).")
    args = ap.parse_args()

    calibration = bool(args.calibration)
    enable_motor = not bool(args.no_motor)

    camera.s_waitingOrderArr = waiting_list_read_file(FILE_OBJECT_ORDER_NAME)
    camera.s_waitingIndex = 0
    print("Containers to detect:", camera.s_waitingOrderArr)

    worker = CameraWorker(calibration=calibration, enable_motor=enable_motor)
    worker.start()

    app = NutApp(worker=worker)
    app.mainloop()


if __name__ == "__main__":
    main()