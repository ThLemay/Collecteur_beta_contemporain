#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0117-NUT — Interface Tkinter (version simple et robuste)
Widgets : uniquement tk.Label / tk.Button / tk.Frame — rien de custom
Thread  : queue.Queue + after() pour la communication thread→UI

Usage :
    python interface_nut_simple.py              # démo sans matériel
    python interface_nut_simple.py --live       # avec caméra + GPIO
    python interface_nut_simple.py --no-motor   # live sans moteurs
    python interface_nut_simple.py -c           # mode calibration
"""

from __future__ import annotations

import argparse
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
import tkinter as tk

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] pip install pillow — flux caméra désactivé")

# ── Modules matériels ─────────────────────────────────────────────────────────
LIVE_MODE = False
camera_mod = trapdoor_mod = fork_mod = None

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
AMBER   = "#f59e0b"
GREEN   = "#10b981"
RED     = "#ef4444"
TEXT    = "#e2e8f0"
TEXT_D  = "#64748b"

SIZES = {0: "XS", 1: "S", 2: "M", 3: "L", 4: "XL"}
RADII = [12.59, 14.46, 16.41, 18.24, 22.4]
DIRS  = {0: "→ Droite", 1: "→ Droite", 2: "← Gauche", 3: "← Gauche", 4: "→ Droite"}

FILE_ORDER         = "awaiting_object_list.txt"
VALIDATION_DELAY_S = 0.3
FONT_MONO          = ("Courier", 11)
FONT_MONO_S        = ("Courier", 9)
FONT_MONO_L        = ("Courier", 14, "bold")
FONT_HUGE          = ("Courier", 52, "bold")


# ══════════════════════════════════════════════════════════════════════════════
#   HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_queue(path: str) -> list[int]:
    try:
        with open(path) as f:
            return [int(l.strip()) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"[WARN] {path} non trouvé — file de démo")
        return [0, 2, 1, 3]


def make_btn(parent, text: str, command, color: str = AMBER,
             width: int = 20, **kw) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        fg=color, bg=CARD,
        activeforeground=BG, activebackground=color,
        font=FONT_MONO, relief="flat",
        bd=0, highlightthickness=1,
        highlightbackground=color,
        width=width, pady=8, cursor="hand2",
        **kw
    )


def make_lbl(parent, text: str = "", color: str = TEXT,
             font=None, **kw) -> tk.Label:
    return tk.Label(
        parent, text=text,
        fg=color, bg=kw.pop("bg", CARD),
        font=font or FONT_MONO,
        **kw
    )


def hsep(parent) -> tk.Frame:
    """Séparateur horizontal."""
    return tk.Frame(parent, bg=TEXT_D, height=1)


# ══════════════════════════════════════════════════════════════════════════════
#   ÉTAT PARTAGÉ  (dataclass copiable)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class State:
    status:      str           = "En attente..."
    expected_id: Optional[int] = None
    detected_id: Optional[int] = None
    dim_cm:      str           = "—"
    dim_px:      str           = "—"
    dirt:        str           = "—"
    error_str:   str           = "—"
    val_pct:     int           = 0
    is_done:     bool          = False
    sorted_n:    int           = 0
    rejected_n:  int           = 0
    error_n:     int           = 0
    queue_idx:   int           = 0


# ══════════════════════════════════════════════════════════════════════════════
#   WORKER  (thread Python standard)
#   Communique avec l'UI via queue.Queue — thread-safe, sans dépendance Qt
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    def __init__(self, *, calibration: bool, enable_motor: bool,
                 queue_list: list[int]):
        super().__init__(daemon=True)
        self.calibration  = calibration
        self.enable_motor = enable_motor
        self._ql          = queue_list
        self._qi          = 0

        self._stop        = threading.Event()
        self._running     = False
        self._lock        = threading.Lock()

        # Files vers l'UI
        self.state_q: "queue.Queue[State]"        = queue.Queue(maxsize=4)
        self.frame_q: "queue.Queue[object]"       = queue.Queue(maxsize=2)

        self._state       = State()
        self._match_ts    = 0.0
        self._disp_once   = True
        self._demo_phase  = 0
        self._demo_ts     = 0.0
        self._demo_ok     = True

    # ── API thread-safe ───────────────────────────────────────────────────────

    def set_running(self, v: bool):
        with self._lock:
            self._running   = v
            self._match_ts  = 0.0
            self._disp_once = True
            if not v:
                self._state.val_pct  = 0
                self._demo_phase     = 0

    def stop(self):
        self._stop.set()

    @property
    def queue_list(self): return self._ql
    @property
    def queue_idx(self):  return self._qi

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self):
        if self.enable_motor and LIVE_MODE:
            try:
                fork_mod.center()
                trapdoor_mod.open()
            except Exception as e:
                print("[MOTOR INIT]", e)

        while not self._stop.is_set():
            try:
                if LIVE_MODE:
                    self._live_step()
                    time.sleep(0.066)   # ~15 FPS
                else:
                    self._demo_step()
                    time.sleep(0.08)
            except Exception as e:
                self._state.status = f"Erreur: {e}"
                self._push_state()
                time.sleep(0.2)

        if LIVE_MODE:
            try: trapdoor_mod.stop()
            except: pass
            try: camera_mod.uninit()
            except: pass

    # ── Live ──────────────────────────────────────────────────────────────────

    def _live_step(self):
        raw     = camera_mod.get_frame()
        resized = camera_mod.resize_and_crop_frame(raw)
        frame, info = camera_mod.find_contours(resized, self.calibration)

        # Frame → PIL Image pour l'UI
        if frame is not None and PIL_OK:
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            if not self.frame_q.full():
                self.frame_q.put_nowait(img)

        self._update_from_info(info)

        with self._lock:
            running = self._running
        if running and not self.calibration:
            self._process_logic(info)

        self._push_state()

    def _update_from_info(self, info):
        s = self._state
        s.expected_id = self._ql[self._qi] if self._ql else None
        s.detected_id = None if info.id == -1 else info.id
        s.dim_cm  = (f"{(info.dimCm[0]+info.dimCm[1])/2:.2f} cm"
                     if info.id != -1 else "—")
        s.dim_px  = (f"{info.dimPx[0]:.0f}×{info.dimPx[1]:.0f} px"
                     if info.id != -1 else "—")
        s.dirt    = f"{info.colorDelta:.1f}" if info.id != -1 else "—"
        if LIVE_MODE:
            from camera import containerErrorCode
            m = {
                containerErrorCode.VALID:          "VALIDE",
                containerErrorCode.NOT_RECOGNIZED: "NON RECONNU",
                containerErrorCode.MISSMATCH_SIZE: "TAILLE INVALIDE",
                containerErrorCode.MISSMATCH_COLOR:"COULEUR INVALIDE",
            }
            s.error_str = m.get(info.error, str(info.error))

    def _process_logic(self, info):
        from camera import containerErrorCode
        exp_id = self._ql[self._qi]
        s = self._state

        if info.id == -1:
            if self._disp_once:
                s.status  = f"En attente de {SIZES.get(exp_id,'?')}..."
                s.val_pct = 0
                self._disp_once = False
            self._match_ts = 0.0
            return

        self._disp_once = True

        if info.id == exp_id and info.error == containerErrorCode.VALID:
            if self._match_ts == 0.0:
                self._match_ts = time.time()
            elapsed   = time.time() - self._match_ts
            s.val_pct = min(100, int(elapsed / VALIDATION_DELAY_S * 100))
            s.status  = f"Validation {s.val_pct}%..."
            if elapsed >= VALIDATION_DELAY_S:
                s.val_pct  = 100
                s.status   = "Ejection en cours..."
                cid = info.id
                self._qi   = (self._qi + 1) % max(len(self._ql), 1)
                s.sorted_n += 1
                s.is_done  = True
                self._running = False
                self._do_dump(cid)
        else:
            s.status  = (f"Mauvais: {SIZES.get(info.id,'?')} "
                         f"(attendu {SIZES.get(exp_id,'?')})")
            s.val_pct = 0
            self._match_ts = 0.0
            s.rejected_n  += 1

    def _do_dump(self, cid: int):
        if not self.enable_motor:
            time.sleep(0.8)
            return
        try:
            trapdoor_mod.smart_close()
            if cid in [0, 1]:
                fork_mod.dumpRightAndReturnCenter()
            else:
                fork_mod.dumpLeftAndReturnCenter()
            trapdoor_mod.open()
        except Exception as e:
            print("[DUMP]", e)
            self._state.error_n += 1

    # ── Démo ─────────────────────────────────────────────────────────────────

    def _demo_step(self):
        with self._lock:
            running = self._running
        if not running:
            return

        s      = self._state
        exp_id = self._ql[self._qi % len(self._ql)] if self._ql else 0

        if self._demo_phase == 0:
            s.status = f"En attente de {SIZES.get(exp_id,'?')}..."
            s.detected_id = None
            s.val_pct = 0
            s.dim_cm = s.dim_px = s.dirt = "—"
            s.error_str = "—"
            if random.random() < 0.04:
                self._demo_ok   = random.random() > 0.25
                self._demo_ts   = time.time()
                self._demo_phase = 1

        elif self._demo_phase == 1:
            det = exp_id if self._demo_ok else (exp_id + 1) % 4
            s.detected_id = det
            s.expected_id = exp_id
            s.dim_cm    = f"{RADII[det]:.2f} cm"
            s.dim_px    = f"{240+random.randint(0,10)}×{240+random.randint(0,10)} px"
            s.dirt      = f"{random.uniform(2,15):.1f}"
            s.error_str = "VALIDE" if self._demo_ok else "TAILLE INVALIDE"

            if self._demo_ok:
                elapsed   = time.time() - self._demo_ts
                s.val_pct = min(100, int(elapsed / VALIDATION_DELAY_S * 100))
                s.status  = f"Validation {s.val_pct}%..."
                if s.val_pct >= 100:
                    s.sorted_n  += 1
                    self._qi     = (self._qi + 1) % max(len(self._ql), 1)
                    s.is_done    = True
                    self._running = False
                    self._demo_phase = 0
            else:
                s.status = (f"Mauvais: {SIZES.get(det,'?')} "
                            f"(attendu {SIZES.get(exp_id,'?')})")
                s.val_pct = 0
                s.rejected_n += 1
                if time.time() - self._demo_ts > 1.0:
                    self._demo_phase = 0

        self._push_state()

    def _push_state(self):
        s = self._state
        s.queue_idx = self._qi
        copy = State(**s.__dict__)
        if not self.state_q.full():
            self.state_q.put_nowait(copy)


# ══════════════════════════════════════════════════════════════════════════════
#   APPLICATION TKINTER
# ══════════════════════════════════════════════════════════════════════════════

class NutApp(tk.Tk):
    def __init__(self, worker: CameraWorker):
        super().__init__()
        self.worker = worker

        self.title("0117-NUT")
        self.configure(bg=BG)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind("<F11>",    lambda e: self.attributes("-fullscreen", True))
        self.bind("<F4>",     lambda e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._screens: dict[str, tk.Frame] = {}
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        for name, builder in [
            ("home",   self._build_home),
            ("scan",   self._build_scan),
            ("thanks", self._build_thanks),
        ]:
            f = tk.Frame(container, bg=BG)
            f.grid(row=0, column=0, sticky="nsew")
            self._screens[name] = f
            builder(f)

        self._show("home")
        self._tick()   # boucle de rafraîchissement UI

    # ── Navigation ───────────────────────────────────────────────────────────

    def _show(self, name: str):
        self._screens[name].tkraise()
        getattr(self, f"_on_show_{name}", lambda: None)()

    def _on_close(self):
        self.worker.stop()
        self.destroy()

    # ══════════════════════════════════════════════════════════════════════════
    #   ÉCRAN 1 — ACCUEIL
    # ══════════════════════════════════════════════════════════════════════════

    def _build_home(self, f: tk.Frame):
        f.configure(bg=BG)

        # Barre amber top
        tk.Frame(f, bg=AMBER, height=3).pack(fill="x")

        center = tk.Frame(f, bg=BG)
        center.pack(expand=True)

        # Logo
        tk.Label(center, text="0117", fg=AMBER, bg=BG,
                 font=("Courier", 72, "bold")).pack()
        tk.Label(center, text="NUT", fg=TEXT_D, bg=BG,
                 font=("Courier", 28)).pack()
        tk.Frame(center, bg=AMBER, height=2, width=260).pack(pady=14)
        tk.Label(center,
                 text="S Y S T E M E   D E   T R I   A U T O M A T I Q U E",
                 fg=TEXT_D, bg=BG,
                 font=("Courier", 10)).pack()

        tk.Frame(center, bg=BG, height=28).pack()

        # Panneau file d'attente
        q_frame = tk.Frame(center, bg=CARD,
                           highlightthickness=1,
                           highlightbackground=TEXT_D)
        q_frame.pack(ipadx=20, ipady=10)
        tk.Label(q_frame, text="FILE D'ATTENTE",
                 fg=TEXT_D, bg=CARD,
                 font=("Courier", 8, "bold")).pack(pady=(8, 4))
        self._home_chips = tk.Frame(q_frame, bg=CARD)
        self._home_chips.pack(padx=16, pady=(0, 8))

        tk.Frame(center, bg=BG, height=20).pack()

        # Bouton démarrer
        make_btn(center, "▶  DEMARRER LA RECONNAISSANCE",
                 self._start_scan, color=GREEN, width=36).pack(pady=4)

        self._home_status = tk.Label(center, text="", fg=AMBER, bg=BG,
                                      font=FONT_MONO_S)
        self._home_status.pack(pady=6)

        # Footer
        tk.Frame(f, bg=BG).pack(fill="both", expand=True)
        foot = tk.Frame(f, bg=CARD)
        foot.pack(fill="x")
        tk.Label(foot,
                 text="0117-NUT  |  F4 = quitter  |  ESC = mode fenetre",
                 fg=TEXT_D, bg=CARD,
                 font=("Courier", 8)).pack(pady=6)

    def _on_show_home(self):
        # Redessiner chips
        for w in self._home_chips.winfo_children():
            w.destroy()
        ql  = self.worker.queue_list
        idx = self.worker.queue_idx
        for i, sid in enumerate(ql):
            fg  = AMBER if i == idx else TEXT_D
            brd = AMBER if i == idx else TEXT_D
            lbl = tk.Label(self._home_chips,
                           text=SIZES.get(sid, "?"),
                           fg=fg, bg=CARD,
                           font=("Courier", 13, "bold"),
                           width=3, height=1,
                           highlightthickness=1,
                           highlightbackground=brd)
            lbl.pack(side="left", padx=4)

        exp = ql[idx % len(ql)] if ql else None
        if exp is not None:
            self._home_status.config(
                text=f"Prochain : {SIZES.get(exp,'?')}  |  "
                     f"~{RADII[exp]:.1f} cm  |  {DIRS.get(exp,'')}")
        else:
            self._home_status.config(text="File vide")

    def _start_scan(self):
        if LIVE_MODE:
            try: trapdoor_mod.open()
            except: pass
        self.worker.set_running(True)
        self._show("scan")

    # ══════════════════════════════════════════════════════════════════════════
    #   ÉCRAN 2 — SCAN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_scan(self, f: tk.Frame):
        f.configure(bg=BG)

        # ── Top bar ──
        topbar = tk.Frame(f, bg=CARD, height=46)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="0117-NUT", fg=AMBER, bg=CARD,
                 font=("Courier", 14, "bold")).pack(side="left", padx=14)
        tk.Label(topbar, text="RECONNAISSANCE", fg=TEXT_D, bg=CARD,
                 font=FONT_MONO_S).pack(side="left")
        self._scan_dot = tk.Label(topbar, text="●", fg=GREEN, bg=CARD,
                                   font=("Courier", 10))
        self._scan_dot.pack(side="left", padx=6)
        self._clock_lbl = tk.Label(topbar, text="--:--:--",
                                    fg=AMBER, bg=CARD,
                                    font=("Courier", 14, "bold"))
        self._clock_lbl.pack(side="right", padx=14)
        self._update_clock()

        # ── Corps ──
        body = tk.Frame(f, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # ─ Caméra ─
        cam_frame = tk.Frame(body, bg=CARD,
                              highlightthickness=1,
                              highlightbackground=TEXT_D)
        cam_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        cam_hdr = tk.Frame(cam_frame, bg=CARD)
        cam_hdr.pack(fill="x")
        tk.Label(cam_hdr, text="FLUX CAMERA", fg=TEXT_D, bg=CARD,
                 font=("Courier", 8, "bold")).pack(side="left", padx=10, pady=5)
        tk.Label(cam_hdr, text="● REC", fg=RED, bg=CARD,
                 font=("Courier", 8, "bold")).pack(side="right", padx=10)

        self._cam_lbl = tk.Label(cam_frame, bg="#000000",
                                  text="FLUX CAMERA NON DISPONIBLE\n\nmode demo actif",
                                  fg=TEXT_D, font=FONT_MONO_S)
        self._cam_lbl.pack(fill="both", expand=True)

        # ─ Panneau infos ─
        info = tk.Frame(body, bg=BG, width=280)
        info.pack(side="right", fill="y")
        info.pack_propagate(False)

        self._build_info_panel(info)

        # ── Barre contrôle ──
        ctrl = tk.Frame(f, bg=CARD)
        ctrl.pack(fill="x")
        self._pause_btn = make_btn(ctrl, "⏸  PAUSE",
                                    self._toggle_pause, color=AMBER, width=14)
        self._pause_btn.pack(side="left", padx=10, pady=6)
        make_btn(ctrl, "✕  RETOUR ACCUEIL",
                 self._cancel_scan, color=RED, width=18).pack(side="left", pady=6)
        self._scan_status = tk.Label(ctrl, text="", fg=TEXT, bg=CARD,
                                      font=("Courier", 10, "bold"))
        self._scan_status.pack(side="left", padx=14)

        self._paused = False

    def _build_info_panel(self, parent: tk.Frame):
        """Tout le panneau de droite en widgets standard."""

        def section(text: str):
            tk.Label(parent, text=text, fg=TEXT_D, bg=CARD,
                     font=("Courier", 8, "bold"),
                     anchor="w").pack(fill="x", padx=10, pady=(8, 2))
            hsep(parent).pack(fill="x", padx=10)

        def row(label: str, key: str, color: str = TEXT):
            f = tk.Frame(parent, bg=CARD)
            f.pack(fill="x", padx=10, pady=2)
            tk.Label(f, text=label, fg=TEXT_D, bg=CARD,
                     font=FONT_MONO_S, anchor="w").pack(side="left")
            v = tk.StringVar(value="—")
            lbl = tk.Label(f, textvariable=v, fg=color, bg=CARD,
                           font=("Courier", 10, "bold"), anchor="e")
            lbl.pack(side="right")
            self._vars[key]  = v
            self._vlbls[key] = lbl

        self._vars:  dict[str, tk.StringVar] = {}
        self._vlbls: dict[str, tk.Label]     = {}

        # Contenant cible
        section("PROCHAIN CONTENANT")
        badge_row = tk.Frame(parent, bg=CARD)
        badge_row.pack(fill="x", padx=10, pady=6)

        self._badge_lbl = tk.Label(badge_row, text="—",
                                    fg=AMBER, bg=CARD,
                                    font=("Courier", 32, "bold"),
                                    width=3)
        self._badge_lbl.pack(side="left")
        self._target_info = tk.Label(badge_row, text="",
                                      fg=TEXT_D, bg=CARD,
                                      font=FONT_MONO_S, justify="left")
        self._target_info.pack(side="left", padx=8)

        # Barre validation (simulée avec tk.Frame — aucun Canvas)
        section("VALIDATION")
        val_row = tk.Frame(parent, bg=CARD)
        val_row.pack(fill="x", padx=10, pady=(4, 2))
        self._val_pct_lbl = tk.Label(val_row, text="0%",
                                      fg=AMBER, bg=CARD, font=FONT_MONO_S)
        self._val_pct_lbl.pack(side="right")

        # Track (fond gris)
        track_wrap = tk.Frame(parent, bg=CARD)
        track_wrap.pack(fill="x", padx=10, pady=(0, 6))
        self._bar_track = tk.Frame(track_wrap, bg=TEXT_D, height=7)
        self._bar_track.pack(fill="x")
        # Fill (amber) par-dessus — on change sa width en %
        self._bar_fill = tk.Frame(self._bar_track, bg=AMBER, height=7)
        self._bar_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)

        # Détails détection
        section("DETAILS DETECTION")
        row("Attendu",      "expected", AMBER)
        row("Detecte",      "detected", TEXT)
        row("Rayon",        "dim_cm",   TEXT)
        row("Dim. pixels",  "dim_px",   TEXT_D)
        row("Delta couleur","dirt",     TEXT)
        row("Statut",       "error",    TEXT_D)

        # KPIs session
        section("SESSION")
        kpi_frame = tk.Frame(parent, bg=CARD)
        kpi_frame.pack(fill="x", padx=10, pady=4)
        self._kpi_vars: dict[str, tk.StringVar] = {}
        for key, label, color in [
            ("sorted",   "Tries",   GREEN),
            ("rejected", "Rejetes", AMBER),
            ("errors",   "Erreurs", RED),
        ]:
            cell = tk.Frame(kpi_frame, bg=CARD,
                            highlightthickness=1, highlightbackground=TEXT_D)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=label, fg=TEXT_D, bg=CARD,
                     font=("Courier", 7)).pack(pady=(4, 0))
            v = tk.StringVar(value="0")
            tk.Label(cell, textvariable=v, fg=color, bg=CARD,
                     font=("Courier", 18, "bold")).pack(pady=(0, 4))
            self._kpi_vars[key] = v

    def _on_show_scan(self):
        self._paused = False
        self._pause_btn.config(text="⏸  PAUSE", fg=AMBER)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.worker.set_running(not self._paused)
        if self._paused:
            self._pause_btn.config(text="▶  REPRENDRE", fg=GREEN)
            self._scan_status.config(text="EN PAUSE", fg=AMBER)
            self._scan_dot.config(fg=AMBER)
        else:
            self._pause_btn.config(text="⏸  PAUSE", fg=AMBER)
            self._scan_status.config(text="")
            self._scan_dot.config(fg=GREEN)

    def _cancel_scan(self):
        self.worker.set_running(False)
        self._cam_lbl.config(image="", text="FLUX CAMERA NON DISPONIBLE\n\nmode demo actif")
        self._cam_lbl._img = None
        self._show("home")

    def _update_clock(self):
        self._clock_lbl.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    # ══════════════════════════════════════════════════════════════════════════
    #   ÉCRAN 3 — MERCI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_thanks(self, f: tk.Frame):
        f.configure(bg=BG)
        center = tk.Frame(f, bg=BG)
        center.pack(expand=True)

        tk.Label(center, text="✓", fg=GREEN, bg=BG,
                 font=("Courier", 96, "bold")).pack(pady=(40, 4))
        tk.Label(center, text="CONTENANT ACCEPTE", fg=GREEN, bg=BG,
                 font=("Courier", 20, "bold")).pack()
        tk.Frame(center, bg=TEXT_D, height=1, width=360).pack(pady=18)
        tk.Label(center, text="Votre contenant a ete analyse et ejecte.",
                 fg=TEXT, bg=BG, font=FONT_MONO).pack()
        tk.Label(center, text="Vous pouvez en presenter un nouveau.",
                 fg=TEXT_D, bg=BG, font=FONT_MONO_S).pack(pady=(4, 32))

        # Barre countdown (tk.Frame simple)
        track = tk.Frame(center, bg=TEXT_D, height=6, width=320)
        track.pack()
        track.pack_propagate(False)
        self._cd_fill = tk.Frame(track, bg=AMBER, height=6)
        self._cd_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)
        self._cd_lbl = tk.Label(center, text="Retour dans 3 s...",
                                 fg=TEXT_D, bg=BG, font=FONT_MONO_S)
        self._cd_lbl.pack(pady=8)

        make_btn(center, "↩  RETOUR ACCUEIL MAINTENANT",
                 self._go_home_now, color=AMBER, width=30).pack(pady=10)

        self._cd_remaining = 3000
        self._cd_job       = None

    def _on_show_thanks(self):
        self._cd_remaining = 3000
        self._cd_fill.place_configure(relwidth=0.0)
        self._cd_lbl.config(text="Retour dans 3 s...")
        if self._cd_job:
            self.after_cancel(self._cd_job)
        self._countdown()

    def _countdown(self):
        self._cd_remaining -= 100
        pct  = max(0.0, (3000 - self._cd_remaining) / 3000)
        secs = max(1, self._cd_remaining // 1000 + 1)
        self._cd_fill.place_configure(relwidth=pct)
        self._cd_lbl.config(text=f"Retour dans {secs} s...")
        if self._cd_remaining <= 0:
            self._go_home_now()
            return
        self._cd_job = self.after(100, self._countdown)

    def _go_home_now(self):
        if self._cd_job:
            self.after_cancel(self._cd_job)
        self.worker.set_running(False)
        if LIVE_MODE:
            try: trapdoor_mod.close()
            except: pass
        self._show("home")

    # ══════════════════════════════════════════════════════════════════════════
    #   BOUCLE DE RAFRAÎCHISSEMENT UI  (after 30 ms)
    # ══════════════════════════════════════════════════════════════════════════

    def _tick(self):
        # ── Frame caméra ──
        try:
            img = self.worker.frame_q.get_nowait()
            if PIL_OK and img is not None:
                w = max(self._cam_lbl.winfo_width(),  320)
                h = max(self._cam_lbl.winfo_height(), 240)
                img.thumbnail((w, h), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                self._cam_lbl.config(image=imgtk, text="")
                self._cam_lbl._img = imgtk   # référence pour éviter le GC
        except queue.Empty:
            pass

        # ── État détection ──
        try:
            state = self.worker.state_q.get_nowait()
            self._apply_state(state)
        except queue.Empty:
            pass

        self.after(30, self._tick)

    def _apply_state(self, s: State):
        # Cible
        ql  = self.worker.queue_list
        idx = s.queue_idx
        if ql:
            tid = ql[idx % len(ql)]
            self._badge_lbl.config(text=SIZES.get(tid, "?"))
            self._target_info.config(
                text=f"{SIZES.get(tid,'?')}\n~{RADII[tid]:.1f} cm\n{DIRS.get(tid,'')}")

        # Validation bar (relwidth entre 0.0 et 1.0)
        pct = s.val_pct
        self._bar_fill.place_configure(relwidth=pct / 100)
        self._val_pct_lbl.config(
            text=f"{pct}%",
            fg=GREEN if pct == 100 else AMBER)

        # Lignes détail
        def set_row(key: str, val: str, color: str = TEXT):
            self._vars[key].set(val)
            self._vlbls[key].config(fg=color)

        exp_s = SIZES.get(s.expected_id, "—") if s.expected_id is not None else "—"
        det_s = SIZES.get(s.detected_id, "—") if s.detected_id is not None else "—"
        set_row("expected", exp_s, AMBER)
        match = (s.detected_id is not None and s.detected_id == s.expected_id)
        set_row("detected", det_s, GREEN if match else RED)
        set_row("dim_cm", s.dim_cm)
        set_row("dim_px", s.dim_px, TEXT_D)
        try:
            d = float(s.dirt)
            set_row("dirt", s.dirt, GREEN if d < 8 else (AMBER if d < 16 else RED))
        except ValueError:
            set_row("dirt", s.dirt)
        set_row("error", s.error_str,
                GREEN if s.error_str == "VALIDE" else
                (RED   if s.error_str not in ("—", "") else TEXT_D))

        # Status bar
        if not self._paused:
            self._scan_status.config(text=s.status, fg=TEXT)

        # KPIs
        self._kpi_vars["sorted"].set(str(s.sorted_n))
        self._kpi_vars["rejected"].set(str(s.rejected_n))
        self._kpi_vars["errors"].set(str(s.error_n))

        # Transition → Merci
        if s.is_done:
            self._show("thanks")


# ══════════════════════════════════════════════════════════════════════════════
#   POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global LIVE_MODE, camera_mod, trapdoor_mod, fork_mod

    ap = argparse.ArgumentParser(description="0117-NUT Interface")
    ap.add_argument("-c", "--calibration", action="store_true")
    ap.add_argument("--live",     action="store_true")
    ap.add_argument("--no-motor", action="store_true")
    args = ap.parse_args()

    if args.live:
        try:
            import camera as _cam
            import trapdoor as _trap
            import fork as _fork
            camera_mod, trapdoor_mod, fork_mod = _cam, _trap, _fork
            LIVE_MODE = True
            print("[INFO] Modules materiels charges.")
        except Exception as e:
            print(f"[WARN] Modules indisponibles ({e}) — mode demo.")

    queue_list = read_queue(FILE_ORDER)
    print("File d'attente :", [SIZES.get(i, i) for i in queue_list])

    worker = CameraWorker(
        calibration=args.calibration,
        enable_motor=not args.no_motor,
        queue_list=queue_list,
    )
    worker.start()

    app = NutApp(worker)
    app.mainloop()
    worker.stop()


if __name__ == "__main__":
    main()
