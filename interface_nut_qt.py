#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ── Fix critique : OpenCV embarque sa propre Qt qui entre en conflit avec PyQt5.
# ── On vide son chemin de plugins AVANT tout import pour forcer PyQt5 à utiliser
# ── les siens. Doit être la toute première chose exécutée.
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# Pour VNC / SSH : forcer le display si non défini
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
"""
0117-NUT — Interface PyQt5
Architecture : 3 écrans (Accueil → Scan → Merci)
Thème        : dark industriel / HUD amber

Installation :
    pip install PyQt5
    # Sur Raspberry Pi :
    sudo apt-get install -y python3-pyqt5

Usage :
    python interface_nut_qt.py              # démo sans matériel
    python interface_nut_qt.py --live       # avec caméra + GPIO
    python interface_nut_qt.py -c           # mode calibration
    python interface_nut_qt.py --no-motor   # moteurs désactivés
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import (Qt, QThread, QTimer, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QColor, QFont, QImage, QPainter, QPen, QPixmap)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

# ── Modules matériels (optionnels) ───────────────────────────────────────────
LIVE_MODE = False
camera_mod = trapdoor_mod = fork_mod = None

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#   PALETTE & STYLE
# ══════════════════════════════════════════════════════════════════════════════

BG      = "#080b0f"
SURFACE = "#0d1117"
CARD    = "#111820"
BORDER  = "#1c2535"
BORDER2 = "#253040"
AMBER   = "#f59e0b"
AMBER_D = "#b45309"
GREEN   = "#10b981"
RED     = "#ef4444"
TEXT    = "#cbd5e1"
TEXT_D  = "#64748b"
TEXT_B  = "#f1f5f9"

QSS = f"""
/* ── Base ── */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Courier New", Courier, monospace;
    font-size: 13px;
}}

/* ── Labels ── */
QLabel {{
    background: transparent;
    color: {TEXT};
}}

/* ── Boutons ── */
QPushButton {{
    background-color: {BORDER};
    color: {TEXT};
    border: 1px solid {BORDER2};
    border-top: 2px solid {AMBER};
    padding: 10px 20px;
    font-family: "Courier New", Courier, monospace;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background-color: {CARD};
    color: {AMBER};
    border-color: {AMBER};
}}
QPushButton:pressed {{
    background-color: {AMBER_D};
    color: {BG};
}}

QPushButton[accent="green"] {{
    border-top: 2px solid {GREEN};
    color: {TEXT};
}}
QPushButton[accent="green"]:hover {{
    color: {GREEN};
    border-color: {GREEN};
}}

QPushButton[accent="red"] {{
    border-top: 2px solid {RED};
    color: {TEXT};
}}
QPushButton[accent="red"]:hover {{
    color: {RED};
    border-color: {RED};
}}

/* ── Frames / Cards ── */
QFrame[role="card"] {{
    background-color: {CARD};
    border: 1px solid {BORDER};
}}
QFrame[role="surface"] {{
    background-color: {SURFACE};
    border: none;
}}
QFrame[role="separator"] {{
    background-color: {BORDER};
    max-height: 1px;
    min-height: 1px;
}}

/* ── Progress bar maison ── */
QFrame[role="bar-track"] {{
    background-color: {BORDER};
    border: none;
    max-height: 8px;
    min-height: 8px;
    border-radius: 0px;
}}
QFrame[role="bar-fill"] {{
    background-color: {AMBER};
    border: none;
    max-height: 8px;
    min-height: 8px;
}}
"""

SIZES = {0: "XS", 1: "S", 2: "M", 3: "L", 4: "XL"}
RADII = [12.59, 14.46, 16.41, 18.24, 22.4]
DIRS  = {0: "→ Droite", 1: "→ Droite", 2: "← Gauche", 3: "← Gauche", 4: "→ Droite"}

FILE_ORDER         = "awaiting_object_list.txt"
VALIDATION_DELAY_S = 0.3


# ══════════════════════════════════════════════════════════════════════════════
#   HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_queue(path: str) -> list[int]:
    try:
        with open(path) as f:
            return [int(l.strip()) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"[WARN] {path} non trouvé — file de démo utilisée")
        return [0, 2, 1, 3]


def lbl(text: str, color: str = TEXT, size: int = 13,
        bold: bool = False) -> QLabel:
    """Raccourci création QLabel stylé."""
    w = QLabel(text)
    weight = "bold" if bold else "normal"
    w.setStyleSheet(
        f"color: {color}; font-size: {size}px; font-weight: {weight};"
        f" background: transparent;")
    return w


def sep() -> QFrame:
    f = QFrame()
    f.setProperty("role", "separator")
    return f


def card_frame() -> QFrame:
    f = QFrame()
    f.setProperty("role", "card")
    return f


# ══════════════════════════════════════════════════════════════════════════════
#   ÉTAT PARTAGÉ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionState:
    status:      str           = "En attente..."
    expected_id: Optional[int] = None
    detected_id: Optional[int] = None
    dim_cm:      str           = "—"
    dim_px:      str           = "—"
    dirt:        str           = "—"
    error_str:   str           = "—"
    val_pct:     int           = 0
    is_done:     bool          = False


# ══════════════════════════════════════════════════════════════════════════════
#   WORKER  (QThread — signaux/slots pour thread-safety garantie)
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(QThread):
    # Signaux émis vers l'UI (thread-safe par définition Qt)
    state_updated  = pyqtSignal(object)          # DetectionState
    frame_ready    = pyqtSignal(object)          # QImage ou None
    stats_updated  = pyqtSignal(int, int, int)   # sorted, rejected, errors

    def __init__(self, *, calibration: bool, enable_motor: bool,
                 queue_list: list[int]):
        super().__init__()
        self.calibration   = calibration
        self.enable_motor  = enable_motor
        self._queue_list   = queue_list
        self._queue_idx    = 0
        self._running      = False
        self._active       = True          # False = arrêter le thread

        self.sorted_count   = 0
        self.rejected_count = 0
        self.error_count    = 0

        # Démo
        self._demo_phase   = 0
        self._demo_ts      = 0.0
        self._demo_correct = True
        self._match_ts     = 0.0
        self._disp_once    = True

        self._state = DetectionState()

    # ── API publique (appelée depuis le thread UI) ────────────────────────────

    def set_running(self, v: bool):
        self._running  = v
        self._match_ts = 0.0
        self._disp_once = True
        if not v:
            self._state.val_pct = 0
            self._demo_phase    = 0
            self.state_updated.emit(DetectionState(**self._state.__dict__))

    def stop_worker(self):
        self._active  = False
        self._running = False
        self.quit()
        self.wait(2000)

    @property
    def queue_list(self): return self._queue_list
    @property
    def queue_idx(self):  return self._queue_idx

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self):
        if self.enable_motor and LIVE_MODE:
            try:
                fork_mod.center()
                trapdoor_mod.open()
            except Exception as e:
                print("[MOTOR INIT]", e)

        while self._active:
            try:
                if LIVE_MODE:
                    self._live_step()
                    # Petit sleep pour ne pas saturer le CPU (~15 FPS max)
                    self.msleep(66)
                else:
                    self._demo_step()
                    self.msleep(80)
            except Exception as e:
                self._state.status = f"Erreur : {e}"
                self.state_updated.emit(DetectionState(**self._state.__dict__))
                self.msleep(150)

        # Cleanup
        if LIVE_MODE:
            try: trapdoor_mod.stop()
            except: pass
            try: camera_mod.uninit()
            except: pass

    # ── Live ──────────────────────────────────────────────────────────────────

    def _live_step(self):
        raw     = camera_mod.get_frame()
        resized = camera_mod.resize_and_crop_frame(raw)
        # find_contours tourne toujours pour afficher le flux
        frame, info = camera_mod.find_contours(resized, self.calibration)

        # Envoyer la frame à l'UI dans tous les cas (preview caméra)
        if frame is not None:
            try:
                import numpy as np
                rgb = frame[:, :, ::-1].copy()
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy())
            except Exception as e:
                print("[FRAME CONVERT]", e)

        # Mettre à jour les infos de détection (même si pas running)
        self._update_state_from_info(info)

        # La logique validation/éjection uniquement si "Démarrer" a été cliqué
        if self._running and not self.calibration:
            self._process_logic(info)

        self.state_updated.emit(DetectionState(**self._state.__dict__))
        self.stats_updated.emit(self.sorted_count,
                                self.rejected_count, self.error_count)

    def _update_state_from_info(self, info):
        s = self._state
        exp = self._queue_list[self._queue_idx] if self._queue_list else None
        s.expected_id = exp
        s.detected_id = None if info.id == -1 else info.id
        s.dim_cm  = (f"{(info.dimCm[0]+info.dimCm[1])/2:.2f} cm"
                     if info.id != -1 else "—")
        s.dim_px  = (f"{info.dimPx[0]:.0f} × {info.dimPx[1]:.0f} px"
                     if info.id != -1 else "—")
        s.dirt    = f"{info.colorDelta:.1f}" if info.id != -1 else "—"
        if LIVE_MODE:
            from camera import containerErrorCode
            m = {containerErrorCode.VALID:          "VALIDE",
                 containerErrorCode.NOT_RECOGNIZED: "NON RECONNU",
                 containerErrorCode.MISSMATCH_SIZE: "TAILLE INVALIDE",
                 containerErrorCode.MISSMATCH_COLOR:"COULEUR INVALIDE"}
            s.error_str = m.get(info.error, str(info.error))

    def _process_logic(self, info):
        from camera import containerErrorCode
        exp_id = self._queue_list[self._queue_idx]
        s = self._state

        if info.id == -1:
            if self._disp_once:
                s.status = f"En attente du contenant {SIZES.get(exp_id,'?')}…"
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
            s.status  = f"Validation {s.val_pct}%…"
            if elapsed >= VALIDATION_DELAY_S:
                s.val_pct = 100
                s.status  = "Éjection en cours…"
                cid = info.id
                self._advance_queue()
                self.sorted_count += 1
                s.is_done = True
                self._running = False
                self._do_dump(cid)
        else:
            s.status = (f"Mauvais contenant : {SIZES.get(info.id,'?')} "
                        f"(attendu {SIZES.get(exp_id,'?')})")
            s.val_pct = 0
            self._match_ts = 0.0
            self.rejected_count += 1

    def _do_dump(self, cid: int):
        if not self.enable_motor:
            self.msleep(800)
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
            self.error_count += 1

    def _advance_queue(self):
        self._queue_idx = (self._queue_idx + 1) % max(len(self._queue_list), 1)

    # ── Simulation démo ───────────────────────────────────────────────────────

    def _demo_step(self):
        if not self._running:
            self.msleep(50)
            return

        s  = self._state
        ql = self._queue_list
        exp_id = ql[self._queue_idx % len(ql)] if ql else 0

        if self._demo_phase == 0:
            s.status = f"En attente du contenant {SIZES.get(exp_id,'?')}…"
            s.detected_id = None
            s.val_pct = 0
            s.dim_cm = s.dim_px = s.dirt = "—"
            s.error_str = "—"
            if random.random() < 0.04:
                self._demo_correct = random.random() > 0.25
                self._demo_ts   = time.time()
                self._demo_phase = 1

        elif self._demo_phase == 1:
            det_id = exp_id if self._demo_correct else (exp_id + 1) % 4
            s.detected_id = det_id
            s.expected_id = exp_id
            s.dim_cm    = f"{RADII[det_id]:.2f} cm"
            s.dim_px    = f"{240 + random.randint(0,10)} × {240 + random.randint(0,10)} px"
            s.dirt      = f"{random.uniform(2, 15):.1f}"
            s.error_str = "VALIDE" if self._demo_correct else "TAILLE INVALIDE"

            if self._demo_correct:
                elapsed   = time.time() - self._demo_ts
                s.val_pct = min(100, int(elapsed / VALIDATION_DELAY_S * 100))
                s.status  = f"Validation {s.val_pct}%…"
                if s.val_pct >= 100:
                    self.sorted_count += 1
                    self._advance_queue()
                    s.is_done    = True
                    self._running = False
                    self._demo_phase = 0
                    self.msleep(400)
            else:
                s.status = (f"Mauvais contenant : {SIZES.get(det_id,'?')} "
                            f"(attendu {SIZES.get(exp_id,'?')})")
                s.val_pct = 0
                self.rejected_count += 1
                if time.time() - self._demo_ts > 1.0:
                    self._demo_phase = 0

        self.state_updated.emit(DetectionState(**self._state.__dict__))
        self.stats_updated.emit(self.sorted_count,
                                self.rejected_count, self.error_count)


# ══════════════════════════════════════════════════════════════════════════════
#   WIDGETS RÉUTILISABLES
# ══════════════════════════════════════════════════════════════════════════════

class ProgressBar(QWidget):
    """Barre de progression amber custom."""
    def __init__(self, parent=None, height: int = 7):
        super().__init__(parent)
        self._pct = 0
        self.setFixedHeight(height)
        self.setStyleSheet(f"background: {BORDER};")

    def set_pct(self, pct: int):
        self._pct = max(0, min(100, pct))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w = int(self.width() * self._pct / 100)
        if w > 0:
            # dégradé amber_d → amber
            p.fillRect(0, 0, w // 2, self.height(), QColor(AMBER_D))
            p.fillRect(w // 2, 0, w - w // 2, self.height(), QColor(AMBER))


class CamView(QWidget):
    """Canvas caméra avec overlay HUD (coins + crosshair)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #000000;")

    def set_frame(self, qimg: QImage):
        self._pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._pixmap:
            # Centrer le pixmap
            x = (w - self._pixmap.width())  // 2
            y = (h - self._pixmap.height()) // 2
            p.drawPixmap(x, y, self._pixmap)
        else:
            # Grille placeholder
            p.setPen(QPen(QColor("#111820"), 1))
            for x in range(0, w, 40):
                p.drawLine(x, 0, x, h)
            for y in range(0, h, 40):
                p.drawLine(0, y, w, y)
            p.setPen(QPen(QColor(TEXT_D), 1))
            p.drawText(
                self.rect(), Qt.AlignCenter,
                "FLUX CAMÉRA NON DISPONIBLE\n\npicamera2 requis  ·  mode démo actif"
            )

        # ── Overlay HUD ──
        amber = QColor(AMBER)
        amber_d = QColor(AMBER_D)
        sz = 22

        # Coins
        p.setPen(QPen(amber, 2))
        for cx, cy, dx, dy in [(2,2,1,1),(w-2,2,-1,1),(2,h-2,1,-1),(w-2,h-2,-1,-1)]:
            p.drawLine(cx, cy, cx + sz * dx, cy)
            p.drawLine(cx, cy, cx, cy + sz * dy)

        # Crosshair central
        mx, my, r = w // 2, h // 2, 18
        p.setPen(QPen(amber_d, 1))
        p.drawEllipse(mx - r, my - r, r * 2, r * 2)
        p.drawLine(mx - 30, my, mx - r - 1, my)
        p.drawLine(mx + r + 1, my, mx + 30, my)
        p.drawLine(mx, my - 30, mx, my - r - 1)
        p.drawLine(mx, my + r + 1, mx, my + 30)


class InfoRow(QWidget):
    """Ligne label gauche + valeur droite."""
    def __init__(self, label_text: str, val_color: str = TEXT_B):
        super().__init__()
        self._val_color = val_color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self._label = lbl(label_text, TEXT_D, 11)
        self._value = lbl("—", val_color, 11, bold=True)
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str = None):
        self._value.setText(text)
        c = color or self._val_color
        self._value.setStyleSheet(
            f"color: {c}; font-size: 11px; font-weight: bold; background: transparent;")


class BadgeLabel(QWidget):
    """Badge carré avec bordure amber pour la taille du contenant."""
    def __init__(self, size: int = 72):
        super().__init__()
        self.setFixedSize(size, size)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self._letter = QLabel("—")
        self._letter.setAlignment(Qt.AlignCenter)
        self._letter.setStyleSheet(
            f"color: {AMBER}; font-size: 26px; font-weight: bold;"
            f" background: transparent;")
        self._sub = QLabel("TAILLE")
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setStyleSheet(
            f"color: {AMBER_D}; font-size: 8px; background: transparent;")
        layout.addStretch()
        layout.addWidget(self._letter)
        layout.addWidget(self._sub)
        layout.addStretch()
        self.setStyleSheet(
            f"background: {CARD}; border: 2px solid {AMBER};")

    def set_letter(self, text: str):
        self._letter.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
#   ÉCRAN 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════

class HomeScreen(QWidget):
    def __init__(self, app: "NutApp"):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Barre accent top
        top_line = QFrame()
        top_line.setFixedHeight(3)
        top_line.setStyleSheet(f"background: {AMBER};")
        root.addWidget(top_line)

        # Zone centrale
        center = QWidget()
        center.setStyleSheet(f"background: {BG};")
        cv = QVBoxLayout(center)
        cv.setAlignment(Qt.AlignCenter)
        cv.setSpacing(4)

        # Logo
        cv.addSpacing(40)
        logo = QLabel("0117")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color: {AMBER}; font-size: 80px; font-weight: bold;")
        cv.addWidget(logo)

        sub = QLabel("NUT")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {TEXT_D}; font-size: 32px;")
        cv.addWidget(sub)

        cv.addSpacing(8)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {AMBER_D};")
        line.setFixedWidth(300)
        cv.addWidget(line, alignment=Qt.AlignCenter)
        cv.addSpacing(8)

        tagline = QLabel("S Y S T È M E   D E   T R I   A U T O M A T I Q U E")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(f"color: {TEXT_D}; font-size: 11px;")
        cv.addWidget(tagline)

        cv.addSpacing(32)

        # Panneau file d'attente
        queue_card = QFrame()
        queue_card.setProperty("role", "card")
        queue_card.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER};")
        qcl = QVBoxLayout(queue_card)
        qcl.setContentsMargins(24, 12, 24, 12)

        ql_title = lbl("FILE D'ATTENTE", TEXT_D, 9, bold=True)
        ql_title.setAlignment(Qt.AlignCenter)
        qcl.addWidget(ql_title)

        self._chips_row = QHBoxLayout()
        self._chips_row.setAlignment(Qt.AlignCenter)
        self._chips_row.setSpacing(8)
        qcl.addLayout(self._chips_row)

        cv.addWidget(queue_card, alignment=Qt.AlignCenter)

        cv.addSpacing(28)

        # Bouton démarrer
        self._start_btn = QPushButton("▶   DÉMARRER LA RECONNAISSANCE")
        self._start_btn.setProperty("accent", "green")
        self._start_btn.setFixedHeight(52)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet(
            self._start_btn.styleSheet())   # force refresh property
        self._start_btn.clicked.connect(self.app.start_scan)
        cv.addWidget(self._start_btn, alignment=Qt.AlignCenter)

        cv.addSpacing(12)
        self._status_lbl = lbl("", AMBER, 11)
        self._status_lbl.setAlignment(Qt.AlignCenter)
        cv.addWidget(self._status_lbl)
        cv.addStretch()

        root.addWidget(center, stretch=1)

        # Footer
        footer = QFrame()
        footer.setStyleSheet(f"background: {SURFACE};")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(16, 6, 16, 6)
        fl.addWidget(lbl("0117-NUT  ·  Firmware v2.4  ·  ESC = quitter plein écran",
                         TEXT_D, 9))
        root.addWidget(footer)

    def on_show(self):
        # Vider et reconstruire les chips
        while self._chips_row.count():
            item = self._chips_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ql  = self.app.worker.queue_list
        idx = self.app.worker.queue_idx
        for i, sid in enumerate(ql):
            is_cur = (i == idx)
            chip = QLabel(SIZES.get(sid, "?"))
            chip.setFixedSize(44, 44)
            chip.setAlignment(Qt.AlignCenter)
            fg  = AMBER  if is_cur else TEXT_D
            brd = AMBER  if is_cur else BORDER2
            chip.setStyleSheet(
                f"color: {fg}; background: {CARD};"
                f" border: 1px solid {brd}; font-size: 14px; font-weight: bold;")
            self._chips_row.addWidget(chip)

        exp = ql[idx % len(ql)] if ql else None
        if exp is not None:
            self._status_lbl.setText(
                f"Prochain : {SIZES.get(exp,'?')}  ·  "
                f"Rayon ~{RADII[exp]:.1f} cm  ·  {DIRS.get(exp,'')}")
        else:
            self._status_lbl.setText("File vide")

        # Forcer le style du bouton (propriété Qt)
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)


# ══════════════════════════════════════════════════════════════════════════════
#   ÉCRAN 2 — SCAN
# ══════════════════════════════════════════════════════════════════════════════

class ScanScreen(QWidget):
    def __init__(self, app: "NutApp"):
        super().__init__()
        self.app     = app
        self._paused = False
        self._build()

        # Timer horloge
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background: {SURFACE};")
        tbl = QHBoxLayout(topbar)
        tbl.setContentsMargins(16, 0, 16, 0)

        tbl.addWidget(lbl("0117-NUT", AMBER, 15, bold=True))
        tbl.addSpacing(16)
        tbl.addWidget(lbl("RECONNAISSANCE ACTIVE", TEXT_D, 10))

        # Dot clignotant simulé via QLabel (on change sa couleur avec QTimer)
        self._dot_lbl = lbl("●", GREEN, 14)
        tbl.addWidget(self._dot_lbl)
        self._dot_state = True
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._blink_dot)
        self._dot_timer.start(900)

        tbl.addStretch()
        self._clock_lbl = lbl("--:--:--", AMBER, 15, bold=True)
        tbl.addWidget(self._clock_lbl)
        root.addWidget(topbar)

        # ── Corps ─────────────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background: {BG};")
        bl = QHBoxLayout(body)
        bl.setContentsMargins(12, 12, 12, 12)
        bl.setSpacing(12)

        # ─ Caméra ─
        cam_card = QFrame()
        cam_card.setStyleSheet(f"background: {CARD}; border: 1px solid {BORDER};")
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(0)

        cam_hdr = QFrame()
        cam_hdr.setFixedHeight(34)
        cam_hdr.setStyleSheet(f"background: {SURFACE}; border: none;")
        chl = QHBoxLayout(cam_hdr)
        chl.setContentsMargins(12, 0, 12, 0)
        chl.addWidget(lbl("FLUX CAMÉRA", TEXT_D, 9, bold=True))
        chl.addStretch()
        chl.addWidget(lbl("● REC", RED, 9, bold=True))
        chl.addSpacing(8)
        chl.addWidget(lbl("10 FPS", GREEN, 9))
        cam_layout.addWidget(cam_hdr)

        self._cam_view = CamView()
        cam_layout.addWidget(self._cam_view, stretch=1)
        bl.addWidget(cam_card, stretch=5)

        # ─ Panneau infos ─
        info_card = QFrame()
        info_card.setStyleSheet(f"background: {CARD}; border: 1px solid {BORDER};")
        info_card.setFixedWidth(300)
        il = QVBoxLayout(info_card)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(0)
        self._build_info_panel(il)
        bl.addWidget(info_card)

        root.addWidget(body, stretch=1)

        # ── Barre de contrôle ─────────────────────────────────────────────────
        ctrl = QFrame()
        ctrl.setStyleSheet(f"background: {SURFACE};")
        ctrl.setFixedHeight(56)
        ctl = QHBoxLayout(ctrl)
        ctl.setContentsMargins(12, 8, 12, 8)

        self._pause_btn = QPushButton("⏸   PAUSE")
        self._pause_btn.setFixedHeight(40)
        self._pause_btn.setCursor(Qt.PointingHandCursor)
        self._pause_btn.clicked.connect(self._toggle_pause)
        ctl.addWidget(self._pause_btn)
        ctl.addSpacing(8)

        back_btn = QPushButton("✕   RETOUR ACCUEIL")
        back_btn.setProperty("accent", "red")
        back_btn.setFixedHeight(40)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._cancel)
        ctl.addWidget(back_btn)
        ctl.addSpacing(16)

        self._status_bar = lbl("", TEXT, 11, bold=True)
        ctl.addWidget(self._status_bar)
        ctl.addStretch()

        root.addWidget(ctrl)

    def _build_info_panel(self, layout: QVBoxLayout):
        def section(title: str):
            w = QWidget()
            w.setStyleSheet(f"background: {SURFACE};")
            hl = QHBoxLayout(w)
            hl.setContentsMargins(12, 6, 12, 6)
            hl.addWidget(lbl(title, TEXT_D, 9, bold=True))
            layout.addWidget(w)

        def padded(widget):
            wrap = QWidget()
            wrap.setStyleSheet(f"background: {CARD};")
            wl = QVBoxLayout(wrap)
            wl.setContentsMargins(14, 8, 14, 8)
            wl.addWidget(widget)
            layout.addWidget(wrap)
            return wrap

        # Prochain contenant
        section("PROCHAIN CONTENANT")
        target_w = QWidget()
        target_w.setStyleSheet(f"background: {CARD};")
        tw = QHBoxLayout(target_w)
        tw.setContentsMargins(14, 10, 14, 10)
        self._badge = BadgeLabel()
        tw.addWidget(self._badge)
        tw.addSpacing(12)
        info_sub = QVBoxLayout()
        self._target_name = lbl("—", TEXT_B, 12, bold=True)
        self._target_meta = lbl("", TEXT_D, 10)
        self._target_meta.setWordWrap(True)
        info_sub.addWidget(self._target_name)
        info_sub.addWidget(self._target_meta)
        tw.addLayout(info_sub)
        layout.addWidget(target_w)

        layout.addWidget(sep())

        # Validation
        val_hdr = QWidget()
        val_hdr.setStyleSheet(f"background: {CARD};")
        vhl = QHBoxLayout(val_hdr)
        vhl.setContentsMargins(14, 8, 14, 4)
        vhl.addWidget(lbl("VALIDATION", TEXT_D, 9, bold=True))
        vhl.addStretch()
        self._val_pct_lbl = lbl("0%", AMBER, 9)
        vhl.addWidget(self._val_pct_lbl)
        layout.addWidget(val_hdr)

        self._val_bar = ProgressBar(height=7)
        bar_wrap = QWidget()
        bar_wrap.setStyleSheet(f"background: {CARD};")
        bwl = QVBoxLayout(bar_wrap)
        bwl.setContentsMargins(14, 0, 14, 10)
        bwl.addWidget(self._val_bar)
        layout.addWidget(bar_wrap)

        layout.addWidget(sep())

        # Détails
        det_w = QWidget()
        det_w.setStyleSheet(f"background: {CARD};")
        dwl = QVBoxLayout(det_w)
        dwl.setContentsMargins(14, 10, 14, 10)
        dwl.addWidget(lbl("DÉTAILS DÉTECTION", TEXT_D, 9, bold=True))
        dwl.addSpacing(6)

        self._rows: dict[str, InfoRow] = {}
        fields = [
            ("Contenant attendu", "expected", AMBER),
            ("Détecté",           "detected", TEXT_B),
            ("Rayon",             "dim_cm",   TEXT_B),
            ("Dimensions px",     "dim_px",   TEXT_D),
            ("Δ Couleur",         "dirt",     TEXT_B),
            ("Statut",            "error",    TEXT_D),
        ]
        for flabel, key, col in fields:
            row = InfoRow(flabel, col)
            dwl.addWidget(row)
            self._rows[key] = row
        layout.addWidget(det_w)

        layout.addWidget(sep())

        # KPIs session
        kpi_w = QWidget()
        kpi_w.setStyleSheet(f"background: {CARD};")
        kwl = QVBoxLayout(kpi_w)
        kwl.setContentsMargins(14, 8, 14, 8)
        kwl.addWidget(lbl("SESSION", TEXT_D, 9, bold=True))
        kwl.addSpacing(4)

        kpi_row = QHBoxLayout()
        self._kpi_sorted   = self._mini_kpi(kpi_row, "Triés",   GREEN)
        self._kpi_rejected = self._mini_kpi(kpi_row, "Rejetés", AMBER)
        self._kpi_errors   = self._mini_kpi(kpi_row, "Erreurs", RED)
        kwl.addLayout(kpi_row)
        layout.addWidget(kpi_w)

        layout.addStretch()

    def _mini_kpi(self, parent_layout: QHBoxLayout, label: str, color: str):
        frame = QFrame()
        frame.setStyleSheet(f"background: {BORDER};")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(0)
        fl.addWidget(lbl(label, TEXT_D, 8))
        v = lbl("0", color, 20, bold=True)
        v.setAlignment(Qt.AlignCenter)
        fl.addWidget(v)
        parent_layout.addWidget(frame)
        return v

    # ── Slots UI ─────────────────────────────────────────────────────────────

    def on_show(self):
        self._paused = False
        self._pause_btn.setText("⏸   PAUSE")
        self._dot_timer.start(900)

    @pyqtSlot(object)
    def on_state(self, state: DetectionState):
        # Cible
        ql  = self.app.worker.queue_list
        idx = self.app.worker.queue_idx
        if ql:
            tid = ql[idx % len(ql)]
            self._badge.set_letter(SIZES.get(tid, "?"))
            self._target_name.setText(f"Contenant {SIZES.get(tid,'?')}")
            self._target_meta.setText(
                f"Rayon ~{RADII[tid]:.1f} cm\n{DIRS.get(tid,'')}")

        # Validation
        pct = state.val_pct
        self._val_bar.set_pct(pct)
        self._val_pct_lbl.setText(f"{pct}%")
        self._val_pct_lbl.setStyleSheet(
            f"color: {GREEN if pct==100 else AMBER}; font-size: 9px;"
            f" background: transparent;")

        # Lignes de détail
        r = self._rows
        exp_s = SIZES.get(state.expected_id, "—") if state.expected_id is not None else "—"
        det_s = SIZES.get(state.detected_id, "—") if state.detected_id is not None else "—"
        r["expected"].set_value(exp_s, AMBER)
        r["detected"].set_value(
            det_s,
            GREEN if (state.detected_id is not None and
                      state.detected_id == state.expected_id) else RED)
        r["dim_cm"].set_value(state.dim_cm)
        r["dim_px"].set_value(state.dim_px)
        try:
            d = float(state.dirt)
            r["dirt"].set_value(state.dirt,
                                GREEN if d < 8 else (AMBER if d < 16 else RED))
        except ValueError:
            r["dirt"].set_value(state.dirt)
        r["error"].set_value(
            state.error_str,
            GREEN if state.error_str == "VALIDE" else RED)

        if not self._paused:
            self._status_bar.setText(state.status)

        if state.is_done:
            self.app.scan_done()

    @pyqtSlot(object)
    def on_frame(self, qimg: QImage):
        self._cam_view.set_frame(qimg)

    @pyqtSlot(int, int, int)
    def on_stats(self, s: int, r: int, e: int):
        self._kpi_sorted.setText(str(s))
        self._kpi_rejected.setText(str(r))
        self._kpi_errors.setText(str(e))

    def _toggle_pause(self):
        self._paused = not self._paused
        self.app.worker.set_running(not self._paused)
        if self._paused:
            self._pause_btn.setText("▶   REPRENDRE")
            self._status_bar.setText("⏸  EN PAUSE")
            self._dot_lbl.setStyleSheet(
                f"color: {AMBER}; font-size: 14px; background: transparent;")
        else:
            self._pause_btn.setText("⏸   PAUSE")
            self._status_bar.setText("")

    def _cancel(self):
        self.app.worker.set_running(False)
        self._cam_view.clear()
        self.app.back_home()

    def _update_clock(self):
        self._clock_lbl.setText(time.strftime("%H:%M:%S"))

    def _blink_dot(self):
        self._dot_state = not self._dot_state
        c = GREEN if self._dot_state else BG
        self._dot_lbl.setStyleSheet(
            f"color: {c}; font-size: 14px; background: transparent;")


# ══════════════════════════════════════════════════════════════════════════════
#   ÉCRAN 3 — MERCI
# ══════════════════════════════════════════════════════════════════════════════

class ThanksScreen(QWidget):
    def __init__(self, app: "NutApp"):
        super().__init__()
        self.app   = app
        self._job  = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)
        root.setSpacing(8)

        check = QLabel("✓")
        check.setAlignment(Qt.AlignCenter)
        check.setStyleSheet(
            f"color: {GREEN}; font-size: 100px; font-weight: bold;")
        root.addWidget(check)

        accepted = lbl("CONTENANT ACCEPTÉ", GREEN, 24, bold=True)
        accepted.setAlignment(Qt.AlignCenter)
        root.addWidget(accepted)

        line = QFrame()
        line.setFixedHeight(1)
        line.setFixedWidth(400)
        line.setStyleSheet(f"background: {BORDER2};")
        root.addWidget(line, alignment=Qt.AlignCenter)
        root.addSpacing(8)

        l1 = lbl("Votre contenant a été analysé et éjecté.", TEXT, 13)
        l1.setAlignment(Qt.AlignCenter)
        root.addWidget(l1)

        l2 = lbl("Vous pouvez en présenter un nouveau.", TEXT_D, 12)
        l2.setAlignment(Qt.AlignCenter)
        root.addWidget(l2)

        root.addSpacing(32)

        self._cd_bar = ProgressBar(height=6)
        self._cd_bar.setFixedWidth(320)
        root.addWidget(self._cd_bar, alignment=Qt.AlignCenter)
        root.addSpacing(6)

        self._cd_lbl = lbl("Retour dans 3 s…", TEXT_D, 11)
        self._cd_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._cd_lbl)

        root.addSpacing(16)

        back_btn = QPushButton("↩   RETOUR ACCUEIL MAINTENANT")
        back_btn.setFixedWidth(320)
        back_btn.setFixedHeight(44)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._go_now)
        root.addWidget(back_btn, alignment=Qt.AlignCenter)

        # Timer countdown
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._remaining = 3000
        self._total     = 3000

    def on_show(self):
        self._remaining = 3000
        self._cd_bar.set_pct(0)
        self._cd_lbl.setText("Retour dans 3 s…")
        self._timer.start(100)

    def _tick(self):
        self._remaining -= 100
        pct  = int((self._total - self._remaining) / self._total * 100)
        secs = max(1, self._remaining // 1000 + 1)
        self._cd_bar.set_pct(pct)
        self._cd_lbl.setText(f"Retour dans {secs} s…")
        if self._remaining <= 0:
            self._go_now()

    def _go_now(self):
        self._timer.stop()
        self.app.back_home()


# ══════════════════════════════════════════════════════════════════════════════
#   APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class NutApp(QMainWindow):
    def __init__(self, worker: CameraWorker):
        super().__init__()
        self.worker = worker
        self.setWindowTitle("0117-NUT")
        self.setStyleSheet(QSS)

        # Plein écran
        self.showFullScreen()

        # Raccourcis clavier
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence("Escape"), self,
                  activated=lambda: self.showNormal())
        QShortcut(QKeySequence("F11"), self,
                  activated=lambda: self.showFullScreen())
        QShortcut(QKeySequence("F4"), self,
                  activated=self.close)

        # Écrans
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._home   = HomeScreen(self)
        self._scan   = ScanScreen(self)
        self._thanks = ThanksScreen(self)

        self._stack.addWidget(self._home)
        self._stack.addWidget(self._scan)
        self._stack.addWidget(self._thanks)

        # Connecter signaux worker → slots ScanScreen
        # IMPORTANT : connexions AVANT worker.start() pour ne perdre aucune frame
        worker.state_updated.connect(self._scan.on_state)
        worker.frame_ready.connect(self._scan.on_frame)
        worker.stats_updated.connect(self._scan.on_stats)

        # Démarrer le thread APRÈS les connexions
        worker.start()

        self._show(self._home)

    def _show(self, screen: QWidget):
        self._stack.setCurrentWidget(screen)
        if hasattr(screen, "on_show"):
            screen.on_show()

    def start_scan(self):
        if LIVE_MODE:
            try: trapdoor_mod.open()
            except: pass
        self.worker.set_running(True)
        self._show(self._scan)

    def scan_done(self):
        self._show(self._thanks)

    def back_home(self):
        self.worker.set_running(False)
        if LIVE_MODE:
            try: trapdoor_mod.close()
            except: pass
        self._scan._cam_view.clear()
        self._show(self._home)

    def closeEvent(self, event):
        self.worker.stop_worker()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#   POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global LIVE_MODE, camera_mod, trapdoor_mod, fork_mod

    ap = argparse.ArgumentParser(description="0117-NUT Interface PyQt5")
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
            print("[INFO] Modules matériels chargés.")
        except Exception as e:
            print(f"[WARN] Modules matériels indisponibles ({e}) — mode démo.")

    queue_list = read_queue(FILE_ORDER)
    print("File d'attente :", [SIZES.get(i, i) for i in queue_list])

    app = QApplication(sys.argv)
    app.setApplicationName("0117-NUT")

    # Worker créé APRÈS QApplication (exigence Qt)
    worker = CameraWorker(
        calibration=args.calibration,
        enable_motor=not args.no_motor,
        queue_list=queue_list,
    )
    win = NutApp(worker)   # start() appelé dans NutApp après connexion des signaux
    ret = app.exec_()
    worker.stop_worker()
    sys.exit(ret)


if __name__ == "__main__":
    main()
