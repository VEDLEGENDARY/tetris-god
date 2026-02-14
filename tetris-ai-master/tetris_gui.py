"""
Tetris AI - Deep Q-Learning Neural Network
"""

# ─── Application Constants ───────────────────────────────────────────────────
VERSION = "3.5.0"
APP_TITLE = "Tetris AI - Deep Q-Learning"

# ─── TensorFlow Setup (before any TF imports) ─────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TF logs (INFO/WARNING/ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable CPU optimizations

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
import sys
import glob
import json
import shutil
import numpy as np
from datetime import datetime, timedelta
from statistics import mean
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, FixedLocator

# AI modules
from tetris import Tetris
from dqn_agent import DQNAgent

# ─── DIRECTORIES (always relative to this script) ─────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")
STATE_FILE = os.path.join(MODELS_DIR, "_training_state.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
C = {
    "bg":           "#0a0a12",
    "bg_grad":      "#0e0e1a",
    "card":         "#14142a",
    "card_hover":   "#191935",
    "card_border":  "#28284a",
    "card_border_h":"#3d3d6a",
    "surface":      "#1c1c38",
    "surface_hover":"#242450",
    "primary":      "#6366f1",
    "primary_hover":"#818cf8",
    "primary_dim":  "#4f46e5",
    "primary_glow": "#6366f140",
    "accent":       "#22d3ee",
    "accent_hover": "#67e8f9",
    "accent_dim":   "#0891b2",
    "text":         "#f0f0f8",
    "text_sec":     "#a0a0c0",
    "text_muted":   "#6a6a8a",
    "text_dim":     "#45455a",
    "green":        "#4ade80",
    "green_hover":  "#86efac",
    "green_dim":    "#166534",
    "red":          "#f87171",
    "red_hover":    "#fca5a5",
    "red_dim":      "#991b1b",
    "yellow":       "#facc15",
    "yellow_dim":   "#854d0e",
    "blue":         "#60a5fa",
    "blue_dim":     "#1e3a5f",
    "orange":       "#fb923c",
    "purple":       "#c084fc",
    "border":       "#2e2e50",
    "border_light": "#3a3a5e",
    "input_bg":     "#0f0f1e",
    "input_border": "#2a2a48",
    "input_focus":  "#6366f1",
    "cell_empty":   "#121228",
    "cell_border":  "#1a1a35",
    "danger_bg":    "#2a1020",
    "danger_hover": "#3d1525",
}

PIECE_COLORS = {
    0: "#22d3ee",   # I
    1: "#c084fc",   # T
    2: "#fb923c",   # L
    3: "#60a5fa",   # J
    4: "#f87171",   # Z
    5: "#4ade80",   # S
    6: "#facc15",   # O
}
PIECE_SHAPES = {
    0: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    1: [[0,0,0,0],[0,1,0,0],[1,1,1,0],[0,0,0,0]],
    2: [[0,0,0,0],[0,0,1,0],[1,1,1,0],[0,0,0,0]],
    3: [[0,0,0,0],[1,0,0,0],[1,1,1,0],[0,0,0,0]],
    4: [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
    5: [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
    6: [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def darken(hex_color, factor=0.6):
    h = hex_color.lstrip('#')
    r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

def lighten(hex_color, factor=0.3):
    h = hex_color.lstrip('#')
    r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:6],16)
    r = min(255, int(r+(255-r)*factor))
    g = min(255, int(g+(255-g)*factor))
    b = min(255, int(b+(255-b)*factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def blend(hex1, hex2, t=0.5):
    h1, h2 = hex1.lstrip('#'), hex2.lstrip('#')
    r = int(int(h1[:2],16)*(1-t) + int(h2[:2],16)*t)
    g = int(int(h1[2:4],16)*(1-t) + int(h2[2:4],16)*t)
    b = int(int(h1[4:6],16)*(1-t) + int(h2[4:6],16)*t)
    return f"#{min(255,r):02x}{min(255,g):02x}{min(255,b):02x}"

def fmt_time(secs):
    if secs < 0:
        return "--:--"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def get_last_saved_episode():
    """Get the last completed episode from state file, and the last saved model episode."""
    # Check state file for the true last episode
    state = load_training_state()
    last_episode = state.get("last_episode", 0) if state else 0
    
    # Find the last saved model (which might be earlier than last_episode)
    episodes = []
    for f in glob.glob(os.path.join(MODELS_DIR, "episode_*.keras")):
        try:
            num = int(os.path.basename(f).replace("episode_", "").replace(".keras", ""))
            if num == 1 or num % 50 == 0:
                episodes.append(num)
        except ValueError:
            pass
    last_saved_model = max(episodes) if episodes else 0
    
    # Return the true last episode (for resuming) and last saved model (for loading)
    return last_episode, last_saved_model

def load_training_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_training_state(state_dict):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state_dict, f)
    except Exception:
        pass


# ─── Hover / Focus helpers ───────────────────────────────────────────────────

def add_hover(widget, normal_fg, hover_fg, normal_border=None, hover_border=None):
    """Bind smooth hover effect to a CTkFrame-based widget."""
    def _enter(e):
        try:
            widget.configure(fg_color=hover_fg)
            if hover_border:
                widget.configure(border_color=hover_border)
        except Exception:
            pass
    def _leave(e):
        try:
            widget.configure(fg_color=normal_fg)
            if normal_border:
                widget.configure(border_color=normal_border)
        except Exception:
            pass
    widget.bind("<Enter>", _enter)
    widget.bind("<Leave>", _leave)

def add_entry_focus(entry, normal_border=None, focus_border=None):
    """Bind focus glow to a CTkEntry."""
    if normal_border is None:
        normal_border = C["input_border"]
    if focus_border is None:
        focus_border = C["input_focus"]
    def _focus_in(e):
        try:
            entry.configure(border_color=focus_border)
        except Exception:
            pass
    def _focus_out(e):
        try:
            entry.configure(border_color=normal_border)
        except Exception:
            pass
    entry.bind("<FocusIn>", _focus_in)
    entry.bind("<FocusOut>", _focus_out)

def add_btn_press(btn, normal_fg, press_fg, normal_hover=None):
    """Bind press-darken feedback to a CTkButton."""
    def _press(e):
        try:
            btn.configure(fg_color=press_fg)
        except Exception:
            pass
    def _release(e):
        try:
            btn.configure(fg_color=normal_fg)
        except Exception:
            pass
    btn.bind("<ButtonPress-1>", _press)
    btn.bind("<ButtonRelease-1>", _release)


# ─── Glass Card ───────────────────────────────────────────────────────────────

class GlassCard(ctk.CTkFrame):
    """Glassmorphic card with hover border glow."""
    def __init__(self, master, hoverable=True, **kw):
        super().__init__(
            master,
            fg_color=C["card"],
            corner_radius=18,
            border_width=1,
            border_color=C["card_border"],
            **kw,
        )
        if hoverable:
            add_hover(self, C["card"], C["card_hover"],
                      C["card_border"], C["card_border_h"])


# ─── Animated Dot ────────────────────────────────────────────────────────────

class PulsingDot(ctk.CTkFrame):
    def __init__(self, master, size=8, color_on=None, color_off=None):
        if color_on is None:
            color_on = C["green"]
        if color_off is None:
            color_off = C["text_dim"]
        super().__init__(master, width=size, height=size, corner_radius=size//2,
                         fg_color=color_off)
        self._on = color_on
        self._off = color_off
        self._active = False
        self._phase = 0

    def set_active(self, active, color=None):
        self._active = active
        if color:
            self._on = color
        if not active:
            self.configure(fg_color=self._off)
        else:
            self._pulse()

    def _pulse(self):
        if not self._active:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        self._phase = (self._phase + 1) % 20
        t = abs(self._phase - 10) / 10.0
        col = blend(self._on, lighten(self._on, 0.4), t)
        try:
            self.configure(fg_color=col)
        except Exception:
            return
        self.after(80, self._pulse)


# ─── Section Title ────────────────────────────────────────────────────────────

class SectionLabel(ctk.CTkFrame):
    def __init__(self, master, text, right_widget=None):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=text, font=("Inter", 12, "bold"),
                     text_color=C["text_muted"]).pack(side="left")
        if right_widget:
            right_widget.pack(side="right")


# ─── Header ──────────────────────────────────────────────────────────────────

class Header(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True)

        icon_card = GlassCard(left)
        icon_card.pack(side="left", padx=(0, 14))
        ctk.CTkLabel(icon_card, text="\U0001f9e0", font=("Segoe UI Emoji", 36),
                     text_color=C["primary"]).pack(padx=14, pady=10)

        title_frame = ctk.CTkFrame(left, fg_color="transparent")
        title_frame.pack(side="left")
        row1 = ctk.CTkFrame(title_frame, fg_color="transparent")
        row1.pack(anchor="w")
        ctk.CTkLabel(row1, text="Tetris AI", font=("Inter", 36, "bold"),
                     text_color=C["text"]).pack(side="left")
        ctk.CTkLabel(row1, text=" \u2728", font=("Segoe UI Emoji", 26),
                     text_color=C["yellow"]).pack(side="left", padx=(6, 0))
        ctk.CTkLabel(title_frame, text="Neural Network Training Visualization",
                     font=("Inter", 16), text_color=C["text_muted"]).pack(anchor="w", pady=(2, 0))

        # Stats card
        stats = GlassCard(self)
        stats.pack(side="right")
        inner = ctk.CTkFrame(stats, fg_color="transparent")
        inner.pack(padx=20, pady=10)

        def _stat_col(parent, label, initial, color):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(side="left", padx=10)
            ctk.CTkLabel(f, text=label, font=("Inter", 12, "bold"),
                         text_color=C["text_muted"]).pack(anchor="e")
            lbl = ctk.CTkLabel(f, text=initial,
                         font=("JetBrains Mono", 26, "bold"), text_color=color)
            lbl.pack(anchor="e")
            return lbl

        self.best_lbl = _stat_col(inner, "BEST SCORE", "0", C["accent"])
        ctk.CTkFrame(inner, fg_color=C["border"], width=1, height=36).pack(side="left", padx=6)
        self.ep_lbl = _stat_col(inner, "EPISODES", "0", C["text"])
        ctk.CTkFrame(inner, fg_color=C["border"], width=1, height=36).pack(side="left", padx=6)
        self.saved_lbl = _stat_col(inner, "SAVED", "0", C["green"])

    def update_stats(self, best_score, episodes, saved=None):
        self.best_lbl.configure(text=f"{best_score:,}")
        self.ep_lbl.configure(text=f"{episodes:,}")
        if saved is not None:
            self.saved_lbl.configure(text=f"{saved:,}")


# ─── Control Panel ────────────────────────────────────────────────────────────

class ControlPanel(GlassCard):
    # Default parameter values
    DEFAULTS = {
        "episodes": "3000",
        "batch_size": "128",
        "epsilon_stop": "2000",
        "discount": "0.95",
        "mem_size": "1000",
        "epochs": "1",
        "train_every": "1",
        "max_score": "0"
    }

    def __init__(self, master, app):
        super().__init__(master, hoverable=False)
        self.app = app

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=32, pady=24)  # Increased by ~15%

        ctk.CTkLabel(inner, text="Control Panel", font=("Inter", 24, "bold"),
                 text_color=C["text"]).pack(anchor="w")
        ctk.CTkLabel(inner, text="Manage AI Training and Visualization",
                 font=("Inter", 16), text_color=C["text_muted"]).pack(anchor="w", pady=(2, 18))

        # ── Training ──
        SectionLabel(inner, "TRAINING SESSION").pack(fill="x", pady=(0, 6))

        self.train_btn = ctk.CTkButton(
            inner, text="\u25b6  Start Training", font=("Inter", 16, "bold"),
            fg_color=C["primary"], hover_color=C["primary_hover"],
            text_color="white", height=50, corner_radius=14,
            command=lambda: app.toggle_learning()
        )
        self.train_btn.pack(fill="x", pady=(0, 14))
        add_btn_press(self.train_btn, C["primary"], C["primary_dim"])

        # ── Hyperparameters ──
        SectionLabel(inner, "HYPERPARAMETERS").pack(fill="x", pady=(0, 8))

        pf = ctk.CTkFrame(inner, fg_color="transparent")
        pf.pack(fill="x", pady=(0, 16))  # Increased width by ~15%
        pf.grid_columnconfigure(1, weight=1)
        pf.grid_columnconfigure(3, weight=1)

        # Load saved parameters or use defaults
        saved_state = load_training_state()
        saved_params = saved_state.get("parameters", {}) if saved_state else {}

        self._ep_entry    = self._param(pf, "Total Episodes:",    saved_params.get("episodes", self.DEFAULTS["episodes"]), 0, 0)
        self._batch_entry = self._param(pf, "Batch Size:",        saved_params.get("batch_size", self.DEFAULTS["batch_size"]),  1, 0)
        self._eps_entry   = self._param(pf, "Epsilon Stop:", saved_params.get("epsilon_stop", self.DEFAULTS["epsilon_stop"]), 2, 0)
        self._disc_entry  = self._param(pf, "Discount:",    saved_params.get("discount", self.DEFAULTS["discount"]), 3, 0)
        self._mem_entry   = self._param(pf, "Memory Size:",        saved_params.get("mem_size", self.DEFAULTS["mem_size"]), 0, 2)
        self._epoch_entry = self._param(pf, "Epochs:",   saved_params.get("epochs", self.DEFAULTS["epochs"]),    1, 2)
        self._tevery_entry= self._param(pf, "Train Every:",        saved_params.get("train_every", self.DEFAULTS["train_every"]),    2, 2)
        self._limit_entry = self._param(pf, "Max Score:",    saved_params.get("max_score", self.DEFAULTS["max_score"]),    3, 2)


        # ── Visualization ──
        ctk.CTkFrame(inner, fg_color=C["border"], height=1).pack(fill="x", pady=(0, 14))
        SectionLabel(inner, "VISUALIZE EPISODE").pack(fill="x", pady=(0, 8))

        # Pause/Resume Toggle Button
        self.pause_toggle_btn = ctk.CTkButton(
            inner, text="⏸  Pause Visualization", font=("Inter", 16, "bold"),
            fg_color="transparent", hover_color=C["surface_hover"],
            text_color=C["yellow"], height=50, corner_radius=14,
            border_width=2, border_color=C["accent_dim"],
            command=lambda: app.toggle_pause_visualization()
        )
        self.pause_toggle_btn.pack(fill="x", pady=(0, 8))
        add_btn_press(self.pause_toggle_btn, "transparent", C["surface"])
        self.pause_toggle_btn.configure(state="disabled")

        ep_frame = ctk.CTkFrame(inner, fg_color="transparent")
        ep_frame.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(ep_frame, text="Episode #", font=("Inter", 16),
                     text_color=C["text_muted"]).pack(side="left")
        self.vis_entry = ctk.CTkEntry(
            ep_frame, font=("JetBrains Mono", 16), width=90,
            fg_color=C["input_bg"], border_color=C["input_border"],
            text_color=C["text"], height=36, corner_radius=10,
            placeholder_text="e.g. 50", justify="center"
        )
        self.vis_entry.pack(side="left", padx=(8, 0), fill="x", expand=True)
        add_entry_focus(self.vis_entry)

        self.vis_btn = ctk.CTkButton(
            inner, text="\U0001f441  Visualize Episode", font=("Inter", 16, "bold"),
            fg_color="transparent", hover_color=C["surface_hover"],
            text_color=C["accent"], height=50, corner_radius=14,
            border_width=2, border_color=C["accent_dim"],
            command=lambda: app.toggle_visualization()
        )
        self.vis_btn.pack(fill="x", pady=(0, 14))
        add_btn_press(self.vis_btn, "transparent", C["surface"])

        # ── Speed Control ──
        ctk.CTkFrame(inner, fg_color=C["border"], height=1).pack(fill="x", pady=(0, 14))
        SectionLabel(inner, "PLAYBACK SPEED").pack(fill="x", pady=(0, 6))
        speed_frame = ctk.CTkFrame(inner, fg_color="transparent")
        speed_frame.pack(fill="x", pady=(0, 14))
        self.speed_label = ctk.CTkLabel(speed_frame, text="1.0x",
                     font=("JetBrains Mono", 16, "bold"), text_color=C["accent"])
        self.speed_label.pack(side="right")
        self.speed_slider = ctk.CTkSlider(
            speed_frame, from_=0.1, to=5.0, number_of_steps=49,
            fg_color=C["surface"], progress_color=C["primary"],
            button_color=C["primary_hover"], button_hover_color=C["accent"],
            height=16, corner_radius=8,
            command=self._on_speed_change
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # ── Reset Button ──
        ctk.CTkFrame(inner, fg_color=C["border"], height=1).pack(fill="x", pady=(0, 14))
        self.reset_btn = ctk.CTkButton(
            inner, text="\U0001f5d1  Reset All Data", font=("Inter", 24, "bold"),
            fg_color=C["danger_bg"], hover_color=C["danger_hover"],
            text_color=C["red"], height=44, corner_radius=12,
            border_width=1, border_color="#3d1525",
            command=lambda: app.reset_all()
        )
        self.reset_btn.pack(fill="x", pady=(0, 14))
        add_btn_press(self.reset_btn, C["danger_bg"], C["red_dim"])

        # Status indicators removed

    def _param(self, parent, label, default, row, col):
        # Double the padding between label and entry, and between rows
        ctk.CTkLabel(parent, text=label, font=("Inter", 14),
                     text_color=C["text_muted"]).grid(row=row, column=col,
                     sticky="w", padx=(0, 5), pady=6)
        e = ctk.CTkEntry(parent, font=("JetBrains Mono", 12), width=400,
                          fg_color=C["input_bg"], border_color=C["input_border"],
                          text_color=C["text"], height=32, corner_radius=8)
        e.grid(row=row, column=col+1, sticky="ew", padx=(0, 5), pady=4)
        e.insert(0, default)
        add_entry_focus(e)
        return e

    def _on_speed_change(self, val):
        self.speed_label.configure(text=f"{val:.1f}x")

    @property
    def speed(self):
        return self.speed_slider.get()

    def get_params(self):
        try:
            return dict(
                episodes=int(self._ep_entry.get()),
                batch_size=int(self._batch_entry.get()),
                epsilon_stop_episode=int(self._eps_entry.get()),
                discount=float(self._disc_entry.get()),
                mem_size=int(self._mem_entry.get()),
                epochs=int(self._epoch_entry.get()),
                train_every=int(self._tevery_entry.get()),
                piece_limit=int(self._limit_entry.get()),
            )
        except ValueError:
            return None

    def save_params(self):
        """Save current parameter values to state file."""
        saved_state = load_training_state() or {}
        saved_state["parameters"] = {
            "episodes": self._ep_entry.get(),
            "batch_size": self._batch_entry.get(),
            "epsilon_stop": self._eps_entry.get(),
            "discount": self._disc_entry.get(),
            "mem_size": self._mem_entry.get(),
            "epochs": self._epoch_entry.get(),
            "train_every": self._tevery_entry.get(),
            "max_score": self._limit_entry.get()
        }
        save_training_state(saved_state)

    def reset_params(self):
        """Reset all parameters to default values."""
        self._ep_entry.delete(0, "end")
        self._ep_entry.insert(0, self.DEFAULTS["episodes"])
        self._batch_entry.delete(0, "end")
        self._batch_entry.insert(0, self.DEFAULTS["batch_size"])
        self._eps_entry.delete(0, "end")
        self._eps_entry.insert(0, self.DEFAULTS["epsilon_stop"])
        self._disc_entry.delete(0, "end")
        self._disc_entry.insert(0, self.DEFAULTS["discount"])
        self._mem_entry.delete(0, "end")
        self._mem_entry.insert(0, self.DEFAULTS["mem_size"])
        self._epoch_entry.delete(0, "end")
        self._epoch_entry.insert(0, self.DEFAULTS["epochs"])
        self._tevery_entry.delete(0, "end")
        self._tevery_entry.insert(0, self.DEFAULTS["train_every"])
        self._limit_entry.delete(0, "end")
        self._limit_entry.insert(0, self.DEFAULTS["max_score"])

    def _set_params_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for e in [self._ep_entry, self._batch_entry, self._eps_entry,
                  self._disc_entry, self._mem_entry, self._epoch_entry,
                  self._tevery_entry, self._limit_entry]:
            e.configure(state=state)

    def set_learning(self, active, episode=0):
        if active:
            self.train_btn.configure(text="\u25a0  Stop Learning",
                fg_color=C["red"], hover_color=C["red_hover"])
            add_btn_press(self.train_btn, C["red"], C["red_dim"])
            self._set_params_enabled(False)
            self.reset_btn.configure(state="disabled")
        else:
            self.train_btn.configure(text="\u25b6  Start Learning",
                fg_color=C["primary"], hover_color=C["primary_hover"])
            add_btn_press(self.train_btn, C["primary"], C["primary_dim"])
            self._set_params_enabled(True)
            self.reset_btn.configure(state="normal")

    def set_visualizing(self, active, ep_num=0):
        if active:
            self.vis_btn.configure(text="\u25a0  Stop Visualization",
                fg_color=C["red"], hover_color=C["red_hover"],
                text_color="white", border_color=C["red"])
            add_btn_press(self.vis_btn, C["red"], C["red_dim"])
            self.pause_toggle_btn.configure(state="normal", text="⏸  Pause Visualization", text_color=C["yellow"])
        else:
            self.vis_btn.configure(text="\U0001f441  Visualize Episode",
                fg_color="transparent", hover_color=C["surface_hover"],
                text_color=C["accent"], border_color=C["accent_dim"])
            add_btn_press(self.vis_btn, "transparent", C["surface"])
            self.pause_toggle_btn.configure(state="disabled", text="⏸  Pause Visualization", text_color=C["yellow"])

    def set_paused(self, paused):
        if paused:
            self.pause_toggle_btn.configure(text="▶  Resume Visualization", text_color=C["green"])
        else:
            self.pause_toggle_btn.configure(text="⏸  Pause Visualization", text_color=C["yellow"])

    def get_vis_episode(self):
        txt = self.vis_entry.get().strip()
        if not txt:
            return None
        try:
            return int(txt)
        except ValueError:
            return None


# ─── Training Status Panel ────────────────────────────────────────────────────

class TrainingStatusPanel(GlassCard):
    def __init__(self, master):
        super().__init__(master)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=16)

        row = ctk.CTkFrame(inner, fg_color="transparent")
        row.pack(fill="x")

        icon = ctk.CTkFrame(row, fg_color=C["surface"], corner_radius=14,
                            width=48, height=48)
        icon.pack(side="left", padx=(0, 14))
        icon.pack_propagate(False)
        self._brain = ctk.CTkLabel(icon, text="\U0001f9e0", font=("Segoe UI Emoji", 24),
                     text_color=C["text_muted"])
        self._brain.pack(expand=True)

        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True)

        self._status = ctk.CTkLabel(info, text="TRAINING IDLE",
                     font=("Inter", 12, "bold"), text_color=C["text_muted"])
        self._status.pack(anchor="w")
        self._main = ctk.CTkLabel(info, text="No training started",
                     font=("JetBrains Mono", 18), text_color=C["text_muted"])
        self._main.pack(anchor="w")
        self._sub = ctk.CTkLabel(info, text="",
                     font=("Inter", 14), text_color=C["text_muted"])
        self._sub.pack(anchor="w")

        self.progress = ctk.CTkProgressBar(inner, height=6, corner_radius=3,
                     fg_color=C["surface"], progress_color=C["primary"])
        self.progress.pack(fill="x", pady=(12, 6))
        self.progress.set(0)

        extra = ctk.CTkFrame(inner, fg_color="transparent")
        extra.pack(fill="x")
        self._elapsed = ctk.CTkLabel(extra, text="", font=("Inter", 14),
                     text_color=C["text_muted"])
        self._elapsed.pack(side="left")
        self._eta = ctk.CTkLabel(extra, text="", font=("Inter", 14),
                     text_color=C["text_muted"])
        self._eta.pack(side="right")

    def update(self, active, ep=0, total=0, elapsed=0, epsilon=0, avg_50=0):
        if active:
            self._brain.configure(text_color=C["green"])
            self._status.configure(text="\u27f3 TRAINING", text_color=C["green"])
            self._main.configure(text=f"Episode {ep:,} of {total:,}",
                                 text_color=C["text"])
            self._sub.configure(text=f"Avg (last 50): {avg_50:,.0f}  |  "
                                        f"Epsilon: {epsilon:.3f}  |  "
                                        f"Elapsed: {fmt_time(elapsed)}")
            self.progress.set(ep / total if total > 0 else 0)
        else:
            self._brain.configure(text_color=C["text_muted"])
            self._status.configure(text="TRAINING IDLE", text_color=C["text_muted"])
            if ep > 0:
                self._main.configure(text=f"Completed: Episode #{ep}",
                                     font=("JetBrains Mono", 18),
                                     text_color=C["text_sec"])
                self._sub.configure(text=f"Elapsed: {fmt_time(elapsed)}")
            else:
                self._main.configure(text="No training started",
                                     font=("JetBrains Mono", 18),
                                     text_color=C["text_muted"])
                self._sub.configure(text="")
            self.progress.set(0)
            self._elapsed.configure(text="")
            self._eta.configure(text="")

    def reset(self):
        self._main.configure(text="No training started",
                                 text_color=C["text_muted"])
        self._sub.configure(text="Avg (last 50): 0")
        self.progress.set(0)


# ─── Tetris Board (Visualization Only) ────────────────────────────────────────

class TetrisBoardWidget(GlassCard):
    CELL = 44
    GAP = 2
    ROWS = 20
    COLS = 10

    def __init__(self, master):
        super().__init__(master, hoverable=False)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(padx=14, pady=14)

        # Label row
        lf = ctk.CTkFrame(inner, fg_color="transparent")
        lf.pack(pady=(0, 10))
        self._label = ctk.CTkLabel(lf, text="VISUALIZATION",
                     font=("Inter", 16, "bold"), text_color=C["text_muted"])
        self._label.pack(side="left")
        self._dot = PulsingDot(lf, size=8, color_on=C["accent"])
        self._dot.pack(side="left", padx=(10, 0))

        # Canvas
        cw = self.COLS * (self.CELL + self.GAP) + self.GAP
        ch = self.ROWS * (self.CELL + self.GAP) + self.GAP
        self.canvas = tk.Canvas(inner, width=cw, height=ch,
                                bg=C["card"], highlightthickness=0, bd=0)
        self.canvas.pack()
        self._draw_idle()

    def _draw_idle(self):
        """Draw the idle state: empty grid with a subtle 'waiting' message."""
        self.canvas.delete("all")
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x1 = self.GAP + c * (self.CELL + self.GAP)
                y1 = self.GAP + r * (self.CELL + self.GAP)
                x2, y2 = x1 + self.CELL, y1 + self.CELL
                self.canvas.create_rectangle(x1, y1, x2, y2,
                    fill=C["cell_empty"], outline=C["cell_border"], width=1)

        # Center message
        cx = (self.COLS * (self.CELL + self.GAP) + self.GAP) // 2
        cy = (self.ROWS * (self.CELL + self.GAP) + self.GAP) // 2
        self.canvas.create_text(cx, cy - 14,
            text="No Visualization Active",
            fill=C["text_dim"], font=("Inter", 16, "bold"), anchor="center")
        self.canvas.create_text(cx, cy + 12,
            text="Enter an episode # and click Visualize",
            fill=C["text_dim"], font=("Inter", 12), anchor="center")

    def _draw_board(self, board, piece_id=None):
        # Optimized rendering: only redraw when board state changes
        # Store placed blocks (blue blocks) and only render those
        try:
            self.canvas.delete("all")
            
            # Validate board dimensions
            if board and (len(board) != self.ROWS or any(len(row) != self.COLS for row in board)):
                print(f"[ERROR] Invalid board dimensions: {len(board)}x{len(board[0]) if board else 0}")
                return
            
            # Draw empty grid background first
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    x1 = self.GAP + c * (self.CELL + self.GAP)
                    y1 = self.GAP + r * (self.CELL + self.GAP)
                    x2, y2 = x1 + self.CELL, y1 + self.CELL
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                        fill=C["cell_empty"], outline=C["cell_border"], width=1)
            
            # Only render placed blocks (MAP_BLOCK) in blue
            if board:
                for r in range(min(len(board), self.ROWS)):
                    for c in range(min(len(board[r]), self.COLS)):
                        if board[r][c] == Tetris.MAP_BLOCK:
                            x1 = self.GAP + c * (self.CELL + self.GAP)
                            y1 = self.GAP + r * (self.CELL + self.GAP)
                            x2, y2 = x1 + self.CELL, y1 + self.CELL
                            self._draw_cell(x1, y1, x2, y2, C["primary"])
        except Exception as e:
            print(f"[ERROR] Board rendering failed: {e}")
            import traceback
            traceback.print_exc()

    def _draw_cell(self, x1, y1, x2, y2, color):
        dk = darken(color, 0.55)
        lt = lighten(color, 0.35)
        mid = lighten(color, 0.1)
        glow = lighten(color, 0.2)

        self.canvas.create_rectangle(x1-1, y1-1, x2+1, y2+1,
            fill="", outline=glow, width=1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=dk, width=1)
        self.canvas.create_line(x1+2, y1+2, x2-2, y1+2, fill=lt, width=2)
        self.canvas.create_line(x1+2, y1+2, x1+2, y2-2, fill=mid, width=1)
        self.canvas.create_line(x1+2, y2-2, x2-2, y2-2, fill=dk, width=1)

    def update_board(self, env, label=None, active=False):
        try:
            board = env._get_complete_board()
            if board is None:
                print("[WARN] Board is None, skipping render")
                return
            self._draw_board(board, env.current_piece)
            if label:
                self._label.configure(text=label)
            self._dot.set_active(active)
        except Exception as e:
            print(f"[ERROR] update_board failed: {e}")
            import traceback
            traceback.print_exc()

    def clear(self):
        self._draw_idle()
        self._label.configure(text="VISUALIZATION")
        self._dot.set_active(False)


# ─── Next Piece ───────────────────────────────────────────────────────────────

class NextPieceWidget(GlassCard):
    SZ = TetrisBoardWidget.CELL
    GAP = TetrisBoardWidget.GAP

    def __init__(self, master):
        super().__init__(master)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(padx=26, pady=22)
        ctk.CTkLabel(inner, text="NEXT PIECE", font=("Inter", 16, "bold"),
                     text_color=C["text_muted"]).pack(anchor="w", pady=(0, 10))
        s = 4 * (self.SZ + self.GAP) + self.GAP
        self.canvas = tk.Canvas(inner, width=s, height=s,
                                bg=C["card"], highlightthickness=0)
        self.canvas.pack()
        self._draw_piece(None)

    def _draw_piece(self, pid):
        self.canvas.delete("all")
        shape = PIECE_SHAPES.get(pid)
        color = PIECE_COLORS.get(pid)
        for r in range(4):
            for c in range(4):
                x1 = self.GAP + c*(self.SZ+self.GAP)
                y1 = self.GAP + r*(self.SZ+self.GAP)
                x2, y2 = x1+self.SZ, y1+self.SZ
                if shape and shape[r][c]:
                    dk, lt = darken(color, 0.6), lighten(color, 0.35)
                    glow = lighten(color, 0.2)
                    self.canvas.create_rectangle(x1-1, y1-1, x2+1, y2+1,
                        fill="", outline=glow, width=1)
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                        fill=color, outline=dk, width=1)
                    self.canvas.create_line(x1+2, y1+2, x2-2, y1+2, fill=lt, width=2)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                        fill=C["cell_empty"], outline=C["cell_border"], width=1)

    def set(self, pid):
        self._draw_piece(pid)


# ─── Stats Panel ─────────────────────────────────────────────────────────────

class StatsPanelWidget(GlassCard):
    def __init__(self, master):
        super().__init__(master)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=14)

        self._ep_lbl = ctk.CTkLabel(inner, text="Visualization\u2014",
                     font=("Inter", 16, "bold"), text_color=C["text_muted"])
        self._ep_lbl.pack(anchor="w", pady=(0, 10))

        grid = ctk.CTkFrame(inner, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        self._score  = self._row(grid, "\U0001f3af", "Score",      "0", C["accent"], row=0, col=0)
        self._high   = self._row(grid, "\U0001f3c6", "High Score", "0", C["yellow"], row=0, col=1)
        self._lines  = self._row(grid, "\u26a1",     "Pieces",     "0", C["green"], row=1, col=0)
        self._pps    = self._row(grid, "\U0001f4ca", "Pieces/Sec", "0", C["blue"], row=1, col=1)

    def _row(self, parent, icon, label, val, color, row=None, col=None):
        f = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=10)
        if row is None:
            f.pack(fill="x", pady=3)
        else:
            f.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
        add_hover(f, "transparent", C["surface"])

        ibg = ctk.CTkFrame(f, fg_color=darken(color, 0.25), corner_radius=9,
                           width=30, height=30)
        ibg.pack(side="left", padx=(6, 10), pady=4)
        ibg.pack_propagate(False)
        ctk.CTkLabel(ibg, text=icon, font=("Segoe UI Emoji", 14)).pack(expand=True)
        info = ctk.CTkFrame(f, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, pady=4)
        ctk.CTkLabel(info, text=label, font=("Inter", 16, "bold"),
                     text_color=C["text_muted"]).pack(anchor="w")
        v = ctk.CTkLabel(info, text=val, font=("JetBrains Mono", 16, "bold"),
                     text_color=C["text"])
        v.pack(anchor="w")
        return v

    def set(self, score, high, pieces, episode, pps=0):
        self._ep_lbl.configure(text=f"Visualizing Ep #{episode}" if episode else "Visualization\u2014")
        self._score.configure(text=f"{score:,}")
        self._high.configure(text=f"{high:,}")
        self._lines.configure(text=f"{pieces:,}")
        self._pps.configure(text=f"{pps:.1f}")


# ─── Score Chart ──────────────────────────────────────────────────────────────

class ScoreChartWidget(GlassCard):
    TICK_VALUES = [10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000]

    def __init__(self, master):
        super().__init__(master, hoverable=False)
        card_rgb = tuple(int(C["card"].lstrip('#')[i:i+2], 16)/255 for i in (0,2,4))
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=card_rgb)
        self.ax = self.fig.add_subplot(111)
        self.setup_styles()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        cw = self.canvas.get_tk_widget()
        cw.configure(bg=C["card"], highlightthickness=0)
        cw.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=2, pady=2)

        self.eps = deque(maxlen=500)
        self.avgs = deque(maxlen=500)
        self.maxs = deque(maxlen=500)

        # Hover marker dots (drawn on the axes)
        self._avg_dot = None
        self._max_dot = None

        # Tooltip label
        self._tooltip_label = ctk.CTkLabel(self, text="", corner_radius=8,
                        fg_color=C["bg"], text_color=C["text"],
                        font=("JetBrains Mono", 16))
        self._tooltip_label.place_forget()

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self.plot()

    @staticmethod
    def _score_fmt(x, pos):
        """Format tick labels as clean integers: 1, 5, 10, 20, 50, 100 ..."""
        if int(x) in ScoreChartWidget.TICK_VALUES:
            return f"{int(x):,}"
        if x >= 1000:
            return f"{x/1000:.0f}k" if x % 1000 == 0 else f"{x/1000:.1f}k"
        return f"{int(x)}"

    def setup_styles(self):
        card_rgb = tuple(int(C["card"].lstrip('#')[i:i+2], 16)/255 for i in (0,2,4))
        self.ax.set_facecolor(card_rgb)
        self.fig.patch.set_facecolor(card_rgb)
        self.fig.patch.set_alpha(1.0)
        self.ax.tick_params(axis='x', colors=C["text_muted"], labelsize=9)
        self.ax.tick_params(axis='y', colors=C["text_muted"], labelsize=9)
        self.ax.set_ylabel("Score (Logarithmic)", color=C["text_muted"], fontsize=16)
        self.ax.set_xlabel("Episode", color=C["text_muted"], fontsize=16)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(C["card_border"])
        self.ax.spines['left'].set_color(C["card_border"])
        self.ax.grid(color=C["card_border"], linestyle='--', linewidth=0.5,
                     alpha=0.5)
        # Logarithmic y-axis with clean integer labels
        self.ax.set_yscale('symlog', linthresh=1)
        self.ax.yaxis.set_major_locator(FixedLocator(self.TICK_VALUES))
        self.ax.yaxis.set_minor_locator(FixedLocator([]))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._score_fmt))
        self.ax.yaxis.set_minor_formatter(FuncFormatter(self._score_fmt))
        self.fig.tight_layout(pad=0.5)

    def _on_hover(self, event):
        if not event.inaxes or event.xdata is None or not self.eps:
            self._on_leave(event)
            return

        # Find closest data point
        lines = self.ax.get_lines()
        if len(lines) < 2 or self._avg_dot is None or self._max_dot is None:
            return
        line_avg, line_max = lines[:2]
        x_data, y_data_avg = line_avg.get_data()
        _, y_data_max = line_max.get_data()

        # Find index of closest x_data based on mouse x-coordinate
        idx = np.searchsorted(x_data, event.xdata)
        if idx == len(x_data): idx -= 1
        if idx > 0 and abs(x_data[idx-1] - event.xdata) < abs(x_data[idx] - event.xdata):
            idx = idx - 1

        ep, avg, max_s = x_data[idx], y_data_avg[idx], y_data_max[idx]

        # Update hover dots
        self._avg_dot.set_data([ep], [avg])
        self._max_dot.set_data([ep], [max_s])
        self._avg_dot.set_visible(True)
        self._max_dot.set_visible(True)

        # Update and place tooltip at top center of graph
        tooltip_text = f"Ep {ep:,}:  Avg {avg:,.0f} | Max {max_s:,.0f}"
        self._tooltip_label.configure(text=tooltip_text)

        # Place tooltip at top center of graph (fixed position)
        canvas_widget = self.canvas.get_tk_widget()
        self._tooltip_label.update_idletasks()
        widget_width = canvas_widget.winfo_width()
        tooltip_width = self._tooltip_label.winfo_reqwidth()
        rel_x = (widget_width - tooltip_width) // 2  # Center horizontally
        rel_y = 10  # Fixed 10px from top

        self._tooltip_label.place(x=rel_x, y=rel_y, anchor="nw")

        self.canvas.draw_idle()

    def _on_leave(self, event):
        if self._avg_dot:
            self._avg_dot.set_visible(False)
        if self._max_dot:
            self._max_dot.set_visible(False)
        self._tooltip_label.place_forget()
        self.canvas.draw_idle()

    def load_data(self, eps, avgs, maxs):
        self.eps.clear()
        self.avgs.clear()
        self.maxs.clear()
        self.eps.extend(eps)
        self.avgs.extend(avgs)
        self.maxs.extend(maxs)
        self.plot()

    def plot(self):
        self.ax.clear()
        self.setup_styles()
        if len(self.eps) > 1:
            self.ax.plot(self.eps, self.avgs, color=C["primary"], linewidth=2)
            self.ax.plot(self.eps, self.maxs, color=C["accent"], linewidth=1.5,
                         linestyle='--')
            self.ax.fill_between(self.eps, self.avgs, self.maxs,
                                 color=C["primary"], alpha=0.1)
        self._avg_dot, = self.ax.plot([], [], 'o', ms=8, mfc=C["primary"], mec='white', mew=1.5, visible=False)
        self._max_dot, = self.ax.plot([], [], 'o', ms=8, mfc=C["accent"], mec='white', mew=1.5, visible=False)
        self.canvas.draw()

    def add_data_point(self, ep, avg_score, max_score):
        # If the new episode is less than or equal to the last one, we've restarted.
        if self.eps and ep <= self.eps[-1]:
            # Find where the new episode would fit in the existing data
            idx = np.searchsorted(self.eps, ep)
            # Clear all data from that point forward
            self.eps = deque(list(self.eps)[:idx])
            self.avgs = deque(list(self.avgs)[:idx])
            self.maxs = deque(list(self.maxs)[:idx])

        self.eps.append(ep)
        self.avgs.append(avg_score)
        self.maxs.append(max_score)
        self.plot()

    def clear_plot(self):
        self.eps.clear()
        self.avgs.clear()
        self.maxs.clear()
        self.plot()


# ─── Generation List ─────────────────────────────────────────────────────────

class GenerationListWidget(GlassCard):
    def __init__(self, master):
        super().__init__(master, hoverable=False)
        self._gens = deque(maxlen=20)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=18)

        ctk.CTkLabel(inner, text="Recent Episodes", font=("Inter", 24, "bold"),
                     text_color=C["text"]).pack(anchor="w")
        ctk.CTkLabel(inner, text="Last completed training runs",
                     font=("Inter", 16), text_color=C["text_muted"]).pack(anchor="w", pady=(2, 14))

        self._list = ctk.CTkScrollableFrame(inner, fg_color="transparent",
                     height=220, corner_radius=0,
                     scrollbar_button_color=C["border"],
                     scrollbar_button_hover_color=C["text_dim"])
        self._list.pack(fill="both", expand=True)
        self._show_empty()

    def _show_empty(self):
        for w in self._list.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._list, text="No episodes completed yet",
                     font=("Inter", 16), text_color=C["text_muted"]).pack(pady=40)

    def add(self, ep_range, avg_score, max_score):
        self._gens.appendleft(dict(range=ep_range, avg=avg_score, max=max_score))
        self._rebuild()

    def _rebuild(self):
        for w in self._list.winfo_children():
            w.destroy()
        if not self._gens:
            self._show_empty()
            return

        items = list(self._gens)
        for i, g in enumerate(items):
            latest = (i == 0)
            nxt = items[i+1] if i+1 < len(items) else None

            normal_fg = C["surface"] if not latest else blend(C["primary"], C["card"], 0.88)
            hover_fg = C["surface_hover"] if not latest else blend(C["primary"], C["card"], 0.78)
            bw = 1 if latest else 0
            bc = blend(C["primary"], C["card"], 0.5) if latest else C["card_border"]

            row = ctk.CTkFrame(self._list, fg_color=normal_fg, corner_radius=14,
                              border_width=bw, border_color=bc)
            row.pack(fill="x", pady=3)
            add_hover(row, normal_fg, hover_fg)

            ri = ctk.CTkFrame(row, fg_color="transparent")
            ri.pack(fill="x", padx=14, pady=10)

            ebg = blend(C["primary"], C["card"], 0.7) if latest else C["card"]
            ef = ctk.CTkFrame(ri, fg_color=ebg, corner_radius=10,
                              width=78, height=50)
            ef.pack(side="left", padx=(0, 12))
            ef.pack_propagate(False)
            ctk.CTkLabel(ef, text=f"{g['range']}", font=("JetBrains Mono", 16, "bold"),
                        text_color=C["primary"] if latest else C["text_muted"]
                        ).pack(expand=True)

            det = ctk.CTkFrame(ri, fg_color="transparent")
            det.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(det, text=f"Avg {g['avg']:,} | Max {g['max']:,}",
                        font=("Inter", 36, "bold"), text_color=C["text"],
                        ).pack(anchor="w", pady=(2, 0))

            if nxt:
                diff = g['avg'] - nxt['avg']
                if diff > 0:
                    txt, col = f"\u2191 +{diff:,}", C["green"]
                elif diff < 0:
                    txt, col = f"\u2193 {diff:,}", C["red"]
                else:
                    txt, col = "\u2014 0", C["text_muted"]
                ctk.CTkLabel(ri, text=txt, font=("Inter", 16, "bold"),
                            text_color=col).pack(side="right")

    def clear(self):
        self._gens.clear()
        self._show_empty()


# =====================================================================
#  Main Application
# =====================================================================

class TetrisAIApp(ctk.CTk):
    def toggle_pause_visualization(self):
        if self.is_visualizing:
            if not getattr(self, '_vis_paused', False):
                self.pause_visualization()
            else:
                self.resume_visualization()
    def pause_visualization(self):
        if self.is_visualizing and not getattr(self, '_vis_paused', False):
            self._vis_paused = True
            self.controls.set_paused(True)

    def resume_visualization(self):
        if self.is_visualizing and getattr(self, '_vis_paused', False):
            self._vis_paused = False
            self.controls.set_paused(False)

    def __init__(self):
        super().__init__()
        self.title("Tetris AI \u2014 Neural Network Training Visualization")
        # Maximize window (keeps title bar with min/max/close buttons)
        self.after(10, lambda: self.state("zoomed"))
        self.minsize(1100, 700)
        self.configure(fg_color=C["bg"])
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # State
        self.is_learning = False
        self.is_visualizing = False
        self._stop_learn = threading.Event()
        self._stop_vis = threading.Event()
        self._learn_thread = None
        self._vis_thread = None
        self.current_episode = 0
        self.best_score = 0
        self._train_start_time = 0
        self._recent_batches = []

        self._build()
        self._load_existing_state()

    def _build(self):
        self.header = Header(self, self)
        self.header.pack(fill="x", padx=28, pady=(18, 10))

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=28, pady=(6, 18))
        main.grid_columnconfigure(0, weight=3, minsize=300)
        main.grid_columnconfigure(1, weight=2, minsize=280)  # Reduced from weight=4
        main.grid_columnconfigure(2, weight=5, minsize=400)  # Increased from weight=4
        main.grid_rowconfigure(0, weight=1)

        # Left column
        left = ctk.CTkScrollableFrame(main, fg_color="transparent",
                scrollbar_button_color=C["border"],
                scrollbar_button_hover_color=C["text_dim"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.controls = ControlPanel(left, self)
        self.controls.pack(fill="x", pady=(0, 10))
        self.train_status = TrainingStatusPanel(left)
        self.train_status.pack(fill="x")

        # Center column
        center = ctk.CTkFrame(main, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=10)
        center.grid_rowconfigure(0, weight=3) # Give more weight to board
        center.grid_rowconfigure(1, weight=1)

        self.board = TetrisBoardWidget(center)
        self.board.pack(pady=(0, 10), fill="both", expand=True)

        bot = ctk.CTkFrame(center, fg_color="transparent")
        bot.pack(fill="x")
        self.next_piece = NextPieceWidget(bot)
        self.next_piece.pack(side="left", padx=(0, 14))
        self.stats = StatsPanelWidget(bot)
        self.stats.pack(side="left")

        # Right column - 50/50 split for graph and recent episodes
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        right.grid_rowconfigure(0, weight=1)  # Chart takes 50%
        right.grid_rowconfigure(1, weight=1)  # Gen list takes 50%
        right.grid_columnconfigure(0, weight=1)

        self.chart = ScoreChartWidget(right)
        self.chart.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.gen_list = GenerationListWidget(right)
        self.gen_list.grid(row=1, column=0, sticky="nsew")

    # ── Load existing state ───────────────────────────────────────────────

    def _load_existing_state(self):
        saved = load_training_state()
        if saved:
            self.best_score = saved.get("best_score", 0)
            self.current_episode = saved.get("last_episode", 0)
            chart = saved.get("chart_data", {})
            if chart.get("eps"):
                self.chart.load_data(chart["eps"], chart["avgs"], chart["maxs"])
            gens = saved.get("recent_episodes", [])
            normalized = []
            for g in gens:
                if "range" in g:
                    normalized.append(dict(range=g["range"],
                                           avg=g.get("avg", 0),
                                           max=g.get("max", 0),
                                           steps=g.get("steps", 0)))
                else:
                    score = g.get("score", 0)
                    normalized.append(dict(range=f"{g['ep']}-{g['ep']}",
                                           avg=score, max=score, steps=0))

            self._recent_batches = normalized[-20:]
            # Add in reverse order so newest episodes appear first
            for g in reversed(self._recent_batches):
                self.gen_list.add(g["range"], g["avg"], g["max"])
        model_count = len(glob.glob(os.path.join(MODELS_DIR, "episode_*.keras")))
        self.header.update_stats(self.best_score, self.current_episode, model_count)

    # ── Training (runs independently of visualization) ────────────────────

    def toggle_learning(self):
        if self.is_learning:
            self._stop_learn.set()
            self.is_learning = False
            self.controls.set_learning(False, self.current_episode)
            self.train_status.update(False, self.current_episode)
        else:
            self._start_learning()

    def _start_learning(self):
        params = self.controls.get_params()
        if params is None:
            messagebox.showerror("Invalid Parameters",
                                 "Enter valid numeric hyperparameters.")
            return

        # Save parameters before starting training
        self.controls.save_params()

        self.is_learning = True
        self._stop_learn.clear()
        self.controls.set_learning(True, 0)

        # NOTE: We do NOT stop visualization — they run simultaneously
        self._learn_thread = threading.Thread(target=self._learn_loop,
                                              args=(params,), daemon=True)
        self._learn_thread.start()

    def _ui_interval(self, episode_num):
        if episode_num <= 300:
            return 1.0
        if episode_num <= 800:
            return 2.0
        if episode_num <= 1300:
            return 4.0
        if episode_num <= 2000:
            return 5.0
        return 7.0

    def _learn_loop(self, params):
        env = Tetris()
        total_episodes = params["episodes"]
        eps_stop = params["epsilon_stop_episode"]
        mem_size = params["mem_size"]
        discount = params["discount"]
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        train_every = params["train_every"]
        piece_limit = params.get("piece_limit", 0)  # 0 means no limit
        log_every = 10
        save_model_every = 50  # Save .keras file every N episodes (not every one)
        n_neurons = [32, 32, 32]
        activations = ['relu', 'relu', 'relu', 'linear']
        replay_start_size = min(mem_size, 1000)  # Original: 1000

        # Resume from last saved episode
        last_episode, last_saved_model = get_last_saved_episode()
        # Use last_episode to determine where to start, last_saved_model to load the model
        start_ep = last_episode
        last_model = os.path.join(MODELS_DIR, f"episode_{last_saved_model}.keras") if last_saved_model > 0 else None

        if last_saved_model > 0 and last_model and os.path.exists(last_model):
            agent = DQNAgent(env.get_state_size(), modelFile=last_model,
                             epsilon_stop_episode=eps_stop, mem_size=mem_size,
                             discount=discount, replay_start_size=replay_start_size)
            if eps_stop > 0 and start_ep < eps_stop:
                agent.epsilon = max(agent.epsilon_min,
                                   1.0 - (1.0 - agent.epsilon_min) * start_ep / eps_stop)
            elif start_ep >= eps_stop:
                agent.epsilon = agent.epsilon_min
        else:
            agent = DQNAgent(env.get_state_size(),
                             n_neurons=n_neurons, activations=activations,
                             epsilon_stop_episode=eps_stop, mem_size=mem_size,
                             discount=discount, replay_start_size=replay_start_size)
            start_ep = 0

        if start_ep >= total_episodes:
            self.after(0, lambda: messagebox.showinfo("Training Complete",
                f"All {total_episodes} episodes already completed.\n"
                f"Increase episode count or reset to retrain."))
            self.after(0, self._training_done)
            return

        scores = deque(maxlen=50) # Store last 50 scores for max/avg windows
        self.scores = scores
        chart_eps = list(self.chart.eps)
        chart_avgs = list(self.chart.avgs)
        chart_maxs = list(self.chart.maxs)
        batch_scores = []
        batch_steps = []
        batch_window = 50  # For both graph and recent panel
        recent_batches = list(getattr(self, "_recent_batches", []))
        model_count = len(glob.glob(os.path.join(MODELS_DIR, "episode_*.keras")))
        self._train_start_time = time.time()
        last_ui_update = 0  # Throttle UI updates
        last_best_save = 0  # Track when we last saved best model to reduce disk I/O
        
        # Consistency tracking for max score achievement
        max_score_achievements = deque(maxlen=10)  # Track last 10 episodes that reached max score
        consistency_threshold = 7  # Need 7 out of 10 to be considered consistent

        for episode in range(start_ep, total_episodes):
            if self._stop_learn.is_set():
                break

            ep_num = episode + 1
            self.current_episode = ep_num
            current_state = env.reset()
            done = False
            steps = 0

            # Game loop — NO board rendering, NO sleep during training
            while not done:
                if self._stop_learn.is_set():
                    break
                # Check if score reaches or exceeds max score
                if piece_limit > 0 and env.get_game_score() >= piece_limit:
                    done = True
                    break
                nxt = {tuple(v): k for k, v in env.get_next_states().items()}
                best = agent.best_state(nxt.keys())
                act = nxt[best]
                reward, done = env.play(act[0], act[1], render=False, piece_limit=piece_limit)
                agent.add_to_memory(current_state, best, reward, done)
                current_state = best
                steps += 1

            if self._stop_learn.is_set():
                # Save current episode for proper resume
                self.current_episode = ep_num
                # Only save if it's a multiple of 50 or 1
                if ep_num == 1 or ep_num % save_model_every == 0:
                    agent.save_model(os.path.join(MODELS_DIR, f"episode_{ep_num}.keras"))
                    model_count += 1
                break

            game_score = env.get_game_score()
            scores.append(game_score)
            batch_scores.append(game_score)
            batch_steps.append(steps)
            
            # Track consistency for max score achievement
            if piece_limit > 0 and game_score >= piece_limit:
                max_score_achievements.append(1)
                # Check if consistently reaching max score
                if len(max_score_achievements) == 10 and sum(max_score_achievements) >= consistency_threshold:
                    # Save consistency model
                    consistency_model_path = os.path.join(MODELS_DIR, f"consistent_ep_{ep_num}.keras")
                    agent.save_model(consistency_model_path)
                    print(f"[INFO] Consistency achieved! Saved model at episode {ep_num}")
                    # Also update best model if this score is higher
                    if game_score > self.best_score:
                        agent.save_model(os.path.join(_SCRIPT_DIR, "best.keras"))
            else:
                max_score_achievements.append(0)

            if ep_num == 1 or ep_num % save_model_every == 0:
                agent.save_model(os.path.join(MODELS_DIR, f"episode_{ep_num}.keras"))
                model_count += 1

            if game_score > self.best_score:
                old_best = self.best_score
                self.best_score = game_score
                # Only save best model every 10 episodes or if it's a major improvement (>20% better)
                # This reduces disk I/O which can slow down training
                episodes_since_save = ep_num - last_best_save
                is_major_improvement = old_best > 0 and game_score > old_best * 1.2
                if episodes_since_save >= 10 or is_major_improvement or old_best == 0:
                    agent.save_model(os.path.join(MODELS_DIR, "best.keras"))
                    agent.save_model(os.path.join(_SCRIPT_DIR, "best.keras"))
                    last_best_save = ep_num
                    # Only save episode if it's a multiple of 50 or 1
                    if (ep_num == 1 or ep_num % save_model_every == 0) and not (ep_num == 1 and save_model_every != 1):
                        model_count += 1  # Already saved above

            # Batch update for recent panel and graph
            if ep_num % batch_window == 0:
                avg_batch = int(round(mean(batch_scores))) if batch_scores else 0
                max_batch = max(batch_scores) if batch_scores else 0
                avg_steps = mean(batch_steps) if batch_steps else 0
                start_ep = max(0, ep_num - batch_window)
                recent_batches.append({
                    "range": f"{start_ep}-{ep_num}",
                    "avg": avg_batch,
                    "max": max_batch,
                    "steps": avg_steps
                })
                if len(recent_batches) > 10:
                    recent_batches = recent_batches[-10:]
                avg_50 = int(round(mean(scores))) if scores else 0
                max_50 = max(scores) if scores else 0
                chart_eps.append(ep_num)
                chart_avgs.append(avg_50)
                chart_maxs.append(max_50)
                self.after(0, self.chart.add_data_point, ep_num, avg_50, max_50)
                batch_scores.clear()
                batch_steps.clear()

            # Train the neural network
            if episode % train_every == 0:
                agent.train(batch_size=batch_size, epochs=epochs)

            now = time.time()
            interval = self._ui_interval(ep_num)
            if now - last_ui_update >= interval:
                last_ui_update = now
                elapsed = now - self._train_start_time
                epsilon_val = getattr(agent, 'epsilon', 0)
                avg_50 = int(round(mean(scores))) if scores else 0
                _ep, _mc, _bs = ep_num, model_count, self.best_score
                # Show last 5 batches in recent panel
                self.after(0, self._ui_batch_update, _ep, total_episodes,
                          elapsed, epsilon_val, _mc, _bs, recent_batches[-5:], avg_50)
                # Yield GIL very briefly so UI thread can process (reduced from 0.05)
                time.sleep(0.001)

            # Save training state periodically
            if ep_num % 50 == 0:
                save_training_state(dict(
                    last_episode=ep_num,
                    best_score=self.best_score,
                    chart_data=dict(eps=chart_eps, avgs=chart_avgs, maxs=chart_maxs),
                    recent_episodes=recent_batches[-20:],
                ))

        # Save final state with current episode
        self.current_episode = self.current_episode if self._stop_learn.is_set() else ep_num
        save_training_state(dict(
            last_episode=self.current_episode,
            best_score=self.best_score,
            chart_data=dict(eps=chart_eps, avgs=chart_avgs, maxs=chart_maxs),
            recent_episodes=recent_batches[-20:],
        ))
        self._recent_batches = recent_batches[-20:]
        self.after(0, self._training_done)

    def _ui_batch_update(self, ep, total, elapsed, epsilon, model_count, best_score, recent, avg_50):
        """Single batched UI update — interval scales with episode count."""
        self.controls.set_learning(True, ep)
        self.train_status.update(True, ep, total, elapsed, epsilon, avg_50)
        self.header.update_stats(best_score, ep, model_count)
        if recent:
            self.gen_list.clear()
            for g in recent:
                self.gen_list.add(g["range"], g["avg"], g["max"])

    def _training_done(self):
        self.is_learning = False
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        model_count = len([f for f in glob.glob(os.path.join(MODELS_DIR, "episode_*.keras")) if os.path.basename(f) == "episode_1.keras" or int(os.path.basename(f).replace("episode_", "").replace(".keras", "")) % 50 == 0])
        avg_50 = int(round(mean(self.scores))) if getattr(self, "scores", None) else 0
        self.header.update_stats(self.best_score, self.current_episode, model_count)
        self.controls.set_learning(False, self.current_episode)
        self.train_status.update(False, self.current_episode, elapsed=elapsed, avg_50=avg_50)

    # ── Visualization (runs independently of training) ────────────────────

    def toggle_visualization(self):
        if self.is_visualizing:
            self._stop_vis.set()
            self.is_visualizing = False
            self.controls.set_visualizing(False)
            # Do NOT clear the board or stats here; leave as is until new visualization
        else:
            self._start_vis()

    def _start_vis(self):
        ep_num = self.controls.get_vis_episode()
        if ep_num is None:
            messagebox.showerror("No Episode Entered",
                "Please enter an episode number to visualize.")
            return

        if ep_num is not None and ep_num > 0 and not (ep_num == 1 or ep_num % 50 == 0):
            messagebox.showinfo("Invalid Episode",
                "Please enter episode 1 or a multiple of 50.")
            return

        if ep_num == -1:
            model_path = os.path.join(MODELS_DIR, "best.keras")
        else:
            model_path = os.path.join(MODELS_DIR, f"episode_{ep_num}.keras")

        if not os.path.exists(model_path):
            root_path = f"episode_{ep_num}.keras" if ep_num > 0 else "best.keras"
            if os.path.exists(root_path):
                model_path = root_path
            else:
                messagebox.showerror("Model Not Found",
                    f"No model found for episode #{ep_num}.\n"
                    f"Looked for: {model_path}")
                return

        # NOTE: We do NOT stop training — they run simultaneously
        self.is_visualizing = True
        self._stop_vis.clear()
        self.controls.set_visualizing(True, ep_num if ep_num > 0 else 0)

        self._vis_thread = threading.Thread(target=self._vis_loop,
                                             args=(model_path, ep_num), daemon=True)
        self._vis_thread.start()

    def _vis_loop(self, model_path, ep_num):
        self._vis_paused = False
        try:
            env = Tetris()
            agent = DQNAgent(env.get_state_size(), modelFile=model_path)

            label = f"Ep #{ep_num}" if ep_num > 0 else "Best Model"
            
            # Track visualization-specific stats
            vis_best_score = 0
            frame_count = 0
            update_every = 1  # Update UI every frame at 5fps

            while not self._stop_vis.is_set():
                env.reset()
                done = False
                steps = 0
                game_start = time.time()

                while not done and not self._stop_vis.is_set():
                    # Pause logic
                    while getattr(self, '_vis_paused', False) and not self._stop_vis.is_set():
                        time.sleep(0.05)
                    try:
                        # Get next states - if empty, game is over
                        next_states = env.get_next_states()
                        if not next_states:
                            print("[INFO] Visualization: No valid moves available, game over.")
                            done = True
                            break

                        nxt = {tuple(v): k for k, v in next_states.items()}
                        best = agent.best_state(nxt.keys())
                        act = nxt.get(best)
                        
                        # Validate action
                        if not act or len(act) != 2:
                            print("[WARN] Visualization: Invalid action, ending episode.")
                            done = True
                            break

                        # Place piece directly without animation (5fps = 0.2s per piece)
                        env.current_pos = [act[0], 0]
                        env.current_rotation = act[1]
                        piece = env._get_rotated_piece()
                        
                        # Validate piece
                        if not piece or len(piece) == 0:
                            print("[WARN] Visualization: Invalid piece, ending episode.")
                            done = True
                            break

                        # Drop piece to final position
                        drop_count = 0
                        while not env._check_collision(piece, env.current_pos):
                            env.current_pos[1] += 1
                            drop_count += 1
                            if drop_count > env.BOARD_HEIGHT * 2:  # Safety limit
                                print("[ERROR] Infinite drop detected, breaking")
                                done = True
                                break
                        
                        if done:
                            break
                            
                        env.current_pos[1] -= 1

                        if self._stop_vis.is_set():
                            break

                        # Safety: if after drop, piece is still out of bounds, skip this step
                        if env.current_pos[1] < 0 or env.current_pos[1] > env.BOARD_HEIGHT:
                            print(f"[WARN] Visualization: piece landed out of bounds at y={env.current_pos[1]}, ending.")
                            done = True
                            break

                        # Place piece on board
                        env.board = env._add_piece_to_board(env._get_rotated_piece(), env.current_pos)
                        if env.board is None:
                            print("[ERROR] Board became None after adding piece")
                            done = True
                            break
                            
                        lc, env.board = env._clear_lines(env.board)
                        sc = 1 + (lc ** 2) * Tetris.BOARD_WIDTH
                        env.score += sc
                        env._new_round()
                        done = env.game_over
                        steps += 1
                        
                        # Update visualization best score
                        if env.score > vis_best_score:
                            vis_best_score = env.score

                        # Update UI at 5fps with frame skipping for efficiency
                        frame_count += 1
                        if frame_count % update_every == 0:
                            elapsed = time.time() - game_start
                            pps = steps / elapsed if elapsed > 0 else 0
                            
                            # Use update_idletasks to prevent UI queue overflow
                            try:
                                self.after(0, self._ui_vis_board, env, label, True)
                                self.after(0, self.stats.set, env.get_game_score(),
                                          vis_best_score, steps,
                                          ep_num if ep_num > 0 else 0, pps)
                                self.after(0, self.next_piece.set, env.next_piece)
                            except Exception as ui_err:
                                print(f"[ERROR] UI update failed: {ui_err}")

                        time.sleep(0.2)  # 5fps
                        
                    except Exception as e:
                        print(f"[ERROR] Visualization game loop error at step {steps}: {e}")
                        import traceback
                        traceback.print_exc()
                        done = True
                        break

                if not self._stop_vis.is_set():
                    # Final update with episode-specific stats
                    try:
                        self.after(0, self.stats.set, env.get_game_score(),
                                  vis_best_score, steps,
                                  ep_num if ep_num > 0 else 0, 0)
                    except Exception as e:
                        print(f"[ERROR] Final stats update failed: {e}")
                    time.sleep(2.0)

        except Exception as e:
            print(f"[FATAL] Visualization loop crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.after(0, self._vis_done)

    def _ui_vis_board(self, env, label, active):
        """Update board — only called from visualization."""
        try:
            if env is None or env.board is None:
                print("[WARN] Skipping board update - env or board is None")
                return
            self.board.update_board(env, label, active)
            if hasattr(env, 'next_piece') and env.next_piece is not None:
                self.next_piece.set(env.next_piece)
        except Exception as e:
            print(f"[ERROR] _ui_vis_board failed: {e}")
            import traceback
            traceback.print_exc()

    def _vis_done(self):
        self.is_visualizing = False
        self.controls.set_visualizing(False)
        # Do NOT clear the board here; leave as is until new visualization

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset_all(self):
        if self.is_learning or self.is_visualizing:
            messagebox.showwarning("Active Process",
                "Stop training and visualization before resetting.")
            return

        confirm = messagebox.askyesno("Reset All Data",
            "This will delete ALL saved models and training data.\n\n"
            "Are you sure?", icon="warning")
        if not confirm:
            return

        if os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
            os.makedirs(MODELS_DIR, exist_ok=True)

        root_best = os.path.join(_SCRIPT_DIR, "best.keras")
        if os.path.exists(root_best):
            os.remove(root_best)

        self.best_score = 0
        self.current_episode = 0

        # Reset parameters to defaults
        self.controls.reset_params()

        self.header.update_stats(0, 0, 0)
        self.chart.clear_plot()
        self.gen_list.clear()
        self.stats.set(0, 0, 0, 0)
        self.board.clear()
        self.next_piece.set(None)
        self.train_status.reset()
        self._train_start_time = 0
        self._recent_batches = []
 

# =====================================================================

if __name__ == "__main__":
    app = TetrisAIApp()
    app.mainloop()
