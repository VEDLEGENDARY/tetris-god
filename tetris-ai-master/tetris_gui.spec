# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Tetris AI
Bundles all dependencies into a single-file executable
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the directory of this spec file
spec_root = os.path.abspath(SPECPATH)

# Collect all necessary data files
datas = []

# Include the best model if it exists
best_model = os.path.join(spec_root, 'best.keras')
if os.path.exists(best_model):
    datas.append((best_model, '.'))

# Include the models directory (with all episode models)
models_dir = os.path.join(spec_root, 'models')
if os.path.exists(models_dir):
    datas.append((models_dir, 'models'))

# Collect hidden imports for TensorFlow, Keras, and other dependencies
hiddenimports = [
    'tensorflow',
    'keras',
    'keras.models',
    'keras.layers',
    'h5py',
    'customtkinter',
    'PIL',
    'PIL._tkinter_finder',
    'matplotlib',
    'matplotlib.backends', 'matplotlib.backends.backend_tkagg',
    'numpy',
    'cv2',
]

# Binary exclusions to reduce size
excludes = [
    'scipy',
    'pandas',
    'jupyter',
    'notebook',
    'IPython',
    'pytest',
    'torch',  # Exclude PyTorch to reduce size
]

a = Analysis(
    ['tetris_gui.py'],
    pathex=[spec_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TetrisAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add an icon file here if you have one (e.g., 'icon.ico')
)
