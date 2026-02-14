@echo off
echo ===================================
echo  Tetris AI - Building Executable
echo ===================================
echo.
echo This will take 5-15 minutes...
echo TensorFlow is a large library!
echo.

cd /d "%~dp0"

python -m PyInstaller ^
  --name TetrisAI ^
  --onefile ^
  --windowed ^
  --add-data "best.keras;." ^
  --add-data "models;models" ^
  --hidden-import tensorflow ^
  --hidden-import keras ^
  --hidden-import customtkinter ^
  --hidden-import matplotlib.backends.backend_tkagg ^
  --hidden-import PIL ^
  --hidden-import numpy ^
  --hidden-import cv2 ^
  --exclude-module torch ^
  --exclude-module scipy ^
  --exclude-module pandas ^
  --noconfirm ^
  --clean ^
  tetris_gui.py

echo.
echo ===================================
echo Build Complete!
echo.
echo Your executable is at:
echo   dist\TetrisAI.exe
echo.
echo File size: ~500-800 MB
echo ===================================
pause
