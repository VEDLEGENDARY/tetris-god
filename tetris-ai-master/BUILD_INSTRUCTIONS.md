# Building the Executable

## Quick Build Instructions

### Option 1: Use the Build Script (Easiest)

1. **Double-click** `build_exe.bat` in the `tetris-ai-master` folder
2. **Wait 5-15 minutes** - TensorFlow is a very large library!
3. **Find your executable** at: `dist/TetrisAI.exe`

### Option 2: Manual Build

Open PowerShell or Command Prompt in the `tetris-ai-master` folder and run:

```powershell
python -m PyInstaller --name TetrisAI --onefile --windowed --add-data "best.keras;." --add-data "models;models" --hidden-import tensorflow --hidden-import keras --hidden-import customtkinter --hidden-import matplotlib.backends.backend_tkagg --exclude-module torch --exclude-module scipy --exclude-module pandas --noconfirm --clean tetris_gui.py
```

## Build Details

- **Build Time**: 5-15 minutes (TensorFlow dependency analysis is slow)
- **Output**: `dist/TetrisAI.exe` 
- **Size**: ~500-800 MB (all dependencies bundled)
- **Standalone**: No Python or libraries needed to run

## Distribution

Once built:
- Upload `TetrisAI.exe` to GitHub Releases
- Users can download and run it directly
- No installation required!
- The exe will create a `models/` folder to save training data

## Troubleshooting

- **Build taking too long?** This is normal! TensorFlow has thousands of files.
- **Out of space?** You need ~2-3 GB free during the build process.
- **Antivirus blocking?** Temporarily disable it during build.
- **Windows Defender?** It may scan the exe on first run (normal).
