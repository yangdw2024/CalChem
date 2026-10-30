"""
CalChem GUI build script - create exe using PyInstaller
"""

import os
import sys
import shutil
import subprocess

APP_NAME = "CalChem"
MAIN_SCRIPT = "main.py"
DIST_DIR = "dist"
BUILD_DIR = "build"
CALCHEM_PATH = r"D:\YDW\Trae_Gaussian"
GUI_DIR = r"D:\YDW\CalChem\gui"

def clean_build():
    """Clean build directories."""
    for d in [DIST_DIR, BUILD_DIR]:
        path = os.path.join(GUI_DIR, d)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Cleaned: {path}/")
    
    spec_file = os.path.join(GUI_DIR, f"{APP_NAME}.spec")
    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"Cleaned: {spec_file}")

def build_exe():
    """Build exe using PyInstaller."""
    print("=" * 60)
    print("CalChem Windows Exe Builder")
    print("=" * 60)
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Change to GUI directory
    os.chdir(GUI_DIR)
    print(f"Working directory: {os.getcwd()}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--add-data", f"{CALCHEM_PATH};calchem",
        "--add-data", "styles;styles",
        "--hidden-import", "numpy",
        "--hidden-import", "scipy",
        "--hidden-import", "matplotlib",
        "--hidden-import", "PyQt6",
        "--exclude-module", "tkinter",
        MAIN_SCRIPT
    ]
    
    print("\nRunning build...")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        exe_path = os.path.join(GUI_DIR, DIST_DIR, f"{APP_NAME}.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"\nBuild successful!")
            print(f"Output: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
            return exe_path
    
    print("Build failed!")
    return None

if __name__ == "__main__":
    clean_build()
    build_exe()