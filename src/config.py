"""
Configuration file for FTS-xRooFit-Demo
========================================
Modify these paths to match your local ROOT/xRooFit installation.
"""

import os
import sys
import platform

# Detect operating system
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ROOT Installation Paths (modify these for your system)
# -----------------------------------------------------

if IS_MACOS:
    # macOS with Homebrew
    ROOT_BASE = os.environ.get('ROOTSYS', '/opt/homebrew/opt/root')
    ROOT_LIB = os.path.join(ROOT_BASE, 'lib')
    ROOT_PYTHON = os.path.join(ROOT_BASE, 'lib', 'root')
    
elif IS_LINUX:
    # Linux (typical installation)
    ROOT_BASE = os.environ.get('ROOTSYS', '/usr/local/root')
    ROOT_LIB = os.path.join(ROOT_BASE, 'lib')
    ROOT_PYTHON = os.path.join(ROOT_BASE, 'lib')
    
else:
    # Windows or other OS - user must set ROOTSYS environment variable
    ROOT_BASE = os.environ.get('ROOTSYS', '')
    if not ROOT_BASE:
        print("Warning: ROOTSYS environment variable not set!")
        print("Please set ROOTSYS to your ROOT installation directory")
    ROOT_LIB = os.path.join(ROOT_BASE, 'lib') if ROOT_BASE else ''
    ROOT_PYTHON = os.path.join(ROOT_BASE, 'lib') if ROOT_BASE else ''

# xRooFit Custom Build (optional)
# --------------------------------
# If you have a custom xRooFit build, set the path here
# Otherwise, leave as None to use the standard ROOT version
XROOFIT_LIB = os.environ.get('XROOFIT_LIB', None)

# Python Path Configuration
# -------------------------
def setup_root_paths():
    """Add ROOT paths to Python sys.path"""
    paths_to_add = [ROOT_LIB, ROOT_PYTHON]
    
    for path in paths_to_add:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            
    # Set PYTHONPATH environment variable
    if ROOT_PYTHON and os.path.exists(ROOT_PYTHON):
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if ROOT_PYTHON not in current_pythonpath:
            os.environ['PYTHONPATH'] = f"{ROOT_PYTHON}:{current_pythonpath}"

# Validation
# ----------
def validate_root_installation():
    """Check if ROOT is properly installed and accessible"""
    try:
        import ROOT
        print(f"✓ ROOT found: {ROOT.gROOT.GetVersion()}")
        return True
    except ImportError:
        print("✗ ROOT not found. Please check your installation.")
        print(f"  Expected ROOT at: {ROOT_BASE}")
        print("  To fix:")
        print("  1. Install ROOT: https://root.cern/install/")
        print("  2. Set ROOTSYS environment variable")
        print("  3. Or modify config.py with correct paths")
        return False

# Helper functions for notebook compatibility
# ----------------------------------------

def get_root_paths():
    """Get ROOT library paths for manual sys.path setup"""
    return ROOT_LIB, ROOT_PYTHON

def get_pythonpath():
    """Get PYTHONPATH string for environment setup"""
    if ROOT_PYTHON and os.path.exists(ROOT_PYTHON):
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if ROOT_PYTHON not in current_pythonpath:
            return f"{ROOT_PYTHON}:{current_pythonpath}"
    return os.environ.get('PYTHONPATH', '')

def get_project_root():
    """Get the project root directory"""
    # This file is in src/, so project root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-setup when imported
if __name__ != "__main__":
    setup_root_paths()