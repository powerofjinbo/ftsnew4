#!/usr/bin/env python3
"""
Environment Setup Helper for FTS-xRooFit-Demo
==============================================
Run this script to check your environment and set up paths.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    print("✓ Python version OK")
    return True

def check_python_packages():
    """Check required Python packages"""
    required = ['numpy', 'matplotlib', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing")
            missing.append(package)
    
    if missing:
        print(f"\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def check_root_installation():
    """Check ROOT installation"""
    # Try importing config which sets up paths
    try:
        import config
        config.setup_root_paths()
        import ROOT
        print(f"✓ ROOT {ROOT.gROOT.GetVersion()} found")
        
        # Check for xRooFit
        try:
            ws = ROOT.xRooNode("RooWorkspace", "test", "test")
            print("✓ xRooFit available")
            return True
        except:
            print("⚠️  xRooFit not found (optional for standalone version)")
            return True
            
    except ImportError:
        print("✗ ROOT not found")
        print("\nROOT is optional but required for full functionality.")
        print("To install ROOT:")
        print("  macOS: brew install root")
        print("  Linux: See https://root.cern/install/")
        return False

def check_jupyter():
    """Check Jupyter installation"""
    try:
        import jupyter
        import notebook
        print("✓ Jupyter notebook installed")
        return True
    except ImportError:
        print("⚠️  Jupyter not found (optional)")
        print("  To install: pip install jupyter notebook")
        return False

def main():
    print("=" * 60)
    print("FTS-xRooFit-Demo Environment Check")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_python_packages),
        ("ROOT Installation", check_root_installation),
        ("Jupyter Notebook", check_jupyter),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        print("-" * 40)
        results.append(check_func())
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all(results[:2]):  # First two are required
        print("✓ Core requirements satisfied")
        print("\nYou can run the standalone version:")
        print("  python FTS_Academic_Implementation.py")
        
        if results[2]:  # ROOT
            print("\nYou can also run the full xRooFit integration:")
            print("  jupyter notebook FTS_plus.ipynb")
    else:
        print("✗ Some requirements missing")
        print("Please install missing components and try again.")
    
    print("\nFor detailed setup instructions, see README.md")

if __name__ == "__main__":
    main()