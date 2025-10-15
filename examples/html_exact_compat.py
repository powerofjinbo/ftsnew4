#!/usr/bin/env python3
"""
html_exact_compat.py - xRooFit Environment Compatibility Layer
=============================================================

This module provides compatibility with the xRooFit demonstration environment
as shown in the HTML demos. It sets up the necessary paths and loads the
xRooFit library for seamless integration.

Equivalent to the HTML demo environment setup.
"""

import os
import sys

def setup_xroofit_environment():
    """Set up xRooFit environment paths and library loading"""
    
    # Set up library paths for the built xRooFit
    xroofit_build_path = "/Users/victorzhang/Library/CloudStorage/Dropbox/FTS/xroofit_build"
    root_lib_path = "/opt/homebrew/Cellar/root/6.34.08_1/lib/root"
    
    # Update DYLD_LIBRARY_PATH for dynamic library loading
    current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
    if xroofit_build_path not in current_dyld:
        os.environ['DYLD_LIBRARY_PATH'] = f"{xroofit_build_path}:{current_dyld}"
    
    # Update LD_LIBRARY_PATH for compatibility
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if xroofit_build_path not in current_ld:
        os.environ['LD_LIBRARY_PATH'] = f"{xroofit_build_path}:{current_ld}"
    
    # Update PYTHONPATH for ROOT
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [root_lib_path, xroofit_build_path]
    for path in paths_to_add:
        if path not in current_pythonpath:
            os.environ['PYTHONPATH'] = f"{path}:{current_pythonpath}"
            current_pythonpath = os.environ['PYTHONPATH']
    
    # Add paths to Python sys.path
    for path in [root_lib_path, xroofit_build_path]:
        if path not in sys.path:
            sys.path.insert(0, path)

def load_xroofit_library():
    """Load the xRooFit library"""
    try:
        import ROOT
        
        # Load xRooFit library from our build
        xroofit_lib = "/Users/victorzhang/Library/CloudStorage/Dropbox/FTS/xroofit_build/libxRooFit.dylib"
        if os.path.exists(xroofit_lib):
            result = ROOT.gSystem.Load(xroofit_lib)
            if result == 0:
                print("✅ xRooFit library loaded successfully")
                print("🎯 xRooFit -- Create/Explore/Modify Workspaces")
                return True
            else:
                print(f"⚠️  xRooFit library load returned code {result}")
                return False
        else:
            print(f"❌ xRooFit library not found at: {xroofit_lib}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import ROOT: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading xRooFit: {e}")
        return False

# Auto-setup when module is imported
setup_xroofit_environment()
load_success = load_xroofit_library()

# Module metadata
__version__ = "1.0.0"
__author__ = "FTS Implementation Team"
__description__ = "xRooFit HTML demo environment compatibility layer"

# Export setup status
XROOFIT_AVAILABLE = load_success