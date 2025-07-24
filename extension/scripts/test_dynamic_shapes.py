#!/usr/bin/env python3
"""
Test script to see if onnx_tool supports -1 for dynamic dimensions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'onnx-tool-experiment'))

try:
    import onnx_tool
    import numpy as np
    
    print("Testing onnx_tool with -1 for dynamic dimensions...")
    
    # Test with a simple model path (we'll pass a dummy path)
    # The key question is: does onnx_tool.Model().graph.shape_infer() accept -1?
    
    print("onnx_tool imported successfully")
    print("Testing if numpy.zeros((-1, 128)) works...")
    
    try:
        # This should fail because numpy can't create arrays with -1
        test_array = np.zeros((-1, 128))
        print("numpy.zeros with -1 works!")
    except Exception as e:
        print(f"numpy.zeros with -1 fails: {e}")
    
    print("Testing if numpy array shapes can be (-1, 128)...")
    try:
        # Alternative: use symbolic approach
        print("We need to check onnx_tool documentation for symbolic shape handling")
    except Exception as e:
        print(f"Error: {e}")
        
except ImportError as e:
    print(f"Failed to import onnx_tool: {e}")
