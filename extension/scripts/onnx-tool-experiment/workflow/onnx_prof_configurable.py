from typing import List, Dict, Tuple, Optional
import onnx_tool
import numpy
from onnx_tool.tensor import Tensor
from onnxsim import simplify
import onnx
import os
import sys

def detect_dynamic_shapes(onnx_model) -> Dict[str, bool]:
    """
    Detect which inputs have dynamic shapes and return info about them.
    """
    dynamic_info = {}
    
    for input_proto in onnx_model.graph.input:
        input_name = input_proto.name
        input_shape_dims = input_proto.type.tensor_type.shape.dim
        has_dynamic = any(d.dim_value <= 0 or d.dim_param for d in input_shape_dims)
        dynamic_info[input_name] = has_dynamic
        
        if has_dynamic:
            print(f"[INFO] Detected dynamic input '{input_name}' with shape: {[d.dim_value if d.dim_value > 0 else d.dim_param or 'dynamic' for d in input_shape_dims]}")
    
    return dynamic_info

def get_safe_input_shape(input_proto, default_batch_size: int = 1, default_seq_length: int = 128) -> Tuple:
    """
    Get a safe input shape for profiling, handling dynamic dimensions intelligently.
    """
    input_shape_dims = input_proto.type.tensor_type.shape.dim
    shape = []
    
    for i, d in enumerate(input_shape_dims):
        if d.dim_value > 0:
            # Fixed dimension
            shape.append(d.dim_value)
        elif d.dim_param:
            # Named dynamic dimension
            dim_name = d.dim_param.lower()
            if 'batch' in dim_name:
                shape.append(default_batch_size)
            elif 'seq' in dim_name or 'length' in dim_name:
                shape.append(default_seq_length)
            elif 'time' in dim_name:
                shape.append(default_seq_length)
            else:
                # Generic dynamic dimension
                if i == 0:  # First dimension often batch
                    shape.append(default_batch_size)
                elif i == 1:  # Second dimension often sequence length in NLP models
                    shape.append(default_seq_length)
                else:
                    shape.append(1)
        else:
            # Unnamed dynamic dimension (dim_value <= 0)
            if i == 0:  # First dimension often batch
                shape.append(default_batch_size)
            elif i == 1:  # Second dimension often sequence length in NLP models
                shape.append(default_seq_length)
            else:
                shape.append(1)
    
    return tuple(shape)

def try_multiple_shapes(model_path: str, input_name: str, input_proto) -> Optional[Tuple]:
    """
    Try multiple shape configurations to find one that works for profiling.
    Returns None if all attempts fail, indicating we should skip shape inference.
    """
    # Strategy 1: Try common reasonable shapes
    batch_sizes = [1, 2]
    seq_lengths = [1, 32, 128]
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            try:
                test_shape = get_safe_input_shape(input_proto, batch_size, seq_length)
                print(f"[INFO] Trying shape {test_shape} for input '{input_name}'")
                
                # Quick test with onnx_tool
                m = onnx_tool.Model(model_path)
                m.graph.shape_infer({input_name: numpy.zeros(test_shape)})
                
                print(f"[SUCCESS] Shape {test_shape} works for input '{input_name}'")
                return test_shape
                
            except Exception as e:
                print(f"[DEBUG] Shape {test_shape} failed: {str(e)[:100]}")
                continue
    
    # Strategy 2: Try skipping shape inference entirely
    try:
        print("[INFO] Trying to skip shape inference for dynamic model")
        m = onnx_tool.Model(model_path)
        m.graph.shape_infer(None)  # Skip shape inference
        print("[SUCCESS] Shape inference skipped successfully")
        return "SKIP_SHAPE_INFERENCE"  # Special return value
    except Exception as e:
        print(f"[DEBUG] Skipping shape inference failed: {str(e)[:100]}")
    
    return None

def profile_model(modelpath: str, results_base_dir: str = None, skip_simplification: bool = False, 
                 enable_dynamic_shape_handling: bool = True):
    """
    Profile an ONNX model using onnx_tool with enhanced dynamic shape handling
    
    Args:
        modelpath: Path to the ONNX model file
        results_base_dir: Base directory for saving results (optional)
        skip_simplification: Skip the internal simplification step if already simplified
        enable_dynamic_shape_handling: Enable intelligent dynamic shape handling
    """
    
    # Use intelligent default results directory if not provided
    if results_base_dir is None:
        # Try to use a directory relative to the model location first
        model_dir = os.path.dirname(os.path.abspath(modelpath))
        results_base_dir = os.path.join(model_dir, "onnx_analysis_results")
        
        # If that's not writable, fall back to temp directory
        try:
            os.makedirs(results_base_dir, exist_ok=True)
            # Test if we can write to this directory
            test_file = os.path.join(results_base_dir, ".qtron_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError):
            # Fall back to temp directory
            import tempfile
            results_base_dir = os.path.join(tempfile.gettempdir(), "qtron_onnx_analysis")
    
    results_dir = os.path.join(results_base_dir, os.path.basename(modelpath).replace('.onnx',''))
    results_dir = results_dir + "/"
    
    print(f"Results directory: {results_dir}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Profiling ONNX model: {modelpath}")
    onnx_model = onnx.load(modelpath)
    
    # Detect dynamic shapes
    dynamic_info = detect_dynamic_shapes(onnx_model)
    has_dynamic_inputs = any(dynamic_info.values())
    
    if has_dynamic_inputs and enable_dynamic_shape_handling:
        print("[INFO] Model has dynamic inputs, using enhanced shape handling")
    
    # Skip simplification if requested (model is already simplified)
    if not skip_simplification:
        try:
            onnx_model = simplify(onnx_model)[0]  # optional simplification step
            onnx.save(onnx_model, modelpath)  # overwrite with simplified model
        except Exception as e:
            print(f"[WARNING] Simplification failed: {e}")
            # Continue with original model
    
    # Enhanced input shape handling
    input_proto = onnx_model.graph.input[0]
    input_name = input_proto.name
    
    if has_dynamic_inputs and enable_dynamic_shape_handling:
        # Try multiple shape configurations
        shape_result = try_multiple_shapes(modelpath, input_name, input_proto)
        
        if shape_result == "SKIP_SHAPE_INFERENCE":
            # Skip shape inference entirely
            print("[INFO] Using model without shape inference for dynamic inputs")
            input_shape = None
        elif shape_result is None:
            # If all attempts failed, use basic fallback
            print("[WARNING] All dynamic shape attempts failed, using basic fallback")
            input_shape_dims = input_proto.type.tensor_type.shape.dim
            input_shape = tuple(d.dim_value if (d.dim_value > 0) else 1 for d in input_shape_dims)
        else:
            # Use the successful shape
            input_shape = shape_result
        
    else:
        # Original logic for static shapes or when dynamic handling is disabled
        input_shape_dims = input_proto.type.tensor_type.shape.dim
        input_shape = tuple(d.dim_value if (d.dim_value > 0) else 1 for d in input_shape_dims)
    
    if input_shape is None:
        print(f"[INFO] Profiling model without shape inference")
    else:
        print(f"[INFO] Using input shape {input_shape} for profiling")
    
    try:
        m = onnx_tool.Model(modelpath)
        
        # Apply shape inference based on our strategy
        if input_shape is None:
            # Skip shape inference for dynamic models
            m.graph.shape_infer(None)
        else:
            # Use concrete shape
            m.graph.shape_infer({input_name: numpy.zeros(input_shape)})
        
        m.graph.profile()
        
        # Save results
        txt_path = results_dir + os.path.basename(modelpath.replace('.onnx','.txt'))
        csv_path = results_dir + os.path.basename(modelpath.replace('.onnx','.csv'))
        shapes_path = results_dir + os.path.basename(modelpath.replace('.onnx','_shapes_only.onnx'))
        
        m.graph.print_node_map(txt_path)  # save file
        m.graph.print_node_map(csv_path)  # csv file
        m.save_model(shapes_path, shape_only=True)   # save model with updated shapes
        
        # Apply simplification on the _shapes_only.onnx model
        try:
            simplified_model = simplify(onnx.load(shapes_path))[0]
            onnx.save(simplified_model, shapes_path)
            print("[INFO] Shape-only model simplified successfully")
        except Exception as e:
            print(f"[WARNING] Shape-only model simplification failed: {e}")
        
        print(f"[SUCCESS] Profiling completed. Results saved to: {results_dir}")
        
    except Exception as e:
        print(f"[ERROR] Profiling failed with shape {input_shape}: {e}")
        raise  # Re-raise to trigger fallback in calling code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx_prof_configurable.py <model_path> [results_base_dir] [skip_simplification]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results_base_dir = sys.argv[2] if len(sys.argv) > 2 else None
    skip_simplification = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
    
    profile_model(model_path, results_base_dir, skip_simplification)
