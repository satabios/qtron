# Dynamic Shape Handling Fix - Version 1.0.4

## Issue Identified
The dynamic shape handling feature wasn't working because the TypeScript extension wasn't properly passing all required arguments to the Python script. 

### Root Cause
In the original implementation, when `onnxToolResultsPath` was empty (default behavior), the argument wasn't being properly passed to the Python subprocess, causing the `enable_dynamic_shapes` parameter to be missing.

### Symptoms
- Dynamic shape handling appeared to be disabled even when enabled in settings
- Error messages like: "The input tensor [name]'s shape [batch_size] is not valid"
- onnx_tool analysis would fail immediately without trying dynamic shape strategies

## Fix Applied

### 1. TypeScript Argument Handling
**File**: `src/onnx_viewer.ts`

**Problem**: Empty `onnxToolResultsPath` was causing argument indexing issues
```typescript
// Before: Could result in missing arguments
scriptArgs.push(onnxToolResultsPath); // Could be empty string
scriptArgs.push(enableDynamicShapeHandling ? 'true' : 'false');
```

**Solution**: Always pass explicit argument values
```typescript
// After: Always pass valid arguments
const resultsDir = onnxToolResultsPath && onnxToolResultsPath.trim() ? onnxToolResultsPath : 'DEFAULT';
scriptArgs.push(resultsDir); // Always non-empty
scriptArgs.push(enableDynamicShapeHandling ? 'true' : 'false');
```

### 2. Python Argument Parsing
**File**: `scripts/simplify_onnx.py`

**Enhancement**: Better handling of "DEFAULT" placeholder
```python
# Handle both empty strings and DEFAULT placeholder
results_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] not in ['', 'DEFAULT'] else None
```

### 3. Enhanced Debugging
Added comprehensive argument logging to help diagnose future issues:
```typescript
outputChannel.appendLine(`[QTron] Arguments: [${scriptArgs.map(arg => `"${arg}"`).join(', ')}]`);
```

```python
print(f"[DEBUG] Arguments received: {sys.argv}")
print(f"[DEBUG] enable_profiling: {enable_profiling}")
print(f"[DEBUG] results_dir: {results_dir}")
print(f"[DEBUG] enable_dynamic_shapes: {enable_dynamic_shapes}")
```

## Verification

The fix was verified by testing argument parsing:
```bash
python scripts/simplify_onnx.py test.onnx output.onnx true DEFAULT true
```

**Output**:
```
[DEBUG] Arguments received: ['scripts/simplify_onnx.py', 'test.onnx', 'output.onnx', 'true', 'DEFAULT', 'true']
[DEBUG] enable_profiling: True
[DEBUG] results_dir: None
[DEBUG] enable_dynamic_shapes: True
Dynamic shape handling: enabled
```

## Expected Behavior After Fix

When processing models with dynamic shapes (like `state_encoder.onnx` with `[batch_size]` dimension):

1. **Detection**: Script will detect dynamic dimensions correctly
2. **Multi-Strategy**: Will try multiple shape combinations:
   - Strategy 1: Intelligent mapping (`batch_size` → 1, 2)
   - Strategy 2: Skip shape inference entirely  
   - Strategy 3: Basic fallback (all dynamic → 1)
3. **Logging**: Will show "Dynamic shape handling: enabled" in output
4. **Success**: Should achieve much higher success rates on dynamic models

## Deployment

- **Version**: 1.0.4
- **Package**: `qtron-1.0.4.vsix`
- **Compatibility**: Backwards compatible, no settings changes required
- **Default**: Dynamic shape handling remains enabled by default

## Next Steps

1. Install the updated extension (`qtron-1.0.4.vsix`)
2. Test with the problematic `state_encoder.onnx` model
3. Verify the debug output shows dynamic shape handling as enabled
4. Monitor success rates with dynamic models
