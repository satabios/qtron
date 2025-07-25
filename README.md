# QTron - ONNX Viewer with Integrated Analysis

A lightweight VS Code extension for viewing and analyzing ONNX models with seamless onnx_tool integration.  
Inspired by [Netron](https://github.com/lutzroeder/netron) and [ONNX Viewer](https://github.com/lijian736/onnx_viewer) but optimized for autoaisys workflows.

Why was this developed?
 * Load Large models faster (we are impatient like that :P); Strip the weights.
 * Profile the ONNX model (lets make our lives simpler).
 * I would like to automate mundane tasks in my life. Thus automated the flow.

All this happens with just a click to view an ONNX file—yep, that’s what I’ve been dreaming about lately, and I finally cooked this extension!

NOTE: Loading a model for the first time might take a might more time, but post to which you can happily load the lightweight profiled onnx model.
```
📁 ONNX File →      🚀 strip_and_profile_model() →         👁️ Visualize
     ↓                            ↓                                 ↓
  Original              [Simplify + Analyze]                   Interactive
   Model                          ↓                              Viewer   
                       📦 Weights Stripped              [LightWeight Profiled Model]
                       Profiled for 
                       MAC, Memory, Params (.onnx)             
                       📄 Reports (.txt, .csv)                                                          

```

# Qtron in Action
![Qtron](images/qtron.gif)

<p align="center">
  <img src="images/weight_stripped_profiled.png" alt="QTron: Weight Stripped/Profiled" width="500"/>
</p>
<p align="center"><em>Weight Stripped/Profiled ONNX</em></p>

![Profiling Info](images/profiling.png)
*Layer Wise Profiling Data*

![Simplified View](images/simplified.png)
*Simplified ONNX*

## ✨ Key Features

**🎯 One-Click Analysis**: Simply open any `.onnx` file in VS Code
- Automatic model simplification for cleaner visualization
- Integrated onnx_tool profiling with detailed metrics

**📊 Comprehensive Analysis**:
- Layer-wise MAC (Multiply-Accumulate) operations counting
- Parameter count and memory size analysis
- Export to TXT and CSV formats for further processing
- Shape-only ONNX files for efficient analysis
- **NEW**: Advanced dynamic shape handling for models with variable inputs

**⚡ Smart Defaults**:
- Results saved next to your ONNX file automatically
- Graceful fallback when advanced tools unavailable
- Intelligent handling of dynamic batch sizes and sequence lengths
- Configurable through VS Code settings

## 🔧 Configuration

Access settings via VS Code preferences (`qtron.*`):

| Setting | Description | Default |
|---------|-------------|---------|
| `qtron.pythonPath` | Python interpreter path | `"python"` |
| `qtron.enableSimplification` | Enable ONNX simplification | `true` |
| `qtron.enableOnnxToolProfiling` | Enable onnx_tool profiling | `true` |
| `qtron.onnxToolResultsPath` | Analysis results directory | Auto-detected |
| `qtron.enableDynamicShapeHandling` | Smart handling of dynamic input shapes | `true` |

## 📁 Analysis Output

When onnx_tool profiling is enabled, you'll get organized results:

```
your_model_directory/
├── your_model.onnx                    # Original model
└── onnx_analysis_results/
    └── your_model/
        ├── your_model_profile.txt      # Human-readable summary
        ├── your_model_profile.csv      # Detailed metrics (Excel-ready)
        └── your_model_shapes_only.onnx # Shape-optimized model
```

## 🔄 Processing Workflow

QTron follows an optimized processing pipeline:

**Integrated Workflow (onnx_tool enabled):**
```
📁 ONNX File →      🚀 strip_and_profile_model()     → 👁️ Visualize
     ↓                        ↓                            ↓
  Original            [Simplify + Analyze]             Interactive
   Model                      ↓                           Viewer                              
                        📄 Reports (.txt, .csv)  [ LightWeight Profiled Model]
                        📦 Weights Stripped/
                        Profiled Model (.onnx)                                                                   

```

**Fallback Workflow (simplification only):**
```
📁 ONNX File → 🔧 Simplify → 👁️ Visualize
     ↓              ↓          ↓
  Original      Simplified  Interactive
   Model         Model      Viewer
```

**Optimization Benefits:**
- ✅ Single-step processing eliminates duplicate simplification
- ✅ Automatic fallback handling for maximum compatibility
- ✅ Preserves original files while generating analysis

## 📦 Installation

   **Build**: 
   ```bash
   git clone https://gitlab.qualcomm.com/sathyapr/qtron
   cd onnx_viewer
   ./regen.sh
   ```
   **Installation**
   ```bash
   # If you are feeling lazy download the .vsix I have genearted
   ./extension/qtron-*.vsix
   ```

## 🚀 Usage

1. Open any `.onnx` file in VS Code
2. QTron automatically processes the model:
   - Simplifies for better visualization
   - Runs comprehensive analysis (if available)
   - Opens interactive viewer
3. Find analysis results in `onnx_analysis_results/` directory

## 🔍 Advanced Features

For detailed information about the integration architecture and workflow optimization, see:
- [Processing Flow Diagram](extension/ONNX_PROCESSING_FLOW.md)
- [Architecture Overview](extension/ARCHITECTURE_DIAGRAM.md)
- [Dynamic Shape Handling](extension/DYNAMIC_SHAPE_IMPROVEMENTS.md)

## 💡 Contributing

Have suggestions or found a bug? We'd love to hear from you!
- Open an issue on GitHub
- Submit a pull request
- Try to fix it on your own or ping me on Team and buy me a coffee, we can debug together :P
