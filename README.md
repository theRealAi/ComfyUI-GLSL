# ComfyUI GLSL GPU Processor

A production-ready ComfyUI custom node package providing a GPU-native GLSL compute environment for image processing.

## Overview

This package provides two primary nodes:

### Node 1 — GLSL GPU Processor
Production-oriented shader execution from `.glsl` files.
- Loads `.glsl` shaders from a shader library.
- Supports reusable/versionable shaders.
- Automatically exposes shader parameters.
- Uses shader compilation and pipeline caching.
- Designed for stable production workflows.

### Node 2 — GLSL Shader
Interactive GLSL authoring and experimentation.
- Allows users to write GLSL directly inside the ComfyUI node.
- Provides an embedded shader editor.
- Supports inline shader compilation.
- Displays compiler errors.
- Automatically exposes shader parameters.
- Supports simple and advanced shader modes.
- Can optionally save an inline shader into the shader library.

Both nodes share the same underlying GLSL runtime.

## Architecture

```text
                         ComfyUI
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     GLSL GPU Processor             GLSL Shader
      File-based shader             Inline shader
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  GLSL Runtime   │
                   ├─────────────────┤
                   │ Shader Parser   │
                   │ GLSL Compiler   │
                   │ SPIR-V Cache    │
                   │ Reflection      │
                   │ Validation      │
                   │ Uniform System  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Vulkan Backend  │
                   ├─────────────────┤
                   │ GPU Context     │
                   │ Resources       │
                   │ Synchronization │
                   │ Interop         │
                   └────────┬────────┘
                            │
                            ▼
                      NVIDIA GPU
```

The fundamental rule is:
> **The ComfyUI nodes are UI/orchestration layers. The GLSL runtime performs all shader execution.**

## Features

- ✅ Vulkan Compute + GLSL → SPIR-V backend
- ✅ File-based shader library (examples, production, user)
- ✅ Inline shader authoring with metadata parsing
- ✅ Simple Mode (auto-wrapper generation) and Advanced Mode
- ✅ Automatic uniform UI from metadata (`@uniform`)
- ✅ Shader caching and pipeline reuse
- ✅ Compile status display with error reporting
- ✅ Save inline shaders to library
- ✅ Diagnostics node for system info
- ✅ Zero-copy GPU interop (where supported)
- ✅ HDR image support (RGBA32F)
- ✅ Color management agnostic

## Installation

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd path/to/ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUI-GLSL.git
   ```

3. Install dependencies in your ComfyUI venv:
   ```bash
   pip install glsl-compiler pyvulkan
   ```

## Usage

### GLSL GPU Processor Node

Select a shader from the library, connect an image and optional mask, then enable or disable processing.

### GLSL Shader Node

Write GLSL in the inline editor with metadata comments:
```glsl
/*
@name Exposure
@description GPU exposure adjustment
@version 1.0.0

@input image IMAGE

@uniform exposure float 0.0 min=-10 max=10 step=0.01
*/
vec4 process(vec4 color, ivec2 pixel)
{
    color.rgb *= exp2(exposure);
    return color;
}
```

Then click "Compile" to test and "Save Shader" to persist to `shaders/user/`.

## Development

See [Production Vibe Coding Specification](comfyui-glsl.md) for full implementation details.

## License

MIT