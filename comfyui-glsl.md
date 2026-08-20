# ComfyUI GLSL GPU Processor

## Production Vibe Coding Specification — v1.1

---

# 1. Objective

Build a production-ready ComfyUI custom node package providing a GPU-native GLSL compute environment for image processing.

The system provides two primary nodes:

### Node 1 — GLSL GPU Processor

Production-oriented shader execution.

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

Both nodes must use the **same underlying GLSL runtime**.

The architecture must NOT duplicate shader execution logic between the two nodes.

---

# 2. Core Architecture

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

---

# 3. GPU Backend

Primary backend:

**Vulkan Compute + GLSL → SPIR-V**

The architecture must support future backends:

```text
VulkanBackend
CUDABackend
OpenGLBackend
DirectXBackend
MetalBackend
```

Only Vulkan is required for v1.

Do not implement CPU GLSL execution.

---

# 4. Package Structure

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-GLSL/
        │
        ├── __init__.py
        ├── nodes.py
        ├── README.md
        ├── LICENSE
        ├── requirements.txt
        ├── pyproject.toml
        │
        ├── shaders/
        │   ├── examples/
        │   │   ├── passthrough.glsl
        │   │   ├── invert.glsl
        │   │   ├── grayscale.glsl
        │   │   ├── exposure.glsl
        │   │   ├── reinhard.glsl
        │   │   └── chromatic_aberration.glsl
        │   │
        │   ├── production/
        │   │
        │   └── user/
        │
        ├── src/
        │   ├── __init__.py
        │   │
        │   ├── backend/
        │   │   ├── __init__.py
        │   │   ├── base.py
        │   │   └── vulkan.py
        │   │
        │   ├── shader/
        │   │   ├── __init__.py
        │   │   ├── parser.py
        │   │   ├── compiler.py
        │   │   ├── cache.py
        │   │   ├── reflection.py
        │   │   └── validator.py
        │   │
        │   ├── editor/
        │   │   ├── __init__.py
        │   │   ├── editor.py
        │   │   └── autocomplete.py
        │   │
        │   ├── gpu/
        │   │   ├── __init__.py
        │   │   ├── context.py
        │   │   ├── buffers.py
        │   │   ├── textures.py
        │   │   ├── synchronization.py
        │   │   └── interop.py
        │   │
        │   ├── uniforms/
        │   │   ├── __init__.py
        │   │   ├── types.py
        │   │   ├── parser.py
        │   │   └── uploader.py
        │   │
        │   ├── runtime/
        │   │   ├── __init__.py
        │   │   ├── executor.py
        │   │   └── batch.py
        │   │
        │   └── utils/
        │       ├── logging.py
        │       ├── paths.py
        │       └── diagnostics.py
        │
        ├── tests/
        │   ├── test_parser.py
        │   ├── test_compiler.py
        │   ├── test_uniforms.py
        │   ├── test_shader_validation.py
        │   ├── test_inline_shader.py
        │   └── test_runtime.py
        │
        └── docs/
            ├── shader_authoring.md
            ├── inline_shaders.md
            ├── architecture.md
            └── troubleshooting.md
```

---

# 5. Node 1 — GLSL GPU Processor

This node is intended for production shader libraries.

Display name:

```text
GLSL GPU Processor
```

Inputs:

```text
image              IMAGE
mask               MASK / optional
shader             shader selector
enabled            BOOLEAN
```

Dynamic inputs are generated from shader metadata.

Example:

```text
┌─────────────────────────────────────┐
│ GLSL GPU Processor                  │
├─────────────────────────────────────┤
│ Shader                              │
│ [ production/reinhard.glsl      ▼ ] │
│                                     │
│ Image        [ IMAGE ]              │
│ Mask         [ MASK ]               │
│                                     │
│ Exposure     [ 0.00 ]               │
│ White Point  [ 4.00 ]               │
│ Strength     [ 1.00 ]               │
│                                     │
│ Enabled      [ TRUE ]               │
│                                     │
│              IMAGE                  │
└─────────────────────────────────────┘
```

This node should NOT contain a GLSL editor.

---

# 6. Node 2 — GLSL Shader

Display name:

```text
GLSL Shader
```

This is the interactive shader authoring node.

Purpose:

- rapid shader development
- experimentation
- prototyping
- one-off processing
- custom client/project operations
- technical R&D
- creating shaders without touching the filesystem

Example:

```text
┌─────────────────────────────────────────────┐
│ GLSL Shader                                 │
├─────────────────────────────────────────────┤
│ Mode: [ Simple ▼ ]                          │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ vec4 process(vec4 color, ivec2 pixel)  │ │
│ │ {                                       │ │
│ │     color.rgb *= exposure;              │ │
│ │     return color;                       │ │
│ │ }                                       │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ [ Compile ] [ Save Shader ]                 │
│                                             │
│ Exposure       [ 1.00 ]                     │
│ Strength       [ 1.00 ]                     │
│                                             │
│ Status: ✓ Compiled                          │
│                                             │
│ Image        [ IMAGE ]                      │
│ Mask         [ MASK ]                       │
│                                             │
│                    IMAGE                    │
└─────────────────────────────────────────────┘
```

---

# 7. GLSL Shader Modes

The inline shader node must support two modes.

## 7.1 Simple Mode

The user writes only the processing function.

Example:

```glsl
vec4 process(vec4 color, ivec2 pixel)
{
    color.rgb *= 2.0;
    return color;
}
```

The runtime automatically generates the required compute shader wrapper.

The wrapper provides:

- input image
- output image
- resolution
- pixel coordinates
- uniforms
- mask
- helper functions

This is the preferred mode for most users.

---

# 8. Advanced Mode

Advanced users can provide a complete compute shader.

Example:

```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outputImage;

layout(push_constant) uniform Parameters
{
    vec2 resolution;
    float exposure;
} params;

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    if (pixel.x >= int(params.resolution.x) ||
        pixel.y >= int(params.resolution.y))
        return;

    vec4 color = imageLoad(inputImage, pixel);

    color.rgb *= exp2(params.exposure);

    imageStore(outputImage, pixel, color);
}
```

In Advanced Mode the runtime must NOT modify the user's shader code except where explicitly required by the shader interface contract.

---

# 9. Inline Shader Metadata

Simple Mode must support metadata.

Example:

```glsl
/*
@name Exposure
@description GPU exposure adjustment
@version 1.0.0

@input image IMAGE
@input mask MASK

@uniform exposure float 0.0 min=-10 max=10 step=0.01
@uniform strength float 1.0 min=0 max=1 step=0.01
*/
```

The same metadata parser used by file-based shaders must be used by inline shaders.

There must be only one metadata specification.

---

# 10. Automatic Uniform UI

For:

```glsl
@uniform exposure float 0.0 min=-10 max=10 step=0.01
```

create:

```text
Exposure [ 0.00 ]
```

For:

```glsl
@uniform strength float 1.0 min=0 max=1 step=0.01
```

create:

```text
Strength [ 1.00 ]
```

Support:

```text
float
int
uint
bool

vec2
vec3
vec4

ivec2
ivec3
ivec4

uvec2
uvec3
uvec4

mat3
mat4
```

---

# 11. Inline Shader Inputs

Support:

```glsl
@input image IMAGE
@input mask MASK
@input depth IMAGE
@input normal IMAGE
@input secondary IMAGE
```

The UI should automatically create the corresponding ComfyUI inputs.

Example:

```text
Image       IMAGE
Mask        MASK
Depth       IMAGE
Secondary   IMAGE
```

---

# 12. Compile Button

The inline node must provide:

```text
[ Compile ]
```

Compilation should:

```text
GLSL source
    ↓
metadata parse
    ↓
shader validation
    ↓
GLSL → SPIR-V
    ↓
SPIR-V validation
    ↓
pipeline creation
    ↓
cache
```

Compilation should occur only when necessary.

Do not recompile an unchanged shader on every workflow execution.

---

# 13. Compile Status

The node should maintain a visible compile status.

States:

```text
Not Compiled
Compiling...
Compiled
Compilation Failed
Validation Failed
Runtime Error
```

Example:

```text
Status: ✓ Compiled
```

or:

```text
Status: ✕ Compilation Failed
Line 42: undeclared identifier 'exposure'
```

---

# 14. Compiler Error Presentation

Never expose only a raw Python traceback.

Display:

```text
GLSL COMPILATION ERROR

Shader: Inline GLSL Shader

Line 42:
    color.rgb *= expoosure;

Error:
    undeclared identifier 'expoosure'

No GPU execution was performed.
```

The original compiler output should remain available in the ComfyUI console/log.

---

# 15. Automatic Compilation

The inline node may automatically compile when the workflow executes.

However:

- Explicit Compile should be available.
- Cached valid shaders should not compile again.
- Failed shaders should not repeatedly compile on every execution.
- Source changes must invalidate the cache.

---

# 16. Shader Save

The inline node must support:

```text
[ Save Shader ]
```

When clicked, prompt for:

```text
Shader Name
Category
```

Example:

```text
Name:
Reinhard Extended

Category:
user
```

Save as:

```text
shaders/user/Reinhard_Extended.glsl
```

The saved shader must include its metadata.

---

# 17. Save Workflow

The intended workflow is:

```text
Write GLSL
     ↓
Compile
     ↓
Test
     ↓
Tune parameters
     ↓
Save Shader
     ↓
Shader Library
     ↓
GLSL GPU Processor
```

This creates a natural transition from experimentation to production.

---

# 18. Shader Library Integration

After saving a shader:

```text
shaders/user/
```

must automatically become available to:

```text
GLSL GPU Processor
```

The user should not have to write Python code.

---

# 19. Shader Naming

Sanitize filenames.

Reject:

```text
../../shader
```

Do not allow:

```text
/
\
:
*
?
"
<
>
|
```

where inappropriate for the target filesystem.

Shader names must not permit arbitrary filesystem writes.

---

# 20. Inline Shader Persistence

The complete GLSL source must be serialized into the ComfyUI workflow when the node is saved.

The workflow must remain reproducible even if the original `.glsl` file does not exist.

This is critical.

The inline node should therefore contain:

```text
shader source
shader metadata
shader mode
shader version
```

inside the node's serialized configuration.

---

# 21. Production Reproducibility

For file-based shaders, store enough metadata to identify the exact shader used.

For example:

```text
shader:
    production/reinhard.glsl

shader_hash:
    SHA256...

shader_version:
    1.2.0
```

The workflow should be able to detect if the shader file changed since the workflow was created.

---

# 22. Shader Hashing

Hash:

```text
shader source
metadata
compiler configuration
```

Use SHA-256 or another cryptographically strong hash.

The hash is used for:

- cache lookup
- reproducibility
- change detection
- diagnostics

---

# 23. Shader Cache

Cache:

```text
GLSL
 ↓
SPIR-V
 ↓
VkShaderModule
 ↓
VkComputePipeline
```

The cache key should include:

```text
shader hash
compiler version
SPIR-V version
GPU/device
pipeline configuration
```

---

# 24. Shader Hot Reload

Support:

```text
COMFYUI_GLSL_DEV=1
```

In development mode:

- detect shader source changes
- invalidate shader cache
- recompile automatically
- refresh shader list
- preserve the node's selected shader

Production mode should avoid constant filesystem polling.

---

# 25. GLSL Runtime API

Create a shared runtime interface:

```python
class GLSLRuntime:

    def compile(self, source, metadata):
        ...

    def validate(self, shader):
        ...

    def execute(
        self,
        shader,
        inputs,
        uniforms,
        width,
        height,
        batch
    ):
        ...

    def save_shader(self, shader, path):
        ...
```

Both nodes must use this runtime.

---

# 26. Backend Interface

```python
class GPUBackend:

    def initialize(self):
        ...

    def compile_shader(self, source):
        ...

    def create_pipeline(self, shader):
        ...

    def create_texture(self, ...):
        ...

    def execute(self, pipeline, resources, uniforms):
        ...

    def synchronize(self):
        ...

    def shutdown(self):
        ...
```

The runtime must never directly depend on Vulkan implementation details.

---

# 27. Vulkan Backend

Implement:

```text
VulkanBackend
```

Responsibilities:

- Vulkan initialization
- GPU selection
- command queues
- command buffers
- descriptor sets
- compute pipelines
- images
- buffers
- synchronization
- memory
- CUDA/Vulkan interoperability

---

# 28. GPU Image Processing

Primary execution path:

```text
PyTorch CUDA IMAGE
        ↓
GPU Interop
        ↓
Vulkan Image
        ↓
GLSL Compute
        ↓
Vulkan Image
        ↓
GPU Interop
        ↓
PyTorch CUDA IMAGE
```

Avoid:

```text
GPU
 ↓
CPU
 ↓
NumPy
 ↓
GPU
```

for image processing.

---

# 29. Zero-Copy Requirement

The preferred implementation must use GPU-resident interoperability.

Do not claim zero-copy unless the implementation actually achieves it.

If the target environment cannot support true zero-copy interoperability, implement the fastest available GPU fallback and clearly identify the transfer path.

---

# 30. Image Format

Default:

```text
RGBA32F
```

Preserve floating-point data.

Do not automatically:

- clamp HDR values
- convert to 8-bit
- apply gamma
- apply sRGB
- apply tone mapping

---

# 31. Color Management

The GLSL runtime must be color-management agnostic.

No automatic:

```text
sRGB
Rec.709
ACES
gamma 2.2
tone mapping
```

The shader controls the mathematical transformation.

---

# 32. Standard Shader Interface

Provide:

```text
resolution
width
height
invResolution
frameIndex
batchIndex
time
```

where applicable.

Shaders must bounds-check their dispatch dimensions.

---

# 33. Compute Shader Default

Default workgroup:

```glsl
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
```

Allow this to be overridden by advanced shaders.

Do not assume image dimensions are divisible by 16.

---

# 34. Mask Behavior

Mask value:

```text
0.0
```

means original image.

Mask value:

```text
1.0
```

means fully processed image.

Conceptually:

```glsl
output = mix(original, processed, mask);
```

Mask dimensions must match the input image.

---

# 35. Batch Processing

Support:

```text
[B, H, W, C]
```

ComfyUI image batches.

Output batch size must match input batch size.

The first implementation may dispatch one image at a time if necessary for correctness.

Future versions may optimize using texture arrays or 3D dispatch.

---

# 36. Multiple Texture Inputs

Support multiple image resources.

Example:

```text
binding 0 → image
binding 1 → mask
binding 2 → depth
binding 3 → secondary
binding 4 → normal
```

The binding system must be centralized.

Do not hardcode bindings in individual shaders.

---

# 37. GPU Resource Management

Implement:

```text
TexturePool
BufferPool
PipelineCache
DescriptorPool
```

Reuse resources where possible.

Avoid allocation/destruction on every execution.

---

# 38. Synchronization

Use explicit Vulkan synchronization.

Do not use:

```python
time.sleep()
```

as a synchronization mechanism.

Correctly synchronize:

```text
PyTorch → Vulkan
Vulkan execution
Vulkan → PyTorch
```

---

# 39. Performance

Prioritize:

1. GPU residency
2. minimal synchronization
3. shader caching
4. pipeline caching
5. descriptor reuse
6. resource pooling
7. asynchronous execution

Do not process individual pixels in Python.

Do not use NumPy/PIL for the production image-processing path.

---

# 40. Diagnostics Node

Add:

```text
GLSL GPU Diagnostics
```

The node reports:

```text
Backend
Vulkan version
GPU
Driver
Compute support
GLSL compiler
SPIR-V
Interop
Supported formats
Cache
Shader directories
```

Example:

```text
GLSL GPU Diagnostics

Backend: Vulkan
GPU: NVIDIA RTX 5090
Vulkan: OK
GLSL Compiler: OK
SPIR-V: OK
Interop: OK
Cache: OK
```

---

# 41. Runtime Diagnostics

Optional debug logging:

```text
[GLSL]
[GLSL][VULKAN]
[GLSL][SHADER]
[GLSL][CACHE]
[GLSL][GPU]
```

Debug information may include:

```text
GPU
resolution
batch
dispatch dimensions
shader
shader hash
pipeline
execution time
memory
```

---

# 42. Performance Timing

Optional GPU timing:

```text
Shader: Reinhard Extended
Resolution: 3840x2160
GPU: RTX 5090

GPU execution: 0.42 ms
Transfer: 0.08 ms
Total: 0.51 ms
```

Do not introduce mandatory GPU synchronization solely for timing in production.

---

# 43. Security

Inline shaders are executable GPU programs.

Therefore:

- Do not execute Python from shader metadata.
- Do not download remote shaders automatically.
- Restrict file-based shader loading to configured directories.
- Sanitize save paths.
- Prevent directory traversal.
- Do not execute arbitrary filesystem commands.
- Treat metadata as data.

---

# 44. ComfyUI Workflow Serialization

The inline `GLSL Shader` node must serialize:

```text
source
mode
metadata
shader name
shader version
shader hash
```

The workflow must remain self-contained.

A user opening the workflow on another machine should still have access to the inline shader source.

---

# 45. File Shader Reproducibility

The production processor should serialize:

```text
shader path
shader hash
shader version
```

Optionally provide:

```text
Embed Shader
```

which stores the shader source directly in the workflow.

This allows a production workflow to be portable.

---

# 46. Simple Mode Runtime Wrapper

The runtime should internally generate something conceptually equivalent to:

```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f)
readonly uniform image2D inputImage;

layout(binding = 1, rgba32f)
writeonly uniform image2D outputImage;

layout(push_constant) uniform Parameters
{
    vec2 resolution;
    float time;
    int frameIndex;
} params;

vec4 process(vec4 color, ivec2 pixel);

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    if (
        pixel.x >= int(params.resolution.x) ||
        pixel.y >= int(params.resolution.y)
    )
        return;

    vec4 color = imageLoad(inputImage, pixel);

    color = process(color, pixel);

    imageStore(outputImage, pixel, color);
}
```

The actual implementation may use a more sophisticated generated interface.

---

# 47. Simple Mode Helper Functions

Provide optional helpers:

```glsl
vec2 uv(ivec2 pixel);
vec4 sampleInput(vec2 uv);
vec4 sampleMask(vec2 uv);
vec2 getResolution();
float getTime();
int getFrameIndex();
```

These helpers should be documented.

---

# 48. Advanced Mode

Advanced Mode should provide maximum control.

The user may define:

- bindings
- workgroup sizes
- buffers
- images
- push constants
- shared memory
- multiple resources

However, the runtime must validate the shader against supported interfaces.

---

# 49. Shader Validation

Validate:

- GLSL syntax
- SPIR-V validity
- supported GLSL version
- descriptor bindings
- image formats
- push constants
- workgroup dimensions
- metadata compatibility

Provide useful errors.

---

# 50. Shader Reflection

Use SPIR-V reflection where appropriate to verify:

```text
metadata
shader resources
bindings
uniforms
```

Detect mismatches.

Example:

```text
Shader metadata declares:

@uniform exposure float

but shader contains no corresponding parameter.

Result:

WARNING:
Uniform 'exposure' is declared in metadata but not consumed by the shader.
```

---

# 51. Example Inline Shader

Simple mode:

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

The node automatically exposes:

```text
Exposure [ 0.00 ]
```

---

# 52. Example Inline Reinhard

```glsl
/*
@name Reinhard Extended
@description Extended Reinhard tone mapping
@version 1.0.0

@input image IMAGE

@uniform exposure float 0.0 min=-10 max=10 step=0.01
@uniform whitePoint float 4.0 min=0.01 max=32 step=0.01
@uniform strength float 1.0 min=0 max=1 step=0.01
*/

vec4 process(vec4 color, ivec2 pixel)
{
    vec3 x = color.rgb * exp2(exposure);

    vec3 mapped =
        x * (1.0 + x / (whitePoint * whitePoint))
        / (1.0 + x);

    color.rgb = mix(color.rgb, mapped, strength);

    return color;
}
```

Preserve alpha.

Do not gamma-correct the result.

---

# 53. Example Production Shader

Include:

```text
shaders/examples/reinhard_extended.glsl
```

and equivalent inline example.

This ensures the two systems demonstrate identical shader functionality.

---

# 54. Shader Editor

The inline node should provide a proper code editor where the ComfyUI frontend allows it.

Desired features:

- GLSL syntax highlighting
- line numbers
- indentation
- bracket matching
- basic autocomplete
- compile action
- compiler error location
- copy/paste
- undo/redo

Do not build a full IDE.

The editor should remain lightweight.

---

# 55. Editor Implementation

Keep the editor frontend isolated from the Python runtime.

The backend should receive:

```text
shader source
```

and return:

```text
success
SPIR-V/cache identifier
compiler diagnostics
```

The frontend should not implement shader compilation logic.

---

# 56. Compile Workflow

```text
User edits shader
       │
       ▼
Compile
       │
       ▼
Parse metadata
       │
       ▼
Generate wrapper if Simple Mode
       │
       ▼
Validate GLSL
       │
       ▼
Compile to SPIR-V
       │
       ▼
Validate SPIR-V
       │
       ▼
Reflect resources
       │
       ▼
Create/cache pipeline
       │
       ▼
Status = READY
```

---

# 57. Runtime Workflow

```text
ComfyUI IMAGE
      │
      ▼
Validate input
      │
      ▼
Acquire GPU resource
      │
      ▼
Bind textures
      │
      ▼
Upload uniforms
      │
      ▼
Dispatch compute shader
      │
      ▼
Synchronize
      │
      ▼
Return IMAGE
```

---

# 58. Error States

Handle:

```text
ShaderParseError
ShaderCompileError
ShaderValidationError

VulkanInitializationError
VulkanDeviceError
VulkanAllocationError
VulkanExecutionError

InvalidImageError
InvalidMaskError
InvalidUniformError
InvalidShaderInterfaceError
```

Errors must be readable and actionable.

---

# 59. Graceful Startup

If Vulkan is unavailable:

**ComfyUI must still start.**

Do not initialize Vulkan during package import.

Initialize lazily on first execution or diagnostics request.

---

# 60. Configuration

Support:

```text
COMFYUI_GLSL_BACKEND=vulkan

COMFYUI_GLSL_GPU=0

COMFYUI_GLSL_GPU_NAME="RTX 5090"

COMFYUI_GLSL_SHADER_PATH=

COMFYUI_GLSL_CACHE_PATH=

COMFYUI_GLSL_DEV=0

COMFYUI_GLSL_DEBUG=0

COMFYUI_GLSL_FP16=0
```

Normal installation should require no configuration.

---

# 61. Testing

Test:

### Parser

- metadata
- inputs
- uniforms
- invalid metadata

### Inline shader

- simple mode
- advanced mode
- compilation
- compilation errors
- workflow serialization
- save shader
- reload shader

### Compiler

- valid shader
- invalid GLSL
- cache hit
- cache invalidation

### Runtime

- 512×512
- 1920×1080
- 4K
- odd dimensions
- batch > 1
- mask
- multiple textures

### GPU

- initialization
- device selection
- resource allocation
- synchronization
- cleanup

---

# 62. Stress Testing

Test:

```text
8K
large batches
multiple GLSL nodes
multiple shaders
repeated execution
shader hot reload
invalid shaders
GPU memory pressure
ComfyUI restart
workflow reload
inline shader serialization
```

The system must fail gracefully.

---

# 63. Performance Requirements

The production path must prioritize:

```text
GPU residency
pipeline reuse
shader caching
resource reuse
minimal synchronization
```

Prohibited as the normal processing path:

```text
Python pixel loops
NumPy image processing
PIL image processing
GPU → CPU → GPU image processing
```

unless required for a clearly documented fallback.

---

# 64. VFX Requirements

The system must be suitable for:

```text
EXR
HDR
linear float
4K
8K
plates
mattes
masks
depth
normal maps
motion vectors
AI-generated imagery
```

Never assume display-referred data.

Never automatically clamp HDR values.

---

# 65. Future Capabilities

The architecture should allow:

```text
3D textures
texture arrays
SSBO
UBO
atomic operations
shared memory
multi-pass shaders
ping-pong buffers
temporal processing
motion vectors
depth processing
```

Future functionality must not require rewriting the node architecture.

---

# 66. Future Shader Chain

Eventually support:

```text
GLSL Shader
     ↓
GLSL Shader
     ↓
GLSL Shader
```

while keeping intermediate resources on the GPU.

Potential future node:

```text
GLSL Shader Chain
```

Do not implement this unless it does not compromise the v1 architecture.

---

# 67. Future Shared Runtime

The shader runtime should remain independent enough that it can eventually be reused by:

```text
ComfyUI
Velaris
standalone VFX utilities
batch processing tools
real-time GPU applications
```

Long-term architecture:

```text
                    GLSL Library
                         │
                         ▼
                  Shader Compiler
                         │
                       SPIR-V
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          ComfyUI               Other Apps
              │                     │
              ▼                     ▼
           Vulkan                Vulkan
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                      GPU
```

---

# 68. Definition of Done

The project is complete when:

### Core

- ComfyUI starts normally.
- Nodes register correctly.
- Vulkan initializes.
- GPU is detected.
- Diagnostics work.

### GLSL GPU Processor

- File shaders load.
- Shader metadata works.
- Dynamic uniforms work.
- Masks work.
- Multiple textures work.
- Shader cache works.
- Pipeline cache works.

### GLSL Shader

- Inline editor works.
- Simple Mode works.
- Advanced Mode works.
- Compile button works.
- Compiler errors are displayed.
- Uniforms are automatically exposed.
- Shader source is serialized into workflows.
- Save Shader works.
- Saved shaders appear in GLSL GPU Processor.

### GPU

- 512×512 works.
- 1080p works.
- 4K works.
- 8K is tested.
- Batch processing works.
- Odd resolutions work.
- GPU resources are correctly released.

### Production

- No unnecessary CPU pixel processing.
- Shader caching works.
- Pipeline reuse works.
- Invalid shaders fail gracefully.
- Vulkan failure does not prevent ComfyUI startup.
- Documentation exists.
- Automated tests pass.

---

# 69. Implementation Order

## Phase 1 — Package

Create:

```text
package
node registration
shader discovery
metadata parser
```

Verify ComfyUI startup.

## Phase 2 — Vulkan

Implement:

```text
Vulkan initialization
GPU selection
compute queue
command infrastructure
```

## Phase 3 — Compiler

Implement:

```text
GLSL → SPIR-V
SPIR-V validation
shader cache
```

## Phase 4 — Basic Runtime

Implement:

```text
passthrough shader
GPU image
GPU output
```

## Phase 5 — GLSL GPU Processor

Implement file-based shaders.

## Phase 6 — GLSL Shader

Implement:

```text
inline editor
Simple Mode
Advanced Mode
Compile
errors
```

## Phase 7 — Dynamic Interface

Implement:

```text
uniform metadata
dynamic UI
texture inputs
mask
```

## Phase 8 — Save Shader

Implement:

```text
Save Shader
shader library integration
```

## Phase 9 — CUDA/Vulkan Interop

Implement the best available GPU-resident interoperability mechanism for the target environment.

## Phase 10 — Optimization

Implement:

```text
pipeline caching
resource pooling
descriptor reuse
GPU synchronization optimization
```

## Phase 11 — Production Hardening

Test:

```text
4K
8K
batch
multiple nodes
GPU memory pressure
invalid shaders
workflow serialization
shader hot reload
ComfyUI restart
```

---

# 70. Coding-Agent Instructions

Implement this as a **production software package, not a proof of concept**.

Before implementation:

1. Inspect the actual ComfyUI version/API available in the environment.
2. Inspect installed PyTorch and CUDA versions.
3. Inspect available Vulkan runtime.
4. Inspect available GLSL/SPIR-V compiler.
5. Determine the practical CUDA/Vulkan interoperability mechanism.
6. Do not invent APIs.
7. Keep Vulkan implementation isolated.
8. Keep the ComfyUI nodes thin.
9. Keep the shader compiler independent from the UI.
10. Add tests as functionality is implemented.
11. Document platform limitations.
12. Never claim zero-copy unless verified.

Do not silently substitute CPU processing.

If GPU interoperability requires a fallback, clearly identify:

```text
GPU-native path
```

versus:

```text
fallback transfer path
```

---

# 71. Critical Design Principle

The system is fundamentally:

```text
             SHADER AUTHORING
                    │
         ┌──────────┴──────────┐
         │                     │
     Inline GLSL          GLSL Library
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
             GLSL Runtime
                    │
              GLSL → SPIR-V
                    │
                    ▼
              Vulkan Compute
                    │
                    ▼
                  GPU
                    │
                    ▼
                 IMAGE
```

The **GLSL Shader node is the R&D environment**.

The **GLSL GPU Processor is the production environment**.

They must share the same runtime.

The intended workflow is:

```text
               EXPLORE
                  │
                  ▼
          GLSL Shader Node
                  │
                  ▼
              COMPILE
                  │
                  ▼
               TEST
                  │
                  ▼
               TUNE
                  │
                  ▼
            SAVE SHADER
                  │
                  ▼
             GLSL Library
                  │
                  ▼
             PRODUCTION
                  │
                  ▼
        GLSL GPU Processor
```

This separation is intentional: artists and technical directors can experiment directly inside ComfyUI, while approved shaders become stable, reusable, hashable production assets.