# ComfyUI GLSL Nodes

import os
import sys
import json

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .src.shader.parser import parse_metadata
from .src.runtime.executor import GLSLRuntime
from .src.utils.paths import discover_shader_directories, find_glsl_files

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GLSLGPUProcessor": GLSLGPUProcessor,
    "GLSLShaderNode": GLSLShaderNode,
    "GLSLGPUDiagnostics": GLSLGPUDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLSLGPUProcessor": "GLSL GPU Processor",
    "GLSLShaderNode": "GLSL Shader",
    "GLSLGPUDiagnostics": "GLSL GPU Diagnostics",
}

# Global runtime instance
glsl_runtime = None

def get_glsl_runtime():
    global glsl_runtime
    if glsl_runtime is None:
        glsl_runtime = GLSLRuntime()
    return glsl_runtime


class GLSLGPUProcessor:
    """Production-oriented shader execution from file-based shaders."""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Discover available shaders
        shader_dirs = discover_shader_directories()
        shader_files = []
        for directory in shader_dirs:
            shader_files.extend(find_glsl_files(directory))
        
        # Normalize paths to relative names for UI selector
        shader_options = [os.path.relpath(f, os.path.join(os.path.dirname(__file__), 'shaders')) for f in shader_files]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "shader": (shader_options,),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "GLSL"

    def process(self, image, shader, enabled, mask=None):
        if not enabled:
            return (image,)
        
        # Get runtime
        runtime = get_glsl_runtime()
        
        # Load and compile shader from file
        shader_path = os.path.join(os.path.dirname(__file__), 'shaders', shader)
        with open(shader_path, 'r') as f:
            source = f.read()
            
        try:
            metadata = parse_metadata(source)
            # TODO: Dynamic inputs based on metadata would go here in a full implementation
            result = runtime.execute(source, metadata, image, mask)
            return (result,)
        except Exception as e:
            print(f"GLSL GPU Processor Error: {e}")
            return (image,)


class GLSLShaderNode:
    """Interactive GLSL authoring node."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shader_source": ("STRING", {"multiline": True, "default": ""}),
                "shader_mode": (["simple", "advanced"], {"default": "simple"}),
                "image": ("IMAGE",),
                "compile": ("BOOLEAN", {"default": False}),
                "save_shader": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "GLSL"

    def process(self, shader_source, shader_mode, image, compile, save_shader, mask=None):
        # Get runtime
        runtime = get_glsl_runtime()
        
        if compile:
            try:
                metadata = parse_metadata(shader_source)
                result = runtime.execute(shader_source, metadata, image, mask, mode=shader_mode)
                return (result,)
            except Exception as e:
                print(f"GLSL Shader Compile Error: {e}")
                # Return original image on error
                return (image,)
        
        # Default behavior if not compiling
        return (image,)


class GLSLGPUDiagnostics:
    """Reports system diagnostics for GLSL GPU backend."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "report"
    CATEGORY = "GLSL"

    def report(self):
        runtime = get_glsl_runtime()
        try:
            diagnostics = runtime.get_diagnostics()
            return (json.dumps(diagnostics, indent=2),)
        except Exception as e:
            return (f"Error: {e}",)