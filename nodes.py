# ComfyUI GLSL Nodes

import json

from .src.shader.ui_schema import (
    SHADER_NODE_SLOT_INPUTS,
    SHADER_NODE_SLOT_UNIFORMS,
    collect_extra_image_inputs,
    collect_union_uniform_inputs,
    enrich_metadata,
    extract_uniform_values,
    list_shader_relpaths,
    load_shader_source,
)
from .src.runtime.executor import GLSLRuntime

glsl_runtime = None

DEFAULT_SIMPLE_SHADER = """/*
@name Inline
@description Inline GLSL shader
@version 1.0.0

@input image IMAGE
*/
vec4 process(vec4 color, ivec2 pixel)
{
    // Node slots: params.float_1 / float_2, int_1 / int_2, vec2_1 / vec2_2
    color.rgb = mix(color.rgb, 1.0 - color.rgb, clamp(params.float_1, 0.0, 1.0));
    return color;
}
"""

# Also accepted: GLES fragment shaders (#version 300 es + fragColor)


def get_glsl_runtime():
    global glsl_runtime
    if glsl_runtime is None:
        glsl_runtime = GLSLRuntime()
    return glsl_runtime


def _merge_shader_node_slots(metadata, kwargs):
    """Attach fixed authoring slots and collect their values."""
    by_name = {u["name"]: dict(u) for u in metadata.get("uniforms", [])}
    for slot in SHADER_NODE_SLOT_UNIFORMS:
        by_name.setdefault(slot["name"], dict(slot))
    metadata = dict(metadata)
    metadata["uniforms"] = list(by_name.values())
    values = extract_uniform_values(metadata, kwargs)
    return metadata, values


class GLSLGPUProcessor:
    """Production-oriented shader execution from file-based shaders."""

    @classmethod
    def INPUT_TYPES(cls):
        shader_options = list_shader_relpaths()
        if not shader_options:
            shader_options = ["(no shaders found)"]

        optional = {
            "mask": ("MASK",),
        }
        optional.update(collect_extra_image_inputs())
        optional.update(collect_union_uniform_inputs())

        return {
            "required": {
                "image": ("IMAGE",),
                "shader": (shader_options,),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "GLSL"

    def process(self, image, shader, enabled, mask=None, **kwargs):
        if not enabled:
            return (image,)
        if shader == "(no shaders found)":
            raise RuntimeError("GLSL GPU Processor: no .glsl files under shaders/")

        runtime = get_glsl_runtime()
        source = load_shader_source(shader)
        metadata = enrich_metadata(source)
        uniforms = extract_uniform_values(metadata, kwargs)
        if mask is None:
            for inp in metadata.get("inputs", []):
                if inp.get("type") == "MASK" and inp.get("name") in kwargs:
                    mask = kwargs.get(inp["name"])
                    break

        result = runtime.execute(
            source,
            metadata,
            image,
            mask,
            mode="simple",
            uniforms=uniforms,
        )
        return (result,)


class GLSLShaderNode:
    """Interactive GLSL authoring node (simple process(), compute, or GLES fragment)."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "mask": ("MASK",),
        }
        optional.update(SHADER_NODE_SLOT_INPUTS)
        return {
            "required": {
                "shader_source": ("STRING", {"multiline": True, "default": DEFAULT_SIMPLE_SHADER}),
                "shader_mode": (["auto", "simple", "advanced", "fragment"], {"default": "auto"}),
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "time": ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6, "step": 0.01}),
                "save_shader": ("BOOLEAN", {"default": False}),
                "save_name": ("STRING", {"default": "Inline_Shader"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "process"
    CATEGORY = "GLSL"

    def process(
        self,
        shader_source,
        shader_mode,
        image,
        enabled,
        time,
        save_shader,
        save_name,
        mask=None,
        **kwargs,
    ):
        if not enabled:
            return (image, "Skipped (disabled)")
        if not shader_source.strip():
            return (image, "Skipped (empty shader)")

        runtime = get_glsl_runtime()
        metadata = enrich_metadata(shader_source)
        metadata, uniforms = _merge_shader_node_slots(metadata, kwargs)
        mode = "simple" if shader_mode in ("auto", "fragment", "simple") else "advanced"

        try:
            result = runtime.execute(
                shader_source,
                metadata,
                image,
                mask,
                mode=mode,
                uniforms=uniforms,
                time=float(time),
            )
            status = f"OK ({shader_mode})"
            if save_shader:
                runtime.save_shader(shader_source, f"user/{save_name}")
                status = f"OK + saved user/{save_name}.glsl"
            return (result, status)
        except Exception as e:
            raise RuntimeError(f"GLSL Shader Error: {e}") from e


class GLSLGPUDiagnostics:
    """Reports system diagnostics for GLSL GPU backend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

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
