"""
Simple-mode compute shader wrapper generation and push-constant packing.
"""

from __future__ import annotations

import re
import struct
from typing import Any, Dict, List, Optional, Tuple


STANDARD_HEADER_FIELDS = [
    ("resolution", "vec2"),
    ("time", "float"),
    ("frameIndex", "int"),
]


def has_process_function(source: str) -> bool:
    return re.search(r"\bvec4\s+process\s*\(", source) is not None


def is_full_compute_shader(source: str) -> bool:
    """True only for Vulkan compute shaders (not GLES fragment)."""
    from .adapter import is_vulkan_compute_shader

    return is_vulkan_compute_shader(source)


def build_push_constant_glsl(uniforms: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Build GLSL push-constant block members and ordered (name, type) fields.
    Layout: resolution, time, frameIndex, then metadata uniforms.
    """
    fields: List[Tuple[str, str]] = list(STANDARD_HEADER_FIELDS)
    for u in uniforms:
        name = u["name"]
        if name in ("resolution", "time", "frameIndex"):
            continue
        fields.append((name, u["type"]))

    lines = ["layout(push_constant) uniform Parameters {"]
    for name, typ in fields:
        lines.append(f"    {typ} {name};")
    lines.append("} params;")
    return "\n".join(lines), fields


def pack_push_constants(
    fields: List[Tuple[str, str]],
    width: int,
    height: int,
    uniforms: Dict[str, Any],
    time: float = 0.0,
    frame_index: int = 0,
) -> bytes:
    """
    Pack push constants with std140-like alignment for scalars/vectors.
    """
    values = {
        "resolution": (float(width), float(height)),
        "time": float(time),
        "frameIndex": int(frame_index),
    }
    values.update(uniforms)

    offset = 0
    chunks: List[bytes] = []

    def align(n: int) -> None:
        nonlocal offset
        pad = (n - (offset % n)) % n
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad

    def write_f32(*vals: float) -> None:
        nonlocal offset
        data = struct.pack("<" + "f" * len(vals), *[float(v) for v in vals])
        chunks.append(data)
        offset += len(data)

    def write_i32(v: int) -> None:
        nonlocal offset
        data = struct.pack("<i", int(v))
        chunks.append(data)
        offset += len(data)

    def write_u32(v: int) -> None:
        nonlocal offset
        data = struct.pack("<I", int(v))
        chunks.append(data)
        offset += len(data)

    for name, typ in fields:
        val = values.get(name)
        if typ == "float":
            align(4)
            write_f32(0.0 if val is None else float(val))
        elif typ == "int":
            align(4)
            write_i32(0 if val is None else int(val))
        elif typ == "uint":
            align(4)
            write_u32(0 if val is None else int(val))
        elif typ == "bool":
            align(4)
            write_i32(1 if val else 0)
        elif typ == "vec2":
            align(8)
            if val is None:
                write_f32(0.0, 0.0)
            elif isinstance(val, (list, tuple)):
                write_f32(val[0], val[1])
            else:
                write_f32(float(val), float(val))
        elif typ == "vec3":
            align(16)
            if val is None:
                write_f32(0.0, 0.0, 0.0)
            else:
                write_f32(val[0], val[1], val[2])
            # std140 vec3 takes 16 bytes
            align(16)
        elif typ == "vec4":
            align(16)
            if val is None:
                write_f32(0.0, 0.0, 0.0, 0.0)
            else:
                write_f32(val[0], val[1], val[2], val[3])
        elif typ == "ivec2":
            align(8)
            if val is None:
                write_i32(0)
                write_i32(0)
            else:
                write_i32(val[0])
                write_i32(val[1])
        else:
            raise Exception(f"Unsupported push-constant uniform type: {typ}")

    align(4)
    # Vulkan push constants require size multiple of 4
    size = offset
    if size == 0:
        return b"\x00\x00\x00\x00"
    return b"".join(chunks)


def wrap_simple_shader(source: str, metadata: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], int]:
    """
    Wrap a simple `process()` body into a full compute shader.

    Returns (glsl, push_fields, push_size).
    """
    # Strip metadata comment for embedding user code, keep process() source as-is
    user_code = source.strip()
    uniforms = metadata.get("uniforms", [])
    push_glsl, fields = build_push_constant_glsl(uniforms)
    packed_size = len(pack_push_constants(fields, 1, 1, {}))

    has_mask = any(i.get("type") == "MASK" for i in metadata.get("inputs", []))

    mask_decl = ""
    mask_apply = ""
    if has_mask:
        mask_decl = "layout(binding = 2, rgba32f) readonly uniform image2D maskImage;"
        mask_apply = """
    float m = imageLoad(maskImage, pixel).r;
    color = mix(original, color, clamp(m, 0.0, 1.0));
"""

    # If user already included version/main, treat as advanced
    if is_full_compute_shader(user_code):
        # Still need field list for defaults-only advanced shaders that use our layout
        return user_code, fields, max(packed_size, 16)

    wrapper = f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outputImage;
{mask_decl}

{push_glsl}

{user_code}

void main()
{{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(params.resolution.x) || pixel.y >= int(params.resolution.y))
        return;

    vec4 original = imageLoad(inputImage, pixel);
    vec4 color = process(original, pixel);
{mask_apply}
    imageStore(outputImage, pixel, color);
}}
"""
    return wrapper, fields, packed_size


def prepare_shader_source(source: str, metadata: Dict[str, Any], mode: str = "simple"):
    """
    Prepare final GLSL for compilation based on mode / dialect.

    Returns (prepared, fields, push_size) where prepared is str or List[str]
    (List for GIPS multi-pass filters).
    """
    from .adapter import adapt_fragment_to_compute, is_fragment_shader, is_vulkan_compute_shader
    from .gips import adapt_gips_to_compute, is_gips_shader

    if is_gips_shader(source):
        return adapt_gips_to_compute(source, metadata)

    if is_fragment_shader(source):
        return adapt_fragment_to_compute(source, metadata)

    if mode == "advanced" or is_vulkan_compute_shader(source):
        _, fields = build_push_constant_glsl(metadata.get("uniforms", []))
        size = len(pack_push_constants(fields, 1, 1, {}))
        if "imageSize" in source and "params.resolution" not in source and "Parameters" not in source:
            fields = [("imageSize", "ivec2")]
            size = 8
        return source, fields, size

    if has_process_function(source) or mode == "simple":
        return wrap_simple_shader(source, metadata)

    if re.search(r"\bvoid\s+main\s*\(", source):
        return adapt_fragment_to_compute(source, metadata)

    return wrap_simple_shader(source, metadata)


def defaults_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for u in metadata.get("uniforms", []):
        out[u["name"]] = u.get("default", 0)
    return out
