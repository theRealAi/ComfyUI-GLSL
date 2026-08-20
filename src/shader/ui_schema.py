"""
Helpers for building ComfyUI INPUT_TYPES from shader metadata.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .parser import parse_metadata
from ..utils.paths import discover_shader_directories, find_glsl_files, _package_root


RESERVED_INPUT_NAMES = {
    "image",
    "shader",
    "enabled",
    "mask",
    "shader_source",
    "shader_mode",
    "save_shader",
    "save_name",
    "time",
}

# Fixed authoring slots for GLSL Shader (not the full library uniform union).
SHADER_NODE_SLOT_UNIFORMS: List[Dict[str, Any]] = [
    {"name": "float_1", "type": "float", "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01},
    {"name": "float_2", "type": "float", "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01},
    {"name": "int_1", "type": "int", "default": 0, "min": -1000000, "max": 1000000, "step": 1},
    {"name": "int_2", "type": "int", "default": 0, "min": -1000000, "max": 1000000, "step": 1},
    {"name": "vec2_1", "type": "vec2", "default": (0.0, 0.0), "min": -1000.0, "max": 1000.0, "step": 0.01},
    {"name": "vec2_2", "type": "vec2", "default": (0.0, 0.0), "min": -1000.0, "max": 1000.0, "step": 0.01},
]

_VEC_COMPONENTS = {
    "vec2": "xy",
    "vec3": "xyz",
    "vec4": "xyzw",
    "ivec2": "xy",
    "ivec3": "xyz",
    "ivec4": "xyzw",
    "uvec2": "xy",
    "uvec3": "xyz",
    "uvec4": "xyzw",
}


def list_shader_relpaths() -> List[str]:
    shaders_root = os.path.join(_package_root(), "shaders")
    files = []
    for directory in discover_shader_directories():
        files.extend(find_glsl_files(directory))
    return sorted(os.path.relpath(f, shaders_root).replace("\\", "/") for f in files)


def load_shader_source(relpath: str) -> str:
    path = os.path.join(_package_root(), "shaders", relpath)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def enrich_metadata(source: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge dialect-discovered uniforms (GIPS / fragment / Processing) into
    @uniform metadata so Processor UI and execution see the same fields.
    """
    meta = dict(metadata or parse_metadata(source))
    by_name: Dict[str, Dict[str, Any]] = {
        u["name"]: dict(u) for u in meta.get("uniforms", []) if u.get("name")
    }

    try:
        from .gips import discover_gips_uniforms, is_gips_shader

        if is_gips_shader(source):
            for u in discover_gips_uniforms(source):
                name = u["name"]
                if name not in by_name:
                    by_name[name] = dict(u)
                else:
                    for key, val in u.items():
                        if key == "name":
                            continue
                        if key not in by_name[name] or by_name[name][key] in (None, ""):
                            by_name[name][key] = val
    except Exception:
        pass

    if not by_name:
        try:
            from .adapter import _discover_uniforms

            for u in _discover_uniforms(source):
                by_name.setdefault(u["name"], dict(u))
        except Exception:
            pass

    meta["uniforms"] = list(by_name.values())
    return meta


def metadata_for_shader(relpath: str) -> Dict[str, Any]:
    return enrich_metadata(load_shader_source(relpath))


def _scalar_default(default: Any, index: int = 0) -> float:
    if isinstance(default, (list, tuple)):
        if index < len(default):
            return float(default[index])
        return float(default[0]) if default else 0.0
    if isinstance(default, (int, float)):
        return float(default)
    return 0.0


def expand_uniform_widgets(uniform: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand vecN uniforms into per-component float widgets (name_x, name_y, …)
    so ComfyUI can expose tweakable fields.
    """
    typ = uniform.get("type", "float")
    comps = _VEC_COMPONENTS.get(typ)
    if not comps:
        return [dict(uniform)]

    name = uniform["name"]
    default = uniform.get("default", 0.0)
    widgets = []
    for i, c in enumerate(comps):
        w = {
            "name": f"{name}_{c}",
            "type": "float",
            "default": _scalar_default(default, i),
            "parent": name,
            "component": i,
        }
        for key in ("min", "max", "step"):
            if key in uniform:
                w[key] = uniform[key]
        widgets.append(w)
    return widgets


def uniform_to_comfy_input(uniform: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    typ = uniform.get("type", "float")
    default = uniform.get("default", 0)
    widget = uniform.get("widget")

    if widget == "toggle" and "off" not in uniform and "on" not in uniform:
        return (
            "BOOLEAN",
            {"default": bool(_scalar_default(default) > 0.5)},
        )

    if typ == "float" or typ.startswith("vec") or typ.startswith("ivec") or typ.startswith("uvec"):
        return (
            "FLOAT",
            {
                "default": _scalar_default(default),
                "min": float(uniform.get("min", -1000000)),
                "max": float(uniform.get("max", 1000000)),
                "step": float(uniform.get("step", 0.01)),
            },
        )
    if typ in ("int", "uint"):
        return (
            "INT",
            {
                "default": int(default) if default is not None else 0,
                "min": int(uniform.get("min", -1000000)),
                "max": int(uniform.get("max", 1000000)),
                "step": int(uniform.get("step", 1)),
            },
        )
    if typ == "bool":
        return ("BOOLEAN", {"default": bool(default)})
    return ("FLOAT", {"default": 0.0, "min": -1000000, "max": 1000000, "step": 0.01})


def collect_union_uniform_inputs() -> Dict[str, tuple]:
    """Union of tweakable uniforms across the shader library (incl. GIPS)."""
    result: Dict[str, tuple] = {}
    for rel in list_shader_relpaths():
        try:
            meta = metadata_for_shader(rel)
        except Exception:
            continue
        for u in meta.get("uniforms", []):
            for widget in expand_uniform_widgets(u):
                name = widget.get("name")
                if not name or name in RESERVED_INPUT_NAMES or name in result:
                    continue
                comfy_type, opts = uniform_to_comfy_input(widget)
                result[name] = (comfy_type, opts)
    return result


def collect_extra_image_inputs() -> Dict[str, tuple]:
    """Optional IMAGE sockets declared via @input (except primary image/mask)."""
    result: Dict[str, tuple] = {}
    for rel in list_shader_relpaths():
        try:
            meta = metadata_for_shader(rel)
        except Exception:
            continue
        for inp in meta.get("inputs", []):
            name = inp.get("name")
            typ = inp.get("type")
            if not name or name in RESERVED_INPUT_NAMES or name in result:
                continue
            if typ == "MASK":
                result[name] = ("MASK",)
            elif typ == "IMAGE":
                result[name] = ("IMAGE",)
    return result


def shader_widget_names(meta: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for u in meta.get("uniforms", []):
        for widget in expand_uniform_widgets(u):
            names.append(widget["name"])
    return names


def shader_ui_schema(relpath: str) -> Dict[str, Any]:
    """Schema consumed by the web UI to show/hide widgets."""
    meta = metadata_for_shader(relpath)
    uniform_names = shader_widget_names(meta)
    return {
        "path": relpath,
        "name": meta.get("name", relpath),
        "description": meta.get("description", ""),
        "version": meta.get("version", ""),
        "uniforms": meta.get("uniforms", []),
        "inputs": meta.get("inputs", []),
        "uniform_names": uniform_names,
        "input_names": [i["name"] for i in meta.get("inputs", [])],
    }


def extract_uniform_values(metadata: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pull widget values from Comfy kwargs, reassembling expanded vec components."""
    values: Dict[str, Any] = {}
    for u in metadata.get("uniforms", []):
        name = u["name"]
        typ = u.get("type", "float")
        comps = _VEC_COMPONENTS.get(typ)
        if comps:
            keys = [f"{name}_{c}" for c in comps]
            if all(k in kwargs and kwargs[k] is not None for k in keys):
                values[name] = tuple(float(kwargs[k]) for k in keys)
            elif name in kwargs and kwargs[name] is not None:
                values[name] = kwargs[name]
            continue

        if name not in kwargs or kwargs[name] is None:
            continue
        val = kwargs[name]
        if u.get("widget") == "toggle":
            if isinstance(val, bool):
                on = float(u.get("on", 1.0))
                off = float(u.get("off", 0.0))
                values[name] = on if val else off
            else:
                values[name] = val
        else:
            values[name] = val
    return values


def _build_shader_node_slot_inputs() -> Dict[str, tuple]:
    result: Dict[str, tuple] = {}
    for slot in SHADER_NODE_SLOT_UNIFORMS:
        for widget in expand_uniform_widgets(slot):
            name = widget["name"]
            result[name] = uniform_to_comfy_input(widget)
    return result


SHADER_NODE_SLOT_INPUTS = _build_shader_node_slot_inputs()
