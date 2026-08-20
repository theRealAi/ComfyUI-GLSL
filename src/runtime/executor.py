"""
GLSL Runtime Executor

Main GLSL runtime interface for shader compilation and execution.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .backend.vulkan import VulkanBackend
from ..shader.compiler import ShaderCompiler
from ..shader.parser import parse_metadata
from ..shader.wrapper import defaults_from_metadata, pack_push_constants, prepare_shader_source
from ..shader.adapter import UNIFORM_DEFAULTS

logger = logging.getLogger(__name__)


class GLSLRuntime:
    """Shared GLSL runtime used by all ComfyUI-GLSL nodes."""

    def __init__(self):
        self.backend: Optional[VulkanBackend] = None
        self.shader_cache: Dict[str, bytes] = {}
        self.pipeline_cache_meta: Dict[str, Dict[str, Any]] = {}
        self.compiler = ShaderCompiler()
        self._initialized = False

    def _initialize_backend(self) -> None:
        if self._initialized:
            return
        try:
            self.backend = VulkanBackend()
            self.backend.initialize()
            self._initialized = True
            logger.info("Vulkan backend initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize Vulkan backend: %s", e)
            self.backend = None
            self._initialized = False

    def compile(self, source: str, metadata: Dict[str, Any], mode: str = "simple") -> Dict[str, Any]:
        """Compile shader and return SPIR-V plus push-constant layout metadata."""
        self._initialize_backend()
        prepared, fields, push_size = prepare_shader_source(source, metadata, mode=mode)
        if isinstance(prepared, str):
            prepared_list = [prepared]
        else:
            prepared_list = list(prepared)

        passes = []
        for idx, prep in enumerate(prepared_list):
            shader_key = self._get_shader_hash(prep + f"#pass{idx}", metadata)
            if shader_key in self.shader_cache:
                spirv = self.shader_cache[shader_key]
            else:
                spirv = self.compiler.compile(prep, metadata)
                self.shader_cache[shader_key] = spirv
            passes.append({"key": shader_key, "spirv": spirv, "prepared_source": prep})

        meta = {
            "key": passes[0]["key"],
            "spirv": passes[0]["spirv"],
            "fields": fields,
            "push_size": push_size,
            "prepared_source": prepared_list[0],
            "passes": passes,
        }
        self.pipeline_cache_meta[meta["key"]] = meta
        return meta

    def validate(self, shader_source: str, metadata: Dict[str, Any]) -> bool:
        from ..shader.validator import validate_shader

        validate_shader(shader_source, metadata)
        return True

    def execute(
        self,
        shader_source: str,
        metadata: Dict[str, Any],
        image_input: Any,
        mask: Optional[Any] = None,
        mode: str = "simple",
        uniforms: Optional[Dict[str, Any]] = None,
        time: float = 0.0,
        frame_index: int = 0,
    ) -> Any:
        """
        Execute a GLSL compute shader on a ComfyUI IMAGE tensor [B,H,W,C].
        """
        self._initialize_backend()
        if not self._initialized or self.backend is None:
            raise Exception("Vulkan backend is not available")

        if not torch.is_tensor(image_input):
            raise Exception(f"Expected torch IMAGE tensor, got {type(image_input)}")

        compiled = self.compile(shader_source, metadata, mode=mode)
        for p in compiled["passes"]:
            self.validate(p["prepared_source"], metadata)

        pipelines = [
            self.backend.get_or_create_pipeline(p["key"], p["spirv"], compiled["push_size"])
            for p in compiled["passes"]
        ]

        values = defaults_from_metadata(metadata)
        # GIPS inline defaults first (typed), then generic Processing defaults
        try:
            from ..shader.gips import discover_gips_uniforms, is_gips_shader

            if is_gips_shader(shader_source):
                for u in discover_gips_uniforms(shader_source):
                    values[u["name"]] = u.get("default", 0)
        except Exception:
            pass
        for name, default in UNIFORM_DEFAULTS.items():
            values.setdefault(name, default)
        if uniforms:
            values.update(uniforms)

        image = image_input
        device = image.device
        dtype = image.dtype
        if image.ndim != 4:
            raise Exception(f"Expected IMAGE shape [B,H,W,C], got {tuple(image.shape)}")

        batch = image.shape[0]
        in_channels = image.shape[-1]
        outputs = []
        for b in range(batch):
            frame = image[b]
            rgba = self._to_rgba_numpy(frame)
            mask_np = None
            if mask is not None:
                mask_np = self._mask_to_rgba_numpy(mask, b, rgba.shape[0], rgba.shape[1])

            fields = compiled["fields"]
            if fields == [("imageSize", "ivec2")]:
                push = struct_pack_image_size(rgba.shape[1], rgba.shape[0])
            else:
                push = pack_push_constants(
                    fields,
                    width=rgba.shape[1],
                    height=rgba.shape[0],
                    uniforms=values,
                    time=time,
                    frame_index=frame_index + b,
                )
                if len(push) < compiled["push_size"]:
                    push = push + b"\x00" * (compiled["push_size"] - len(push))

            out_np = rgba
            for pipe in pipelines:
                out_np = self.backend.dispatch_rgba(
                    pipe,
                    out_np,
                    push_constants=push,
                    mask_rgba=mask_np,
                )
            out_np = np.nan_to_num(out_np, nan=0.0, posinf=10.0, neginf=0.0)
            if in_channels == 3:
                out_np = out_np[..., :3]
            elif in_channels == 1:
                out_np = out_np[..., :1]
            outputs.append(torch.from_numpy(np.ascontiguousarray(out_np)))

        result = torch.stack(outputs, dim=0).to(device=device, dtype=dtype)
        return result

    def _to_rgba_numpy(self, frame: torch.Tensor) -> np.ndarray:
        """Convert HxW[C] tensor to contiguous float32 HxWx4."""
        arr = frame.detach().float().cpu().numpy()
        if arr.ndim != 3:
            raise Exception(f"Expected HxWxC frame, got {arr.shape}")
        h, w, c = arr.shape
        if c == 4:
            rgba = arr
        elif c == 3:
            rgba = np.concatenate([arr, np.ones((h, w, 1), dtype=np.float32)], axis=2)
        elif c == 1:
            rgba = np.repeat(arr, 3, axis=2)
            rgba = np.concatenate([rgba, np.ones((h, w, 1), dtype=np.float32)], axis=2)
        else:
            raise Exception(f"Unsupported channel count: {c}")
        return np.ascontiguousarray(rgba, dtype=np.float32)

    def _mask_to_rgba_numpy(self, mask: Any, batch_index: int, height: int, width: int) -> np.ndarray:
        if not torch.is_tensor(mask):
            raise Exception("Mask must be a torch tensor")
        m = mask
        if m.ndim == 4:
            # [B,H,W,C] or [B,1,H,W]
            if m.shape[-1] in (1, 3, 4) and m.shape[1] != 1:
                frame = m[batch_index]
            else:
                frame = m[batch_index, 0]
        elif m.ndim == 3:
            # [B,H,W] or [H,W,C]
            if m.shape[0] == mask.shape[0] and m.shape[-1] not in (1, 3, 4):
                frame = m[min(batch_index, m.shape[0] - 1)]
            else:
                frame = m
        elif m.ndim == 2:
            frame = m
        else:
            raise Exception(f"Unsupported mask shape: {tuple(m.shape)}")

        arr = frame.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape[0] != height or arr.shape[1] != width:
            # nearest resize via torch
            t = torch.from_numpy(arr)[None, None]
            t = torch.nn.functional.interpolate(t, size=(height, width), mode="nearest")
            arr = t[0, 0].numpy()
        rgba = np.repeat(arr.astype(np.float32)[..., None], 4, axis=2)
        return np.ascontiguousarray(rgba)

    def save_shader(self, shader_source: str, path: str) -> None:
        from ..utils.paths import sanitize_shader_name, _package_root

        name = sanitize_shader_name(os.path.basename(path))
        if not name.endswith(".glsl"):
            name += ".glsl"
        category = os.path.dirname(path).replace("\\", "/").strip("/")
        if not category:
            category = "user"
        full_dir = os.path.join(_package_root(), "shaders", category)
        os.makedirs(full_dir, exist_ok=True)
        full_path = os.path.join(full_dir, name)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(shader_source)
        logger.info("Saved shader to %s", full_path)

    def get_diagnostics(self) -> Dict[str, Any]:
        self._initialize_backend()
        diagnostics = {
            "backend": "Vulkan" if self.backend and self.backend.is_available() else "None",
            "initialized": self._initialized,
            "shader_cache_size": len(self.shader_cache),
            "pipeline_cache_size": len(self.backend._pipeline_cache) if self.backend else 0,
            "compiler": self.compiler.get_diagnostics(),
        }
        if self.backend and self.backend.is_available():
            diagnostics["gpu_info"] = self.backend.get_gpu_info()
            diagnostics["spirv_support"] = True
            diagnostics["glsl_compiler"] = "glslc" if self.compiler.compiler_available else "unavailable"
        return diagnostics

    def _get_shader_hash(self, source: str, metadata: Dict[str, Any]) -> str:
        combined = source + str(metadata)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def create_passthrough_shader(self) -> str:
        return """/*
@name Passthrough
@description Simple passthrough shader
@version 1.0.0

@input image IMAGE
*/
vec4 process(vec4 color, ivec2 pixel)
{
    return color;
}
"""


def struct_pack_image_size(width: int, height: int) -> bytes:
    import struct

    return struct.pack("<ii", int(width), int(height))
