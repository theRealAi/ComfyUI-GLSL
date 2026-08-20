"""
Adapt common GLSL dialects into Vulkan compute shaders.

Supports:
- GLSL ES 3.00 / WebGL2 fragment
- WebGL1 / classic GLSL
- GLSL Sandbox / Shadertoy-like (time, resolution, mouse)
- Processing texture shaders (vertTexCoord, texture, texOffset)
  e.g. https://github.com/genekogan/Processing-Shader-Examples
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


TIME_ALIASES = {"utime", "u_time", "itime", "time", "ftime", "i_time"}
RESOLUTION_ALIASES = {"resolution", "uresolution", "u_resolution", "iresolution"}
MOUSE_ALIASES = {"mouse", "umouse", "u_mouse", "imouse"}

# Provided automatically in the compute preamble (not push constants)
AUTO_LOCALS = {"textoffset", "verttexcoord", "vertcolor"}

# Sensible defaults for Processing / filter uniforms when not set in the UI
UNIFORM_DEFAULTS: Dict[str, Any] = {
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "threshold": 0.5,
    "hue": 0.0,
    "sigma": 3.0,
    "blurSize": 9,
    "horizontalPass": 1,
    "pixels": (64.0, 64.0),
    "pixelsPerRow": 48,
    "rad": 3,
    "brt": 0.05,
    "turns": 2.0,
    "radius": 1.0,
    "radTwist": 1.0,
    "rollRate": 2.0,
    "rollAmount": 0.1,
    "row": 0.5,
    "col": 0.5,
    "modr": 0.1,
    "modg": 0.1,
    "modb": 0.1,
    "density": 1.0,
    "frequency": 40.0,
    "rbias": (0.0, 0.0),
    "gbias": (0.0, 0.0),
    "bbias": (0.0, 0.0),
    "rmult": (1.0, 1.0),
    "gmult": (1.0, 1.0),
    "bmult": (1.0, 1.0),
}


def is_vulkan_compute_shader(source: str) -> bool:
    body = _strip_comments(source)
    return bool(
        re.search(r"\blocal_size_x\b", body)
        or re.search(r"\bgl_GlobalInvocationID\b", body)
        or re.search(r"\bimageStore\b", body)
    )


def is_fragment_shader(source: str) -> bool:
    body = _strip_comments(source)
    if is_vulkan_compute_shader(body):
        return False
    if re.search(r"\bvec4\s+process\s*\(", body):
        return False
    return bool(
        re.search(r"\bfragColor\b", body)
        or re.search(r"\bgl_FragColor\b", body)
        or re.search(r"#version\s+300\s+es", body)
        or re.search(r"\bgl_FragCoord\b", body)
        or re.search(r"\bout\s+vec4\b", body)
        or re.search(r"\btexture2D\s*\(", body)
        or re.search(r"\bvarying\b", body)
        or re.search(r"\buniform\s+sampler2D\b", body)
        or re.search(r"\buniform\s+(float\s+time|vec2\s+resolution)\b", body)
        or re.search(r"PROCESSING_TEXTURE_SHADER", body)
        or re.search(r"\bvertTexCoord\b", body)
    )


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*[\s\S]*?\*/", "", source)
    source = re.sub(r"//.*?$", "", source, flags=re.M)
    return source


def _discover_uniforms(source: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"\buniform\s+(float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;"
    )
    for typ, name, default in pattern.findall(source):
        lower = name.lower()
        if lower in TIME_ALIASES or lower in RESOLUTION_ALIASES or lower in AUTO_LOCALS:
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "type": typ,
            "default": UNIFORM_DEFAULTS.get(name, 0),
        }
        if default:
            try:
                if typ == "float":
                    entry["default"] = float(default.strip())
                elif typ in ("int", "uint"):
                    entry["default"] = int(default.strip())
                elif typ == "bool":
                    entry["default"] = default.strip().lower() in ("true", "1")
            except ValueError:
                pass
        found.append(entry)
    return found


def _merge_uniforms(metadata: Dict[str, Any], discovered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = {u["name"]: dict(u) for u in metadata.get("uniforms", [])}
    for u in discovered:
        if u["name"] not in existing:
            existing[u["name"]] = u
        else:
            # Fill missing defaults from Processing table
            if "default" not in existing[u["name"]] or existing[u["name"]].get("default") in (None, 0, 0.0):
                if u["name"] in UNIFORM_DEFAULTS:
                    existing[u["name"]]["default"] = UNIFORM_DEFAULTS[u["name"]]
    return list(existing.values())


def _replace_ident(body: str, name: str, replacement: str) -> str:
    return re.sub(rf"(?<!params\.)\b{re.escape(name)}\b", replacement, body)


def _rewrite_all_function_bodies(source: str, rewriter) -> str:
    """
    Apply rewriter to each function body, leaving signatures untouched.
    Passes the function's parameter names so they are not rewritten.
    """
    pattern = re.compile(
        r"\b(?:void|float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4|mat2|mat3|mat4)\s+\w+\s*\(([^;{}]*)\)\s*\{"
    )
    out = []
    last = 0
    for match in pattern.finditer(source):
        params_blob = match.group(1)
        param_names = set(re.findall(r"\b\w+\s+(\w+)\s*(?:,|$)", params_blob))
        # also catch "in float x" style
        param_names.update(re.findall(r"\b(?:in|out|inout)?\s*\w+\s+(\w+)\s*(?:=[^,]+)?(?:,|$)", params_blob))
        sig_end = match.end()
        out.append(source[last:sig_end])
        depth = 1
        i = sig_end
        while i < len(source) and depth:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        body = source[sig_end : i - 1]
        out.append(rewriter(body, param_names))
        out.append("}")
        last = i
    out.append(source[last:])
    return "".join(out)


def _rewrite_texture_lookups(body: str) -> str:
    """Map texture2D / texture sampling onto _sampleInput, including helpers."""
    # Named bilinear helper used by some Processing/WebGL demos
    body = re.sub(r"\btexture2D_bilinear\b", "_sampleBilinear", body)
    # Calls: _sampleBilinear(texture, st, dims, one) -> _sampleBilinear(st, dims, one)
    body = re.sub(
        r"\b_sampleBilinear\s*\(\s*\w+\s*,",
        "_sampleBilinear(",
        body,
    )
    # Rewrite helper signatures that take sampler2D
    body = re.sub(
        r"\bvec4\s+_sampleBilinear\s*\(\s*sampler2D\s+\w+\s*,\s*",
        "vec4 _sampleBilinear(",
        body,
    )
    # Standard texture2D(sampler, uv) / texture(sampler, uv)
    body = re.sub(r"\btexture2D\s*\(\s*\w+\s*,", "_sampleInput(", body)
    body = re.sub(r"\btexture\s*\(\s*\w+\s*,", "_sampleInput(", body)
    return body


def adapt_fragment_to_compute(source: str, metadata: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], int]:
    from .wrapper import build_push_constant_glsl, pack_push_constants

    discovered = _discover_uniforms(source)
    uniforms = _merge_uniforms(metadata, discovered)
    if re.search(r"\bmouse\b", source, re.I) and not any(
        u["name"].lower() in MOUSE_ALIASES for u in uniforms
    ):
        uniforms = list(uniforms) + [{"name": "mouse", "type": "vec2", "default": (0.5, 0.5)}]

    push_glsl, fields = build_push_constant_glsl(uniforms)
    push_size = len(pack_push_constants(fields, 1, 1, {}))

    body = source.strip()
    body = re.sub(r"#version[^\n]*\n?", "", body)
    body = re.sub(r"#extension[^\n]*\n?", "", body)
    body = re.sub(r"#ifdef\s+GL_ES\b[\s\S]*?#endif", "", body)
    body = re.sub(r"#define\s+PROCESSING_\w+[^\n]*\n?", "", body)
    body = re.sub(r"precision\s+\w+\s+(float|int)\s*;", "", body)

    # Drop interface declarations replaced by compute preamble
    body = re.sub(r"\bout\s+vec4\s+\w+\s*;", "", body)
    body = re.sub(r"\buniform\s+sampler2D\s+\w+\s*;", "", body)
    body = re.sub(
        r"\buniform\s+(float|int|uint|bool|vec2|vec3|vec4|ivec2|ivec3|ivec4)\s+\w+\s*(=\s*[^;]+)?\s*;",
        "",
        body,
    )
    body = re.sub(r"\bvarying\s+\w+\s+\w+\s*;", "", body)
    body = re.sub(r"\bin\s+\w+\s+\w+\s*;", "", body)
    body = re.sub(r"\battribute\s+\w+\s+\w+\s*;", "", body)

    body = body.replace("gl_FragColor", "fragColor")
    body = re.sub(r"\bgl_FragCoord\.xy\b", "(vec2(pixel) + 0.5)", body)
    body = re.sub(r"\bgl_FragCoord\b", "vec4(vec2(pixel) + 0.5, 0.0, 1.0)", body)

    for name in ("vUv", "v_uv", "vUV", "v_texCoord", "vTexCoord", "texCoord"):
        body = _replace_ident(body, name, "texUV")

    body = _rewrite_texture_lookups(body)

    # Rewrite uniform identifiers inside every function body (not just main),
    # so helpers like aastep() can read Processing uniforms safely.
    def _rewrite_idents(region: str, param_names=None) -> str:
        skip = set(param_names or ())
        for alias in ("uTime", "u_time", "iTime", "i_time", "fTime", "Time", "time"):
            if alias not in skip:
                region = _replace_ident(region, alias, "params.time")
        for alias in ("uResolution", "u_resolution", "iResolution", "resolution"):
            if alias not in skip:
                region = _replace_ident(region, alias, "params.resolution")
        for alias in ("uMouse", "u_mouse", "iMouse", "mouse"):
            if alias not in skip:
                region = _replace_ident(region, alias, "params.mouse")
        for u in uniforms:
            name = u["name"]
            if name in skip:
                continue
            if name.lower() in TIME_ALIASES | RESOLUTION_ALIASES | MOUSE_ALIASES | AUTO_LOCALS:
                continue
            region = _replace_ident(region, name, f"params.{name}")
        return region

    body = _rewrite_all_function_bodies(body, _rewrite_idents)
    body = _stabilize_dynamic_loops(body)

    has_mask = any(i.get("type") == "MASK" for i in metadata.get("inputs", []))
    mask_decl = ""
    mask_apply = ""
    if has_mask:
        mask_decl = "layout(binding = 2, rgba32f) readonly uniform image2D maskImage;"
        mask_apply = """
    float m = imageLoad(maskImage, pixel).r;
    fragColor = mix(original, fragColor, clamp(m, 0.0, 1.0));
"""

    helpers = """
vec4 _sampleInput(vec2 sampleUV)
{
    vec2 uvClamped = clamp(sampleUV, vec2(0.0), vec2(1.0));
    ivec2 size = imageSize(inputImage);
    ivec2 p = ivec2(floor(uvClamped * vec2(size)));
    p = clamp(p, ivec2(0), size - ivec2(1));
    return imageLoad(inputImage, p);
}

vec4 _sampleBilinear(vec2 st, vec2 dims, vec2 one)
{
    vec2 uv = st * dims;
    vec2 uv00 = floor(uv - vec2(0.5));
    vec2 uvlerp = uv - uv00 - vec2(0.5);
    vec2 st00 = (uv00 + vec2(0.5)) * one;
    vec4 texel00 = _sampleInput(st00);
    vec4 texel10 = _sampleInput(st00 + vec2(one.x, 0.0));
    vec4 texel01 = _sampleInput(st00 + vec2(0.0, one.y));
    vec4 texel11 = _sampleInput(st00 + one);
    vec4 texel0 = mix(texel00, texel01, uvlerp.y);
    vec4 texel1 = mix(texel10, texel11, uvlerp.y);
    return mix(texel0, texel1, uvlerp.x);
}
"""

    # Processing locals: keep vertTexCoord.st / texOffset.st working.
    # Use texUV (not uv) so user shaders may declare their own vec2 uv.
    preamble = """void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(params.resolution.x) || pixel.y >= int(params.resolution.y))
        return;
    vec2 texUV = (vec2(pixel) + 0.5) / params.resolution;
    vec4 original = imageLoad(inputImage, pixel);
    vec4 fragColor = original;
    vec4 vertTexCoord = vec4(texUV, 0.0, 1.0);
    vec4 vertColor = vec4(1.0);
    vec2 texOffset = vec2(1.0) / params.resolution;
"""

    main_re = re.compile(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)\s*\{")
    if main_re.search(body):
        # If user defined their own texture2D_bilinear / _sampleBilinear, drop ours duplicate later
        user_has_bilinear = bool(re.search(r"\bvec4\s+_sampleBilinear\s*\(", body))
        body = main_re.sub(preamble, body, count=1)
        helper_block = helpers
        if user_has_bilinear:
            # Keep only _sampleInput; user helper already present (rewritten)
            helper_block = """
vec4 _sampleInput(vec2 sampleUV)
{
    vec2 uvClamped = clamp(sampleUV, vec2(0.0), vec2(1.0));
    ivec2 size = imageSize(inputImage);
    ivec2 p = ivec2(floor(uvClamped * vec2(size)));
    p = clamp(p, ivec2(0), size - ivec2(1));
    return imageLoad(inputImage, p);
}
"""
        wrapped = f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outputImage;
{mask_decl}

{push_glsl}
{helper_block}
{body}
"""
        wrapped = _inject_before_main_end(
            wrapped,
            f"""{mask_apply}
    imageStore(outputImage, pixel, fragColor);
""",
        )
        return wrapped, fields, push_size

    wrapped = f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outputImage;
{mask_decl}

{push_glsl}
{helpers}
void main()
{{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(params.resolution.x) || pixel.y >= int(params.resolution.y))
        return;
    vec2 texUV = (vec2(pixel) + 0.5) / params.resolution;
    vec4 original = imageLoad(inputImage, pixel);
    vec4 fragColor = original;
    vec4 vertTexCoord = vec4(texUV, 0.0, 1.0);
    vec4 vertColor = vec4(1.0);
    vec2 texOffset = vec2(1.0) / params.resolution;
    {{
{body}
    }}
{mask_apply}
    imageStore(outputImage, pixel, fragColor);
}}
"""
    return wrapped, fields, push_size


def _stabilize_dynamic_loops(body: str) -> str:
    """
    SPIR-V often rejects loops whose bounds depend on uniforms.
    Rewrite common Processing patterns to use a fixed iteration cap.
    """
    # neon-style: for (i = -rad; i < rad; i++) {
    body = re.sub(
        r"for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*-\s*params\.rad\s*;\s*\1\s*<\s*params\.rad\s*;\s*\1\s*\+\+\s*\)\s*\{",
        r"for (int \1 = -12; \1 < 12; \1++) { if (abs(\1) >= params.rad) continue; ",
        body,
    )
    body = re.sub(
        r"for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*-\s*rad\s*;\s*\1\s*<\s*rad\s*;\s*\1\s*\+\+\s*\)\s*\{",
        r"for (int \1 = -12; \1 < 12; \1++) { if (abs(\1) >= rad) continue; ",
        body,
    )
    # blur-style: for (float i = 1.0; i <= numBlurPixelsPerSide; i++)
    body = re.sub(
        r"for\s*\(\s*float\s+(\w+)\s*=\s*1\.0\s*;\s*\1\s*<=\s*numBlurPixelsPerSide\s*;\s*\1\s*\+\+\s*\)\s*\{",
        r"for (float \1 = 1.0; \1 <= 32.0; \1++) { if (\1 > numBlurPixelsPerSide) break; ",
        body,
    )
    return body


def _inject_before_main_end(source: str, injection: str) -> str:
    match = re.search(r"\bvoid\s+main\s*\(", source)
    if not match:
        return source + "\n" + injection
    i = source.find("{", match.end())
    if i < 0:
        return source
    depth = 0
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[:j] + injection + source[j:]
    return source
