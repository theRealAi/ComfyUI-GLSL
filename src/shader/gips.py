"""
Adapt GIPS (GLSL Image Processing System) filters to Vulkan compute.

See: https://github.com/kajott/GIPS/blob/main/ShaderFormat.md
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def is_gips_shader(source: str) -> bool:
    if re.search(r"@gips_version\s*=", source):
        return True
    if re.search(r"\b(?:vec[234])\s+run(?:_pass[1-4])?\s*\(", source):
        return True
    return False


def _parse_coord_filter(source: str, pass_name: Optional[str] = None) -> Tuple[str, str]:
    coord, filt = "none", "on"
    for m in re.finditer(r"//[^\n]*", source):
        line = m.group(0)
        cm = re.search(r"@coord\s*=\s*(\w+)", line)
        if cm:
            v = cm.group(1).lower()
            if v == "pixel":
                coord = "pixel"
            elif v in ("rel", "relative"):
                coord = "rel"
            else:
                coord = "none"
        fm = re.search(r"@filter\s*=\s*(\w+)", line)
        if fm:
            v = fm.group(1).lower()
            filt = "off" if v in ("0", "off", "nearest", "point") else "on"

    if pass_name:
        fn = re.search(rf"\b(?:vec[234])\s+{re.escape(pass_name)}\s*\(", source)
        if fn:
            preceding = source[max(0, fn.start() - 400) : fn.start()]
            for line in reversed(preceding.splitlines()):
                if "@coord" in line or "@filter" in line:
                    cm = re.search(r"@coord\s*=\s*(\w+)", line)
                    if cm:
                        v = cm.group(1).lower()
                        if v == "pixel":
                            coord = "pixel"
                        elif v in ("rel", "relative"):
                            coord = "rel"
                        else:
                            coord = "none"
                    fm = re.search(r"@filter\s*=\s*(\w+)", line)
                    if fm:
                        v = fm.group(1).lower()
                        filt = "off" if v in ("0", "off", "nearest", "point") else "on"
                    break
    return coord, filt


def _parse_vec_literal(text: str) -> Any:
    text = text.strip()
    m = re.match(r"vec([234])\s*\((.*)\)\s*$", text, re.S)
    if not m:
        try:
            return float(text)
        except ValueError:
            return 0.0
    n = int(m.group(1))
    parts = [p.strip() for p in m.group(2).split(",")]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(0.0)
    while len(vals) < n:
        vals.append(vals[-1] if vals else 0.0)
    return tuple(vals[:n])


def _apply_gips_annotations(entry: Dict[str, Any], comment: str) -> None:
    """Parse GIPS trailing // annotations (@min/@max/@angle/@toggle/…)."""
    if not comment:
        return
    for key in ("min", "max", "step", "off", "on"):
        m = re.search(rf"@{key}\s*=\s*([-+0-9.eE]+)", comment)
        if m:
            try:
                entry[key] = float(m.group(1))
            except ValueError:
                pass
    if re.search(r"@angle\b", comment):
        entry.setdefault("min", -3.14159265)
        entry.setdefault("max", 3.14159265)
        entry.setdefault("step", 0.01)
    if re.search(r"@(?:toggle|switch)\b", comment):
        entry["widget"] = "toggle"
        if "default" not in entry or entry["default"] in (None,):
            entry["default"] = float(entry.get("off", 0.0))


def discover_gips_uniforms(source: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    line_re = re.compile(
        r"^\s*uniform\s+(float|vec2|vec3|vec4)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;"
        r"\s*(?://(.*))?$"
    )
    for line in source.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        typ, name, default, comment = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        if name.startswith("gips_"):
            continue
        entry: Dict[str, Any] = {"name": name, "type": typ, "default": 0.0}
        if default:
            entry["default"] = _parse_vec_literal(default)
        elif typ == "vec2":
            entry["default"] = (0.0, 0.0)
        elif typ == "vec3":
            entry["default"] = (0.0, 0.0, 0.0)
        elif typ == "vec4":
            entry["default"] = (0.0, 0.0, 0.0, 0.0)
        _apply_gips_annotations(entry, comment)
        found.append(entry)
    return found


def _find_functions(source: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"\b(vec[234]|float|void)\s+(\w+)\s*\(([^)]*)\)\s*\{")
    funcs = []
    for m in pattern.finditer(source):
        ret, name, args = m.group(1), m.group(2), m.group(3).strip()
        start = m.start()
        brace = source.find("{", m.end() - 1)
        depth = 0
        i = brace
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        funcs.append({"ret": ret, "name": name, "args": args, "start": start, "end": i})
    return funcs


def _classify_run_sig(args: str, ret: str) -> str:
    args = re.sub(r"\bin\s+", "", args).strip()
    pm = re.match(r"(vec[234]|float)\s+\w+", args)
    if not pm:
        return "unknown"
    arg_t = pm.group(1)
    if arg_t == "vec2":
        return f"pos->{ret}"
    if arg_t == "vec4":
        return f"color4->{ret}"
    if arg_t == "vec3":
        return f"color3->{ret}"
    return "unknown"


def _pixel_helpers(coord: str, filt: str) -> Tuple[str, str]:
    if coord == "pixel":
        to_uv = """
vec2 _gips_to_uv(vec2 pos)
{
    return pos / max(params.resolution, vec2(1.0));
}
"""
        pos_expr = "vec2(pixel) + 0.5"
    elif coord == "rel":
        to_uv = """
vec2 _gips_to_uv(vec2 pos)
{
    float shortEdge = min(params.resolution.x, params.resolution.y);
    vec2 pixelPos = pos * (shortEdge * 0.5) + params.resolution * 0.5;
    return pixelPos / max(params.resolution, vec2(1.0));
}
"""
        pos_expr = (
            "((vec2(pixel) + 0.5) - params.resolution * 0.5) / "
            "(min(params.resolution.x, params.resolution.y) * 0.5)"
        )
    else:
        to_uv = """
vec2 _gips_to_uv(vec2 pos)
{
    return pos;
}
"""
        pos_expr = "(vec2(pixel) + 0.5) / params.resolution"

    sample = """
vec4 _sampleInput(vec2 sampleUV)
{
    vec2 uvClamped = clamp(sampleUV, vec2(0.0), vec2(1.0));
    ivec2 size = imageSize(inputImage);
    ivec2 p = ivec2(floor(uvClamped * vec2(size)));
    p = clamp(p, ivec2(0), size - ivec2(1));
    return imageLoad(inputImage, p);
}
"""
    if filt == "on":
        sample += """
vec4 pixel(vec2 pos)
{
    vec2 sampleUV = clamp(_gips_to_uv(pos), vec2(0.0), vec2(1.0));
    vec2 size = vec2(imageSize(inputImage));
    vec2 uv = sampleUV * size - vec2(0.5);
    vec2 uv00 = floor(uv);
    vec2 f = fract(uv);
    vec2 one = vec2(1.0) / size;
    vec2 st00 = (uv00 + vec2(0.5)) / size;
    vec4 t00 = _sampleInput(st00);
    vec4 t10 = _sampleInput(st00 + vec2(one.x, 0.0));
    vec4 t01 = _sampleInput(st00 + vec2(0.0, one.y));
    vec4 t11 = _sampleInput(st00 + one);
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}
"""
    else:
        sample += """
vec4 pixel(vec2 pos)
{
    return _sampleInput(_gips_to_uv(pos));
}
"""
    return to_uv + sample, pos_expr


def _stabilize_gips_loops(body: str) -> str:
    body = re.sub(
        r"for\s*\(\s*float\s+(\w+)\s*=\s*1\.0\s*;\s*\1\s*<\s*(\w+)\s*;\s*\1\s*\+=\s*1\.0\s*\)\s*\{",
        r"for (float \1 = 1.0; \1 < 64.0; \1 += 1.0) { if (\1 >= \2) break; ",
        body,
    )
    body = re.sub(
        r"for\s*\(\s*float\s+(\w+)\s*=\s*1\.0\s*;\s*\1\s*<\s*params\.(\w+)\s*;\s*\1\s*\+=\s*1\.0\s*\)\s*\{",
        r"for (float \1 = 1.0; \1 < 64.0; \1 += 1.0) { if (\1 >= params.\2) break; ",
        body,
    )
    body = re.sub(
        r"for\s*\(\s*float\s+(\w+)\s*=\s*([^;]+);\s*\1\s*<=\s*(\w+)\s*;\s*\1\s*\+=\s*1\.0\s*\)\s*\{",
        r"for (float \1 = -32.0; \1 <= 32.0; \1 += 1.0) { if (\1 < (\2) || \1 > (\3)) continue; ",
        body,
    )
    return body


def _replace_ident(body: str, name: str, replacement: str) -> str:
    return re.sub(rf"(?<!params\.)\b{re.escape(name)}\b", replacement, body)


def _rewrite_uniforms_in_functions(source: str, uniforms: List[Dict[str, Any]]) -> str:
    names = [u["name"] for u in uniforms]

    def rewriter(region: str, param_names=None) -> str:
        skip = set(param_names or ())
        region = _replace_ident(region, "gips_image_size", "params.resolution")
        for name in names:
            if name in skip:
                continue
            region = _replace_ident(region, name, f"params.{name}")
        return region

    from .adapter import _rewrite_all_function_bodies

    return _rewrite_all_function_bodies(source, rewriter)


def _split_call_args(argstr: str) -> List[str]:
    """Split function-call arguments, respecting nested parentheses."""
    args: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(argstr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(argstr[start:i].strip())
            start = i + 1
    args.append(argstr[start:].strip())
    return args


def _replace_gips_tex_calls(code: str) -> str:
    """Rewrite texture*(gips_tex, ...) with balanced UV args -> _sampleInput(uv)."""
    out: List[str] = []
    i = 0
    while i < len(code):
        m = re.search(
            r"\b(textureLodOffset|textureLod|texture)\s*\(\s*gips_tex\s*,",
            code[i:],
        )
        if not m:
            out.append(code[i:])
            break
        out.append(code[i : i + m.start()])
        kind = m.group(1)
        abs_start = i + m.start()
        call_open = code.find("(", abs_start)
        depth = 0
        j = call_open
        while j < len(code):
            if code[j] == "(":
                depth += 1
            elif code[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            out.append(code[abs_start:])
            break
        inner = code[call_open + 1 : j]
        rest = re.sub(r"^\s*gips_tex\s*,\s*", "", inner, count=1)
        args = _split_call_args(rest)
        if kind == "textureLodOffset" and len(args) >= 3:
            uv, offset = args[0], args[2]
            repl = f"_sampleInput(({uv}) + vec2({offset}) / params.resolution)"
        elif len(args) >= 1:
            repl = f"_sampleInput({args[0]})"
        else:
            out.append(code[abs_start : j + 1])
            i = j + 1
            continue
        out.append(repl)
        i = j + 1
    return "".join(out)


def _rewrite_gips_builtins(code: str, coord: str) -> str:
    """Map GIPS/OpenGL builtins onto our compute helpers."""
    # Local vars named `sign` shadow the builtin; uniforms are already params.*
    code = re.sub(r"\bbvec3\s+sign\b", "bvec3 _signFlag", code)
    code = re.sub(r"(?<!params\.)\bsign\b(?!\s*\()", "_signFlag", code)

    # Reserved word in Vulkan GLSL
    code = re.sub(r"\bvec3\s+sample\b", "vec3 _sampleCol", code)
    code = re.sub(r"\bvec4\s+sample\b", "vec4 _sampleCol", code)
    code = re.sub(r"\bfloat\s+sample\b", "float _sampleCol", code)
    code = re.sub(r"(?<!params\.)\bsample\b", "_sampleCol", code)

    code = code.replace("gl_FragColor", "fragColor")
    # gips_pos is set in main() before calling run*
    code = re.sub(r"\bgl_FragCoord\.xy\b", "gips_pos", code)
    code = re.sub(r"\bgl_FragCoord\.x\b", "gips_pos.x", code)
    code = re.sub(r"\bgl_FragCoord\.y\b", "gips_pos.y", code)
    code = re.sub(r"\bgl_FragCoord\b", "vec4(gips_pos, 0.0, 1.0)", code)

    # Derivatives -> named polyfills
    code = re.sub(r"\bfwidth\s*\(", "_gips_fwidth(", code)
    code = re.sub(r"\bdFdx\s*\(", "_gips_dFdx(", code)
    code = re.sub(r"\bdFdy\s*\(", "_gips_dFdy(", code)

    # Direct gips_tex sampling uses raw UVs (NOT the @coord system)
    code = _replace_gips_tex_calls(code)
    return code


def _derivative_polyfills() -> str:
    return """
float _gips_dFdx(float v) { return 0.0; }
float _gips_dFdy(float v) { return 0.0; }
vec2 _gips_dFdx(vec2 v) { return vec2(0.0); }
vec2 _gips_dFdy(vec2 v) { return vec2(0.0); }
vec3 _gips_dFdx(vec3 v) { return vec3(0.0); }
vec3 _gips_dFdy(vec3 v) { return vec3(0.0); }
float _gips_fwidth(float v) { return 1.0 / max(params.resolution.x, params.resolution.y); }
vec2 _gips_fwidth(vec2 v) { return vec2(_gips_fwidth(0.0)); }
vec3 _gips_fwidth(vec3 v) { return vec3(_gips_fwidth(0.0)); }
"""


def _wrap_pass(
    user_code: str,
    run_name: str,
    ret: str,
    args: str,
    coord: str,
    filt: str,
    push_glsl: str,
    uniforms: List[Dict[str, Any]],
) -> str:
    helper_src, pos_expr = _pixel_helpers(coord, filt)
    sig = _classify_run_sig(args, ret)

    if sig.startswith("pos->"):
        call = f"{run_name}({pos_expr})"
        if ret == "vec3":
            invoke = (
                f"gips_pos = {pos_expr};\n"
                f"    vec3 _r = {call};\n"
                f"    fragColor = vec4(_r, 1.0);"
            )
        else:
            invoke = f"gips_pos = {pos_expr};\n    fragColor = {call};"
    elif sig.startswith("color4->"):
        if ret == "vec3":
            invoke = (
                f"gips_pos = (vec2(pixel) + 0.5);\n"
                f"    vec3 _r = {run_name}(original);\n"
                f"    fragColor = vec4(_r, original.a);"
            )
        else:
            invoke = (
                f"gips_pos = (vec2(pixel) + 0.5);\n"
                f"    fragColor = {run_name}(original);"
            )
    elif sig.startswith("color3->"):
        if ret == "vec4":
            invoke = (
                f"gips_pos = (vec2(pixel) + 0.5);\n"
                f"    fragColor = {run_name}(original.rgb);"
            )
        else:
            invoke = (
                f"gips_pos = (vec2(pixel) + 0.5);\n"
                f"    vec3 _r = {run_name}(original.rgb);\n"
                f"    fragColor = vec4(_r, original.a);"
            )
    else:
        raise Exception(f"Unsupported GIPS run signature: {ret} {run_name}({args})")

    code = re.sub(
        r"\buniform\s+(float|vec2|vec3|vec4|sampler2D)\s+\w+\s*(=\s*[^;]+)?\s*;",
        "",
        user_code,
    )
    # Uniforms before builtins so names like `sign` become params.sign first
    code = _rewrite_uniforms_in_functions(code, uniforms)
    code = _rewrite_gips_builtins(code, coord)
    code = _stabilize_gips_loops(code)

    return f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outputImage;

{push_glsl}
{_derivative_polyfills()}
{helper_src}
vec2 gips_pos;

{code}

void main()
{{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(params.resolution.x) || pixel.y >= int(params.resolution.y))
        return;
    vec4 original = imageLoad(inputImage, pixel);
    vec4 fragColor = original;
    {invoke}
    imageStore(outputImage, pixel, fragColor);
}}
"""


def adapt_gips_to_compute(
    source: str, metadata: Dict[str, Any]
) -> Tuple[List[str], List[Tuple[str, str]], int]:
    from .wrapper import build_push_constant_glsl, pack_push_constants

    discovered = discover_gips_uniforms(source)
    by_name = {u["name"]: u for u in discovered}
    for u in metadata.get("uniforms", []):
        if u["name"] not in by_name:
            by_name[u["name"]] = u
    uniforms = list(by_name.values())

    push_glsl, fields = build_push_constant_glsl(uniforms)
    push_size = len(pack_push_constants(fields, 1, 1, {}))
    if push_size > 128:
        while len(fields) > 3 and len(pack_push_constants(fields, 1, 1, {})) > 128:
            fields = fields[:-1]
        lines = ["layout(push_constant) uniform Parameters {"]
        for name, typ in fields:
            lines.append(f"    {typ} {name};")
        lines.append("} params;")
        push_glsl = "\n".join(lines)
        push_size = len(pack_push_constants(fields, 1, 1, {}))
        keep = {f[0] for f in fields}
        uniforms = [u for u in uniforms if u["name"] in keep]

    funcs = _find_functions(source)
    run_passes = [f for f in funcs if re.match(r"run_pass[1-4]$", f["name"])]
    run_single = [f for f in funcs if f["name"] == "run"]
    if run_passes:
        targets = sorted(run_passes, key=lambda f: f["name"])
    elif run_single:
        targets = run_single
    else:
        raise Exception("GIPS shader has no run() / run_passN() entry point")

    passes = []
    for fn in targets:
        coord, filt = _parse_coord_filter(source, fn["name"])
        passes.append(
            _wrap_pass(
                source,
                fn["name"],
                fn["ret"],
                fn["args"],
                coord,
                filt,
                push_glsl,
                uniforms,
            )
        )
    return passes, fields, max(push_size, 16)
