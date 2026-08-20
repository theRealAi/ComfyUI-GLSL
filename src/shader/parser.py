"""
GLSL Shader Metadata Parser

Parses metadata comments from GLSL shaders to extract:
- Name
- Description
- Version
- Inputs (@input)
- Uniforms (@uniform)
"""

import re
from typing import Dict, List, Any


def parse_metadata(source: str) -> Dict[str, Any]:
    """
    Parse GLSL shader metadata from comment blocks.
    
    Args:
        source (str): Full GLSL source code
        
    Returns:
        dict: Parsed metadata including name, description, version, inputs, uniforms
    """
    # Find comment blocks. Use a non-capturing pattern so findall returns full matches
    # (capturing groups would return tuples and break .strip()).
    comments = re.findall(r'/\*[\s\S]*?\*/', source)

    if not comments:
        return {
            "name": "Unnamed Shader",
            "description": "",
            "version": "1.0.0",
            "inputs": [],
            "uniforms": []
        }

    # Strip /* */ delimiters from the first comment block
    comment_block = comments[0][2:-2]

    # Parse metadata lines
    lines = comment_block.strip().split('\n')
    
    metadata = {
        "name": "Unnamed Shader",
        "description": "",
        "version": "1.0.0",
        "inputs": [],
        "uniforms": []
    }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Check for metadata directives
        if line.startswith("@name"):
            metadata["name"] = line[5:].strip()
        elif line.startswith("@description"):
            metadata["description"] = line[12:].strip()
        elif line.startswith("@version"):
            metadata["version"] = line[8:].strip()
        elif line.startswith("@input"):
            # Parse input directive
            parts = line[6:].strip().split()
            if len(parts) >= 2:
                input_info = {
                    "name": parts[0],
                    "type": parts[1]
                }
                metadata["inputs"].append(input_info)
        elif line.startswith("@uniform"):
            # Parse uniform directive
            try:
                uniform_info = parse_uniform(line[8:].strip())
                if uniform_info:
                    metadata["uniforms"].append(uniform_info)
            except Exception as e:
                print(f"Warning: Failed to parse uniform from '{line}': {e}")
    
    return metadata


def parse_uniform(line: str) -> Dict[str, Any]:
    """
    Parse a @uniform directive line.
    
    Example:
        @uniform exposure float 0.0 min=-10 max=10 step=0.01
    
    Returns:
        dict: Uniform information
    """
    # Split by spaces but be careful with quoted strings (for string types)
    parts = line.split()
    
    if len(parts) < 3:
        return None
    
    name = parts[0]
    type_ = parts[1]
    default = parts[2] if len(parts) > 2 else "0"
    
    # Parse default value based on type
    try:
        if type_ in ("int", "uint"):
            default_value = int(default)
        elif type_ == "bool":
            default_value = bool(default.lower() in ("true", "1"))
        else:  # float, vec2, vec3, vec4, etc.
            default_value = float(default)
    except ValueError:
        default_value = default  # Keep as string if not parseable
    
    uniform_info = {
        "name": name,
        "type": type_,
        "default": default_value
    }
    
    # Parse min/max/step if present
    for i, part in enumerate(parts):
        if part.startswith("min="):
            try:
                uniform_info["min"] = float(part[4:])
            except ValueError:
                pass
        elif part.startswith("max="):
            try:
                uniform_info["max"] = float(part[4:])
            except ValueError:
                pass
        elif part.startswith("step="):
            try:
                uniform_info["step"] = float(part[5:])
            except ValueError:
                pass
    
    return uniform_info


def test_parser():
    """Test the parser with example inputs."""
    
    # Test 1: Simple case
    simple_shader = """
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
"""
    
    metadata = parse_metadata(simple_shader)
    print("Simple Shader Metadata:", metadata)
    
    # Test 2: Complex case with multiple uniforms
    complex_shader = """
/*
@name Reinhard Extended
@description Extended Reinhard tone mapping
@version 1.0.0

@input image IMAGE
@input mask MASK

@uniform exposure float 0.0 min=-10 max=10 step=0.01
@uniform whitePoint float 4.0 min=0.01 max=32 step=0.01
@uniform strength float 1.0 min=0 max=1 step=0.01
@uniform useHDR bool true
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
"""
    
    metadata = parse_metadata(complex_shader)
    print("Complex Shader Metadata:", metadata)


if __name__ == "__main__":
    test_parser()