"""
Unit tests for GLSL Shader Metadata Parser
"""

import unittest
from src.shader.parser import parse_metadata


class TestMetadataParser(unittest.TestCase):
    """Test cases for the metadata parser."""

    def test_simple_shader(self):
        """Test parsing a simple shader with basic metadata."""
        shader = """
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
        
        metadata = parse_metadata(shader)
        
        self.assertEqual(metadata["name"], "Exposure")
        self.assertEqual(metadata["description"], "GPU exposure adjustment")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(len(metadata["inputs"]), 1)
        self.assertEqual(metadata["inputs"][0]["name"], "image")
        self.assertEqual(metadata["inputs"][0]["type"], "IMAGE")
        self.assertEqual(len(metadata["uniforms"]), 1)
        self.assertEqual(metadata["uniforms"][0]["name"], "exposure")
        self.assertEqual(metadata["uniforms"][0]["type"], "float")
        self.assertEqual(metadata["uniforms"][0]["default"], 0.0)
        self.assertEqual(metadata["uniforms"][0]["min"], -10.0)
        self.assertEqual(metadata["uniforms"][0]["max"], 10.0)
        self.assertEqual(metadata["uniforms"][0]["step"], 0.01)

    def test_complex_shader(self):
        """Test parsing a shader with multiple uniforms and inputs."""
        shader = """
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
        
        metadata = parse_metadata(shader)
        
        self.assertEqual(metadata["name"], "Reinhard Extended")
        self.assertEqual(len(metadata["inputs"]), 2)
        self.assertEqual(len(metadata["uniforms"]), 4)
        
        # Check exposure uniform
        exposure = next(u for u in metadata["uniforms"] if u["name"] == "exposure")
        self.assertEqual(exposure["type"], "float")
        self.assertEqual(exposure["default"], 0.0)
        self.assertEqual(exposure["min"], -10.0)
        self.assertEqual(exposure["max"], 10.0)
        
        # Check useHDR uniform
        useHDR = next(u for u in metadata["uniforms"] if u["name"] == "useHDR")
        self.assertEqual(useHDR["type"], "bool")
        self.assertEqual(useHDR["default"], True)

    def test_no_metadata(self):
        """Test parsing a shader with no metadata."""
        shader = """
vec4 process(vec4 color, ivec2 pixel)
{
    return color;
}
"""
        
        metadata = parse_metadata(shader)
        
        self.assertEqual(metadata["name"], "Unnamed Shader")
        self.assertEqual(len(metadata["inputs"]), 0)
        self.assertEqual(len(metadata["uniforms"]), 0)

    def test_empty_shader(self):
        """Test parsing an empty shader."""
        shader = ""
        
        metadata = parse_metadata(shader)
        
        self.assertEqual(metadata["name"], "Unnamed Shader")
        self.assertEqual(len(metadata["inputs"]), 0)
        self.assertEqual(len(metadata["uniforms"]), 0)


if __name__ == '__main__':
    unittest.main()