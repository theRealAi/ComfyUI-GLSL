"""
GLSL Shader Validator

Validates GLSL shaders for:
- Syntax correctness
- SPIR-V validity
- Supported GLSL version
- Descriptor bindings
- Image formats
- Push constants
- Workgroup dimensions
- Metadata compatibility
"""

from typing import Dict, Any, List
import re


class ShaderValidationError(Exception):
    """Custom exception for shader validation errors."""
    pass


def validate_shader(source: str, metadata: Dict[str, Any]) -> bool:
    """
    Validate a GLSL shader.
    
    Args:
        source (str): Full GLSL source code
        metadata (dict): Parsed metadata
        
    Returns:
        bool: True if valid, raises exception otherwise
        
    Raises:
        ShaderValidationError: If validation fails
    """
    # Check for basic syntax
    if not source or not isinstance(source, str):
        raise ShaderValidationError("Shader source is empty or invalid")
    
    # Check for required GLSL version
    if '#version' not in source:
        raise ShaderValidationError("GLSL shader must specify a version (e.g., #version 450)")
    
    # Validate metadata compatibility with shader content
    validate_metadata_compatibility(source, metadata)
    
    return True


def validate_metadata_compatibility(source: str, metadata: Dict[str, Any]) -> None:
    """
    Check that metadata matches the actual shader implementation.
    
    Args:
        source (str): GLSL source code
        metadata (dict): Parsed metadata
        
    Raises:
        ShaderValidationError: If metadata is incompatible with shader
    """
    # This is a placeholder for more complex validation logic
    # In a full implementation, we would:
    # 1. Parse the GLSL AST to find actual uniforms
    # 2. Check if all @uniforms are referenced in the shader
    # 3. Validate that @input bindings match what's used
    # 4. Verify push constants and workgroup sizes
    
    # Example: Check for unused uniforms (simplified)
    declared_uniforms = [u['name'] for u in metadata.get('uniforms', [])]
    
    # Look for usage of each uniform in the source
    for uniform_name in declared_uniforms:
        # Simple check - if it's not found, warn but don't fail
        if uniform_name not in source:
            print(f"Warning: Uniform '{uniform_name}' declared in metadata but not found in shader")


def validate_spirv(spirv_binary: bytes) -> bool:
    """
    Validate SPIR-V binary.
    
    Args:
        spirv_binary (bytes): Compiled SPIR-V bytecode
        
    Returns:
        bool: True if valid SPIR-V
        
    Raises:
        ShaderValidationError: If SPIR-V is invalid
    """
    # In a real implementation, we'd use a SPIR-V validator here
    # For now, this is a placeholder
    if not spirv_binary or not isinstance(spirv_binary, bytes):
        raise ShaderValidationError("Invalid SPIR-V binary")
    
    return True


def validate_bindings(source: str, bindings: List[Dict[str, Any]]) -> None:
    """
    Validate descriptor bindings in shader.
    
    Args:
        source (str): GLSL source code
        bindings (list): List of binding specifications
        
    Raises:
        ShaderValidationError: If bindings are invalid
    """
    # Placeholder for binding validation logic
    pass


def validate_workgroup_size(source: str, size: tuple) -> None:
    """
    Validate workgroup dimensions.
    
    Args:
        source (str): GLSL source code
        size (tuple): Workgroup size (x, y, z)
        
    Raises:
        ShaderValidationError: If workgroup size is invalid
    """
    # Placeholder for workgroup validation logic
    pass


def test_validator():
    """Test the validator with example inputs."""
    
    from .parser import parse_metadata
    
    # Test 1: Valid shader
    valid_shader = """
/*
@name Test Shader
@description A test shader
@version 1.0.0

@input image IMAGE

@uniform exposure float 0.0 min=-10 max=10 step=0.01
*/
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    // Test shader body
}
"""
    
    metadata = parse_metadata(valid_shader)
    try:
        validate_shader(valid_shader, metadata)
        print("✓ Valid shader passed validation")
    except ShaderValidationError as e:
        print(f"✗ Valid shader failed: {e}")


if __name__ == "__main__":
    test_validator()