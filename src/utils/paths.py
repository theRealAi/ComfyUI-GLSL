"""
Shader Path Utilities

Handles discovery, sanitization, and resolution of shader files.
"""

import os
import re
from typing import List


def _package_root() -> str:
    """Return ComfyUI-GLSL package root (parent of src/)."""
    # paths.py lives at src/utils/paths.py -> go up three levels
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_shader_directories() -> List[str]:
    """
    Discover all shader directories in the package.
    
    Returns:
        list: List of absolute paths to shader directories
    """
    shaders_dir = os.path.join(_package_root(), 'shaders')

    # Define standard shader directories
    dirs = [
        os.path.join(shaders_dir, 'examples'),
        os.path.join(shaders_dir, 'production'),
        os.path.join(shaders_dir, 'user')
    ]

    # Filter only existing directories
    return [d for d in dirs if os.path.exists(d)]


def find_glsl_files(directory: str) -> List[str]:
    """
    Find all .glsl files recursively in a directory.
    
    Args:
        directory (str): Absolute path to directory
        
    Returns:
        list: List of absolute paths to .glsl files
    """
    glsl_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.glsl'):
                glsl_files.append(os.path.join(root, file))
                
    return glsl_files


def sanitize_shader_name(name: str) -> str:
    """
    Sanitize shader name to be filesystem-safe.
    
    Args:
        name (str): Raw shader name
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[^\w\-\.]', '_', name)
    
    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip('. _')
    
    # Limit length (filesystem-safe limit)
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
        
    return sanitized


def resolve_shader_path(shader_id: str) -> str:
    """
    Resolve a shader ID to an absolute path.
    
    Args:
        shader_id (str): Shader path relative to shaders/ directory
        
    Returns:
        str: Absolute path to shader file
        
    Raises:
        ValueError: If shader path is invalid or unsafe
    """
    shaders_dir = os.path.join(_package_root(), 'shaders')

    # Normalize path
    normalized = os.path.normpath(shader_id)

    # Prevent directory traversal
    if '..' in normalized:
        raise ValueError("Invalid shader path: directory traversal detected")

    # Check that it's within the shaders directory
    full_path = os.path.join(shaders_dir, normalized)

    # Ensure file exists
    if not os.path.exists(full_path):
        raise ValueError(f"Shader file does not exist: {full_path}")

    return full_path


def create_shader_directories():
    """
    Create standard shader directories if they don't exist.
    """
    shaders_dir = os.path.join(_package_root(), 'shaders')
    for name in ('examples', 'production', 'user'):
        os.makedirs(os.path.join(shaders_dir, name), exist_ok=True)


if __name__ == "__main__":
    # Test path utilities
    print("Discovering shader directories...")
    dirs = discover_shader_directories()
    print("Found directories:", dirs)
    
    print("\nFinding GLSL files...")
    files = find_glsl_files(dirs[0] if dirs else '.')
    print("Found files:", files[:5], "..." if len(files) > 5 else "")
    
    print("\nSanitizing names...")
    test_names = ["My Shader", "Shader/With/Slashes", "Shader:with:colons"]
    for name in test_names:
        sanitized = sanitize_shader_name(name)
        print(f"'{name}' -> '{sanitized}'")