"""
GLSL Shader Compiler

Compiles GLSL shaders to SPIR-V bytecode with validation.
"""

import os
import re
import hashlib
import logging
import shutil
import subprocess
import tempfile
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)


class ShaderCompiler:
    """
    Compiles GLSL shaders to SPIR-V bytecode with validation.
    
    This class handles:
    - GLSL to SPIR-V compilation
    - SPIR-V validation
    - Shader caching
    - Error reporting
    """

    def __init__(self):
        self.cache = {}
        self.glslc_path = self._find_glslc()
        self.compiler_available = self.glslc_path is not None

    def _find_glslc(self) -> Optional[str]:
        """Locate the Vulkan SDK glslc compiler."""
        path = shutil.which("glslc")
        if path:
            return path
        # Common Windows Vulkan SDK install
        sdk_root = os.environ.get("VULKAN_SDK")
        if sdk_root:
            candidate = os.path.join(sdk_root, "Bin", "glslc.exe")
            if os.path.isfile(candidate):
                return candidate
        return None

    def compile(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Compile GLSL shader to SPIR-V bytecode.
        
        Args:
            source (str): GLSL source code
            metadata (dict, optional): Shader metadata
            
        Returns:
            bytes: Compiled SPIR-V bytecode
            
        Raises:
            Exception: If compilation fails
        """
        # Create cache key
        cache_key = self._get_cache_key(source, metadata)
        
        if cache_key in self.cache:
            logger.debug(f"Using cached SPIR-V for shader {cache_key[:8]}...")
            return self.cache[cache_key]
        
        try:
            if not self.compiler_available:
                raise Exception(
                    "glslc not found. Install the Vulkan SDK and ensure glslc is on PATH."
                )
            spirv = self._compile_with_glslc(source)
            
            # Cache result
            self.cache[cache_key] = spirv
            
            return spirv
            
        except Exception as e:
            raise Exception(f"Shader compilation failed: {e}")

    def _strip_metadata_comment(self, source: str) -> str:
        """Remove the leading metadata block comment before compiling."""
        return re.sub(r'^\s*/\*[\s\S]*?\*/', '', source, count=1).lstrip()

    def _compile_with_glslc(self, source: str) -> bytes:
        """
        Compile GLSL compute shader with system glslc.
        
        Args:
            source (str): GLSL source code
            
        Returns:
            bytes: Compiled SPIR-V bytecode
        """
        glsl_source = self._strip_metadata_comment(source)
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "shader.comp")
            out_path = os.path.join(tmp, "shader.spv")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(glsl_source)

            result = subprocess.run(
                [self.glslc_path, "-fshader-stage=compute", src_path, "-o", out_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "unknown glslc error").strip()
                raise Exception(err)

            with open(out_path, "rb") as f:
                spirv = f.read()

            logger.debug(f"Successfully compiled shader to {len(spirv)} bytes SPIR-V")
            return spirv

    def validate_spirv(self, spirv_binary: bytes) -> bool:
        """
        Validate SPIR-V bytecode.
        
        Args:
            spirv_binary (bytes): SPIR-V bytecode
            
        Returns:
            bool: True if valid
        """
        try:
            # In a real implementation, we would validate the SPIR-V
            # For now we'll just check basic structure
            if not spirv_binary or len(spirv_binary) < 4:
                return False
                
            # Check for SPIR-V magic number (0x07230203)
            magic_number = int.from_bytes(spirv_binary[:4], byteorder='little')
            expected_magic = 0x07230203
            
            if magic_number != expected_magic:
                logger.warning(f"Invalid SPIR-V magic number: {magic_number}")
                return False
                
            logger.debug("SPIR-V validation passed")
            return True
        except Exception as e:
            logger.error(f"SPIR-V validation failed: {e}")
            return False

    def _get_cache_key(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for the shader.
        
        Args:
            source (str): Shader source code
            metadata (dict, optional): Shader metadata
            
        Returns:
            str: Cache key as hex string
        """
        # Combine source and metadata for comprehensive caching
        combined = source + str(metadata or {})
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get compiler diagnostics.
        
        Returns:
            dict: Diagnostics information
        """
        return {
            "compiler_available": self.compiler_available,
            "glslc_path": self.glslc_path,
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys())[:5]  # Show first 5 keys
        }


if __name__ == "__main__":
    # Test compiler
    compiler = ShaderCompiler()
    
    test_shader = """
/*
@name Test Shader
@description A test shader for compilation
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
    
    print("Testing Shader Compiler...")
    try:
        # Test compilation
        spirv = compiler.compile(test_shader)
        print(f"✓ Compiled shader to {len(spirv)} bytes SPIR-V")
        
        # Test validation
        is_valid = compiler.validate_spirv(spirv)
        print(f"✓ SPIR-V validation: {'Passed' if is_valid else 'Failed'}")
        
        print("Compiler test completed successfully")
    except Exception as e:
        print(f"✗ Compiler test failed: {e}")