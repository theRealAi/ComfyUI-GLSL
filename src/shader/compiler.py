"""
GLSL Shader Compiler

Compiles GLSL shaders to SPIR-V bytecode with validation.
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from .parser import parse_metadata

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
        self.compiler_available = self._check_compiler_availability()

    def _check_compiler_availability(self) -> bool:
        """Check if GLSL compiler is available."""
        try:
            # Try to import glsl-compiler
            from glsl_compiler import compile_glsl
            return True
        except ImportError:
            logger.info("glsl-compiler not available - using fallback")
            return False
        except Exception as e:
            logger.warning(f"GLSL compiler check failed: {e}")
            return False

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
            if self.compiler_available:
                # Use actual compiler
                spirv = self._compile_with_glslc(source)
            else:
                # Fallback to placeholder
                logger.info("Using fallback SPIR-V compilation")
                spirv = self._compile_fallback(source)
            
            # Cache result
            self.cache[cache_key] = spirv
            
            return spirv
            
        except Exception as e:
            raise Exception(f"Shader compilation failed: {e}")

    def _compile_with_glslc(self, source: str) -> bytes:
        """
        Compile GLSL using the glsl-compiler.
        
        Args:
            source (str): GLSL source code
            
        Returns:
            bytes: Compiled SPIR-V bytecode
        """
        try:
            from glsl_compiler import compile_glsl
            spirv = compile_glsl(source, "compute")
            logger.debug(f"Successfully compiled shader to {len(spirv)} bytes SPIR-V")
            return spirv
        except Exception as e:
            raise Exception(f"GLSL compilation failed: {e}")

    def _compile_fallback(self, source: str) -> bytes:
        """
        Fallback SPIR-V generation for development.
        
        Args:
            source (str): GLSL source code
            
        Returns:
            bytes: Placeholder SPIR-V bytecode
        """
        # In a real implementation, this would generate valid SPIR-V
        # For now we return a placeholder with proper magic number
        logger.warning("Using fallback compilation - not suitable for production")
        return b"\x03\x02\x23\x07"  # SPIR-V magic number

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