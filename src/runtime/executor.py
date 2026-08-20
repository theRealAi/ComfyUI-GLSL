"""
GLSL Runtime Executor

Main GLSL runtime interface for shader compilation and execution.
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional
from .backend.vulkan import VulkanBackend
from ..shader.compiler import ShaderCompiler

# Setup logging
logger = logging.getLogger(__name__)


class GLSLRuntime:
    """
    Main GLSL runtime interface.
    
    This class provides a unified API for:
    - Shader compilation (GLSL -> SPIR-V)
    - Pipeline creation and caching
    - GPU execution
    - Resource management
    
    All nodes share this runtime instance.
    """

    def __init__(self):
        self.backend = None
        self.shader_cache = {}
        self.pipeline_cache = {}
        self.compiler = ShaderCompiler()
        
        # Initialize backend lazily on first use
        self._initialized = False

    def _initialize_backend(self):
        """Initialize the GPU backend (Vulkan) if not already done."""
        if not self._initialized:
            try:
                self.backend = VulkanBackend()
                self.backend.initialize()
                self._initialized = True
                logger.info("Vulkan backend initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Vulkan backend: {e}")
                # Continue without Vulkan, but execution will fail
                self.backend = None

    def compile(self, source: str, metadata: Dict[str, Any]) -> bytes:
        """
        Compile GLSL shader to SPIR-V bytecode.
        
        Args:
            source (str): GLSL source code
            metadata (dict): Shader metadata
            
        Returns:
            bytes: Compiled SPIR-V bytecode
            
        Raises:
            Exception: If compilation fails
        """
        # Lazy backend initialization
        self._initialize_backend()
        
        try:
            # Use the dedicated shader compiler
            spirv = self.compiler.compile(source, metadata)
            
            # Cache SPIR-V in our own cache as well for consistency with the spec
            shader_key = self._get_shader_hash(source, metadata)
            self.shader_cache[shader_key] = spirv
            
            logger.debug(f"Compiled shader to {len(spirv)} bytes SPIR-V")
            return spirv
        except Exception as e:
            raise Exception(f"Shader compilation failed: {e}")

    def validate(self, shader_source: str, metadata: Dict[str, Any]) -> bool:
        """
        Validate shader for correctness and compatibility.
        
        Args:
            shader_source (str): GLSL source code
            metadata (dict): Shader metadata
            
        Returns:
            bool: True if valid
            
        Raises:
            Exception: If validation fails
        """
        from ..shader.validator import validate_shader
        
        try:
            validate_shader(shader_source, metadata)
            logger.debug("Shader validation passed")
            return True
        except Exception as e:
            raise Exception(f"Shader validation failed: {e}")

    def execute(
        self,
        shader_source: str,
        metadata: Dict[str, Any],
        image_input: Any,  # Placeholder for PyTorch tensor or GPU resource
        mask: Optional[Any] = None,
        mode: str = "simple"
    ) -> Any:
        """
        Execute a GLSL shader on an input image.
        
        Args:
            shader_source (str): GLSL source code
            metadata (dict): Shader metadata
            image_input (Any): Input image tensor or GPU resource
            mask (Optional[Any]): Optional mask for blending
            mode (str): "simple" or "advanced"
            
        Returns:
            Any: Processed image
            
        Raises:
            Exception: If execution fails
        """
        # Lazy backend initialization
        self._initialize_backend()
        
        if not self._initialized or self.backend is None:
            logger.warning("Vulkan backend not initialized, returning input image")
            return image_input
        
        try:
            # Compile shader using our dedicated compiler
            spirv = self.compile(shader_source, metadata)
            
            # Validate shader
            self.validate(shader_source, metadata)
            
            # In a full implementation, we would:
            # 1. Create or retrieve pipeline from cache
            # 2. Prepare GPU resources (textures, buffers)
            # 3. Upload uniforms
            # 4. Dispatch compute shader
            # 5. Synchronize and return result
            
            # Placeholder for execution
            logger.debug(f"Executing shader in {mode} mode")
            logger.debug(f"Shader hash: {self._get_shader_hash(shader_source, metadata)}")
            
            return image_input  # Return input as placeholder
        except Exception as e:
            raise Exception(f"Shader execution failed: {e}")

    def save_shader(self, shader_source: str, path: str) -> None:
        """
        Save inline shader to the filesystem.
        
        Args:
            shader_source (str): GLSL source code with metadata
            path (str): Destination path within shaders/ directory
        """
        from ..utils.paths import sanitize_shader_name
        
        # Sanitize filename
        name = sanitize_shader_name(os.path.basename(path))
        if not name.endswith('.glsl'):
            name += '.glsl'
            
        full_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shaders', path, name)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write shader to file
        with open(full_path, 'w') as f:
            f.write(shader_source)
            
        logger.info(f"Saved shader to {full_path}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get system diagnostics for the GLSL runtime.
        
        Returns:
            dict: Diagnostics information
        """
        self._initialize_backend()
        
        diagnostics = {
            "backend": "Vulkan" if self.backend else "None",
            "initialized": self._initialized,
            "shader_cache_size": len(self.shader_cache),
            "pipeline_cache_size": len(self.pipeline_cache),
            "compiler": self.compiler.get_diagnostics()
        }
        
        if self.backend:
            try:
                gpu_info = self.backend.get_gpu_info()
                diagnostics.update({
                    "gpu_info": gpu_info,
                    "spirv_support": True,  # Placeholder
                    "glsl_compiler": "Available"  # Placeholder
                })
            except Exception as e:
                diagnostics["error"] = str(e)
        
        return diagnostics

    def _get_shader_hash(self, source: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a SHA-256 hash of the shader for caching.
        
        Args:
            source (str): Shader source code
            metadata (dict): Shader metadata
            
        Returns:
            str: SHA-256 hash as hex string
        """
        # Combine source, metadata, and version info
        combined = source + str(metadata)
        shader_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        logger.debug(f"Generated shader hash: {shader_hash[:16]}...")
        return shader_hash


if __name__ == "__main__":
    # Test executor
    runtime = GLSLRuntime()
    
    test_shader = """
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
    
    print("Testing GLSL Runtime...")
    try:
        # Test parsing metadata
        from ..shader.parser import parse_metadata
        metadata = parse_metadata(test_shader)
        print("Metadata:", metadata)
        
        # Test compilation hash
        shader_hash = runtime._get_shader_hash(test_shader, metadata)
        print("Shader hash:", shader_hash[:16] + "...")
        
        # Test compiler
        spirv = runtime.compiler.compile(test_shader, metadata)
        print(f"✓ Compiled shader to {len(spirv)} bytes SPIR-V")
        
        print("✓ Runtime initialized successfully")
    except Exception as e:
        print(f"✗ Runtime test failed: {e}")
