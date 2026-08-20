"""
GLSL Runtime Logging

Provides GPU-resident logging layer for diagnostics and debugging.
"""

import os
import sys
from typing import Dict, Any


class GLSLLogger:
    """Logger for GLSL runtime operations."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.log_file = None
        
        # Try to get log file from environment
        log_path = os.environ.get('COMFYUI_GLSL_LOG')
        if log_path:
            try:
                self.log_file = open(log_path, 'w')
            except Exception as e:
                print(f"Failed to open log file {log_path}: {e}")
    
    def debug(self, message: str, *args):
        """Log a debug message."""
        if self.debug:
            formatted = f"[GLSL][DEBUG] {message % args if args else message}"
            print(formatted)
            if self.log_file:
                self.log_file.write(formatted + '\n')
                self.log_file.flush()
    
    def info(self, message: str, *args):
        """Log an info message."""
        formatted = f"[GLSL][INFO] {message % args if args else message}"
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            self.log_file.flush()
    
    def warning(self, message: str, *args):
        """Log a warning message."""
        formatted = f"[GLSL][WARN] {message % args if args else message}"
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            self.log_file.flush()
    
    def error(self, message: str, *args):
        """Log an error message."""
        formatted = f"[GLSL][ERROR] {message % args if args else message}"
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            self.log_file.flush()
    
    def flush(self):
        """Flush log file buffer."""
        if self.log_file:
            self.log_file.flush()


# Global logger instance
logger = GLSLLogger()

# Convenience functions for common use cases
def get_logger() -> GLSLLogger:
    """Get the global logger instance."""
    return logger

def debug(message: str, *args):
    """Convenience debug function."""
    logger.debug(message, *args)

def info(message: str, *args):
    """Convenience info function."""
    logger.info(message, *args)

def warning(message: str, *args):
    """Convenience warning function."""
    logger.warning(message, *args)

def error(message: str, *args):
    """Convenience error function."""
    logger.error(message, *args)


if __name__ == "__main__":
    # Test logging
    logger = get_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")