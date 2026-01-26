"""
Centralized logging system for the application.
Provides logging to both file and UI components.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from logging.handlers import RotatingFileHandler


class AppLogger:
    """
    Singleton application logger with file and UI output.
    """
    _instance: Optional['AppLogger'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._log_callbacks: list[Callable[[str, str], None]] = []
        self._logger = logging.getLogger('ForgeApp')
        self._logger.setLevel(logging.DEBUG)
        
        # Default log directory
        self._log_dir = Path.cwd() / "logs"
        self._log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        self._setup_file_handler()
        
        # Setup console handler for development
        self._setup_console_handler()
        
        self._initialized = True
    
    def _setup_file_handler(self):
        """Setup rotating file handler"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self._log_dir / f"Forge_{timestamp}.log"
        
        # Rotating file handler - 10MB max, keep 5 backups
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Format: [2025-01-26 14:30:45] INFO: Message
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self._logger.addHandler(file_handler)
        self._file_handler = file_handler
    
    def _setup_console_handler(self):
        """Setup console handler for stdout"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(console_handler)
    
    def set_log_directory(self, directory: Path):
        """
        Change the log directory.
        
        Args:
            directory: New directory for log files
        """
        self._log_dir = Path(directory)
        self._log_dir.mkdir(exist_ok=True)
        
        # Remove old file handler
        self._logger.removeHandler(self._file_handler)
        
        # Setup new file handler
        self._setup_file_handler()
        
        self.info(f"Log directory changed to: {self._log_dir}")
    
    def get_log_directory(self) -> Path:
        """Get current log directory"""
        return self._log_dir
    
    def register_callback(self, callback: Callable[[str, str], None]):
        """
        Register a callback for log messages.
        
        Args:
            callback: Function(level, message) to call on each log message
        """
        # Remove if already registered to avoid duplicates
        if callback in self._log_callbacks:
            self._log_callbacks.remove(callback)
        self._log_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[str, str], None]):
        """
        Unregister a log callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._log_callbacks:
            self._log_callbacks.remove(callback)
    
    def _notify_callbacks(self, level: str, message: str):
        """Notify all registered callbacks"""
        for callback in self._log_callbacks:
            try:
                callback(level, message)
            except Exception as e:
                # Don't let callback errors break logging
                print(f"Error in log callback: {e}")
    
    def debug(self, message: str):
        """Log debug message"""
        self._logger.debug(message)
        self._notify_callbacks('DEBUG', message)
    
    def info(self, message: str):
        """Log info message"""
        self._logger.info(message)
        self._notify_callbacks('INFO', message)
    
    def warning(self, message: str):
        """Log warning message"""
        self._logger.warning(message)
        self._notify_callbacks('WARNING', message)
    
    def error(self, message: str):
        """Log error message"""
        self._logger.error(message)
        self._notify_callbacks('ERROR', message)
    
    def critical(self, message: str):
        """Log critical message"""
        self._logger.critical(message)
        self._notify_callbacks('CRITICAL', message)
    
    def exception(self, message: str):
        """Log exception with traceback"""
        self._logger.exception(message)
        self._notify_callbacks('ERROR', message)


# Global logger instance
_app_logger: Optional[AppLogger] = None


def get_logger() -> AppLogger:
    """Get the global application logger"""
    global _app_logger
    if _app_logger is None:
        _app_logger = AppLogger()
    return _app_logger


# Convenience functions for easy access
def debug(message: str):
    """Log debug message"""
    get_logger().debug(message)


def info(message: str):
    """Log info message"""
    get_logger().info(message)


def warning(message: str):
    """Log warning message"""
    get_logger().warning(message)


def error(message: str):
    """Log error message"""
    get_logger().error(message)


def critical(message: str):
    """Log critical message"""
    get_logger().critical(message)


def exception(message: str):
    """Log exception with traceback"""
    get_logger().exception(message)
