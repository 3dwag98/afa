"""
Centralized Logging Utility for AFA Pipeline.

This module provides a unified logging framework with:
- ISO format timestamps
- Module name identification
- Unique run_id (UUID) for DAG/script execution tracking
- Worker/thread ID for parallel process identification
- Dual output to console and rotating file handler
- Proper exception handling with stack trace logging
"""

import logging
import os
import sys
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class ContextualLogger:
    """
    A wrapper around Python's standard logger that adds contextual identifiers
    to every log entry.
    
    Attributes:
        logger: The underlying Python logger instance.
        run_id: Unique UUID for the current execution run.
        module_name: Name of the module using this logger.
        worker_id: Identifier for parallel workers (if applicable).
    """
    
    def __init__(
        self,
        module_name: str,
        log_file: str = "logs/afa_pipeline.log",
        run_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Initialize the contextual logger.
        
        Args:
            module_name: Name of the module (e.g., 'orchestrator', 'data_ingestion').
            log_file: Path to the log file.
            run_id: Unique run identifier (UUID). If None, generates a new one.
            worker_id: Worker/thread identifier for parallel processes.
            level: Logging level (default: INFO).
        """
        self.module_name = module_name
        self.run_id = run_id or str(uuid.uuid4())
        self.worker_id = worker_id or "main"
        self.logger = self._setup_logger(log_file, level)
    
    def _setup_logger(self, log_file: str, level: int) -> logging.Logger:
        """
        Setup logger with both console and rotating file handlers.
        
        Args:
            log_file: Path to the log file.
            level: Logging level.
            
        Returns:
            Configured logger instance.
        """
        logger_name = f"afa.{self.module_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create formatter with contextual identifiers
        # Format: [ISO_TIMESTAMP] [MODULE] [RUN_ID] [WORKER_ID] LEVEL - MESSAGE
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(module_name)s | %(run_id)s | %(worker_id)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        
        # Rotating file handler (10MB max, 5 backup files)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Add filters to inject contextual info into log records
        class ContextFilter(logging.Filter):
            def __init__(self, module_name: str, run_id: str, worker_id: str):
                super().__init__()
                self.module_name = module_name
                self.run_id = run_id
                self.worker_id = worker_id
            
            def filter(self, record):
                record.module_name = self.module_name
                record.run_id = self.run_id
                record.worker_id = self.worker_id
                return True
        
        context_filter = ContextFilter(self.module_name, self.run_id, self.worker_id)
        file_handler.addFilter(context_filter)
        console_handler.addFilter(context_filter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _log_with_context(self, level: int, msg: str, **kwargs):
        """Log a message with automatic exception logging on ERROR level."""
        if level == logging.ERROR and 'exc_info' not in kwargs:
            kwargs['exc_info'] = True
        self.logger.log(level, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log DEBUG level message."""
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log INFO level message."""
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log WARNING level message."""
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log ERROR level message with stack trace."""
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log CRITICAL level message with stack trace."""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with full stack trace."""
        kwargs['exc_info'] = True
        self.logger.error(msg, **kwargs)
    
    def set_worker_id(self, worker_id: str):
        """
        Update the worker_id for this logger (useful for parallel workers).
        
        Args:
            worker_id: New worker identifier.
        """
        self.worker_id = worker_id
        # Update filters on handlers
        for handler in self.logger.handlers:
            for filter_obj in handler.filters:
                if isinstance(filter_obj, logging.Filter):
                    if hasattr(filter_obj, 'worker_id'):
                        filter_obj.worker_id = worker_id
    
    def get_run_id(self) -> str:
        """Return the current run_id."""
        return self.run_id


def get_logger(
    module_name: str,
    log_file: str = "logs/afa_pipeline.log",
    run_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    level: int = logging.INFO
) -> ContextualLogger:
    """
    Factory function to create a ContextualLogger instance.
    
    Args:
        module_name: Name of the module.
        log_file: Path to the log file.
        run_id: Unique run identifier (UUID).
        worker_id: Worker/thread identifier.
        level: Logging level.
        
    Returns:
        ContextualLogger instance.
    """
    return ContextualLogger(
        module_name=module_name,
        log_file=log_file,
        run_id=run_id,
        worker_id=worker_id,
        level=level
    )


class LogContext:
    """
    Context manager for temporary worker_id changes during parallel operations.
    
    Usage:
        with LogContext(logger, worker_id="worker_1"):
            logger.info("Processing in worker 1")
    """
    
    def __init__(self, contextual_logger: ContextualLogger, worker_id: str):
        """
        Initialize the log context.
        
        Args:
            contextual_logger: The ContextualLogger instance.
            worker_id: Temporary worker ID to use within the context.
        """
        self.logger = contextual_logger
        self.worker_id = worker_id
        self.original_worker_id = None
    
    def __enter__(self):
        """Save original worker_id and set new one."""
        self.original_worker_id = self.logger.worker_id
        self.logger.set_worker_id(self.worker_id)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original worker_id."""
        self.logger.set_worker_id(self.original_worker_id)
        # Log exception if it occurred
        if exc_type is not None:
            self.logger.exception(f"Exception in worker {self.worker_id}: {exc_val}")
        return False  # Don't suppress exceptions


# Global default log file path
DEFAULT_LOG_FILE = "logs/afa_pipeline.log"


def setup_global_logger(
    module_name: str,
    run_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    level: int = logging.INFO
) -> ContextualLogger:
    """
    Setup a global logger for a module with default settings.
    
    This is a convenience function for modules that need a quick logger setup.
    
    Args:
        module_name: Name of the module.
        run_id: Unique run identifier.
        worker_id: Worker identifier.
        level: Logging level.
        
    Returns:
        ContextualLogger instance.
    """
    return get_logger(
        module_name=module_name,
        log_file=DEFAULT_LOG_FILE,
        run_id=run_id,
        worker_id=worker_id,
        level=level
    )
