"""
Logging Configuration for Power Market Simulation

Konfiguriert strukturiertes Logging für alle Module.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguriert Logging für die Anwendung.

    Parameters:
    -----------
    level : str
        Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Pfad zur Log-Datei. Wenn None, wird nicht in Datei geloggt.
    console : bool
        Ob auf die Konsole geloggt werden soll
    format_string : str, optional
        Custom Format-String. Wenn None, wird Default verwendet.

    Returns:
    --------
    logging.Logger
        Konfigurierter Logger
    """
    # Root Logger
    logger = logging.getLogger("power_market_sim")
    logger.setLevel(getattr(logging, level.upper()))

    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Remove existing handlers
    logger.handlers.clear()

    # Console Handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (max 10MB, keep 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger für ein spezifisches Modul zurück.

    Parameters:
    -----------
    name : str
        Name des Moduls (z.B. 'network.builder')

    Returns:
    --------
    logging.Logger
        Logger für das Modul
    """
    return logging.getLogger(f"power_market_sim.{name}")


def log_memory_usage(logger: logging.Logger, prefix: str = ""):
    """
    Loggt aktuelle Speichernutzung.

    Parameters:
    -----------
    logger : logging.Logger
        Logger-Instanz
    prefix : str
        Prefix für Log-Nachricht
    """
    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1e9
        logger.info(f"{prefix}Current memory usage: {memory_gb:.2f} GB")
    except ImportError:
        logger.warning("psutil not installed - cannot log memory usage")


def log_performance(logger: logging.Logger, operation: str, duration: float):
    """
    Loggt Performance-Metriken.

    Parameters:
    -----------
    logger : logging.Logger
        Logger-Instanz
    operation : str
        Name der Operation
    duration : float
        Dauer in Sekunden
    """
    logger.info(f"Performance: {operation} completed in {duration:.2f} seconds")


# Convenience function for common logging patterns
class LogContext:
    """
    Context Manager für Logging von Operationen mit automatischer
    Performance- und Memory-Messung.

    Example:
    --------
    with LogContext(logger, "Building network"):
        # Build network code
        pass
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        log_memory: bool = False
    ):
        self.logger = logger
        self.operation = operation
        self.log_memory = log_memory
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation}")
        if self.log_memory:
            log_memory_usage(self.logger, f"[{self.operation}] ")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation} (duration: {duration:.2f}s)"
            )
            if self.log_memory:
                log_memory_usage(self.logger, f"[{self.operation}] ")
        else:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.2f}s - {exc_val}"
            )

        return False  # Don't suppress exceptions


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(
        level="INFO",
        log_file="logs/test.log",
        console=True
    )

    logger.info("Logging system initialized")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")

    # Using LogContext
    with LogContext(logger, "Test operation", log_memory=True):
        import time
        time.sleep(1)
        logger.info("Doing some work...")
